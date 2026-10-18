"""M3 LoRA training on Tinker (Qwen3.5-4B, rank 16). Trains on data/splits/train_m3.jsonl only.

    uv run python -I scripts/train_m3.py probe            # 20 steps x several LRs, pick by dev loss
    uv run python -I scripts/train_m3.py train --lr 2e-4 --epochs 1

Prompt and chat template are IDENTICAL to the baselines (contains.prompt.build_prompt,
enable_thinking=False); the target is the 9 'allergen: yes|no|unsure' lines + <|im_end|>.
Dev loss uses 150 DEV rows (gated, plain yes/no verdicts) + 150 held-out abstain-style rows are NOT used;
dev rows only. Never touches test / frozen traps / traps_dev.
Cost is estimated from tokens x $0.737 / 1M (Tinker returns no $ field).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import load_env, service_client
from contains.labels import ALLERGENS
from contains.prompt import build_prompt

MODEL = "Qwen/Qwen3.5-4B"
RANK = 16
PRICE = 0.737 / 1e6
BATCH = 16
DEV_LOSS_N = 150
OUT = ROOT / "data" / "eval"
PROBE_LRS = (3e-5,) if "--low" in sys.argv else (1e-4, 3e-4, 1e-3)


def answer(verdicts: dict) -> str:
    return "\n".join(f"{a}: {verdicts[a]}" for a in ALLERGENS)


def make_datum(tok, text, verdicts):
    import tinker
    p = tok.apply_chat_template([{"role": "user", "content": build_prompt(text)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    pi = tok.encode(p, add_special_tokens=False)
    ci = tok.encode(answer(verdicts) + "<|im_end|>", add_special_tokens=False)
    toks, w = pi + ci, [0.0] * len(pi) + [1.0] * len(ci)
    d = tinker.Datum(model_input=tinker.ModelInput.from_ints(toks[:-1]),
                     loss_fn_inputs={"target_tokens": toks[1:], "weights": w[1:]})
    return d, len(toks), len(ci)


def load_train():
    return [json.loads(ln) for ln in (ROOT / "data/splits/train_m3.jsonl").read_text(encoding="utf-8").splitlines()]


def load_dev_loss_rows():
    rows = [json.loads(ln) for ln in (ROOT / "data/splits/dev.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(77)
    rows = rng.sample(rows, DEV_LOSS_N)
    return [(r["ingredients_text"], {a: ("yes" if a in r["labels"] else "no") for a in ALLERGENS}) for r in rows]


def loss_of(out, n_comp):
    m = getattr(out, "metrics", {}) or {}
    return float(m["loss:sum"]) / n_comp


def dev_loss(tc, tok, dev):
    ds = [make_datum(tok, t, v) for t, v in dev]
    out = tc.forward([d for d, _, _ in ds], "cross_entropy").result()
    return loss_of(out, sum(c for _, _, c in ds)), sum(n for _, n, _ in ds)


def lr_at(base, step, total):  # linear decay to 10%
    return base * (1 - 0.9 * step / max(total, 1))


def run(tc, tok, data, lr, steps, dev, log):
    import tinker
    tokens = 0
    for s in range(steps):
        batch = data[s * BATCH:(s + 1) * BATCH]
        ds = [make_datum(tok, r["ingredients_text"], r["verdicts"]) for r in batch]
        fb = tc.forward_backward([d for d, _, _ in ds], "cross_entropy").result()
        tc.optim_step(adam_params=tinker.AdamParams(learning_rate=lr_at(lr, s, steps))).result()
        tokens += sum(n for _, n, _ in ds)
        tl = loss_of(fb, sum(c for _, _, c in ds))
        log.append({"step": s + 1, "train_loss": tl, "lr": lr_at(lr, s, steps)})
        if (s + 1) % 25 == 0 or s == 0:
            print(f"  step {s+1}/{steps} train_loss {tl:.4f}", flush=True)
    return tokens


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "train"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--low", action="store_true")
    ap.add_argument("--name", default="contains-m3")
    a = ap.parse_args()
    sc = service_client()
    train, dev = load_train(), load_dev_loss_rows()
    random.Random(5).shuffle(train)

    if a.mode == "probe":
        res, spent = {}, 0
        l0 = None
        for lr in PROBE_LRS:
            tc = sc.create_lora_training_client(base_model=MODEL, rank=RANK)
            tok = tc.get_tokenizer()
            if l0 is None:
                l0, t0 = dev_loss(tc, tok, dev)
                spent += t0
            tk = run(tc, tok, train, lr, 20, dev, [])
            l1, t1 = dev_loss(tc, tok, dev)
            spent += tk + t1
            res[lr] = l1
            print(f"LR {lr:g}: dev loss {l0:.4f} -> {l1:.4f}  (probe tokens so far {spent:,}, ~${spent*PRICE:.3f})", flush=True)
        best = min(res, key=res.get)
        print(f"PICKED LR {best:g}")
        (OUT / ("m3_lr_probe_low.json" if a.low else "m3_lr_probe.json")).write_text(json.dumps({"dev_loss": {str(k): v for k, v in res.items()}, "picked": best,
                                                           "tokens": spent, "est_usd": spent * PRICE}, indent=1))
        return 0

    steps = int(len(train) * a.epochs / BATCH)
    tc = sc.create_lora_training_client(base_model=MODEL, rank=RANK)
    tok = tc.get_tokenizer()
    data = (train * 3)[: steps * BATCH]
    log, tokens = [], 0
    l0, t0 = dev_loss(tc, tok, dev)
    print(f"train {len(train)} rows, {steps} steps x {BATCH}, lr {a.lr:g}, dev loss before {l0:.4f}", flush=True)
    t_start = time.time()
    tokens = run(tc, tok, data, a.lr, steps, dev, log)
    l1, t1 = dev_loss(tc, tok, dev)
    cost = (tokens + t0 + t1) * PRICE
    print(f"dev loss after {l1:.4f}; train tokens {tokens:,}; est cost ${cost:.3f}; {time.time()-t_start:.0f}s", flush=True)
    path = tc.save_weights_for_sampler(name=a.name).result()
    print("sampler path:", getattr(path, "path", path))
    (OUT / "m3_train_log.json").write_text(json.dumps({"lr": a.lr, "steps": steps, "batch": BATCH, "tokens": tokens,
        "est_usd": cost, "dev_loss_before": l0, "dev_loss_after": l1, "sampler_path": str(getattr(path, "path", path)),
        "log": log}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

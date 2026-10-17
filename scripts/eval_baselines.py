"""M2 — zero-shot baselines on the DEV split.

One backend per invocation:
    uv run python scripts/eval_baselines.py rules   --full      # free, full dev
    uv run python scripts/eval_baselines.py rules   --subset    # free, 300-row subset
    uv run python scripts/eval_baselines.py gemma   --subset    # local Ollama
    uv run python scripts/eval_baselines.py qwen    --subset    # Tinker (paid)
    uv run python scripts/eval_baselines.py gptoss  --subset    # Tinker (paid)

Same prompt (contains.prompt) and same strict parser for all four. Unparseable
output scores as all-unsure. Writes a section to docs/RESULTS.md and a raw
predictions file under data/eval/. The 300-row subset is a deterministic seeded
sample of DEV (seed 300) — representative of the natural dev class balance so the
four models are compared on identical, real-distribution rows. Rare classes
(fish, crustacean_shellfish) have tiny support in 300, so their per-class numbers
are indicative only.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _smoke_common import load_env

from contains.labels import ALLERGENS
from contains.metrics import format_report, score
from contains.prompt import build_prompt, parse_response
from contains.rules import detect_naive

ROOT = Path(__file__).resolve().parent.parent
DEV = ROOT / "data" / "splits" / "dev.jsonl"
RESULTS = ROOT / "docs" / "RESULTS.md"
EVAL_DIR = ROOT / "data" / "eval"
SUBSET_N = 300
SUBSET_SEED = 300
MAX_TOKENS = 64

# Qwen3.5 defaults to thinking mode, which burns the 64-token budget on "Thinking Process:"
# and yields 100% unparseable output. Non-thinking = direct answer, matching the fine-tune.
TEMPLATE_KW = {"qwen": {"enable_thinking": False}, "gptoss": {"reasoning_effort": "low"}}
# gpt-oss (harmony format) reasons in an analysis channel before the answer; it needs a
# larger token budget, and only the `final` channel is parsed.
MAX_TOKENS_BY = {"gptoss": 512}
_FINAL = "<|channel|>final<|message|>"

MODEL_IDS = {"qwen": "Qwen/Qwen3.5-4B", "gptoss": "openai/gpt-oss-120b"}


def load_dev(subset: bool) -> list[dict]:
    rows = [json.loads(ln) for ln in DEV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if subset:
        rng = random.Random(SUBSET_SEED)
        rows = rng.sample(rows, min(SUBSET_N, len(rows)))
    return rows


def run_rules(rows):
    # BASELINE uses rules_naive only (direct names); rules_product is never a baseline.
    preds, lat = [], []
    for r in rows:
        t0 = time.perf_counter()
        hits = set(detect_naive(r["ingredients_text"]))
        lat.append((time.perf_counter() - t0) * 1000)
        preds.append({a: ("yes" if a in hits else "no") for a in ALLERGENS})
    return preds, lat, {"in_tok": 0, "out_tok": 0}


def run_gemma(rows):
    import os

    import httpx
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    preds, lat = [], []
    in_tok = out_tok = 0
    for i, r in enumerate(rows, 1):
        t0 = time.perf_counter()
        resp = httpx.post(f"{host}/api/generate", json={
            "model": "gemma3:4b", "prompt": build_prompt(r["ingredients_text"]),
            "stream": False, "options": {"num_predict": MAX_TOKENS, "temperature": 0.0},
        }, timeout=120.0)
        resp.raise_for_status()
        j = resp.json()
        lat.append((time.perf_counter() - t0) * 1000)
        in_tok += j.get("prompt_eval_count", 0)
        out_tok += j.get("eval_count", 0)
        pred, _ = parse_response(j.get("response", ""))
        preds.append(pred)
        if i % 50 == 0:
            print(f"  gemma {i}/{len(rows)}")
    return preds, lat, {"in_tok": in_tok, "out_tok": out_tok}


def run_tinker(rows, model_key):
    import tinker
    from _smoke_common import service_client
    sc = service_client()
    sampling = sc.create_sampling_client(base_model=MODEL_IDS[model_key])
    tok = sampling.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=MAX_TOKENS_BY.get(model_key, MAX_TOKENS), temperature=0.0)
    preds, lat = [], []
    in_tok = out_tok = 0
    for i, r in enumerate(rows, 1):
        text = tok.apply_chat_template(
            [{"role": "user", "content": build_prompt(r["ingredients_text"])}],
            tokenize=False, add_generation_prompt=True, **TEMPLATE_KW.get(model_key, {}))
        ids = tok.encode(text, add_special_tokens=False)
        t0 = time.perf_counter()
        resp = sampling.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1,
                               sampling_params=params).result()
        lat.append((time.perf_counter() - t0) * 1000)
        out_ids = list(resp.sequences[0].tokens)
        in_tok += len(ids)
        out_tok += len(out_ids)
        raw = tok.decode(out_ids)
        if _FINAL in raw:
            raw = raw.split(_FINAL, 1)[1]
        elif model_key == "gptoss":
            raw = ""  # never reached the final channel -> unparseable -> all-unsure
        pred, _ = parse_response(raw)
        preds.append(pred)
        if i % 50 == 0:
            print(f"  {model_key} {i}/{len(rows)}")
    return preds, lat, {"in_tok": in_tok, "out_tok": out_tok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backend", choices=["rules", "gemma", "qwen", "gptoss"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--full", action="store_true")
    g.add_argument("--subset", action="store_true")
    args = ap.parse_args()
    load_env()

    subset = args.subset or not args.full
    rows = load_dev(subset)
    scope = f"dev-subset-{len(rows)}" if subset else f"dev-full-{len(rows)}"
    golds = [set(r["labels"]) for r in rows]

    runner = {"rules": lambda: run_rules(rows), "gemma": lambda: run_gemma(rows),
              "qwen": lambda: run_tinker(rows, "qwen"),
              "gptoss": lambda: run_tinker(rows, "gptoss")}[args.backend]
    t0 = time.time()
    preds, lat, toks = runner()
    wall = time.time() - t0

    lat_sorted = sorted(lat)
    p50 = lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0

    # $ per 1k labels (rows): paid models only; prefill+sample token prices per 1M.
    price = {"qwen": (0.33, 1.005), "gptoss": (0.33, 0.84)}.get(args.backend)
    cost_str = "n/a (free)"
    if price:
        pre, samp = price
        usd = (toks["in_tok"] * pre + toks["out_tok"] * samp) / 1e6
        cost_str = f"${usd / len(rows) * 1000:.3f}  (total ${usd:.4f})"
    extra = {"p50_latency_ms": f"{p50:.1f}", "wall_s": f"{wall:.1f}",
             "in_tok": toks["in_tok"], "out_tok": toks["out_tok"], "usd_per_1k": cost_str}

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / f"{args.backend}_{scope}.jsonl").write_text(
        "\n".join(json.dumps({"id": r["code"], "gold": sorted(g), "pred": p,
                              "has_declaration": r.get("has_declaration", False)})
                  for r, g, p in zip(rows, golds, preds)), encoding="utf-8")

    label = "rules_naive (BASELINE)" if args.backend == "rules" else args.backend
    und = [i for i, r in enumerate(rows) if not r.get("has_declaration", False)]
    res_all = score(golds, preds)
    res_und = score([golds[i] for i in und], [preds[i] for i in und])
    rep_all = format_report(f"{label} [{scope}] ALL ROWS", res_all, extra)
    rep_und = format_report(f"{label} [{scope}] UNDECLARED ONLY (honest)", res_und,
                            {"p50_latency_ms": f"{p50:.1f}"})
    print(rep_all)
    print(rep_und)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write(f"\n#### cmd: `uv run python scripts/eval_baselines.py {args.backend} "
                 f"{'--subset' if subset else '--full'}`\n{rep_all}\n\n{rep_und}\n")
    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1500:])
        print(f"FAIL: eval_baselines — {type(e).__name__}: {e}")
        sys.exit(1)

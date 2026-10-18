"""Score the M3 LoRA on DEV (full, both views) and traps_dev. NEVER test / frozen traps.

    uv run python -I scripts/score_m3.py

Per allergen, on gold positives: caught (yes) / missed (said no) / abstained (said unsure).
Abstentions are never merged into recall. On gold negatives: false alarm (yes) / abstained.
Predictions are cached to data/eval/lora_m3_*.jsonl.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import load_env, service_client
from contains.labels import ALLERGENS
from contains.metrics import score
from contains.prompt import build_prompt, parse_response
from eval_baselines import load_dev

EVAL = ROOT / "data" / "eval"
MODEL = "Qwen/Qwen3.5-4B"
CHUNK = 128


def sample_all(texts, path):
    import tinker
    sc = service_client()
    sampling = sc.create_sampling_client(model_path=path, base_model=MODEL)
    tok = sampling.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=64, temperature=0.0)
    out, in_tok, out_tok, bad = [], 0, 0, 0
    for i in range(0, len(texts), CHUNK):
        futs = []
        for t in texts[i:i + CHUNK]:
            p = tok.apply_chat_template([{"role": "user", "content": build_prompt(t)}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            ids = tok.encode(p, add_special_tokens=False)
            in_tok += len(ids)
            futs.append(sampling.sample(prompt=tinker.ModelInput.from_ints(ids), num_samples=1, sampling_params=params))
        for f in futs:
            ids = list(f.result().sequences[0].tokens)
            out_tok += len(ids)
            pred, ok = parse_response(tok.decode(ids))
            bad += not ok
            out.append(pred)
        print(f"  sampled {min(i + CHUNK, len(texts))}/{len(texts)}", flush=True)
    return out, in_tok, out_tok, bad


def threeway(golds, preds):
    rows, T = [], dict(c=0, m=0, a=0, fa=0, na=0, pos=0)
    for al in ALLERGENS:
        c = m = a = fa = na = 0
        for g, p in zip(golds, preds):
            v = p.get(al, "unsure")
            if al in g:
                c += v == "yes"; m += v == "no"; a += v == "unsure"
            else:
                fa += v == "yes"; na += v == "unsure"
        pos = c + m + a
        rows.append((al, pos, c, m, a, fa, na))
        for k, x in zip(("c", "m", "a", "fa", "na", "pos"), (c, m, a, fa, na, pos)):
            T[k] += x
    return rows, T


def table(title, golds, preds):
    res = score(golds, preds)
    rows, T = threeway(golds, preds)
    L = [f"\n**{title}** (n={len(golds)})", "",
         "| allergen | gold+ | caught | MISSED (said no) | abstained | caught% | missed% | abstain% | false alarms | abstained on neg |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda x, n: f"{x / n:.2f}" if n else "n/a"
    for al, pos, c, m, a, fa, na in rows:
        L.append(f"| {al} | {pos} | {c} | {m} | {a} | {f(c, pos)} | {f(m, pos)} | {f(a, pos)} | {fa} | {na} |")
    p = T["pos"]
    L.append(f"| **all** | {p} | {T['c']} | **{T['m']}** | {T['a']} | {f(T['c'], p)} | {f(T['m'], p)} | {f(T['a'], p)} | {T['fa']} | {T['na']} |")
    L.append(f"\nmacro-F1 (unsure=not flagged) {res['macro_f1']:.3f} · exact-match {res['exact_match']:.3f} · unsure-cell rate {res['unsure_rate']:.4f}")
    return "\n".join(L), T


def main() -> int:
    load_env()
    path = json.loads((EVAL / "m3_train_log.json").read_text())["sampler_path"]
    dev = load_dev(subset=False)
    t0 = time.time()
    preds, i1, o1, bad1 = sample_all([r["ingredients_text"] for r in dev], path)
    (EVAL / "lora_m3_dev-full.jsonl").write_text("\n".join(json.dumps(
        {"id": r["code"], "gold": r["labels"], "pred": p, "has_declaration": r.get("has_declaration", False)})
        for r, p in zip(dev, preds)), encoding="utf-8")
    golds = [set(r["labels"]) for r in dev]
    und = [i for i, r in enumerate(dev) if not r.get("has_declaration", False)]
    sub_ids = {r["code"] for r in load_dev(subset=True)}
    sub = [i for i, r in enumerate(dev) if r["code"] in sub_ids]
    out = [f"### M3 LoRA (Qwen3.5-4B r16) — DEV, {len(dev)} rows, unparseable {bad1}; wall {time.time()-t0:.0f}s"]
    for title, idx in (("DEV full, ALL rows", range(len(dev))), ("DEV full, UNDECLARED only (honest)", und),
                       ("DEV subset-300 (same rows as the baseline table), UNDECLARED only",
                        [i for i in sub if i in set(und)])):
        t, _ = table(title, [golds[i] for i in idx], [preds[i] for i in idx])
        out.append(t)
    # Qwen zero-shot on the same subset-300, undeclared, for the three-way comparison
    qz = [json.loads(l) for l in (EVAL / "qwen_dev-subset-300.jsonl").read_text(encoding="utf-8").splitlines()]
    qz = [r for r in qz if not r["has_declaration"]]
    t, _ = table("REFERENCE Qwen3.5-4B zero-shot, DEV subset-300 UNDECLARED", [set(r["gold"]) for r in qz], [r["pred"] for r in qz])
    out.append(t)

    td = [json.loads(l) for l in (ROOT / "data/traps_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    td = [r for r in td if r["kind"] != "sanity"]
    tp, i2, o2, bad2 = sample_all([r["text"] for r in td], path)
    (EVAL / "lora_m3_traps_dev.jsonl").write_text("\n".join(json.dumps(
        {"id": r["id"], "gold": r["labels"], "pred": p, "kind": r["kind"]}) for r, p in zip(td, tp)), encoding="utf-8")
    t, T = table(f"traps_dev (unparseable {bad2})", [set(r["labels"]) for r in td], tp)
    out.append(t)
    miss = [(r["id"], al) for r, p in zip(td, tp) for al in r["labels"] if p[al] == "no"]
    abst = [(r["id"], al) for r, p in zip(td, tp) for al in r["labels"] if p[al] == "unsure"]
    fa = [(r["id"], al) for r, p in zip(td, tp) for al in ALLERGENS if al not in r["labels"] and p[al] == "yes"]
    out.append(f"\ntraps_dev misses (id, allergen): {miss}\nabstained: {abst}\nfalse alarms: {fa}")
    cost = ((i1 + i2) * 0.33 + (o1 + o2) * 1.005) / 1e6
    out.append(f"\nscoring tokens in {i1+i2:,} out {o1+o2:,} (~${cost:.3f} at base-Qwen sampling prices)")
    txt = "\n".join(out)
    print(txt)
    (EVAL / "m3_dev_report.md").write_text(txt, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

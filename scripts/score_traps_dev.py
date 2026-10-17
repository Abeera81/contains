"""Headroom check: score rules_naive / Qwen3.5-4B / gpt-oss-120b on data/traps_dev.jsonl.

    uv run python -I scripts/score_traps_dev.py

DEV trap set only (never data/verified/traps.jsonl). Sanity rows are run but NOT scored.
Also reports, for each model, WHY it misses: hidden-allergen misses (compound knowledge)
vs 'unsure' answers (abstention) vs false alarms on clean rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import load_env
from eval_baselines import run_rules, run_tinker

from contains.labels import ALLERGENS
from contains.metrics import score

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "RESULTS.md"


def main() -> int:
    load_env()
    allrows = [json.loads(ln) for ln in (ROOT / "data" / "traps_dev.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [{"ingredients_text": r["text"], **r} for r in allrows if r["kind"] != "sanity"]
    san = [{"ingredients_text": r["text"], **r} for r in allrows if r["kind"] == "sanity"]
    golds = [set(r["labels"]) for r in rows]
    runs = {"rules_naive": lambda x: run_rules(x), "qwen3.5-4b": lambda x: run_tinker(x, "qwen"),
            "gpt-oss-120b": lambda x: run_tinker(x, "gptoss")}
    out, res, diag = [], {}, {}
    for name, fn in runs.items():
        preds, lat, toks = fn(rows)
        sp, _, _ = fn(san)
        res[name] = score(golds, preds)
        miss = fn_unsure = fp = 0
        misses = []
        for r, g, p in zip(rows, golds, preds):
            for a in ALLERGENS:
                v = p.get(a, "unsure")
                if a in g and v == "no":
                    miss += 1; misses.append((r["id"], a))
                elif a in g and v == "unsure":
                    fn_unsure += 1
                elif a not in g and v == "yes":
                    fp += 1
        diag[name] = (miss, fn_unsure, fp, misses)
        print(f"\n--- sanity ({name}, unscored) ---")
        for r, p in zip(san, sp):
            print(f"  {r['title']}: {[a for a in ALLERGENS if p.get(a) == 'yes']}"
                  f"  unsure={[a for a in ALLERGENS if p.get(a) == 'unsure']}")
        out.append((name, toks))
    lines = [f"\n### traps_dev headroom check (n={len(rows)} scored: 30 hidden + 10 clean; 3 sanity rows unscored)",
             "cmd: `uv run python -I scripts/build_traps_dev.py && uv run python -I scripts/score_traps_dev.py`",
             "DEV trap set (different compounds from the frozen 62; peanuts has 0 positives here). Per-allergen RECALL (support):", "",
             "| allergen | support | " + " | ".join(res) + " |", "|---|---|" + "---|" * len(res)]
    pc = {k: {c.allergen: c for c in v["per_class"]} for k, v in res.items()}
    for a in ALLERGENS:
        lines.append(f"| {a} | {pc['rules_naive'][a].support} | " +
                     " | ".join(f"{pc[k][a].recall:.2f} (P {pc[k][a].precision:.2f})" for k in res) + " |")
    tot = sum(pc["rules_naive"][a].support for a in ALLERGENS)
    lines += ["", "| overall | recall (micro) | macro F1 | exact-match | unsure rate | missed (said no) | abstained | false alarms |",
              "|---|---|---|---|---|---|---|---|"]
    for k, v in res.items():
        tp = sum(pc[k][a].tp for a in ALLERGENS)
        m, u, f, _ = diag[k]
        lines.append(f"| {k} | {tp/tot:.2f} ({tp}/{tot}) | {v['macro_f1']:.3f} | {v['exact_match']:.3f} | "
                     f"{v['unsure_rate']:.3f} | {m} | {u} | {f} |")
    lines.append("")
    for k in res:
        lines.append(f"{k} missed (id, allergen): {diag[k][3]}")
    txt = "\n".join(lines)
    print(txt)
    with RESULTS.open("a", encoding="utf-8") as fh:
        fh.write("\n" + txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

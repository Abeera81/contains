"""Four-way baseline table from data/eval/*_dev-subset-300.jsonl (both ways).

    uv run python -I scripts/summarize_baselines.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.labels import ALLERGENS
from contains.metrics import score

E = Path(__file__).resolve().parent.parent / "data" / "eval"
NAMES = {"rules": "rules_naive", "gemma": "gemma3:4b", "qwen": "qwen3.5-4b", "gptoss": "gpt-oss-120b"}
data = {k: [json.loads(ln) for ln in (E / f"{k}_dev-subset-300.jsonl").read_text(encoding="utf-8").splitlines()] for k in NAMES}
for scope, keep in (("ALL ROWS", lambda r: True), ("UNDECLARED ONLY (honest)", lambda r: not r["has_declaration"])):
    res = {}
    for k, rows in data.items():
        rr = [r for r in rows if keep(r)]
        res[k] = score([set(r["gold"]) for r in rr], [r["pred"] for r in rr])
    n = len([r for r in data["rules"] if keep(r)])
    print(f"\n### {scope} (n={n})")
    print("| model | macro P | macro R | macro F1 | exact | unsure |")
    print("|---|---|---|---|---|---|")
    for k, r in res.items():
        print(f"| {NAMES[k]} | {r['macro_precision']:.3f} | {r['macro_recall']:.3f} | {r['macro_f1']:.3f} | {r['exact_match']:.3f} | {r['unsure_rate']:.3f} |")
    print("\nper-allergen RECALL / precision (support = gold positives)")
    print("| allergen | support | " + " | ".join(NAMES.values()) + " |")
    print("|---|---|" + "---|" * len(NAMES))
    for a in ALLERGENS:
        pc = {k: {c.allergen: c for c in res[k]["per_class"]} for k in NAMES}
        sup = pc["rules"][a].support
        print(f"| {a} | {sup} | " + " | ".join(f"R {pc[k][a].recall:.2f} / P {pc[k][a].precision:.2f}" for k in NAMES) + " |")

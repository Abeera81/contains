"""Rebuild the frozen-trap report from the SAVED predictions of the single scored run (no API calls).

    uv run python -I scripts/report_frozen_traps.py

(score_frozen_traps.py sampled and saved predictions, then crashed on a console-encoding print;
this script only re-reads data/eval/frozen_traps_*.jsonl.)
"""
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
from score_frozen_traps import table, TRAPS, OUT, SENTINEL

rows = [json.loads(l) for l in TRAPS.read_text(encoding="utf-8").splitlines() if l.strip()]
load = lambda n: [json.loads(l)["pred"] for l in (OUT / f"frozen_traps_{n}.jsonl").read_text(encoding="utf-8").splitlines()]
ship, naive, qwen = load("shipped_lora_rules"), load("rules_naive"), load("qwen_zeroshot")
h = hashlib.sha256(TRAPS.read_bytes()).hexdigest()
assert h.startswith("e62378cc")
out = [f"traps sha256 {h[:8]} - n={len(rows)} - rules.py sha256 {hashlib.sha256((ROOT/'src/contains/rules.py').read_bytes()).hexdigest()[:12]}"]
out.append(table("1. SHIPPED: LoRA + rules_product", rows, ship)[0])
keep = [i for i, r in enumerate(rows) if r["id"] != "trap-021"]
out.append(table("1b. SHIPPED without trap-021", [rows[i] for i in keep], [ship[i] for i in keep])[0])
j = next(i for i, r in enumerate(rows) if r["id"] == "trap-021")
out.append(f"\ntrap-021 ({rows[j]['text']!r}) shipped: tree_nuts={ship[j]['tree_nuts']} milk={ship[j]['milk']}")
out.append(table("2. rules_naive alone (BASELINE)", rows, naive)[0])
out.append(table("3. Qwen3.5-4B zero-shot alone", rows, qwen)[0])
txt = "\n".join(out)
(OUT / "frozen_traps_report.md").write_text(txt, encoding="utf-8")
SENTINEL.write_text(json.dumps({"scored": True}), encoding="utf-8")
sys.stdout.buffer.write(txt.encode("utf-8"))

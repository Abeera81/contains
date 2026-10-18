"""ABLATION (runs AFTER the main frozen-trap scoring; no code/rules/prompt changes in between).

    uv run python -I scripts/score_frozen_lora_only.py

LoRA alone (raw model verdicts, no rules_product, no unsure-downgrade) on the same 62 frozen rows.
Same prompt, parser and sampling as score_m3.py. Completes the ablation; the frozen traps are then closed.
"""
import hashlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from _smoke_common import load_env
from score_m3 import sample_all
from score_frozen_traps import table, TRAPS, OUT

load_env()
h = hashlib.sha256(TRAPS.read_bytes()).hexdigest()
assert h.startswith("e62378cc"), h[:8]
assert (OUT / "frozen_traps_SCORED.json").exists(), "main scoring must have run first"
rows = [json.loads(l) for l in TRAPS.read_text(encoding="utf-8").splitlines() if l.strip()]
preds, i, o, bad = sample_all([r["text"] for r in rows], os.environ["LORA_SAMPLER_PATH"])
(OUT / "frozen_traps_lora_only.jsonl").write_text("\n".join(json.dumps({"id": r["id"], "gold": r["labels"], "pred": p})
                                                           for r, p in zip(rows, preds)), encoding="utf-8")
txt = table(f"4. LoRA ONLY (no rules; unparseable {bad})", rows, preds)[0] + f"\n\ntokens in {i:,} out {o:,}"
(OUT / "frozen_traps_lora_only_report.md").write_text(txt, encoding="utf-8")
sys.stdout.buffer.write(txt.encode("utf-8"))

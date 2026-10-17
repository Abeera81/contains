"""Report/apply declaration stripping on the TRAIN split (input side only; labels kept).

    uv run python -I scripts/strip_train_declarations.py            # report
    uv run python -I scripts/strip_train_declarations.py --write    # data/splits/train_clean.jsonl

Dev/test are NOT stripped (they are reported both ways instead).

Also DROPS (user decision) from the written training file:
  - stripped rows with a label the rules_product detector can no longer find in the text
    (we destroyed the evidence ourselves -> NOT converted to 'unsure'), and
  - stripped rows shorter than 15 characters.
Output: data/splits/train_clean.jsonl (stripped inputs, labels kept, drops removed).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from contains.off import has_declaration, strip_declarations
from contains.rules import detect_product

rows = [json.loads(ln) for ln in (ROOT / "data/splits/train.jsonl").read_text(encoding="utf-8").splitlines()]
aff = [r for r in rows if r["has_declaration"]]
still = lost_ev = short = 0
drop_ids = set()
for r in aff:
    r["ingredients_text_orig"] = r["ingredients_text"]
    r["ingredients_text"] = strip_declarations(r["ingredients_text"])
    still += has_declaration(r["ingredients_text"])
    is_short = len(r["ingredients_text"]) < 15
    no_ev = bool(set(r["labels"]) - set(detect_product(r["ingredients_text"])))
    short += is_short
    lost_ev += no_ev
    if is_short or no_ev:
        drop_ids.add(r["code"])
kept = [r for r in rows if r["code"] not in drop_ids]
print(f"train rows: {len(rows):,}; affected (had declaration): {len(aff):,} = {len(aff)/len(rows):.1%}")
print(f"  still flagged after strip: {still}")
print(f"  DROPPED: {len(drop_ids):,} rows  (no-visible-evidence label: {lost_ev:,}; <15 chars: {short}; overlap counted once)")
print(f"  clean train: {len(kept):,} rows")
if "--write" in sys.argv:
    out = ROOT / "data/splits/train_clean.jsonl"
    out.write_text(chr(10).join(json.dumps(r, ensure_ascii=False) for r in kept), encoding="utf-8")
    print("wrote", out)

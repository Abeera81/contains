"""Report/apply declaration stripping on the TRAIN split (input side only; labels kept).

    uv run python -I scripts/strip_train_declarations.py            # report
    uv run python -I scripts/strip_train_declarations.py --write    # data/splits/train_stripped.jsonl

Dev/test are NOT stripped (they are reported both ways instead).
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
still = lost_ev = emptied = 0
for r in aff:
    r["ingredients_text_orig"] = r["ingredients_text"]
    r["ingredients_text"] = strip_declarations(r["ingredients_text"])
    still += has_declaration(r["ingredients_text"])
    emptied += len(r["ingredients_text"]) < 15
    # label with no rule evidence left in the stripped text (answer was only in the declaration)
    lost_ev += bool(set(r["labels"]) - set(detect_product(r["ingredients_text"])))
print(f"train rows: {len(rows):,}; affected (had declaration): {len(aff):,} = {len(aff)/len(rows):.1%}")
print(f"  still flagged after strip: {still}; shorter than 15 chars after strip: {emptied}")
print(f"  affected rows with >=1 label the rules can no longer see in the stripped text: {lost_ev:,}")
if "--write" in sys.argv:
    out = ROOT / "data/splits/train_stripped.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    print("wrote", out)

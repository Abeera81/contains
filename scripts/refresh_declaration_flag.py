"""Recompute has_declaration (tightened detector) on the working set, rebuild the splits,
and print the corrected declared rates for working set / train / dev / test / 300-subset.

    uv run python -I scripts/refresh_declaration_flag.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pyarrow as pa
import pyarrow.parquet as pq

from contains.off import has_declaration

SRC = ROOT / "data" / "raw" / "off_working.parquet"
rows = pq.read_table(SRC).to_pylist()
old = sum(r["has_declaration"] for r in rows)
for r in rows:
    r["has_declaration"] = has_declaration(r["ingredients_text"])
new = sum(r["has_declaration"] for r in rows)
pq.write_table(pa.Table.from_pylist(rows), SRC, compression="zstd")
print(f"working set: {old:,} -> {new:,} declared of {len(rows):,}  ({old/len(rows):.1%} -> {new/len(rows):.1%})")
subprocess.run([sys.executable, "-I", str(ROOT / "scripts" / "make_splits.py")], check=True, capture_output=True)
sp = {s: [json.loads(ln) for ln in (ROOT / "data/splits" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
      for s in ("train", "dev", "test")}
for s, r in sp.items():
    d = sum(x["has_declaration"] for x in r)
    print(f"{s}: {len(r):,} rows, declared {d:,} = {d/len(r):.1%}")
sub = random.Random(300).sample(sp["dev"], 300)
d = sum(x["has_declaration"] for x in sub)
print(f"300-subset: declared {d} = {d/300:.1%}; undeclared {300-d}")

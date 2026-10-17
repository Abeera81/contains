"""Repair off_working.parquet: collapse doubled text + add has_declaration flag.

No refetch. For each row: collapse exact doubling (the duplicate-STRUCT bug),
re-clean, re-derive labels, compute has_declaration. Then re-dedupe by normalized
text and rewrite with the new column. Prints how many rows were de-doubled and the
declaration percentage of the working set.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.off import clean_text, collapse_doubling, derive_labels, has_declaration, norm_ingredients_key

SRC = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_working.parquet"


def main() -> int:
    if not SRC.exists():
        print(f"FAIL: finalize_working_set — {SRC} not found.")
        return 1
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(SRC).to_pylist()
    n0 = len(rows)
    dedoubled = 0
    for r in rows:
        before = r["ingredients_text"]
        fixed = collapse_doubling(clean_text(before))
        if fixed != before:
            dedoubled += 1
        r["ingredients_text"] = fixed
        r["labels"] = derive_labels(r["allergens_tags"], fixed)
        r["has_declaration"] = has_declaration(fixed)

    rows = [r for r in rows if 15 <= len(r["ingredients_text"]) <= 1500]
    n_len = n0 - len(rows)

    seen: set = set()
    kept = []
    for r in rows:
        k = norm_ingredients_key(r["ingredients_text"])
        if k in seen:
            continue
        seen.add(k)
        kept.append(r)
    n_dup = len(rows) - len(kept)

    pq.write_table(pa.Table.from_pylist(kept), SRC, compression="zstd")
    decl = sum(1 for r in kept if r["has_declaration"])
    size_mb = SRC.stat().st_size / 1e6
    print(f"De-doubled {dedoubled:,}/{n0:,} rows.")
    print(f"Dropped {n_len:,} rows outside the length band, {n_dup:,} new duplicates.")
    print(f"Final: {len(kept):,} rows, {size_mb:.1f} MB.")
    print(f"\nhas_declaration: {decl:,}/{len(kept):,} = {decl / len(kept):.1%} of the WORKING SET.")

    print("\n--- 3 sample rows (post-repair) ---")
    for i, r in enumerate(kept[:3], 1):
        print(f"[{i}] code={r['code']} has_declaration={r['has_declaration']} labels={r['labels']}")
        print(f"    {r['ingredients_text'][:200]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

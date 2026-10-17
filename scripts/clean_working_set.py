"""One-off: clean HTML out of an already-built off_working.parquet (no refetch).

OFF embeds highlighting markup (e.g. <span class="allergen">milk</span>) in some
ingredient text. The first build stored it verbatim. This pass strips the markup,
re-derives labels from the cleaned text (the gluten->wheat rule reads the text),
and re-dedupes by normalized ingredient text (cleaning can expose new duplicates).
Overwrites the parquet and prints removal counts + fresh sample rows.

Idempotent: running it on already-clean text changes nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.off import clean_text, derive_labels, norm_ingredients_key

SRC = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_working.parquet"


def main() -> int:
    if not SRC.exists():
        print(f"FAIL: clean_working_set — {SRC} not found.")
        return 1
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(SRC).to_pylist()
    n0 = len(rows)
    changed = 0
    for r in rows:
        before = r["ingredients_text"]
        cleaned = clean_text(before)
        if cleaned != before:
            changed += 1
        r["ingredients_text"] = cleaned
        r["product_name"] = clean_text(r.get("product_name") or "")
        r["labels"] = derive_labels(r["allergens_tags"], cleaned)

    # drop rows whose cleaned text fell outside the length band
    rows = [r for r in rows if 15 <= len(r["ingredients_text"]) <= 1500]
    n_len = n0 - len(rows)

    # re-dedupe by normalized ingredient text (code dedupe already held)
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
    size_mb = SRC.stat().st_size / 1e6
    print(f"Cleaned {changed:,}/{n0:,} rows of HTML markup.")
    print(f"Dropped {n_len:,} rows now outside the 15-1500 length band.")
    print(f"Dropped {n_dup:,} new duplicates exposed by cleaning.")
    print(f"Final: {len(kept):,} rows, {size_mb:.1f} MB -> {SRC}")

    print("\n--- 3 SAMPLE ROWS (cleaned) ---")
    for i, r in enumerate(kept[:3], 1):
        print(f"[{i}] code={r['code']} brand={r['brands']!r}")
        print(f"    ingredients_text[:200]={r['ingredients_text'][:200]!r}")
        print(f"    allergens_tags={r['allergens_tags']} -> labels={r['labels']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

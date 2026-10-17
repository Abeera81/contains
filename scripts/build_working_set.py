"""M1-A — build data/raw/off_working.parquet ONCE from Open Food Facts.

The HF dataset has no small file-shards: both `main` and `refs/convert/parquet`
expose a single ~8 GB `food.parquet` (4703 parquet row groups, ~1023 rows each).
So we sample ROW GROUPS spread evenly across the whole file (every k-th group),
read only our 6 columns via AUTHENTICATED range requests (HF_TOKEN -> no 429, and
we touch only the selected groups' bytes, not 8 GB), filter, dedupe, and write one
local parquet.

Changes requested by the user:
  1. Spread sampling: strided row groups across the full list (not the first N);
     print the indices used + resulting country distribution + skew warning.
  2. Dedupe on `code` AND on normalized ingredient text; print removals per step.
  3. (Traps live in a separate script.)

Prints the 3 sample rows at the end. Target 30k-80k filtered rows; if the result
lands far outside that band the script stops and says so.
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _smoke_common import load_env

from contains.off import (
    TARGET_COUNTRIES,
    derive_labels,
    en_text,
    norm_ingredients_key,
)

REPO = "openfoodfacts/product-database"
HF_FILE = "datasets/openfoodfacts/product-database@refs/convert/parquet/default/food/0000.parquet"
COLUMNS = ["code", "product_name", "brands", "ingredients_text", "allergens_tags", "countries_tags"]

OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_working.parquet"
TARGET_LOW, TARGET_HIGH, TARGET_MID = 30_000, 80_000, 55_000
PROBE_GROUPS = 48  # evenly spread, to estimate the filter pass-rate
MIN_LEN, MAX_LEN = 15, 1500
TARGET_SET = set(TARGET_COUNTRIES)


def row_record(code, pname, brands, ing, allerg, countries) -> dict | None:
    """Filter + project one OFF row into our record, or None if it fails the filter."""
    ing_en = en_text(ing)
    if not (MIN_LEN <= len(ing_en) <= MAX_LEN):
        return None
    allerg = list(allerg or [])
    if not allerg:
        return None
    countries = list(countries or [])
    if not (set(countries) & TARGET_SET):
        return None
    return {
        "code": code,
        "product_name": en_text(pname),
        "brands": (brands or "").strip(),
        "ingredients_text": ing_en,
        "allergens_tags": allerg,
        "countries_tags": countries,
        "labels": derive_labels(allerg, ing_en),
    }


def read_group(pf, idx) -> list[dict]:
    tbl = pf.read_row_group(idx, columns=COLUMNS)
    cols = {c: tbl.column(c).to_pylist() for c in COLUMNS}
    out = []
    for i in range(tbl.num_rows):
        rec = row_record(
            cols["code"][i], cols["product_name"][i], cols["brands"][i],
            cols["ingredients_text"][i], cols["allergens_tags"][i], cols["countries_tags"][i],
        )
        if rec is not None:
            out.append(rec)
    return out


def spread_indices(n_total: int, stride: int) -> list[int]:
    return list(range(0, n_total, stride))


def main() -> int:
    load_env()
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print("FAIL: build_working_set — HF_TOKEN missing from .env (needed to avoid 429)")
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem(token=tok)
    t0 = time.time()
    pf = pq.ParquetFile(fs.open(HF_FILE))
    n_groups = pf.metadata.num_row_groups
    n_rows = pf.metadata.num_rows
    print(f"Opened {REPO} food parquet: {n_rows:,} rows in {n_groups} row groups "
          f"(footer {time.time() - t0:.0f}s).")

    # 1) Probe: estimate pass-rate from groups spread across the whole file.
    probe_stride = max(1, n_groups // PROBE_GROUPS)
    probe_idx = spread_indices(n_groups, probe_stride)
    print(f"\nProbe: {len(probe_idx)} row groups spread every {probe_stride} (indices "
          f"{probe_idx[:3]}..{probe_idx[-1]}) to estimate pass-rate...")
    probe_cache: dict[int, list[dict]] = {}
    scanned = 0
    for j, idx in enumerate(probe_idx, 1):
        recs = read_group(pf, idx)
        probe_cache[idx] = recs
        scanned += pf.metadata.row_group(idx).num_rows
        if j % 12 == 0:
            print(f"  probed {j}/{len(probe_idx)} groups, {sum(len(v) for v in probe_cache.values())} kept")
    kept_probe = sum(len(v) for v in probe_cache.values())
    pass_rate = kept_probe / scanned if scanned else 0.0
    print(f"Probe: scanned {scanned:,} rows, kept {kept_probe:,} -> pass-rate {pass_rate:.3%}")

    if pass_rate <= 0:
        print("FAIL: build_working_set — probe kept 0 rows; filter or schema is wrong.")
        return 1

    # 2) Choose a stride so strided groups across the FULL file yield ~TARGET_MID rows.
    rows_per_group = n_rows / n_groups
    groups_needed = TARGET_MID / (pass_rate * rows_per_group)
    stride = max(1, round(n_groups / groups_needed))
    selected = spread_indices(n_groups, stride)
    est = len(selected) * rows_per_group * pass_rate
    print(f"\nPlan: stride={stride} -> {len(selected)} groups spread across all {n_groups} "
          f"(indices {selected[:3]}..{selected[-1]}), est ~{est:,.0f} filtered rows.")

    # 3) Read selected groups (reuse probe cache where indices coincide).
    records: list[dict] = []
    used_idx: list[int] = []
    t1 = time.time()
    for j, idx in enumerate(selected, 1):
        recs = probe_cache.get(idx)
        if recs is None:
            recs = read_group(pf, idx)
        records.extend(recs)
        used_idx.append(idx)
        if j % 25 == 0 or j == len(selected):
            print(f"  read {j}/{len(selected)} groups, {len(records):,} rows kept "
                  f"({time.time() - t1:.0f}s)")
    print(f"Collected {len(records):,} filtered rows from {len(used_idx)} row groups.")

    # 4) Dedupe — report each step.
    before = len(records)
    seen_code: set = set()
    step1 = []
    for r in records:
        c = r["code"]
        if c in seen_code:
            continue
        seen_code.add(c)
        step1.append(r)
    removed_code = before - len(step1)

    seen_ing: set = set()
    step2 = []
    for r in step1:
        k = norm_ingredients_key(r["ingredients_text"])
        if k in seen_ing:
            continue
        seen_ing.add(k)
        step2.append(r)
    removed_ing = len(step1) - len(step2)
    records = step2
    print(f"\nDedupe: {before:,} -> drop {removed_code:,} by code -> drop {removed_ing:,} by "
          f"normalized ingredient text -> {len(records):,} unique rows.")

    # 5) Country distribution + skew check (rows count once per target country present).
    dist = Counter()
    for r in records:
        for c in r["countries_tags"]:
            if c in TARGET_SET:
                dist[c] += 1
    total_country_hits = sum(dist.values()) or 1
    print("\nCountry distribution (rows mentioning each target country):")
    for c, _ in dist.most_common():
        print(f"  {c:24s} {dist[c]:>7,}  ({dist[c] / total_country_hits:.1%})")
    if dist:
        top_c, top_n = dist.most_common(1)[0]
        if top_n / total_country_hits > 0.85:
            print(f"  WARNING: distribution is badly skewed to {top_c} "
                  f"({top_n / total_country_hits:.0%}).")

    # 6) Band check BEFORE writing.
    if not (TARGET_LOW <= len(records) <= TARGET_HIGH):
        print(f"\nSTOP: {len(records):,} rows is outside the 30k-80k target band. "
              f"Not writing. Re-run with a different stride, or tell me how to proceed.")
        return 2

    # 7) Write.
    tbl = pa.Table.from_pylist(records)
    pq.write_table(tbl, OUT, compression="zstd")
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nWrote {OUT} — {len(records):,} rows, {size_mb:.1f} MB.")
    print(f"Row-group indices used ({len(used_idx)}): stride {stride}, "
          f"{used_idx[:5]} ... {used_idx[-3:]}")

    # 8) Three sample rows for eyeballing label quality.
    print("\n--- 3 SAMPLE ROWS ---")
    for i, r in enumerate(records[:3], 1):
        ing = r["ingredients_text"][:200]
        print(f"[{i}] code={r['code']} brand={r['brands']!r}")
        print(f"    ingredients_text[:200]={ing!r}")
        print(f"    allergens_tags={r['allergens_tags']}")
        print(f"    -> labels={r['labels']}")
    print("\nPASS: build_working_set — working set written.")
    return 0


if __name__ == "__main__":
    import traceback
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1800:])
        print(f"FAIL: build_working_set — {type(e).__name__}: {e}")
        sys.exit(1)

"""M0 smoke test — OPEN FOOD FACTS.

Reads food.parquet from the Hugging Face dataset openfoodfacts/product-database
with duckdb. Prints column names, total row count, and the count of rows that have
English ingredients_text AND non-empty allergens_tags AND countries including one
of US/UK/CA/IE/AU. Prints 3 sample rows (ingredients_text truncated to 200 chars)
with their allergens_tags.

Uses the remote hf:// path by default. If a local copy exists at
data/raw/food.parquet it is used instead (and we say so). Prints one PASS/FAIL line.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _smoke_common import REPO_ROOT, load_env

# Direct resolve URL + duckdb httpfs range requests: reads only the parquet footer
# and the referenced columns, not the whole 8 GB file. The hf:// path is far
# slower (it hung past 8 min on the metadata read); the https URL takes ~30s.
HF_PATH = "https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet"
# M1 builds data/raw/off_working.parquet (needed columns, filtered rows) from the
# HF refs/convert/parquet shards. Prefer it, then a full food.parquet, if present.
LOCAL_CANDIDATES = (
    REPO_ROOT / "data" / "raw" / "off_working.parquet",
    REPO_ROOT / "data" / "raw" / "food.parquet",
)
LOCAL_PATH = next((p for p in LOCAL_CANDIDATES if p.exists()), LOCAL_CANDIDATES[-1])
TARGET_COUNTRIES = [
    "en:united-states",
    "en:united-kingdom",
    "en:canada",
    "en:ireland",
    "en:australia",
]


def main() -> int:
    load_env()
    import duckdb

    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")

    if LOCAL_PATH.exists():
        src = str(LOCAL_PATH).replace("\\", "/")
        print(f"Source: LOCAL {src}")
    else:
        src = HF_PATH
        print(f"Source: REMOTE {src}")
        print("  (duckdb httpfs range requests — reads footer + referenced columns only)")

    rel = f"read_parquet('{src}')"

    # 1) columns (reads the footer only — fast)
    cols = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    colnames = [c[0] for c in cols]
    print(f"Columns ({len(colnames)}):")
    for name, ctype, *_ in cols:
        print(f"  {name}: {ctype}")

    # 2) total row count
    t0 = time.time()
    total = con.execute(f"SELECT count(*) FROM {rel}").fetchone()[0]
    print(f"Total rows: {total:,}  ({time.time() - t0:.1f}s)")

    # Build expressions adapted to the actual schema.
    typ = {name: ctype.lower() for name, ctype, *_ in cols}

    def ingredients_en_expr() -> str:
        # Case A: list<struct(lang, text)> (current OFF schema)
        it = typ.get("ingredients_text", "")
        if it.startswith(("struct", "map")) or ("[" in it and "struct" in it):
            # list of struct with lang/text fields
            return (
                "list_transform(list_filter(ingredients_text, x -> x.lang = 'en'), x -> x.text)"
            )
        # Case B: plain string column + separate lang
        return "CASE WHEN ingredients_text IS NOT NULL THEN [ingredients_text] ELSE [] END"

    en_list = ingredients_en_expr()
    en_text = f"nullif(array_to_string({en_list}, ' '), '')"

    where = (
        f"{en_text} IS NOT NULL "
        f"AND length({en_text}) BETWEEN 15 AND 1500 "
        f"AND allergens_tags IS NOT NULL AND len(allergens_tags) > 0 "
        f"AND list_has_any(countries_tags, {TARGET_COUNTRIES})"
    )

    # 3) filtered count
    try:
        t0 = time.time()
        filtered = con.execute(f"SELECT count(*) FROM {rel} WHERE {where}").fetchone()[0]
        print(f"Filtered rows (EN ingredients + allergens_tags + target country): {filtered:,}  ({time.time() - t0:.1f}s)")
    except Exception as e:
        msg = str(e)
        if "429" in msg or "Too Many Requests" in msg:
            print(
                "FAIL: OPEN FOOD FACTS — remote filtered scan hit HTTP 429 (HF unauthenticated "
                "range-request rate limit). Schema + total row count succeeded above. "
                "Fix (M1): build data/raw/off_working.parquet from the HF refs/convert/parquet "
                "shards (needed columns + filtered rows); this script then queries that local "
                "file. Do NOT range-scan the 8 GB food.parquet."
            )
        else:
            print(f"FAIL: OPEN FOOD FACTS — filter query failed ({type(e).__name__}: {e}).")
        return 1

    # 4) 3 sample rows
    t0 = time.time()
    samples = con.execute(
        f"SELECT left({en_text}, 200) AS ingredients, allergens_tags "
        f"FROM {rel} WHERE {where} LIMIT 3"
    ).fetchall()
    print(f"Sample rows ({len(samples)}, {time.time() - t0:.1f}s):")
    for i, (ing, allerg) in enumerate(samples, 1):
        print(f"  [{i}] allergens_tags={allerg}")
        print(f"      ingredients_text[:200]={ing!r}")

    if filtered > 0 and samples:
        print(f"PASS: OPEN FOOD FACTS — {total:,} total rows, {filtered:,} usable, {len(samples)} samples shown")
        return 0
    print("FAIL: OPEN FOOD FACTS — no usable rows matched the filter")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("".join(traceback.format_exception(type(e), e, e.__traceback__))[-1800:])
        print(f"FAIL: OPEN FOOD FACTS — {type(e).__name__}: {e}")
        sys.exit(1)

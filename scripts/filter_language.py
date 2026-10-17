"""Quantify and remove non-English rows from off_working.parquet.

Root causes found: (a) OFF's `en` slot sometimes holds foreign text, en_text() falls
back to main/xx entries, and (b) the country filter is "any of", so a product sold in
Germany AND Ireland passes. Detection: langdetect on LOWERCASED text (all-caps English
fools it), a row is dropped when the top language != en with prob >= 0.8.

    uv run python -I scripts/filter_language.py          # report only
    uv run python -I scripts/filter_language.py --apply  # rewrite parquet (+ lang cols)
"""

from __future__ import annotations

import argparse
import collections
import sys
from multiprocessing import Pool
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "data" / "raw" / "off_working.parquet"
THRESH = 0.9
MIN_LEN = 30  # langdetect is unreliable on very short strings; those are judged by vocab only
_ENG = set(["water", "sugar", "salt", "milk", "flour", "oil", "and", "with", "contains", "less", "than", "natural", "flavor", "flavors", "extract", "powder", "starch", "syrup", "acid", "vitamin", "vitamins", "corn", "wheat", "soy", "butter", "cream", "cheese", "egg", "eggs", "cocoa", "spices", "spice", "yeast", "juice", "concentrate", "color", "preservative", "sodium", "calcium", "organic", "whole", "grain", "rice", "oats", "potato", "tomato", "garlic", "onion", "vegetable", "palm", "sunflower", "of", "or", "the", "may", "contain", "from", "added", "modified", "artificial", "sweetener", "lecithin", "citric"])


def eng_hits(text: str) -> int:
    import re
    return sum(1 for t in re.findall(r"[a-z]+", text.lower()) if t in _ENG)


def detect(text: str) -> tuple[str, float]:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
    try:
        top = detect_langs(text.lower())[0]
        return top.lang, round(top.prob, 3)
    except Exception:
        return "??", 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = pq.read_table(SRC).to_pylist()
    with Pool(6) as p:
        out = p.map(detect, [r["ingredients_text"] for r in rows], chunksize=200)
    target = {"en:united-states", "en:united-kingdom", "en:canada", "en:ireland", "en:australia"}
    bad = [i for i, (lg, pr) in enumerate(out)
           if lg != "en" and pr >= THRESH and len(rows[i]["ingredients_text"]) >= MIN_LEN
           and eng_hits(rows[i]["ingredients_text"]) < 3]
    c = collections.Counter(out[i][0] for i in bad)
    print(f"rows {len(rows):,}; non-English (p>={THRESH}, len>={MIN_LEN}, <3 English-vocab hits): {len(bad):,} = {len(bad)/len(rows):.1%}")
    print("by language:", c.most_common(10))
    multi = sum(1 for i in bad if set(rows[i]["countries_tags"]) - target)
    print(f"  of those, listed in a non-target country too: {multi:,}; target-only: {len(bad)-multi:,}")
    allmulti = sum(1 for r in rows if set(r["countries_tags"]) - target)
    print(f"  working-set rows with any non-target country: {allmulti:,} ({allmulti/len(rows):.1%})")
    badset = set(bad)
    if not args.apply:
        for i in list(bad)[:: max(1, len(bad) // 14)][:14]:
            print("  ", out[i], rows[i]["code"], rows[i]["ingredients_text"][:60].encode("ascii", "replace").decode())
        return 0
    kept = []
    for i, r in enumerate(rows):
        if i in badset:
            continue
        r["lang"], r["lang_prob"] = out[i]
        kept.append(r)
    pq.write_table(pa.Table.from_pylist(kept), SRC, compression="zstd")
    print(f"wrote {len(kept):,} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())

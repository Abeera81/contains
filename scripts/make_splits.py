"""M1-B/C — brand-disjoint train/dev/test splits from data/raw/off_working.parquet.

B. Prints per-allergen positive counts (class balance) over the whole working set.
C. Splits by brand so no brand appears in two splits; prints split sizes and
   asserts zero brand overlap programmatically. Empty-brand rows go to train
   (one bucket) to avoid leakage via a shared empty key.

Writes data/splits/{train,dev,test}.jsonl. Deterministic (seeded).
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.labels import ALLERGENS

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "raw" / "off_working.parquet"
OUT_DIR = ROOT / "data" / "splits"
SEED = 1337
DEV_FRAC, TEST_FRAC = 0.10, 0.10
NOBRAND = "__nobrand__"


def brand_key(brands: str) -> str:
    """Group key: first listed brand, lowercased/trimmed; empty -> NOBRAND."""
    if not brands:
        return NOBRAND
    first = brands.split(",")[0].strip().lower()
    return first or NOBRAND


def main() -> int:
    if not SRC.exists():
        print(f"FAIL: make_splits — {SRC} not found; run build_working_set.py first.")
        return 1
    import pyarrow.parquet as pq

    tbl = pq.read_table(SRC)
    rows = tbl.to_pylist()
    print(f"Loaded {len(rows):,} rows from {SRC.name}.")

    # B. class balance
    pos = Counter()
    for r in rows:
        for lab in r["labels"]:
            pos[lab] += 1
    n_any = sum(1 for r in rows if r["labels"])
    print("\nPer-allergen positive counts (class balance):")
    for a in ALLERGENS:
        print(f"  {a:22s} {pos[a]:>7,}  ({pos[a] / len(rows):.1%})")
    print(f"  {'(rows with >=1 label)':22s} {n_any:>7,}  ({n_any / len(rows):.1%})")
    print(f"  {'(rows with NO label)':22s} {len(rows) - n_any:>7,}")

    # C. brand-disjoint split
    by_brand: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_brand[brand_key(r["brands"])].append(r)

    # Assign brands to buckets to approximate the target fractions by ROW count.
    # NOBRAND is forced to train (never split). Other brands shuffled, then
    # greedily filled into test, then dev, until each hits its row target.
    total = len(rows)
    dev_target, test_target = int(total * DEV_FRAC), int(total * TEST_FRAC)
    # Human-verified gold rows must stay in test across rebuilds: pin their brands there.
    gold_path = Path(__file__).resolve().parent.parent / "data" / "verified" / "test_gold.jsonl"
    pinned: set[str] = set()
    if gold_path.exists():
        gold_ids = {json.loads(ln)["id"] for ln in gold_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
        pinned = {brand_key(r["brands"]) for r in rows if r["code"] in gold_ids} - {NOBRAND}
        print(f"Pinned {len(pinned)} gold-row brands to test ({len(gold_ids)} verified gold ids).")
    brands = [b for b in by_brand if b != NOBRAND and b not in pinned]
    rng = random.Random(SEED)
    rng.shuffle(brands)

    split_of: dict[str, str] = {NOBRAND: "train"}
    dev_n = test_n = 0
    for b in pinned:
        split_of[b] = "test"; test_n += len(by_brand[b])
    for b in brands:
        n = len(by_brand[b])
        if test_n < test_target:
            split_of[b] = "test"; test_n += n
        elif dev_n < dev_target:
            split_of[b] = "dev"; dev_n += n
        else:
            split_of[b] = "train"

    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    brands_in: dict[str, set] = {"train": set(), "dev": set(), "test": set()}
    for b, items in by_brand.items():
        s = split_of[b]
        splits[s].extend(items)
        brands_in[s].add(b)

    # zero-overlap assertion (programmatic)
    overlaps = []
    for a in ("train", "dev", "test"):
        for c in ("train", "dev", "test"):
            if a < c:
                inter = (brands_in[a] & brands_in[c]) - {NOBRAND}
                if inter:
                    overlaps.append((a, c, len(inter)))
    assert not overlaps, f"brand overlap detected: {overlaps}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s, items in splits.items():
        with (OUT_DIR / f"{s}.jsonl").open("w", encoding="utf-8") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\nSplit sizes (brand-disjoint):")
    for s in ("train", "dev", "test"):
        print(f"  {s:6s} rows={len(splits[s]):>7,} ({len(splits[s]) / total:.1%})  "
              f"brands={len(brands_in[s]):>6,}")
    print(f"\nBrand-overlap check: PASS (0 shared brands across splits; "
          f"NOBRAND rows={len(by_brand[NOBRAND]):,} all in train).")
    print("PASS: make_splits — wrote data/splits/{train,dev,test}.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())

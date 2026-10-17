"""Quantify the training sanity gate on the TRAIN split (dev/test untouched).

Tech-design rule: only train on rows where the rules layer and the OFF tags do not
flatly contradict each other. OFF under-tags, so "no tag" != "no allergen"; treating
it as such injects false negatives. We therefore DROP a train row when the
high-precision rules layer detects a definitional allergen that the tags omit
(rules ⊄ tags). The reverse (tags assert an allergen our keywords miss) is NOT a
contradiction — our rules deliberately under-detect — so it is reported but not dropped.

Prints: how many rows are dropped, broken down by no-label vs labeled rows.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.rules import detect

TRAIN = Path(__file__).resolve().parent.parent / "data" / "splits" / "train.jsonl"


def main() -> int:
    if not TRAIN.exists():
        print(f"FAIL: analyze_trainable — {TRAIN} not found; run make_splits.py first.")
        return 1
    rows = [json.loads(ln) for ln in TRAIN.read_text(encoding="utf-8").splitlines() if ln.strip()]
    n = len(rows)

    no_label = labeled = 0
    drop_total = drop_nolabel = drop_labeled = 0
    missed_by_tags = Counter()  # allergen the rules found but tags omitted (false-neg direction)
    for r in rows:
        tags = set(r["labels"])
        rule_hits = set(detect(r["ingredients_text"]))
        is_nolabel = not tags
        if is_nolabel:
            no_label += 1
        else:
            labeled += 1
        extra = rule_hits - tags  # rules found, tags missed -> contradiction
        if extra:
            drop_total += 1
            for a in extra:
                missed_by_tags[a] += 1
            if is_nolabel:
                drop_nolabel += 1
            else:
                drop_labeled += 1

    kept = n - drop_total
    print(f"TRAIN split: {n:,} rows ({no_label:,} no-label = {no_label / n:.1%}, {labeled:,} labeled).")
    print("\nTraining sanity gate (drop rows where rules find an allergen the tags omit):")
    print(f"  dropped total : {drop_total:,} ({drop_total / n:.1%})")
    print(f"    of which no-label rows : {drop_nolabel:,} "
          f"({drop_nolabel / no_label:.1%} of all no-label rows)")
    print(f"    of which labeled rows  : {drop_labeled:,} "
          f"({drop_labeled / labeled:.1%} of all labeled rows)")
    print(f"  kept for training eligibility : {kept:,} ({kept / n:.1%})")
    print("\nMost common allergen the tags were missing (rules-detected):")
    for a, c in missed_by_tags.most_common():
        print(f"    {a:22s} {c:>6,}")
    print(f"\nNo-label rows surviving the gate (kept): {no_label - drop_nolabel:,} "
          f"— these have no tag AND no rule-detected allergen, so 'no allergen' is corroborated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

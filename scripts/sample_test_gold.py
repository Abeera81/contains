"""M1-F — sample 200 rows from the test split into data/test_gold_draft.jsonl.

These are a DRAFT for human review with scripts/review_cli.py. Each row carries
the auto-derived `labels` as `proposed` and `verify: false`; nothing here is used
in scoring until a human confirms it. Deterministic (seeded).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data" / "splits" / "test.jsonl"
OUT = ROOT / "data" / "test_gold_draft.jsonl"
N = 200
SEED = 20261017


def main() -> int:
    if not TEST.exists():
        print(f"FAIL: sample_test_gold — {TEST} not found; run make_splits.py first.")
        return 1
    rows = [json.loads(line) for line in TEST.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(SEED)
    picked = rng.sample(rows, min(N, len(rows)))
    with OUT.open("w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps({
                "id": r["code"],
                "text": r["ingredients_text"],
                "brands": r.get("brands", ""),
                "proposed": r["labels"],
                "allergens_tags": r.get("allergens_tags", []),
                "verify": False,
            }, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT} — {len(picked)} rows sampled from test split (of {len(rows):,}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

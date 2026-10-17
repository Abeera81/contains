"""Promote the adjudicated traps draft to data/verified/traps.jsonl.

The 62 traps were adjudicated against the trap governing principle (an allergen
may be asserted only if the DEFINITION of a named ingredient guarantees it) BEFORE
any model was scored against them, so this is not test-set tuning. This writes the
frozen verified set, prints per-allergen positive counts, and records a SHA256 in
data/verified/HASHES.txt so the frozen set is tamper-evident.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.labels import ALLERGENS

ROOT = Path(__file__).resolve().parent.parent
DRAFT = ROOT / "data" / "traps_draft.jsonl"
OUT = ROOT / "data" / "verified" / "traps.jsonl"
HASHES = ROOT / "data" / "verified" / "HASHES.txt"


def main() -> int:
    if not DRAFT.exists():
        print(f"FAIL: verify_traps — {DRAFT} not found; run build_traps_draft.py first.")
        return 1
    order = {a: i for i, a in enumerate(ALLERGENS)}
    rows = [json.loads(ln) for ln in DRAFT.read_text(encoding="utf-8").splitlines() if ln.strip()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pos = Counter()
    with OUT.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            labels = sorted(set(r["expected"]), key=lambda a: order[a])
            for lab in labels:
                pos[lab] += 1
            fh.write(json.dumps({
                "id": f"trap-{i:03d}",
                "text": r["text"],
                "labels": labels,
                "kind": r["kind"],
                "rationale": r["rationale"],
                "verify": True,
            }, ensure_ascii=False) + "\n")

    # SHA256 of the frozen file
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    rel = OUT.relative_to(ROOT).as_posix()
    line = f"{digest}  {rel}\n"
    existing = HASHES.read_text(encoding="utf-8") if HASHES.exists() else ""
    kept = "".join(ln + "\n" for ln in existing.splitlines() if ln and not ln.endswith(rel))
    HASHES.write_text(kept + line, encoding="utf-8")

    n_hidden = sum(1 for r in rows if r["kind"] == "hidden")
    n_clean = sum(1 for r in rows if r["kind"] == "clean_negative")
    print(f"Wrote {OUT} — {len(rows)} verified traps ({n_hidden} hidden, {n_clean} clean negatives), verify=true.")
    print("\nPer-allergen positive counts (verified traps):")
    for a in ALLERGENS:
        print(f"  {a:22s} {pos[a]:>3}")
    print(f"  (rows with >=1 label)  {sum(1 for r in rows if r['expected']):>3}")
    print(f"  (clean negatives)      {n_clean:>3}")
    print(f"\nSHA256: {digest}")
    print(f"Recorded in {HASHES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

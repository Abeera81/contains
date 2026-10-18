"""Build the M3 training mix -> data/splits/train_m3.jsonl, and prove no trap contamination.

    uv run python -I scripts/build_m3_train.py

1. Sanity gate on train_clean (drop rows where rules_product finds an allergen the tags omit).
2. Stratified subsample: EVERY fish / crustacean_shellfish positive, natural-rate sample of the
   remaining gated rows, up to TARGET rows. Verdicts are yes/no only (no 'unsure' on plain rows).
3. Mix in the full train_abstain slice (verdicts carry 'unsure').
4. Verify no row derives from traps.jsonl (frozen) or traps_dev.jsonl.
"""
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from contains.labels import ALLERGENS
from contains.rules import detect_product

SEED = 3030
TARGET = 5000
RARE = {"fish", "crustacean_shellfish"}


def load(p):
    return [json.loads(ln) for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def main() -> int:
    rng = random.Random(SEED)
    clean = load(ROOT / "data/splits/train_clean.jsonl")
    gated = [r for r in clean if not (set(detect_product(r["ingredients_text"])) - set(r["labels"]))]
    print(f"gate: {len(gated):,}/{len(clean):,} pass ({len(gated)/len(clean):.1%})")

    rare = [r for r in gated if RARE & set(r["labels"])]
    rest = [r for r in gated if not (RARE & set(r["labels"]))]
    rng.shuffle(rest)
    keep = rare + rest[: max(0, TARGET - len(rare))]
    rows = [{"id": f"tr-{r['code']}", "source": "plain", "base_code": r["code"],
             "ingredients_text": r["ingredients_text"],
             "verdicts": {a: ("yes" if a in r["labels"] else "no") for a in ALLERGENS}} for r in keep]
    n_plain = len(rows)
    ab = load(ROOT / "data/splits/train_abstain.jsonl")
    for r in ab:
        rows.append({"id": r["id"], "source": r["kind"], "base_code": r["base_code"],
                     "ingredients_text": r["ingredients_text"], "verdicts": r["verdicts"]})
    rng.shuffle(rows)
    (ROOT / "data/splits/train_m3.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")

    print(f"plain {n_plain} (all {len(rare)} fish/crustacean positives + {n_plain-len(rare)} natural-rate) "
          f"+ abstain/contrast {len(ab)} = {len(rows)} rows")
    print("sources:", dict(Counter(r["source"] for r in rows)))
    print("\nclass balance (rows with verdict yes) / unsure:")
    for a in ALLERGENS:
        c = Counter(r["verdicts"][a] for r in rows)
        print(f"  {a:22s} yes {c['yes']:5d}  no {c['no']:5d}  unsure {c['unsure']:4d}")
    cells = Counter(v for r in rows for v in r["verdicts"].values())
    print("overall cells:", dict(cells), f"(total {sum(cells.values())})")
    print("no-allergen rows:", sum(1 for r in rows if "yes" not in r["verdicts"].values() and "unsure" not in r["verdicts"].values()))

    # ---- trap contamination check ----
    frozen = ROOT / "data/verified/traps.jsonl"
    h = hashlib.sha256(frozen.read_bytes()).hexdigest()
    print(f"\nfrozen traps sha256 {h[:8]} (expect e62378cc): {'OK' if h.startswith('e62378cc') else 'MISMATCH'}")
    traps = load(frozen) + load(ROOT / "data/traps_dev.jsonl")
    codes_in_traps = [k for t in traps for k in ("code", "base_code") if k in t]
    train_codes = {r["base_code"] for r in rows}
    tnorm = {norm(t["text"]) for t in traps if t.get("text")}
    exact = [r["id"] for r in rows if norm(r["ingredients_text"]) in tnorm]
    contained = [r["id"] for r in rows if any(len(n) > 25 and n in norm(r["ingredients_text"]) for n in tnorm)]
    print(f"trap rows: {len(traps)}; trap rows carrying a code/base_code field: {len(codes_in_traps)}")
    print(f"training base_codes: {len(train_codes)} (all from train_clean: {train_codes <= {r['code'] for r in clean}})")
    print(f"exact normalised text matches vs traps: {len(exact)}; trap text contained in a train row: {len(contained)}")
    ok = h.startswith("e62378cc") and not codes_in_traps and not exact and not contained
    print("CONTAMINATION CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

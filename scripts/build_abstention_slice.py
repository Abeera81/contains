"""Build the deliberate ABSTENTION slice of the training set -> data/splits/train_abstain.jsonl.

Real TRAIN lines (from train_clean.jsonl; never dev/test) get ONE formulation-dependent
ingredient inserted, and ONLY the allergens that ingredient could hide become 'unsure'
(if the base line has the allergen as a definite 'yes' it stays yes; base 'no' is required
for the unsure targets so the 'unsure' is genuinely uncertain). CONTRAST rows insert the
same kind of ingredient with a NAMED source (sunflower lecithin, canola oil, milk
chocolate...) and get definite answers, so the model learns "unnamed -> unsure,
named -> decide" instead of hedging on a whole product category.

Rows carry `verdicts` (allergen -> yes/no/unsure), `kind` (abstain|contrast), `trigger`.
    uv run python -I scripts/build_abstention_slice.py
"""
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from contains.labels import ALLERGENS

SEED = 4242
# trigger group -> (surface variants, unsure allergens, n rows, base-text blocker regex)
ABSTAIN = {
    "lecithin": (["lecithin", "emulsifier (lecithin)", "lecithin (emulsifier)"], ["soy"], 60, r"soy|lecithin|sunflower"),
    "natural_flavoring": (["natural flavoring", "natural flavor", "natural flavours", "artificial flavor", "flavouring"],
                          ["milk"], 30, r"flavo|extract|vanilla"),
    "unspecified_oil": (["vegetable oil", "cooking oil", "vegetable oil (may vary)", "refined oil", "edible oil"],
                        ["soy", "peanuts"], 70, r"oil|soy|peanut|fat|shortening"),
    "vanaspati": (["vanaspati", "banaspati (hydrogenated vegetable fat)", "banaspati", "vegetable ghee"],
                  ["soy"], 40, r"soy|vanaspati|banaspati|ghee|fat|shortening|oil"),
    "plain_chocolate": (["chocolate", "dark chocolate", "semisweet chocolate", "chocolate chips", "chocolate pieces"],
                        ["milk", "soy"], 80, r"chocolate|cocoa|soy|milk|dairy|cream|butter"),
    "nut_spread": (["hazelnut spread", "pistachio spread", "almond spread", "chocolate hazelnut spread"],
                   ["milk", "soy"], 50, r"milk|dairy|cream|butter|soy|spread"),
    "generic_bbq_sauce": (["barbecue sauce", "bbq sauce", "steak sauce", "chili sauce"], ["soy", "wheat"], 50,
                          r"sauce|soy|wheat|flour"),
    "salad_dressing": (["salad dressing", "ranch dressing", "dressing"], ["egg", "milk", "soy"], 45,
                       r"dressing|egg|milk|soy|mayo|cream|butter"),
    "curry_paste": (["curry paste", "thai curry paste", "laksa paste"], ["crustacean_shellfish", "fish"], 40,
                    r"paste|shrimp|prawn|fish|crab|curry"),
    "stir_fry_sauce": (["stir-fry sauce", "teriyaki-style sauce", "marinade"], ["soy", "wheat", "sesame"], 45,
                       r"sauce|soy|wheat|sesame|marinade|flour"),
    "bouillon": (["bouillon cube", "stock cube", "seasoning mix", "gravy mix"], ["milk", "wheat", "soy"], 40,
                 r"bouillon|stock|milk|wheat|soy|flour|gravy|seasoning|cream"),
}
# (surface, definite yes allergens, applies-to-allergens-that-must-be-base-no, n, blocker)
CONTRAST = {
    "sunflower_lecithin": (["sunflower lecithin"], [], ["soy"], 25, r"soy|lecithin"),
    "soy_lecithin": (["soy lecithin"], ["soy"], ["soy"], 25, r"soy|lecithin"),
    "named_oil_no": (["canola oil", "sunflower oil", "palm oil", "olive oil", "rapeseed oil"], [], ["soy", "peanuts"], 40,
                     r"oil|soy|peanut|fat|shortening"),
    "soybean_oil": (["soybean oil"], ["soy"], ["soy"], 20, r"oil|soy|fat|shortening"),
    "peanut_oil": (["peanut oil"], ["peanuts"], ["peanuts"], 15, r"oil|peanut|fat|shortening"),
    "milk_chocolate": (["milk chocolate", "white chocolate"], ["milk"], ["milk"], 35, r"chocolate|cocoa|milk|dairy|cream|butter"),
    "named_flavor": (["natural vanilla flavor", "vanilla extract", "lemon flavouring", "orange oil"], [], ["milk", "soy", "wheat"], 30,
                     r"flavo|extract|vanilla"),
    "almond_flavor": (["almond flavoring"], ["tree_nuts"], ["tree_nuts"], 15, r"almond|nut|flavo"),
}


def insert(text: str, item: str, rng: random.Random) -> str:
    cuts, depth = [], 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if ch == "," and depth == 0:
            cuts.append(i)
    if not cuts or rng.random() < 0.25:
        body = text.rstrip(". ")
        return f"{body}, {item}."
    i = rng.choice(cuts)
    return f"{text[:i]}, {item}{text[i:]}"


def main() -> int:
    rng = random.Random(SEED)
    base = [json.loads(ln) for ln in (ROOT / "data/splits/train_clean.jsonl").read_text(encoding="utf-8").splitlines()]
    base = [r for r in base if 20 <= len(r["ingredients_text"]) <= 400]
    out, used = [], set()

    def make(kind, trig, variants, yes_add, unsure, n, blocker):
        pool = [r for r in base if r["code"] not in used and not re.search(blocker, r["ingredients_text"], re.IGNORECASE)
                and not (set(unsure) & set(r["labels"])) and not (set(yes_add) & set(r["labels"]))]
        rng.shuffle(pool)
        for r in pool[:n]:
            used.add(r["code"])
            v = {a: ("yes" if a in r["labels"] else "no") for a in ALLERGENS}
            for a in yes_add:
                v[a] = "yes"
            if kind == "abstain":
                for a in unsure:
                    v[a] = "unsure"
            out.append({"id": f"abs-{len(out)+1:04d}", "kind": kind, "trigger": trig,
                        "ingredients_text": insert(r["ingredients_text"], rng.choice(variants), rng),
                        "verdicts": v, "base_code": r["code"]})
    for trig, (variants, unsure, n, blk) in ABSTAIN.items():
        # a nut spread NAMES its nut (definite yes); only its other components are unknown
        make("abstain", trig, variants, ["tree_nuts"] if trig == "nut_spread" else [], unsure, n, blk)
    for trig, (variants, yes_add, must_no, n, blk) in CONTRAST.items():
        make("contrast", trig, variants, yes_add, must_no, n, blk)
    (ROOT / "data/splits/train_abstain.jsonl").write_text(
        chr(10).join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    c = Counter(r["kind"] for r in out)
    print(f"wrote {len(out)} rows: {c['abstain']} abstain + {c['contrast']} contrast (seed {SEED})")
    print("by trigger:", dict(Counter(r["trigger"] for r in out)))
    cells = Counter(v for r in out if r["kind"] == "abstain" for v in r["verdicts"].values())
    print("abstain-row verdict cells:", dict(cells))
    for kind in ("abstain", "contrast"):
        r = next(x for x in out if x["kind"] == kind)
        print(kind, "|", r["ingredients_text"][:140], "|", {a: v for a, v in r["verdicts"].items() if v != "no"})
    return 0


if __name__ == "__main__":
    sys.exit(main())

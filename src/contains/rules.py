"""Keyword/rules allergen detector.

Two jobs:
  1. M2 baseline #1 (the rules-only labeller).
  2. The training-data sanity gate: a row is dropped from TRAINING when the rules
     layer finds a definitional allergen that the OFF tags omit (a likely
     false-negative), per the tech design. Never applied to dev/test.

Design for PRECISION: each allergen has definitional terms (word-boundary) and a
per-allergen neutralize list of phrases that contain a term but do not imply the
allergen ("coconut milk" is not milk, "buckwheat" is not wheat, "eggplant" is not
egg). Neutralize phrases are blanked before term matching. This is deliberately
conservative — it under-detects rather than over-detects.
"""

from __future__ import annotations

import re

from contains.labels import ALLERGENS

# allergen -> (definitional terms, neutralize phrases)
_SPEC: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "milk": (
        ("milk", "whey", "casein", "caseinate", "butter", "buttermilk", "cheese", "cream",
         "ghee", "lactose", "dairy", "yoghurt", "yogurt", "curd", "custard", "parmesan",
         "mozzarella", "cheddar", "ricotta", "mascarpone", "quark", "paneer"),
        ("coconut milk", "coconut cream", "coconut butter", "almond milk", "soy milk",
         "oat milk", "rice milk", "cashew milk", "hazelnut milk", "peanut butter",
         "cocoa butter", "shea butter", "nut butter", "almond butter", "apple butter",
         "cashew butter", "sunflower butter", "seed butter", "body butter",
         "cream of tartar", "creamer"),
    ),
    "egg": (
        ("egg", "eggs", "albumen", "ovalbumin", "mayonnaise", "mayo", "meringue", "aioli"),
        ("eggplant", "egg-free", "egg replacer", "egg substitute"),
    ),
    "fish": (
        ("fish", "anchovy", "anchovies", "cod", "salmon", "tuna", "sardine", "bonito",
         "haddock", "pollock", "tilapia", "mackerel", "surimi", "worcestershire"),
        ("cuttlefish", "jellyfish", "shellfish", "fish-free"),
    ),
    "crustacean_shellfish": (
        ("shrimp", "prawn", "crab", "lobster", "crawfish", "crayfish", "langoustine",
         "krill", "scampi", "crustacean"),
        ("crab apple",),
    ),
    "tree_nuts": (
        ("almond", "hazelnut", "walnut", "cashew", "pecan", "pistachio", "macadamia",
         "brazil nut", "pine nut", "pinenut", "praline", "marzipan", "frangipane",
         "gianduja", "chestnut"),
        ("water chestnut", "coconut", "nutmeg", "butternut", "nutritional", "nut-free",
         "nut free", "chestnut mushroom"),
    ),
    "peanuts": (
        ("peanut", "peanuts", "groundnut", "arachis"),
        ("peanut-free",),
    ),
    "wheat": (
        ("wheat", "semolina", "durum", "spelt", "farro", "kamut", "couscous", "bulgur",
         "seitan", "farina", "graham flour", "graham cracker"),
        ("buckwheat", "wheat-free", "wheatgrass"),
    ),
    "soy": (
        ("soy", "soya", "soybean", "soybeans", "tofu", "edamame", "miso", "tempeh",
         "tamari", "soy lecithin"),
        ("soy-free", "soya-free", "soy free"),
    ),
    "sesame": (
        ("sesame", "tahini", "benne", "gingelly"),
        ("sesame-free",),
    ),
}


def _compile(terms: tuple[str, ...]) -> re.Pattern:
    # word-boundary alternation; multi-word terms keep their internal spaces
    alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z]){alt}(?![a-z])")


_COMPILED: dict[str, tuple[re.Pattern, tuple[str, ...]]] = {
    a: (_compile(terms), neut) for a, (terms, neut) in _SPEC.items()
}


def detect(text: str) -> list[str]:
    """Return the allergens whose definitional terms appear in `text`, FALCPA order."""
    base = text.lower()
    found: set[str] = set()
    for allergen, (pat, neutralize) in _COMPILED.items():
        t = base
        for phrase in neutralize:
            if phrase in t:
                t = t.replace(phrase, " ")
        if pat.search(t):
            found.add(allergen)
    return [a for a in ALLERGENS if a in found]

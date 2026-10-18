"""Keyword/rules allergen detectors — TWO distinct things (see MEMORY.md):

  detect_naive   — THE BASELINE. Direct allergen names + unambiguous synonyms only,
                   NO compound-product knowledge. The honest floor.
  detect_product — part of OUR SHIPPED PRODUCT (not a baseline). Adds compound-
                   ingredient knowledge (worcestershire->fish, marzipan->tree_nuts,
                   miso->soy, ...) and is what supplies the UI evidence words. It
                   shares authorship with the trap set, so it must NEVER be scored
                   against the traps as a competitor — only ever as "LoRA + rules_product".

Both are precision-first: each allergen has a per-allergen neutralize list of phrases
that contain a term but do not imply the allergen ("coconut milk" is not milk,
"buckwheat" is not wheat, "eggplant" is not egg). Neutralize phrases are blanked
before term matching.

detect() is kept as an alias of detect_product for the training-data sanity gate.
"""

from __future__ import annotations

import re

from contains.labels import ALLERGENS

# Shared neutralize (exclusion) phrases per allergen — applied to BOTH detectors so
# neither false-fires on "coconut milk", "cocoa butter", "buckwheat", "eggplant", etc.
_NEUTRALIZE: dict[str, tuple[str, ...]] = {
    "milk": ("coconut milk", "coconut cream", "coconut butter", "almond milk", "soy milk",
             "oat milk", "rice milk", "cashew milk", "hazelnut milk", "peanut butter",
             "cocoa butter", "shea butter", "nut butter", "almond butter", "apple butter",
             "cashew butter", "sunflower butter", "seed butter", "body butter",
             "cream of tartar", "creamer"),
    "egg": ("eggplant", "egg-free", "egg replacer", "egg substitute"),
    "fish": ("cuttlefish", "jellyfish", "shellfish", "fish-free"),
    "crustacean_shellfish": ("crab apple",),
    "tree_nuts": ("water chestnut", "coconut", "nutmeg", "butternut", "nutritional",
                  "nut-free", "nut free", "chestnut mushroom"),
    "peanuts": ("peanut-free",),
    "wheat": ("buckwheat", "wheat-free", "wheatgrass"),
    "soy": ("soy-free", "soya-free", "soy free"),
    "sesame": ("sesame-free",),
}

# (1) BASELINE: direct names + unambiguous synonyms only. No compound products.
_NAIVE_TERMS: dict[str, tuple[str, ...]] = {
    "milk": ("milk", "whey", "casein", "caseinate", "butter", "buttermilk", "cheese",
             "cream", "lactose", "dairy", "yoghurt", "yogurt", "curd"),
    "egg": ("egg", "eggs", "albumen", "ovalbumin"),
    "fish": ("fish", "anchovy", "anchovies", "cod", "salmon", "tuna", "sardine", "bonito",
             "haddock", "pollock", "tilapia", "mackerel"),
    "crustacean_shellfish": ("shrimp", "prawn", "crab", "lobster", "crawfish", "crayfish",
                             "langoustine", "krill", "scampi", "crustacean"),
    "tree_nuts": ("almond", "hazelnut", "walnut", "cashew", "pecan", "pistachio",
                  "macadamia", "brazil nut", "pine nut", "pinenut", "chestnut"),
    "peanuts": ("peanut", "peanuts", "groundnut", "arachis"),
    "wheat": ("wheat", "flour", "semolina", "durum", "spelt", "farro", "kamut",
              "couscous", "bulgur", "farina"),
    "soy": ("soy", "soya", "soybean", "soybeans", "tofu", "edamame"),
    "sesame": ("sesame", "tahini", "benne", "gingelly"),
}

# (2) PRODUCT: naive terms PLUS compound-ingredient knowledge. Supplies UI evidence.
_PRODUCT_EXTRA: dict[str, tuple[str, ...]] = {
    "milk": ("ghee", "custard", "parmesan", "mozzarella", "cheddar", "ricotta",
             "mascarpone", "quark", "paneer", "bechamel", "dulce de leche", "butterscotch"),
    "egg": ("mayonnaise", "mayo", "meringue", "aioli", "remoulade"),
    "fish": ("surimi", "worcestershire", "fish sauce", "dashi", "fish ball"),
    "crustacean_shellfish": (),
    "tree_nuts": ("praline", "marzipan", "frangipane", "gianduja", "pesto", "nougat"),
    "peanuts": ("satay",),
    "wheat": ("wheat flour", "all-purpose flour", "all purpose flour", "plain flour", "bread flour",
              "cake flour", "pastry flour", "self-raising flour", "self-rising flour", "strong flour",
              "maida", "atta", "kunafa", "kunafeh", "knafeh", "kadayif", "kataifi", "phyllo", "filo",
              "seitan", "graham", "breadcrumb", "breadcrumbs", "panko",
              "brioche", "soy sauce", "teriyaki", "hoisin", "tempura", "bechamel"),
    "soy": ("miso", "tempeh", "tamari", "soy lecithin", "soy sauce", "teriyaki", "hoisin",
            "textured vegetable protein", "tvp"),
    "sesame": (),
}


def _compile(terms: tuple[str, ...], plural: bool = False) -> re.Pattern:
    alt = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    tail = "(?:e?s)?" if plural else ""
    return re.compile(rf"(?<![a-z])(?:{alt}){tail}(?![a-z])")


# Terms too ambiguous for the high-precision PRODUCT detector / gate, kept in NAIVE
# (the honest baseline) per spec: bare "flour" fires on rice/almond/corn flour.
_PRODUCT_DROP: dict[str, tuple[str, ...]] = {"wheat": ("flour",)}

_NAIVE = {a: (_compile(_NAIVE_TERMS[a]), _NEUTRALIZE.get(a, ())) for a in ALLERGENS}
# PRODUCT also matches simple plurals ("almonds", "pistachios", "peaches"-style -es); the
# BASELINE deliberately does not, so the M2 baseline numbers stay reproducible.
_PRODUCT = {
    a: (_compile(
        tuple(t for t in _NAIVE_TERMS[a] if t not in _PRODUCT_DROP.get(a, ()))
        + _PRODUCT_EXTRA.get(a, ()), plural=True), _NEUTRALIZE.get(a, ()))
    for a in ALLERGENS
}


def _run(spec, text: str) -> list[str]:
    base = text.lower()
    found: set[str] = set()
    for allergen, (pat, neutralize) in spec.items():
        t = base
        for phrase in neutralize:
            if phrase in t:
                t = t.replace(phrase, " ")
        if pat.search(t):
            found.add(allergen)
    return [a for a in ALLERGENS if a in found]


def _evidence(spec, text: str) -> dict[str, list[str]]:
    base = text.lower()
    out: dict[str, list[str]] = {}
    for allergen, (pat, neutralize) in spec.items():
        t = base
        for phrase in neutralize:
            if phrase in t:
                t = t.replace(phrase, " ")
        hits = list(dict.fromkeys(m.group(0) for m in pat.finditer(t)))
        if hits:
            out[allergen] = hits
    return out


def evidence_product(text: str) -> dict[str, list[str]]:
    """{allergen: [matched words, in order of first appearance]} for the PRODUCT detector."""
    return _evidence(_PRODUCT, text)


def detect_naive(text: str) -> list[str]:
    """BASELINE detector: direct names + unambiguous synonyms only."""
    return _run(_NAIVE, text)


def detect_product(text: str) -> list[str]:
    """PRODUCT detector: naive + compound-ingredient knowledge (not a baseline)."""
    return _run(_PRODUCT, text)


# Alias for the training-data sanity gate (uses the product detector's fuller coverage).
detect = detect_product

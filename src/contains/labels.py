"""The 9 US major food allergens (FALCPA + FASTER Act) and the Open Food Facts tag mapping.

This module is the single source of truth for allergen identity and ordering.
Everything downstream — the prompt, the parser, the "Contains:" line and the
evaluation — imports from here so names and order never drift.
"""

from __future__ import annotations

# FALCPA / FASTER Act order. The generated "Contains:" line uses this order
# and these display names verbatim.
ALLERGENS: tuple[str, ...] = (
    "milk",
    "egg",
    "fish",
    "crustacean_shellfish",
    "tree_nuts",
    "peanuts",
    "wheat",
    "soy",
    "sesame",
)

# Human-facing display names for the printed label line.
DISPLAY_NAMES: dict[str, str] = {
    "milk": "Milk",
    "egg": "Eggs",
    "fish": "Fish",
    "crustacean_shellfish": "Crustacean shellfish",
    "tree_nuts": "Tree nuts",
    "peanuts": "Peanuts",
    "wheat": "Wheat",
    "soy": "Soybeans",
    "sesame": "Sesame",
}

# Open Food Facts `allergens_tags` -> our label.
#
# `en:gluten` is intentionally omitted here: gluten is NOT one-to-one with wheat
# (barley, rye and oats also carry it). It maps to `wheat` ONLY when the
# ingredient text matches a wheat-family term; that conditional rule lives in
# the data pipeline, not in this flat table. See GLUTEN_WHEAT_TERMS below.
OFF_TAG_TO_LABEL: dict[str, str] = {
    "en:milk": "milk",
    "en:eggs": "egg",
    "en:fish": "fish",
    "en:crustaceans": "crustacean_shellfish",
    "en:nuts": "tree_nuts",
    "en:peanuts": "peanuts",
    "en:soybeans": "soy",
    "en:sesame-seeds": "sesame",
}

# OFF tags we deliberately ignore (not US major allergens).
OFF_TAGS_IGNORED: frozenset[str] = frozenset(
    {
        "en:celery",
        "en:mustard",
        "en:sulphur-dioxide-and-sulphites",
        "en:lupin",
        "en:molluscs",
    }
)

# Wheat-family terms that upgrade an `en:gluten` tag to a `wheat` label.
GLUTEN_WHEAT_TERMS: tuple[str, ...] = (
    "wheat",
    "spelt",
    "durum",
    "semolina",
    "farina",
    "einkorn",
    "triticale",
)


def contains_line(present: list[str] | set[str] | tuple[str, ...]) -> str:
    """Render the FALCPA-style 'Contains:' line from a set of present allergens.

    Order follows ALLERGENS; names follow DISPLAY_NAMES. Returns 'Contains: '
    with nothing after it when no major allergen is present.
    """
    present_set = set(present)
    names = [DISPLAY_NAMES[a] for a in ALLERGENS if a in present_set]
    return "Contains: " + ", ".join(names)

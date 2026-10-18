"""Unnamed-source triggers: the ONLY inputs for which an 'unsure' verdict may surface.

These mirror the abstention slice the LoRA was trained on (scripts/build_abstention_slice.py):
an ingredient whose allergen content depends on formulation (unnamed lecithin, vegetable/cooking
oil, unspecified chocolate, banaspati, natural flavouring, bouillon, unspecified sauces/dressings,
nut spreads). Each trigger lists ONLY the allergens it could hide. A model 'unsure' for an
allergen with no matching trigger in the input is downgraded to 'absent' (and counted).
"""

from __future__ import annotations

import re

# (regex, allergens it can hide). Named sources (sunflower lecithin, milk/white chocolate,
# vanilla flavour...) are excluded by the lookbehinds: they are NAMED, so nothing is unknown.
_TRIGGERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (r"(?<!sunflower )(?<!soy )(?<!soya )(?<!soybean )lecithin", ("soy",)),
    (r"(?<!vanilla )(?<!lemon )(?<!orange )(?<!almond )(?<!mint )(?:natural |artificial )?flavou?r(?:ing)?s?", ("milk",)),
    (r"vegetable oil|cooking oil|refined oil|edible oil", ("soy", "peanuts")),
    (r"vanaspati|banaspati|vegetable ghee", ("soy",)),
    (r"(?<!milk )(?<!white )chocolate", ("milk", "soy")),
    (r"(?:hazelnut|pistachio|almond) spread", ("milk", "soy")),
    (r"barbecue sauce|bbq sauce|steak sauce|chili sauce", ("soy", "wheat")),
    (r"dressing", ("egg", "milk", "soy")),
    (r"curry paste|laksa paste", ("crustacean_shellfish", "fish")),
    (r"stir-fry sauce|teriyaki-style sauce|marinade", ("soy", "wheat", "sesame")),
    (r"bouillon|stock cube|stock powder|seasoning mix|gravy mix", ("milk", "wheat", "soy")),
)
_COMPILED = tuple((re.compile(rf"(?<![a-z])(?:{p})(?![a-z])"), al) for p, al in _TRIGGERS)


def unsure_triggers(text: str) -> dict[str, list[str]]:
    """{allergen: [matched unnamed-source words]} for the allergens those words could hide."""
    t = text.lower()
    found: dict[str, dict[str, int]] = {}
    for pat, allergens in _COMPILED:
        for m in pat.finditer(t):
            for a in allergens:
                found.setdefault(a, {}).setdefault(m.group(0), m.start())
    return {a: sorted(w, key=w.get) for a, w in found.items()}  # in order of appearance

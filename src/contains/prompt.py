"""Shared zero-shot prompt and the strict parser used by ALL baselines.

Every baseline (rules, Qwen, Gemma, gpt-oss) sees the SAME prompt and its output is
read by the SAME strict parser, so differences are the model's, not the harness's.
The parser demands exactly one 'allergen: yes|no|unsure' line per allergen; anything
it cannot parse is scored as all-unsure (per the M2 spec).
"""

from __future__ import annotations

import re

from contains.labels import ALLERGENS

INSTRUCTION = (
    "You are a food-allergen labeller for the 9 US major allergens (FALCPA + FASTER Act).\n"
    "Given an ingredient list, decide for EACH allergen whether the product CONTAINS it.\n"
    "Judge only from the ingredients shown; a compound ingredient counts (e.g. "
    "'worcestershire sauce' contains fish via anchovy; 'tahini' is sesame).\n"
    "US scope rules: barley/oat gluten is NOT wheat (only wheat-family grains are); "
    "molluscs (oyster, clam) are NOT crustacean shellfish; coconut is NOT a tree nut.\n"
    "Answer with EXACTLY 9 lines and nothing else, each line 'name: yes' or 'name: no' "
    "or 'name: unsure', in this exact order:\n"
    "milk, egg, fish, crustacean_shellfish, tree_nuts, peanuts, wheat, soy, sesame.\n"
)


def build_prompt(ingredients_text: str) -> str:
    return f"{INSTRUCTION}\nINGREDIENTS: {ingredients_text}\n"


_LINE = {a: re.compile(rf"(?mi)^\s*{re.escape(a)}\s*:\s*(yes|no|unsure)\b") for a in ALLERGENS}


def parse_response(text: str) -> tuple[dict[str, str], bool]:
    """Return ({allergen: 'yes'|'no'|'unsure'}, parseable).

    Parseable only if every one of the 9 allergens has a recognizable verdict.
    If not, returns all-unsure with parseable=False (scored as all-unsure).
    """
    out: dict[str, str] = {}
    for a, pat in _LINE.items():
        m = pat.search(text or "")
        if m:
            out[a] = m.group(1).lower()
    if len(out) == len(ALLERGENS):
        return out, True
    return {a: "unsure" for a in ALLERGENS}, False

"""Open Food Facts row helpers: EN text extraction, label derivation, dedupe keys.

Kept separate from `labels.py` (the allergen schema) so the data pipeline,
training and eval can all derive labels the same way. Pure functions, no I/O.
"""

from __future__ import annotations

import html
import re

from contains.labels import (
    ALLERGENS,
    GLUTEN_WHEAT_TERMS,
    OFF_TAG_TO_LABEL,
)

# OFF country tags we keep (English-language markets).
TARGET_COUNTRIES: tuple[str, ...] = (
    "en:united-states",
    "en:united-kingdom",
    "en:canada",
    "en:ireland",
    "en:australia",
)

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_TAG = re.compile(r"<[^>]+>")


def clean_text(text: str) -> str:
    """Strip OFF's HTML highlighting (e.g. <span class="allergen">milk</span>) and
    unescape entities, so the stored ingredient line matches what a real label
    shows. Collapses the whitespace left behind by removed tags.
    """
    if not text:
        return ""
    t = _TAG.sub(" ", text)
    t = html.unescape(t)
    return _WS.sub(" ", t).strip()


def en_text(struct_list) -> str:
    """Join the English `text` values from an OFF list<struct<lang,text>> column.

    OFF stores localized strings as a list of {lang, text}. We take entries with
    lang == 'en' (falling back to a lang-less/empty entry only if no 'en' exists).
    Returns a single whitespace-joined string, or '' if none.
    """
    if not struct_list:
        return ""
    en: list[str] = []
    other: list[str] = []
    for item in struct_list:
        if item is None:
            continue
        lang = (item.get("lang") or "").lower() if isinstance(item, dict) else ""
        text = (item.get("text") or "") if isinstance(item, dict) else ""
        text = clean_text(text)
        if not text:
            continue
        if lang == "en":
            en.append(text)
        elif lang in ("", "main", "xx"):
            other.append(text)
    chosen = en or other
    return " ".join(chosen).strip()


def norm_ingredients_key(text: str) -> str:
    """Normalized dedupe key: lowercased, punctuation dropped, whitespace collapsed."""
    t = text.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()


def derive_labels(allergens_tags, ingredients_text: str) -> list[str]:
    """Map OFF `allergens_tags` to our 9 US allergens, in FALCPA order.

    `en:gluten` upgrades to `wheat` only when the ingredient text carries a
    wheat-family term (GLUTEN_WHEAT_TERMS) — gluten alone is not wheat-specific.
    """
    present: set[str] = set()
    low = ingredients_text.lower()
    for tag in allergens_tags or []:
        if tag in OFF_TAG_TO_LABEL:
            present.add(OFF_TAG_TO_LABEL[tag])
        elif tag == "en:gluten" and any(term in low for term in GLUTEN_WHEAT_TERMS):
            present.add("wheat")
    return [a for a in ALLERGENS if a in present]

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
    # OFF often stores the SAME text twice (an HTML-highlighted copy + a plain copy);
    # after clean_text they are identical, which would double the ingredient line.
    # Dedupe pieces by normalized key, preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for piece in chosen:
        key = norm_ingredients_key(piece)
        if key and key not in seen:
            seen.add(key)
            unique.append(piece)
    return " ".join(unique).strip()


def norm_ingredients_key(text: str) -> str:
    """Normalized dedupe key: lowercased, punctuation dropped, whitespace collapsed."""
    t = text.lower()
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t)
    return t.strip()


# Explicit allergen declarations a manufacturer appends to the ingredient list.
# Their presence means the answer is written in the input, so such rows are trivially
# solvable and inflate every system's score. Case-insensitive.
#
# Two tiers. ALWAYS phrases are allergy phrasing on their own. CONTAINS phrases are only
# declarations when the clause that follows names an allergen: "contains: cocoa solids 30%
# min, milk solids 14% min" is a composition statement, not an allergen declaration.
_ALWAYS = re.compile(
    r"(?i)("
    r"may\s+(?:also\s+)?contain"                      # 'may contain nuts'
    r"|allergy\s+advice"
    r"|allergens?\s*:"                                 # 'Allergens: milk'
    r"|allergen\s+information"
    r"|produced\s+in\s+a\s+facility"
    r"|made\s+(?:in|on)\s+(?:a\s+)?(?:shared\s+)?(?:facility|equipment|line)"
    r")"
)
_CONTAINS = re.compile(r"(?i)contains\s*:?\s*")
_ALLERGEN_WORD = re.compile(
    r"(?i)\b(?:milk|dairy|eggs?|fish|shellfish|crustaceans?|shrimp|prawns?|crab|lobster|nuts?|"
    r"tree\s+nuts?|almonds?|hazelnuts?|walnuts?|cashews?|pecans?|pistachios?|peanuts?|wheat|"
    r"gluten|soya?|soybeans?|sesame|barley|rye|sulphites?|sulfites?|celery|mustard|lupin|molluscs?)\b"
)


def _contains_is_declaration(text: str, m: re.Match) -> bool:
    """A 'contains' hit is an allergen declaration iff it is 'contains:' / 'contains <allergen>'
    and the clause up to the end of the sentence names an allergen with no % / 'solids'."""
    clause = re.split(r"[.;)]", text[m.end(): m.end() + 120])[0]
    colon = ":" in m.group(0)
    if "%" in clause or re.search(r"(?i)solids|one or more|following|or less", clause):
        return False
    first = _ALLERGEN_WORD.match(clause.lstrip())
    if colon:
        return bool(_ALLERGEN_WORD.search(clause))
    return bool(first)  # 'contains wheat' without a colon: allergen must come right after


def _find_declaration(text: str, pos: int = 0):
    """Earliest allergen declaration at/after pos as a (start, end_of_match) pair, or None."""
    best = None
    m = _ALWAYS.search(text, pos)
    if m:
        best = (m.start(), m.end())
    for m in _CONTAINS.finditer(text, pos):
        if best and m.start() >= best[0]:
            break
        if _contains_is_declaration(text, m):
            best = (m.start(), m.end())
            break
    return best


def has_declaration(text: str) -> bool:
    """True if the ingredient text embeds an explicit allergen statement (answer leakage)."""
    return _find_declaration(text or "") is not None


def collapse_doubling(text: str) -> str:
    """Collapse an exactly-doubled ingredient line ('A. A.') back to one copy.

    OFF stored duplicate STRUCT entries; the first build joined them. This detects a
    line whose two halves are identical after normalization (ignoring punctuation-only
    tokens left by HTML stripping) and returns just the first half. Non-doubled text is
    returned unchanged. The real fix is in en_text(); this repairs already-built files.
    """
    if not text:
        return text
    toks = text.split()
    norms = [norm_ingredients_key(t) for t in toks]
    kept = [(i, n) for i, n in enumerate(norms) if n]
    nn = [n for _, n in kept]
    if len(nn) < 2 or len(nn) % 2 != 0:
        return text
    h = len(nn) // 2
    if nn[:h] != nn[h:]:
        return text
    cut = kept[h][0]  # original-token index where the second copy begins
    return " ".join(toks[:cut]).strip()


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


def strip_declarations(text: str) -> str:
    """Remove embedded allergen statements ('Contains: wheat.', 'May contain nuts.',
    'Allergy advice: ...') from the INPUT side of a training example, keeping the rest
    of the ingredient list. A statement inside parentheses is cut only to its closing
    paren; otherwise it is cut to the end of its sentence ('.' or end of text).
    Leftover empty parentheses and stray separators are tidied.
    """
    t = text or ""
    for _ in range(20):  # a text can carry several statements
        hit = _find_declaration(t)
        if not hit:
            break
        s, mend = hit
        depth = t.count("(", 0, s) - t.count(")", 0, s)
        if depth > 0:
            e = s
            while e < len(t) and depth > 0:
                depth += (t[e] == "(") - (t[e] == ")")
                e += 1
            e = max(e - 1, s + 1)  # keep the closing paren
        else:
            e = t.find(".", mend)
            e = len(t) if e == -1 else e + 1
        t = t[:s] + " " + t[e:]
        t = re.sub(r"\(\s*\)", " ", t)
        t = re.sub(r"\s+([,;.])", r"\1", t)
        t = re.sub(r"([,;])\s*(?=[,;.])", "", t)
        t = _WS.sub(" ", t).strip(" ,;")
    return t

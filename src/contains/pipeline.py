"""Text -> per-allergen verdicts (model layer + rules_product layer) -> US 'Contains:' line.

Merge rule (safety-first):
  contains : rules_product fired OR the model said yes
  unsure   : model said unsure and rules found nothing
  absent   : otherwise
A rules hit beats a model 'no' / 'unsure'. If the sampler fails, times out or returns
unparseable output we fall back to rules-only and say so in `banner`; a result is ALWAYS returned.
Ingredient text is never logged.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from contains.labels import ALLERGENS, DISPLAY_NAMES
from contains.rules import evidence_product

MAX_CHARS = 6000
MODEL_TIMEOUT_S = 25.0

# matched word (singular, lowercase) -> FDA-style species name. Only terms whose DEFINITION fixes the
# species are listed; anything else matched (generic "fish", pesto, praline, nougat, surimi, fish
# sauce...) is "not determinable" and falls back to the category name plus a note.
_SPECIES = {
    "tree_nuts": {"almond": "almonds", "hazelnut": "hazelnuts", "walnut": "walnuts", "cashew": "cashews",
                  "pecan": "pecans", "pistachio": "pistachios", "macadamia": "macadamia nuts",
                  "brazil nut": "Brazil nuts", "pine nut": "pine nuts", "pinenut": "pine nuts",
                  "chestnut": "chestnuts", "marzipan": "almonds", "frangipane": "almonds",
                  "gianduja": "hazelnuts"},
    "fish": {"anchovy": "anchovy", "anchovie": "anchovy", "anchovies": "anchovy", "cod": "cod",
             "salmon": "salmon", "tuna": "tuna", "sardine": "sardine", "bonito": "bonito",
             "haddock": "haddock", "pollock": "pollock", "tilapia": "tilapia", "mackerel": "mackerel",
             "worcestershire": "anchovy", "dashi": "bonito"},
    "crustacean_shellfish": {"shrimp": "shrimp", "prawn": "prawn", "crab": "crab", "lobster": "lobster",
                             "crawfish": "crawfish", "crayfish": "crayfish", "langoustine": "langoustine",
                             "krill": "krill", "scampi": "scampi"},
}
_SPECIES_CATEGORIES = tuple(_SPECIES)


def _species_for(allergen: str, word: str) -> str | None:
    table = _SPECIES[allergen]
    w = word.lower()
    for cand in (w, w[:-2] if w.endswith("es") else w, w[:-1] if w.endswith("s") else w):
        if cand in table:
            return table[cand]
    return None


class _Cache:
    """In-memory model-verdict cache keyed by a hash of the text (nothing is written to disk)."""

    def __init__(self, size: int = 256):
        self.size, self.d = size, OrderedDict()

    def get(self, k):
        if k in self.d:
            self.d.move_to_end(k)
            return self.d[k]

    def put(self, k, v):
        self.d[k] = v
        self.d.move_to_end(k)
        while len(self.d) > self.size:
            self.d.popitem(last=False)


_cache = _Cache()
_pool = ThreadPoolExecutor(max_workers=4)


def _model_verdicts(text: str, sampler, timeout: float) -> tuple[dict | None, str | None]:
    """Return (verdicts, None) or (None, reason). Never raises."""
    if sampler is None or getattr(sampler, "name", "") == "rules_only":
        return None, None  # intentional rules-only: not a failure, no banner
    key = (getattr(sampler, "name", "?"), hashlib.sha256(text.encode("utf-8")).hexdigest())
    hit = _cache.get(key)
    if hit is not None:
        return hit, None
    def work():
        v = sampler.verdicts(text)
        _cache.put(key, v)  # a late result is still kept, so a retry is instant and costs nothing
        return v

    try:
        return _pool.submit(work).result(timeout=timeout), None
    except FutureTimeout:
        return None, "timed out"
    except Exception as e:  # noqa: BLE001 - any sampler failure must degrade to rules-only
        return None, type(e).__name__


def analyze(text: str, sampler, timeout: float = MODEL_TIMEOUT_S) -> dict:
    text = (text or "").strip()
    truncated = len(text) > MAX_CHARS
    text = text[:MAX_CHARS]
    if not text:
        return {"empty": True, "banner": None, "contains_line": "", "verify_line": "", "notes": [],
                "allergens": [], "sampler": "none", "truncated": False}
    rules = evidence_product(text)
    model, fail = _model_verdicts(text, sampler, timeout)
    banner = None
    if fail:
        banner = ("The model layer is unavailable, so this output is rules-only: it can miss allergens "
                  "hidden in compound ingredients and cannot flag formulation-dependent items. "
                  "Confirm every ingredient with your supplier.")
    elif model is None:
        banner = "Rules-only mode (no model layer configured): hidden-allergen coverage is limited."

    items, notes, line_parts, verify = [], [], [], []
    for a in ALLERGENS:
        words = rules.get(a, [])
        m = model.get(a) if model else None
        layers, status = [], "absent"
        if words:
            status = "contains"
            layers.append("rules")
            if m == "yes":
                layers.append("model")
        elif m == "yes":
            status, layers = "contains", ["model"]
        elif m == "unsure":
            status, layers = "unsure", ["model"]
        species, note = None, None
        if status == "contains" and a in _SPECIES_CATEGORIES:
            found = list(dict.fromkeys(s for s in (_species_for(a, w) for w in words) if s))
            undetermined = (not words) or any(_species_for(a, w) is None for w in words)
            species = found
            if undetermined or not found:
                note = (f"{DISPLAY_NAMES[a]}: species not determinable from the ingredient list"
                        + (" for every source" if found else "") + " — confirm with supplier.")
                notes.append(note)
        name = DISPLAY_NAMES[a]
        if status == "contains":
            line_parts.append(f"{name} ({', '.join(species)})" if species else name)
        elif status == "unsure":
            verify.append(name)
        items.append({"key": a, "name": name, "status": status, "layers": layers, "evidence": words,
                      "model_inference_only": status == "contains" and not words,
                      "species": species, "note": note})
    return {
        "empty": False, "banner": banner, "truncated": truncated,
        "contains_line": ("Contains: " + ", ".join(line_parts) + ".") if line_parts else "No major allergens detected.",
        "verify_line": ("Verify with supplier: " + ", ".join(verify) + ".") if verify else "",
        "notes": notes, "allergens": items,
        "sampler": getattr(sampler, "name", "rules_only") if (model is not None) else "rules_only",
    }

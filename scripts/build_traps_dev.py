"""Build data/traps_dev.jsonl — a DEV trap set (~40 rows) for headroom checks.

DIFFERENT compounds from the frozen 62 (data/verified/traps.jsonl, never touched here).
Same principle: a label asserts only what the DEFINITION of the named ingredient
guarantees, never a brand formulation. Production label-line format. Dev-use only:
it may be looked at and iterated on; the frozen set is for the single final run.
Also writes 3 sanity rows from data/user_session/cake_inputs.txt (kind=sanity, NOT scored).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from contains.labels import ALLERGENS

FROZEN = ROOT / "data" / "verified" / "traps.jsonl"
OUT = ROOT / "data" / "traps_dev.jsonl"

HIDDEN = [
    ("rice, XO sauce, scallion, ginger, sunflower oil", ["crustacean_shellfish"],
     "XO sauce is defined by dried shrimp (and scallop)"),
    ("lamb, dukkah, olive oil, lemon, salt", ["tree_nuts", "sesame"],
     "dukkah = ground hazelnuts/nuts + sesame seeds + spices"),
    ("chicken, panko crumbs, egg wash, vegetable oil", ["egg", "wheat"],
     "panko is wheat bread crumb; egg wash is egg"),
    ("asparagus, béarnaise sauce, salt, pepper", ["milk", "egg"],
     "béarnaise = clarified butter emulsified with egg yolk"),
    ("sugar, halva, pistachio pieces", ["sesame", "tree_nuts"],
     "halva is made from tahini (sesame); pistachio is a tree nut"),
    ("poached salmon, hollandaise sauce, dill, lemon", ["fish", "milk", "egg"],
     "hollandaise = butter + egg yolk; salmon is fish"),
    ("cucumber, tzatziki, olive oil, mint", ["milk"], "tzatziki is strained yoghurt based"),
    ("potato, cauliflower, raita, cumin, coriander", ["milk"], "raita is yoghurt based"),
    ("spinach, paneer, tomato, cream, garam masala", ["milk"], "paneer is a milk cheese"),
    ("mascarpone, ladyfingers, espresso, cocoa, sugar (tiramisu)", ["milk", "egg", "wheat"],
     "tiramisu: mascarpone (milk), ladyfingers (egg + wheat flour)"),
    ("phyllo pastry, walnuts, butter, honey, cinnamon", ["wheat", "tree_nuts", "milk"],
     "baklava: phyllo is wheat, walnuts tree nut, butter milk"),
    ("potato gnocchi, sage, olive oil, salt", ["wheat"], "gnocchi is bound with wheat flour"),
    ("udon noodles, dashi, scallion, mirin", ["wheat", "fish"],
     "udon is wheat noodle; dashi is made from bonito (fish)"),
    ("orzo, tomato, basil, olive oil", ["wheat"], "orzo is durum wheat pasta"),
    ("seitan, soy sauce, ginger, garlic", ["wheat", "soy"], "seitan is vital wheat gluten; soy sauce is soy"),
    ("rice, katsuobushi, nori, sesame seeds", ["fish", "sesame"], "katsuobushi is dried bonito (fish)"),
    ("rice, vegetables, shrimp paste, chili, lime", ["crustacean_shellfish"],
     "shrimp paste is fermented shrimp"),
    ("cream, lobster bisque, brandy, tomato paste", ["crustacean_shellfish", "milk"],
     "bisque is a lobster/shellfish and cream soup"),
    ("pita, hummus, olive oil, paprika", ["wheat", "sesame"],
     "pita is wheat bread; hummus (bi tahini) contains tahini"),
    ("red peppers, romesco sauce, garlic, vinegar", ["tree_nuts"],
     "romesco is thickened with ground almonds/hazelnuts"),
    ("milk chocolate, gianduja, cocoa butter", ["tree_nuts", "milk"],
     "gianduja is hazelnut chocolate; milk chocolate is milk"),
    ("vodka, lemon, orgeat syrup, ice", ["tree_nuts"], "orgeat is almond syrup"),
    ("sugar, hazelnut spread, palm oil", ["tree_nuts"], "hazelnut spread is made from hazelnuts"),
    ("flour, butter, eggs, pâte à choux, sugar", ["wheat", "milk", "egg"],
     "choux pastry = flour, butter, eggs"),
    ("vanilla, crème anglaise, sugar", ["milk", "egg"], "crème anglaise = milk/cream + egg yolks"),
    ("potato, tartar sauce, parsley, capers", ["egg"], "tartar sauce is mayonnaise based (egg)"),
    ("rice, natto, scallion, mustard", ["soy"], "natto is fermented soybeans"),
    ("braided challah, sesame topping, honey", ["wheat", "egg", "sesame"],
     "challah is an egg-enriched wheat bread; sesame topping"),
    ("shredded kataifi, pistachios, sugar syrup", ["wheat", "tree_nuts"], "kataifi is wheat pastry"),
    ("water, doenjang, tofu, zucchini, onion", ["soy"], "doenjang is fermented soybean paste"),
]
CLEAN = [
    ("jackfruit, onion, garlic, coconut aminos, smoked paprika", [],
     "coconut aminos is soy-free; jackfruit is not a nut"),
    ("water, kombu, shiitake, salt", [], "seaweed is not fish; mushrooms clean"),
    ("calamari, olive oil, garlic, lemon", [],
     "squid is a mollusc, not a US major allergen (out of scope; EU-regulated)"),
    ("oyster mushrooms, thyme, olive oil, salt", [], "oyster mushroom is a fungus"),
    ("water chestnuts, bamboo shoots, ginger, rice vinegar", [], "water chestnut is a vegetable, not a tree nut"),
    ("quinoa, pumpkin seeds, maple syrup, cinnamon", [], "seeds/pseudocereal; none of the 9"),
    ("palm sugar, coconut cream, pandan, tapioca pearls", [], "coconut is not a tree nut (FDA Ed. 5)"),
    ("oat milk, sunflower seed butter, agave syrup", [], "plant 'milk'/'butter' without milk or nuts"),
    ("tamarind paste, jaggery, cumin, salt", [], "no allergen"),
    ("rice, arrowroot, nutritional yeast, turmeric", [], "nutritional yeast is not a nut"),
]


def sanity_rows():
    lines = (ROOT / "data" / "user_session" / "cake_inputs.txt").read_text(encoding="utf-8").splitlines()
    out, i = [], 0
    while i < len(lines) and len(out) < 3:
        if lines[i].strip() and i + 1 < len(lines) and lines[i + 1].strip():
            out.append({"id": f"sanity-{len(out)+1:02d}", "title": lines[i].strip(),
                        "text": lines[i + 1].strip(), "labels": None, "kind": "sanity"})
            i += 2
        else:
            i += 1
    return out


def main() -> int:
    frozen = {json.loads(ln)["text"] for ln in FROZEN.read_text(encoding="utf-8").splitlines()}
    rows, n = [], 0
    for kind, items in (("hidden", HIDDEN), ("clean_negative", CLEAN)):
        for text, labels, why in items:
            assert text not in frozen, text
            assert all(a in ALLERGENS for a in labels), labels
            n += 1
            rows.append({"id": f"tdev-{n:03d}", "text": text,
                         "labels": [a for a in ALLERGENS if a in labels],
                         "kind": kind, "rationale": why})
    rows += sanity_rows()
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    sc = [r for r in rows if r["kind"] != "sanity"]
    print(f"wrote {len(rows)} rows ({len(sc)} scored: {len(HIDDEN)} hidden + {len(CLEAN)} clean; 3 sanity)")
    from collections import Counter
    c = Counter(a for r in sc for a in r["labels"])
    print({a: c[a] for a in ALLERGENS})
    return 0


if __name__ == "__main__":
    sys.exit(main())

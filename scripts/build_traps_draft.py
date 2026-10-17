"""M1-D — write data/traps_draft.jsonl.

~60 hidden-allergen cases where the allergen lives inside a COMPOUND ingredient,
plus risky-looking clean negatives. Each trap is written as a realistic label
ingredient LINE (what the model sees in production), with the compound ingredient
embedded among ordinary ingredients — not a bare ingredient name.

Each row: {text, expected, rationale, verify:false, kind}.
  text     — production-format ingredient line
  expected — our 9-allergen labels (FALCPA order), [] for clean negatives
  rationale— one line explaining the hidden allergen (or why it is clean)
  verify   — always false: these are a DRAFT for human review, never auto-scored
  kind     — "hidden" | "clean_negative"

Labels are validated against contains.labels.ALLERGENS before writing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from contains.labels import ALLERGENS

OUT = Path(__file__).resolve().parent.parent / "data" / "traps_draft.jsonl"

# (text, expected_labels, rationale)
HIDDEN: list[tuple[str, list[str], str]] = [
    ("wheat flour, butter, worcestershire sauce, salt, spices",
     ["milk", "fish", "wheat"],
     "worcestershire sauce contains anchovies (fish); butter is milk; wheat flour is wheat"),
    ("potatoes, sunflower oil, basil pesto, salt",
     ["milk", "tree_nuts"],
     "traditional pesto contains pine nuts (tree nut) and parmesan (milk)"),
    ("sugar, couverture chocolate, cocoa, vanilla",
     ["milk", "soy"],
     "milk couverture contains milk solids and soy lecithin as emulsifier"),
    ("surimi, water, wheat starch, crab extract, salt",
     ["fish", "crustacean_shellfish", "wheat"],
     "surimi is a fish paste; crab extract is crustacean; wheat starch is wheat"),
    ("chickpeas, tahini, garlic, lemon juice, olive oil, salt",
     ["sesame"],
     "tahini is ground sesame seeds"),
    ("sugar, almonds, marzipan, invert sugar syrup",
     ["tree_nuts"],
     "marzipan is made from ground almonds (tree nut)"),
    ("romaine lettuce, caesar dressing, croutons, parmesan",
     ["milk", "egg", "fish", "wheat"],
     "caesar dressing has anchovy (fish), egg yolk (egg), parmesan (milk); croutons are wheat"),
    ("rice, clarified butter (ghee), cumin, turmeric, salt",
     ["milk"],
     "ghee is clarified butter, a milk product"),
    ("water, whey protein concentrate, sugar, cocoa, natural flavor",
     ["milk"],
     "whey is a milk-derived protein"),
    ("casein, calcium phosphate, sunflower oil, salt",
     ["milk"],
     "casein is the principal milk protein"),
    ("semolina, water, salt",
     ["wheat"],
     "semolina is coarsely milled durum wheat"),
    ("water, soybeans, rice, miso paste, seaweed, scallion",
     ["soy"],
     "miso is fermented soybean paste"),
    ("shrimp, tempura batter, vegetable oil, salt",
     ["crustacean_shellfish", "egg", "wheat"],
     "tempura batter is wheat flour and egg; shrimp is crustacean"),
    ("water, sugar, fish sauce, chili, garlic, lime",
     ["fish"],
     "fish sauce is fermented anchovy extract"),
    ("cabbage, carrot, oyster sauce, sesame oil, ginger",
     ["crustacean_shellfish", "sesame"],
     "oyster sauce is made from oysters/mollusc extract historically listed with shellfish; sesame oil"),
    ("sugar, egg white, almonds, honey, nougat",
     ["egg", "tree_nuts"],
     "nougat contains whipped egg white and almonds (tree nut)"),
    ("sugar, hazelnuts, praline, cocoa butter, milk",
     ["milk", "tree_nuts"],
     "praline is caramelized nuts (hazelnut/almond); milk listed directly"),
    ("durum wheat semolina, water, pesto alla genovese, salt",
     ["milk", "tree_nuts", "wheat"],
     "pesto genovese = pine nuts + parmesan; durum semolina is wheat"),
    ("potato, rapeseed oil, aioli, chives, salt",
     ["egg"],
     "aioli is an egg-yolk emulsion"),
    ("water, hydrolyzed wheat protein, soy sauce, sugar, vinegar",
     ["soy", "wheat"],
     "soy sauce is brewed from soy and wheat; hydrolyzed wheat protein is wheat"),
    ("rice flour, gram flour, asafoetida, cumin, salt",
     ["wheat"],
     "commercial asafoetida (hing) is usually cut with wheat flour as an anti-caking carrier"),
    ("tomato, mozzarella, basil, balsamic glaze, pine nuts",
     ["milk", "tree_nuts"],
     "mozzarella is milk; pine nuts are a tree nut"),
    ("water, sugar, graham cracker crumbs, butter, vanilla",
     ["milk", "wheat"],
     "graham crackers are wheat; butter is milk"),
    ("chicken, teriyaki glaze, scallion, sesame seeds, rice",
     ["soy", "wheat", "sesame"],
     "teriyaki glaze is soy sauce based (soy + wheat); sesame seeds on top"),
    ("beef, water, hoisin sauce, five spice, scallion",
     ["soy", "wheat"],
     "hoisin sauce is fermented soybean paste, usually with wheat"),
    ("potato, vegetable oil, béchamel sauce, nutmeg, cheese",
     ["milk", "wheat"],
     "béchamel is a butter+flour+milk roux (milk + wheat); cheese is milk"),
    ("flour, sugar, frangipane, apricot, butter",
     ["milk", "egg", "tree_nuts", "wheat"],
     "frangipane is an almond (tree nut) + egg + butter cream; flour is wheat"),
    ("rice, nori, imitation crab, cucumber, sesame seeds",
     ["fish", "crustacean_shellfish", "wheat", "sesame"],
     "imitation crab is surimi (fish + wheat binder) flavored as crab; sesame seeds"),
    ("water, sugar, malt extract, cocoa, milk",
     ["wheat", "milk"],
     "malt extract is derived from barley/wheat; here the wheat-family rule applies via malt wheat"),
    ("peanuts, satay sauce, coconut milk, chili, lime",
     ["peanuts", "soy"],
     "satay sauce is peanut based and usually contains soy sauce; peanuts listed directly"),
    ("chicken, breadcrumbs, parmesan crust, herbs, egg wash",
     ["milk", "egg", "wheat"],
     "breadcrumbs/crust are wheat; parmesan is milk; egg wash is egg"),
    ("sugar, butterscotch, cream, salt, vanilla",
     ["milk"],
     "butterscotch and cream are both milk-based"),
    ("water, textured vegetable protein, soy lecithin, caramel color",
     ["soy"],
     "textured vegetable protein is soy; soy lecithin confirms soy"),
    ("rice noodles, fish balls, spring onion, chili oil, stock",
     ["fish"],
     "fish balls are minced fish"),
    ("flour, sugar, cocoa, chocolate chips, natural vanilla flavor",
     ["wheat"],
     "flour is wheat; plain chocolate chips need not contain milk, so only wheat is certain"),
    ("water, anchovy paste, olive oil, garlic, black pepper",
     ["fish"],
     "anchovy paste is fish"),
    ("potato, cheese sauce, chives, cream, black pepper",
     ["milk"],
     "cheese sauce and cream are milk"),
    ("wheat flour, yeast, water, salt, sesame seed topping",
     ["wheat", "sesame"],
     "wheat flour; sesame seed topping is sesame"),
    ("rice, egg, soy sauce, peas, scallion, sesame oil",
     ["egg", "soy", "wheat", "sesame"],
     "egg fried rice: egg; soy sauce is soy+wheat; sesame oil"),
    ("sugar, condensed milk, dulce de leche, butter",
     ["milk"],
     "dulce de leche and condensed milk are both milk"),
    ("chickpea flour, water, tahini, cumin, parsley",
     ["sesame"],
     "tahini is sesame"),
    ("water, kamut flour, olive oil, rosemary, salt",
     ["wheat"],
     "kamut is a wheat variety (Khorasan wheat)"),
    ("water, spelt flour, sunflower seeds, salt, yeast",
     ["wheat"],
     "spelt is a wheat species"),
    ("vegetables, prawn crackers, vegetable oil, salt",
     ["crustacean_shellfish"],
     "prawn crackers contain prawn (crustacean)"),
    ("water, mirin, white miso, dashi, tofu",
     ["fish", "soy"],
     "dashi is made with bonito (fish); miso and tofu are soy"),
    ("flour, cocoa, marshmallow, gelatin, sugar",
     ["wheat"],
     "flour is wheat; gelatin is not a major allergen"),
    ("beef patty, brioche bun, cheddar, remoulade, pickles",
     ["milk", "egg", "wheat"],
     "brioche is wheat (+egg), cheddar is milk, remoulade is egg-based"),
    ("sugar, almond paste, egg white, powdered sugar (macaron)",
     ["egg", "tree_nuts"],
     "macaron shell is almond flour (tree nut) + egg white"),
    ("water, tamari, ginger, garlic, scallion",
     ["soy"],
     "tamari is a (usually wheat-free) soy sauce — soy only"),
    ("oats, barley malt, raisins, sunflower oil, honey",
     [],
     "oats and barley carry gluten but are NOT wheat; no wheat-family term present -> no US major allergen"),
]

# risky-looking clean negatives (text, rationale)
CLEAN: list[tuple[str, str]] = [
    ("sugar, cocoa butter, cocoa mass, vanilla",
     "cocoa butter is a fat with no milk protein; dark chocolate here has no milk"),
    ("water, coconut milk, coconut oil, guar gum, salt",
     "coconut is NOT a tree-nut major allergen per FDA Edition 5 guidance (2025-01-06)"),
    ("flour substitute: buckwheat flour, water, salt",
     "buckwheat is a pseudocereal unrelated to wheat; not a US major allergen (flour here is buckwheat only)"),
    ("butternut squash, olive oil, sage, black pepper, salt",
     "butternut squash is a vegetable; 'nut' in the name only"),
    ("cream of tartar, baking soda, cornstarch",
     "cream of tartar is potassium bitartrate; no dairy despite 'cream'"),
    ("shea butter, fragrance, tocopherol",
     "shea is technically a tree seed but not an FDA major tree-nut allergen; no milk despite 'butter'"),
    ("sugar, nutmeg, cinnamon, clove, ginger",
     "nutmeg is a seed spice, not a tree nut for allergen labeling"),
    ("water, cocoa butter, sunflower lecithin, sugar",
     "cocoa butter + sunflower lecithin: no milk, no soy"),
    ("coconut water, natural flavor, vitamin c",
     "coconut is not a major allergen (FDA Edition 5)"),
    ("watermelon, salt, lime juice",
     "no allergens; included as an easy clean control"),
    ("potato starch, rice flour, xanthan gum, salt",
     "gluten-free starches; no wheat"),
    ("water, pea protein, sunflower oil, beet juice, salt",
     "pea protein is a legume but not peanut/soy; no US major allergen"),
]


def validate(labels: list[str]) -> None:
    for lab in labels:
        if lab not in ALLERGENS:
            raise ValueError(f"unknown label {lab!r} (not in ALLERGENS)")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    order = {a: i for i, a in enumerate(ALLERGENS)}
    rows = []
    for text, labels, rationale in HIDDEN:
        validate(labels)
        rows.append({
            "text": text,
            "expected": sorted(set(labels), key=lambda a: order[a]),
            "rationale": rationale,
            "verify": False,
            "kind": "hidden",
        })
    for text, rationale in CLEAN:
        rows.append({
            "text": text,
            "expected": [],
            "rationale": rationale,
            "verify": False,
            "kind": "clean_negative",
        })

    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_hidden = sum(1 for r in rows if r["kind"] == "hidden")
    n_clean = sum(1 for r in rows if r["kind"] == "clean_negative")
    print(f"Wrote {OUT} — {len(rows)} trap rows ({n_hidden} hidden, {n_clean} clean negatives).")
    print("All rows have verify=false (DRAFT; not used in scoring until human-verified).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

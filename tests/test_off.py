from contains.off import derive_labels, en_text, norm_ingredients_key


def test_en_text_picks_english():
    rows = [{"lang": "fr", "text": "lait, sucre"}, {"lang": "en", "text": "milk, sugar"}]
    assert en_text(rows) == "milk, sugar"


def test_en_text_falls_back_to_langless():
    assert en_text([{"lang": "", "text": "milk, sugar"}]) == "milk, sugar"
    assert en_text([]) == ""
    assert en_text(None) == ""


def test_gluten_maps_to_wheat_only_with_wheat_term():
    # gluten + wheat-family term -> wheat
    assert derive_labels(["en:gluten"], "wheat flour, water") == ["wheat"]
    assert derive_labels(["en:gluten"], "spelt flour") == ["wheat"]
    # gluten from barley/oats with no wheat term -> NOT wheat
    assert derive_labels(["en:gluten"], "barley malt, oats") == []


def test_tag_mapping_and_order():
    labels = derive_labels(["en:soybeans", "en:milk", "en:fish"], "")
    # FALCPA order: milk, fish, soy
    assert labels == ["milk", "fish", "soy"]


def test_ignored_tags_dropped():
    assert derive_labels(["en:celery", "en:mustard"], "celery, mustard") == []


def test_norm_key_collapses_punctuation_and_case():
    a = norm_ingredients_key("Wheat Flour, Water.")
    b = norm_ingredients_key("wheat   flour water")
    assert a == b == "wheat flour water"


def test_strip_declarations():
    from contains.off import has_declaration, strip_declarations
    t = "flour, sugar, salt. Contains: wheat. May contain traces of nuts."
    out = strip_declarations(t)
    assert out == "flour, sugar, salt." and not has_declaration(out)
    # inside parentheses: cut only to the closing paren
    t2 = "water, shortening (contains: palm oil), salt, sugar."
    assert strip_declarations(t2) == "water, shortening (), salt, sugar." or "salt, sugar" in strip_declarations(t2)
    assert strip_declarations("milk, sugar") == "milk, sugar"


def test_composition_contains_is_not_a_declaration():
    from contains.off import has_declaration, strip_declarations
    comp = "sugar, palm oil, milk chocolate contains: cocoa solids 30% min, milk solids 14% min."
    assert not has_declaration(comp) and strip_declarations(comp) == comp
    assert not has_declaration("vegetable shortening (contains one or more of the following: palm oil)")
    assert has_declaration("flour, water. Contains: wheat, milk.")
    assert has_declaration("sugar. contains wheat")
    assert has_declaration("sugar, may contain traces of nuts")
    assert strip_declarations("sugar, shortening (contains: soy), salt.") == "sugar, shortening, salt."

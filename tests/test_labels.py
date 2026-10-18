from contains.labels import (
    ALLERGENS,
    OFF_TAG_TO_LABEL,
    contains_line,
)


def test_nine_allergens_in_falcpa_order():
    assert ALLERGENS == (
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
    assert len(ALLERGENS) == 9


def test_contains_line_follows_falcpa_order_not_input_order():
    # Given out-of-order input, the line must come back in FALCPA order.
    line = contains_line(["soy", "milk", "wheat"])
    assert line == "Contains: Milk, Wheat, Soybeans"


def test_contains_line_empty_when_no_allergen():
    assert contains_line([]) == "Contains: "


def test_off_tags_map_to_known_labels():
    for tag, label in OFF_TAG_TO_LABEL.items():
        assert tag.startswith("en:")
        assert label in ALLERGENS
    # en:gluten is intentionally NOT a flat mapping (conditional wheat rule).
    assert "en:gluten" not in OFF_TAG_TO_LABEL

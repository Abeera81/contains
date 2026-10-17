from contains.rules import detect


def test_definitional_hits():
    assert detect("wheat flour, butter, worcestershire sauce, salt") == ["milk", "fish", "wheat"]
    assert detect("chickpeas, tahini, lemon") == ["sesame"]
    assert detect("semolina, water, salt") == ["wheat"]
    assert detect("shrimp, rice, scallion") == ["crustacean_shellfish"]


def test_exclusions_do_not_false_fire():
    assert "milk" not in detect("coconut milk, guar gum, salt")
    assert "milk" not in detect("sugar, cocoa butter, cocoa mass, vanilla")
    assert "tree_nuts" not in detect("coconut water, natural flavor")
    assert "tree_nuts" not in detect("butternut squash, olive oil, sage")
    assert "wheat" not in detect("buckwheat flour, water, salt")
    assert "egg" not in detect("eggplant, olive oil, garlic")


def test_peanut_butter_is_peanut_not_milk():
    labels = detect("peanut butter, sugar, salt")
    assert "peanuts" in labels
    assert "milk" not in labels


def test_clean_line_is_empty():
    assert detect("watermelon, salt, lime juice") == []

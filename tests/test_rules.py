from contains.rules import detect_naive, detect_product


def test_naive_hits_direct_names():
    assert detect_naive("wheat flour, butter, salt") == ["milk", "wheat"]
    assert detect_naive("semolina, water, salt") == ["wheat"]
    assert detect_naive("shrimp, rice, scallion") == ["crustacean_shellfish"]
    assert detect_naive("chickpeas, tahini, lemon") == ["sesame"]


def test_naive_has_no_compound_knowledge():
    # worcestershire (anchovy), pesto (pine nut), surimi, miso, marzipan, satay
    assert detect_naive("tomato, worcestershire sauce, salt") == []
    assert detect_naive("potatoes, basil pesto, salt") == []
    assert detect_naive("surimi, water, crab extract") == ["crustacean_shellfish"]  # 'crab' is direct
    assert detect_naive("rice, miso paste, seaweed") == []
    assert detect_naive("sugar, marzipan, syrup") == []


def test_product_knows_compounds():
    assert detect_product("tomato, worcestershire sauce, salt") == ["fish"]
    assert "tree_nuts" in detect_product("potatoes, basil pesto, salt")
    assert "soy" in detect_product("rice, miso paste, seaweed")
    assert "tree_nuts" in detect_product("sugar, marzipan, syrup")


def test_exclusions_do_not_false_fire_in_either():
    for d in (detect_naive, detect_product):
        assert "milk" not in d("coconut milk, guar gum, salt")
        assert "milk" not in d("sugar, cocoa butter, cocoa mass, vanilla")
        assert "tree_nuts" not in d("coconut water, natural flavor")
        assert "tree_nuts" not in d("butternut squash, olive oil, sage")
        assert "wheat" not in d("buckwheat, water, salt")
        assert "egg" not in d("eggplant, olive oil, garlic")


def test_peanut_butter_is_peanut_not_milk():
    labels = detect_naive("peanut butter, sugar, salt")
    assert "peanuts" in labels
    assert "milk" not in labels


def test_clean_line_is_empty():
    assert detect_naive("watermelon, salt, lime juice") == []
    assert detect_product("watermelon, salt, lime juice") == []

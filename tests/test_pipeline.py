import time

from fastapi.testclient import TestClient

from contains import app as appmod
from contains.labels import ALLERGENS
from contains.pipeline import analyze
from contains.sampler import RulesOnly, SamplerError, get_sampler


class Fake:
    def __init__(self, verdicts=None, exc=None, delay=0.0, name="fake"):
        self.name, self.v, self.exc, self.delay = name, verdicts, exc, delay

    def verdicts(self, text):
        time.sleep(self.delay)
        if self.exc:
            raise self.exc
        return {a: "no" for a in ALLERGENS} | (self.v or {})


def by(res):
    return {a["key"]: a for a in res["allergens"]}


def test_fda_names_and_species():
    r = analyze("almonds, anchovies, wheat flour, soy sauce, shrimp", Fake())
    assert r["contains_line"] == ("Contains: Fish (anchovy), Crustacean shellfish (shrimp), "
                                  "Tree nuts (almonds), Wheat, Soybeans.")
    assert by(r)["soy"]["evidence"] and by(r)["soy"]["layers"] == ["rules"]


def test_undeterminable_species_falls_back_with_note():
    r = analyze("pesto, fish sauce", Fake())
    assert r["contains_line"] == "Contains: Fish, Tree nuts."
    assert any("Tree nuts" in n and "species not determinable" in n for n in r["notes"])
    assert any(n.startswith("Fish") for n in r["notes"])


def test_merge_rule_model_only_yes_and_unsure():
    r = analyze("mystery ingredient", Fake({"sesame": "yes", "milk": "unsure"}))
    s = by(r)
    assert s["sesame"]["status"] == "contains" and s["sesame"]["model_inference_only"] and s["sesame"]["layers"] == ["model"]
    assert s["milk"]["status"] == "unsure"
    assert r["verify_line"] == "Verify with supplier: Milk."


def test_rules_hit_beats_model_no_and_unsure_and_both_layers():
    r = analyze("butter, eggs", Fake({"milk": "unsure", "egg": "yes"}))
    s = by(r)
    assert s["milk"]["status"] == "contains" and s["milk"]["layers"] == ["rules"]
    assert s["egg"]["layers"] == ["rules", "model"]


def test_sampler_failure_timeout_and_unparseable_fall_back_with_banner():
    for smp in (Fake(exc=RuntimeError("boom")), Fake(delay=0.5), Fake(exc=SamplerError("unparseable"))):
        r = analyze("butter, pesto", smp, timeout=0.1)
        assert r["banner"] and "rules-only" in r["banner"]
        assert by(r)["milk"]["status"] == "contains" and r["sampler"] == "rules_only"


def test_rules_only_is_not_a_failure_banner():
    r = analyze("butter", RulesOnly())
    assert "unavailable" not in r["banner"]


def test_plurals_match_in_product_not_in_naive():
    from contains.rules import detect_naive, detect_product
    assert "tree_nuts" in detect_product("roasted pistachios") and "tree_nuts" not in detect_naive("roasted pistachios")


def test_get_sampler_config():
    assert get_sampler({"SAMPLER": "rules_only"}).name == "rules_only"
    assert get_sampler({"SAMPLER": "tinker_lora"}).name == "tinker_lora"  # misconfigured -> fails at call time
    try:
        get_sampler({"SAMPLER": "tinker_lora"}).verdicts("x")
        assert False
    except SamplerError:
        pass


def test_api_never_errors(monkeypatch):
    appmod.app.state.sampler = Fake(exc=RuntimeError("x"))
    c = TestClient(appmod.app, raise_server_exceptions=False)
    assert c.post("/api/label", json={"text": "wheat flour"}).status_code == 200
    assert c.post("/api/label", content=b"not json").json()["empty"] is True
    j = c.post("/api/label", json={"text": "wheat flour"}).json()
    assert j["banner"] and j["contains_line"] == "Contains: Wheat."
    assert c.get("/").status_code == 200 and c.get("/healthz").status_code == 200


def test_photo_path_uses_same_pipeline(monkeypatch):
    appmod.app.state.sampler = Fake()
    monkeypatch.setattr(appmod.ocr, "extract_text", lambda b: "butter, tahini")
    c = TestClient(appmod.app)
    j = c.post("/api/photo", content=b"img").json()
    assert j["ocr_text"] == "butter, tahini" and j["contains_line"] == "Contains: Milk, Sesame."

    def bad(b):
        raise appmod.ocr.OCRError("no ingredient list found in the photo")
    monkeypatch.setattr(appmod.ocr, "extract_text", bad)
    assert "Couldn't read" in c.post("/api/photo", content=b"img").json()["ocr_error"]


def test_named_wheat_flours_and_kunafa_are_wheat():
    from contains.rules import detect_product
    for t in ("all-purpose flour, sugar", "roasted kunafa", "plain flour"):
        assert "wheat" in detect_product(t), t
    assert "wheat" not in detect_product("almond flour, rice flour")

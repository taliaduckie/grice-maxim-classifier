from src.labels import MAXIMS, ZS_HYPOTHESES, MAXIM_DESCRIPTIONS

def test_all_maxims_have_descriptions():
    for m in MAXIMS:
        assert m in MAXIM_DESCRIPTIONS

def test_all_maxims_have_hypotheses():
    for m in MAXIMS:
        assert m in ZS_HYPOTHESES

def test_cooperative_is_first():
    assert MAXIMS[0] == "Cooperative"

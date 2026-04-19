from src.labels import MAXIMS, ZS_HYPOTHESES, MAXIM_DESCRIPTIONS, VIOLATION_TYPES, RELATION_NOTE


def test_all_maxims_have_descriptions():
    for m in MAXIMS:
        assert m in MAXIM_DESCRIPTIONS


def test_all_maxims_have_hypotheses():
    for m in MAXIMS:
        assert m in ZS_HYPOTHESES


def test_cooperative_is_first():
    assert MAXIMS[0] == "Cooperative"


def test_relation_note_exists():
    assert isinstance(RELATION_NOTE, str)
    assert len(RELATION_NOTE) > 0


def test_violation_types_include_required():
    for vtype in ["none", "flouting", "violating", "failed_flout", "opting_out", "clash", "unknown"]:
        assert vtype in VIOLATION_TYPES

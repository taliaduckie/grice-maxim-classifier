"""The annotation tool's non-UI logic.

The last hand-annotated file arrived with shifted columns and out-of-vocabulary
labels; these tests pin the guarantees that prevent a repeat.
"""

import csv

import pytest

from src.annotate import (
    OUT_COLUMNS, append_annotation, default_vtype, done_ids, work_order,
)


ROW = {"row_id": "abc123", "context": "Why were you late?",
       "utterance": "The weather is nice today."}


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_the_schema_agreement_py_reads(tmp_path):
    out = tmp_path / "talia.csv"
    append_annotation(out, ROW, "Relation", "unknown", 4, "")
    rows = read(out)
    assert list(rows[0]) == OUT_COLUMNS
    assert rows[0]["maxim"] == "Relation"
    assert rows[0]["confidence_1_to_5"] == "4"


def test_rejects_out_of_vocabulary_labels(tmp_path):
    out = tmp_path / "a.csv"
    with pytest.raises(ValueError):
        append_annotation(out, ROW, "Sarcasm", "flouting", 3, "")
    with pytest.raises(ValueError):
        append_annotation(out, ROW, "Manner", "flout", 3, "")   # short form
    assert not out.exists()


def test_cooperative_forces_vtype_none(tmp_path):
    """§3: none is reserved for Cooperative, and Cooperative always gets it."""
    out = tmp_path / "a.csv"
    append_annotation(out, ROW, "Cooperative", "flouting", 3, "")
    assert read(out)[0]["violation_type"] == "none"


def test_clash_requires_a_note(tmp_path):
    out = tmp_path / "a.csv"
    with pytest.raises(ValueError, match="notes"):
        append_annotation(out, ROW, "Quantity", "clash", 3, "  ")
    append_annotation(out, ROW, "Quantity", "clash", 3, "quantity vs quality")
    assert len(read(out)) == 1


def test_confidence_outside_1_to_5_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="confidence"):
        append_annotation(tmp_path / "a.csv", ROW, "Manner", "violating", 7, "")


def test_relation_defaults_to_unknown():
    """§3 standing rule — intent isn't recoverable from the surface."""
    assert default_vtype("Relation") == "unknown"
    assert default_vtype("Cooperative") == "none"
    assert default_vtype("Quality") is None


def test_resume_skips_only_finished_items(tmp_path):
    out = tmp_path / "talia.csv"
    rows = [{"row_id": f"r{i}", "context": "c?", "utterance": "u"} for i in range(6)]
    append_annotation(out, rows[0], "Manner", "violating", 3, "")
    append_annotation(out, rows[3], "Cooperative", "", 5, "")
    queue = work_order(rows, "talia", done_ids(out))
    assert {r["row_id"] for r in queue} == {"r1", "r2", "r4", "r5"}


def test_order_is_stable_per_annotator_and_differs_between_them():
    rows = [{"row_id": f"r{i}", "context": "c?", "utterance": "u"} for i in range(30)]
    a1 = [r["row_id"] for r in work_order(rows, "talia", set())]
    a2 = [r["row_id"] for r in work_order(rows, "talia", set())]
    b = [r["row_id"] for r in work_order(rows, "sam", set())]
    assert a1 == a2          # resuming shows the same order
    assert a1 != b           # two annotators don't share thread sequencing
    assert sorted(a1) == sorted(b)

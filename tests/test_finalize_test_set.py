"""Folding annotations into a frozen test set.

Run against fake annotator files so nothing depends on the real ones existing.
"""

import csv

import pytest

from src import finalize_test_set as F


def write(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


ANN_COLS = ["row_id", "context", "utterance", "maxim", "violation_type",
            "confidence_1_to_5", "notes"]


def annotator(tmp_path, name, labels):
    """labels: {row_id: (maxim, vtype)}; blank vtype allowed."""
    rows = [{"row_id": rid, "context": "q?", "utterance": f"a-{rid}",
             "maxim": m, "violation_type": v, "confidence_1_to_5": "4", "notes": ""}
            for rid, (m, v) in labels.items()]
    p = tmp_path / "annotations" / f"{name}.csv"
    write(p, rows, ANN_COLS)
    return p


# --- resolution -------------------------------------------------------------

def test_unanimous_label_is_used():
    votes = [("a", "Manner", "flouting"), ("b", "Manner", "flouting")]
    assert F.resolve("x", votes, {}, 1) == ("Manner", "flouting", "unanimous", 2)


def test_disagreement_is_unresolved():
    votes = [("a", "Manner", "flouting"), ("b", "Quality", "flouting")]
    assert F.resolve("x", votes, {}, 1) is None


def test_adjudication_overrides_a_disagreement():
    votes = [("a", "Manner", "flouting"), ("b", "Quality", "flouting")]
    adj = {"x": ("Quality", "violating")}
    assert F.resolve("x", votes, adj, 1) == ("Quality", "violating", "adjudicated", 2)


def test_single_annotator_is_flagged_not_hidden():
    votes = [("a", "Cooperative", "none")]
    assert F.resolve("x", votes, {}, 1) == ("Cooperative", "none", "single-annotator", 1)


def test_min_annotators_leaves_single_votes_unresolved():
    votes = [("a", "Cooperative", "none")]
    assert F.resolve("x", votes, {}, 2) is None


def test_violation_type_tie_falls_back_to_unknown():
    """Both agree on the maxim, split on intent — the maxim survives."""
    votes = [("a", "Relation", "flouting"), ("b", "Relation", "violating")]
    maxim, vtype, *_ = F.resolve("x", votes, {}, 1)
    assert maxim == "Relation" and vtype == "unknown"


def test_violation_type_majority_wins_with_three():
    votes = [("a", "Quality", "flouting"), ("b", "Quality", "flouting"),
             ("c", "Quality", "violating")]
    assert F.resolve("x", votes, {}, 1)[1] == "flouting"


# --- schema validation -----------------------------------------------------

def test_cooperative_requires_none():
    assert F.validate("Cooperative", "flouting") is not None
    assert F.validate("Cooperative", "none") is None


def test_violation_requires_a_type_other_than_none():
    assert F.validate("Manner", "none") is not None
    assert F.validate("Manner", "unknown") is None


def test_out_of_schema_labels_are_rejected():
    assert F.validate("Sarcasm", "flouting") is not None


# --- annotation files ---------------------------------------------------------

def test_blank_and_placeholder_labels_are_dropped(tmp_path):
    p = annotator(tmp_path, "a", {"r1": ("Manner", "flouting"),
                                  "r2": ("", ""), "r3": ("?", "")})
    votes = F.load_annotations([p])
    assert set(votes) == {"r1"}


def test_short_violation_forms_are_normalised(tmp_path):
    p = annotator(tmp_path, "a", {"r1": ("Quality", "flout")})
    assert F.load_annotations([p])["r1"][0][2] == "flouting"


def test_adjudication_sheet_reads_only_decided_rows(tmp_path):
    p = tmp_path / "adj.csv"
    write(p, [
        {"row_id": "r1", "adjudicated_maxim": "Quality", "adjudicated_violation_type": "flout"},
        {"row_id": "r2", "adjudicated_maxim": "", "adjudicated_violation_type": ""},
    ], ["row_id", "adjudicated_maxim", "adjudicated_violation_type"])
    assert F.load_adjudications(p) == {"r1": ("Quality", "flouting")}


# --- training pull ------------------------------------------------------------

def test_training_rows_matching_test_text_are_removed(tmp_path):
    """The same comment is a context under the old pairing; it must go."""
    train = tmp_path / "train.csv"
    write(train, [
        {"utterance": "reply one", "context": "THE ANSWER", "maxim": "Manner"},
        {"utterance": "THE ANSWER", "context": "old question?", "maxim": "Quality"},
        {"utterance": "unrelated", "context": "unrelated q?", "maxim": "Cooperative"},
    ], ["utterance", "context", "maxim"])
    test_rows = [{"utterance": "the  answer"}]     # case/whitespace-insensitive
    kept, removed = F.pull_from_training(test_rows, train)
    assert removed == 2
    assert [r["utterance"] for r in kept] == ["unrelated"]

"""Mechanical failure flags.

Only the derivable half of the error analysis is testable — the judgmental
categories are deliberately left to the coding sheets.
"""

import pytest

from src.error_analysis import mechanical_flags, per_class_recall


def item(gold="Quantity", by_config=None, modal=None, n_correct=0, n_runs=12,
         utt_tokens=20, ctx_tokens=30, truncated=False):
    by_config = by_config or {"A": "Quality", "B": "Quality",
                              "C": "Cooperative", "D": "Cooperative"}
    modal = modal or {"A": "Quality", "B": "Quality",
                      "C": "Cooperative", "D": "Cooperative"}
    return {
        "gold": gold, "by_config": by_config, "modal_by_config": modal,
        "n_correct": n_correct, "n_runs": n_runs,
        "accuracy": n_correct / n_runs, "truncated": truncated,
        "utt_tokens": utt_tokens, "ctx_tokens": ctx_tokens,
    }


def test_never_correct_is_flagged():
    assert "never-correct" in mechanical_flags(item(n_correct=0), {})


def test_mostly_wrong_is_distinct_from_never_correct():
    flags = mechanical_flags(item(n_correct=3), {})
    assert "mostly-wrong" in flags
    assert "never-correct" not in flags


def test_a_mostly_right_item_gets_neither():
    flags = mechanical_flags(item(n_correct=9), {})
    assert "mostly-wrong" not in flags
    assert "never-correct" not in flags


def test_collapse_when_every_config_emits_its_fallback():
    """All four configs predicted their own modal class, none matched gold."""
    assert "collapse-to-modal-class" in mechanical_flags(item(), {})


def test_partial_collapse_names_the_configs():
    flags = mechanical_flags(item(
        by_config={"A": "Quality", "B": "Quality", "C": "Relation", "D": "Manner"},
    ), {})
    assert "collapse-in-A,B" in flags
    assert "collapse-to-modal-class" not in flags


def test_no_collapse_flag_when_the_fallback_is_the_right_answer():
    """Predicting the modal class is not a failure if the modal class is gold."""
    flags = mechanical_flags(item(gold="Quality", n_correct=6, by_config={
        "A": "Quality", "B": "Quality", "C": "Quality", "D": "Quality"},
        modal={"A": "Quality", "B": "Quality", "C": "Quality", "D": "Quality"}), {})
    assert not any(f.startswith("collapse") for f in flags)


def test_config_dependent_when_some_configs_are_right():
    flags = mechanical_flags(item(gold="Quality", n_correct=6, by_config={
        "A": "Quality", "B": "Quality", "C": "Cooperative", "D": "Cooperative"}), {})
    assert "config-dependent" in flags


def test_not_config_dependent_when_all_configs_are_wrong():
    assert "config-dependent" not in mechanical_flags(item(), {})


def test_systematically_missed_class_uses_recall():
    assert "systematically-missed-class" in mechanical_flags(
        item(gold="Quantity"), {"Quantity": 0.06})
    assert "systematically-missed-class" not in mechanical_flags(
        item(gold="Quantity"), {"Quantity": 0.60})


def test_truncation_and_context_dominance_are_separate_signals():
    """A pair can be over the window without the context dwarfing the utterance."""
    long_both = mechanical_flags(
        item(utt_tokens=70, ctx_tokens=70, truncated=True), {})
    assert "truncated-at-128" in long_both
    assert "context-dominates-length" not in long_both

    lopsided = mechanical_flags(
        item(utt_tokens=5, ctx_tokens=40, truncated=False), {})
    assert "context-dominates-length" in lopsided
    assert "truncated-at-128" not in lopsided


def test_per_class_recall_pools_over_runs():
    items = [
        {"gold": "Quantity", "n_runs": 10, "n_correct": 1},
        {"gold": "Quantity", "n_runs": 10, "n_correct": 3},
        {"gold": "Manner", "n_runs": 10, "n_correct": 10},
    ]
    recall = per_class_recall(items)
    assert recall["Quantity"] == pytest.approx(0.2)
    assert recall["Manner"] == pytest.approx(1.0)


def test_per_class_recall_handles_an_absent_class():
    """test_natural has no Cooperative items; it must not appear with a score."""
    recall = per_class_recall([{"gold": "Manner", "n_runs": 4, "n_correct": 2}])
    assert "Cooperative" not in recall

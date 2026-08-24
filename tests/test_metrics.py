"""Shared metrics.

These feed adjacent tables in results/foundation_report.md, so the properties
that make those tables comparable are pinned here.
"""

import numpy as np
import pytest

from src.metrics import bootstrap_ci, macro_f1, mcnemar, present_labels


def test_present_labels_reports_only_gold_classes():
    """test_natural has no Cooperative rows; it must not appear in the average."""
    y = ["Quantity", "Manner", "Quantity"]
    assert present_labels(y) == ["Manner", "Quantity"]


def test_macro_f1_ignores_a_class_absent_from_gold():
    """Predicting an unseen class costs precision, but adds no empty F1 term."""
    y_true = ["Quantity", "Quantity", "Manner", "Manner"]
    with_coop = macro_f1(y_true, ["Cooperative", "Quantity", "Manner", "Manner"])
    assert 0.0 < with_coop < 1.0
    assert macro_f1(y_true, y_true) == pytest.approx(1.0)


def test_macro_f1_is_zero_when_every_prediction_is_wrong():
    assert macro_f1(["Quantity", "Manner"], ["Manner", "Quantity"]) == pytest.approx(0.0)


def test_bootstrap_ci_is_deterministic_for_a_given_seed():
    y = np.array(["Quantity", "Manner"] * 25)
    p = np.array(["Quantity", "Quantity"] * 25)
    assert bootstrap_ci(y, p, 200, seed=1) == bootstrap_ci(y, p, 200, seed=1)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    labs = np.array(["Quantity", "Manner", "Quality", "Relation"])
    y = rng.choice(labs, 60)
    p = np.where(rng.random(60) < 0.6, y, rng.choice(labs, 60))
    lo, hi = bootstrap_ci(y, p, 500)
    assert lo <= macro_f1(y, p) <= hi


def test_bootstrap_ci_accepts_lists_and_arrays_alike():
    y = ["Quantity", "Manner"] * 20
    p = ["Quantity", "Quantity"] * 20
    assert bootstrap_ci(y, p, 100) == bootstrap_ci(np.array(y), np.array(p), 100)


def test_mcnemar_counts_discordant_pairs_in_the_right_direction():
    y = ["A", "A", "A", "A"]
    a = ["A", "A", "B", "B"]   # right on 2
    b = ["B", "B", "B", "A"]   # right on 1
    b_count, c_count, p = mcnemar(y, a, b)
    assert b_count == 2   # a right, b wrong
    assert c_count == 1   # b right, a wrong
    assert 0.0 <= p <= 1.0


def test_mcnemar_returns_p_one_when_models_never_disagree():
    y = ["A", "B", "A"]
    assert mcnemar(y, y, y) == (0, 0, 1.0)


def test_mcnemar_is_significant_for_a_lopsided_split():
    y = ["A"] * 30
    a = ["A"] * 30
    b = ["B"] * 30
    _, _, p = mcnemar(y, a, b)
    assert p < 0.001

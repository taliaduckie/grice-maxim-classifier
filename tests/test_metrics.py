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


# --- cluster bootstrap -------------------------------------------------------

from src.metrics import cluster_bootstrap_ci


def _clustered(n_threads, per_thread, seed=0):
    rng = np.random.default_rng(seed)
    labs = np.array(["Quantity", "Manner", "Quality", "Relation"])
    y = rng.choice(labs, n_threads * per_thread)
    p = np.where(rng.random(len(y)) < 0.6, y, rng.choice(labs, len(y)))
    clusters = np.repeat(np.arange(n_threads), per_thread)
    return y, p, clusters


def test_cluster_ci_is_wider_than_item_ci_with_few_threads():
    """4 threads x 12 items — the situation test_natural was actually in."""
    y, p, c = _clustered(4, 12)
    lo_i, hi_i = bootstrap_ci(y, p, 1000)
    lo_c, hi_c = cluster_bootstrap_ci(y, p, c, 1000)
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_cluster_ci_approaches_item_ci_when_every_item_is_its_own_cluster():
    y, p, _ = _clustered(60, 1)
    c = np.arange(60)
    lo_i, hi_i = bootstrap_ci(y, p, 2000)
    lo_c, hi_c = cluster_bootstrap_ci(y, p, c, 2000)
    assert abs((hi_c - lo_c) - (hi_i - lo_i)) < 0.06


def test_cluster_ci_brackets_the_point_estimate():
    y, p, c = _clustered(10, 6)
    lo, hi = cluster_bootstrap_ci(y, p, c, 500)
    assert lo <= macro_f1(y, p) <= hi


def test_cluster_ci_is_deterministic_for_a_seed():
    y, p, c = _clustered(8, 5)
    assert cluster_bootstrap_ci(y, p, c, 300, seed=3) == cluster_bootstrap_ci(y, p, c, 300, seed=3)


def test_cluster_ci_accepts_string_cluster_ids():
    y, p, _ = _clustered(6, 5)
    c = np.repeat([f"thread-{i}?" for i in range(6)], 5)
    lo, hi = cluster_bootstrap_ci(y, p, c, 200)
    assert 0.0 <= lo <= hi <= 1.0

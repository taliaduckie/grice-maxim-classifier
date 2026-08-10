"""Reference values for the agreement coefficients.

The alpha figures below were cross-checked against the `krippendorff` package
and the Cohen figures against `sklearn.metrics.cohen_kappa_score`. Neither is a
dependency of this project, so the expected values are pinned here instead.
"""

import pytest

from src.agreement import (
    cohens_kappa, fleiss_kappa, krippendorff_alpha,
    percent_agreement, per_label_agreement,
)


def table(*values):
    """Build {item_id: label} from a sequence, skipping None."""
    return {str(i): v for i, v in enumerate(values) if v is not None}


# --- Krippendorff's alpha, nominal ----------------------------------------

def test_alpha_matches_reference_15_unit_example():
    a = table(None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3)
    b = table(1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None)
    c = table(None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4)
    alpha, n = krippendorff_alpha({"a": a, "b": b, "c": c})
    assert n == 12
    assert alpha == pytest.approx(0.691358, abs=1e-6)


def test_alpha_matches_reference_12_unit_example():
    a = table(1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None)
    b = table(1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3)
    c = table(None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None)
    alpha, _ = krippendorff_alpha({"a": a, "b": b, "c": c})
    assert alpha == pytest.approx(0.675258, abs=1e-6)


def test_alpha_is_one_for_perfect_agreement():
    a = table(1, 2, 3, 4, 5, 1, 2, 3)
    alpha, _ = krippendorff_alpha({"a": a, "b": dict(a)})
    assert alpha == pytest.approx(1.0)


def test_alpha_is_negative_for_systematic_disagreement():
    alpha, _ = krippendorff_alpha({"a": table(1, 1, 1, 1), "b": table(2, 2, 2, 2)})
    assert alpha == pytest.approx(-0.75)


def test_alpha_ignores_units_with_a_single_label():
    """An item only one annotator reached carries no agreement information."""
    a = table(1, 2, 3)
    b = table(1, 2, None)
    _, n = krippendorff_alpha({"a": a, "b": b})
    assert n == 2


# --- Cohen and Fleiss ------------------------------------------------------

def test_cohens_kappa_is_one_for_identical_tables():
    a = table("Quality", "Manner", "Relation", "Quality")
    assert cohens_kappa(a, dict(a)) == pytest.approx(1.0)


def test_cohens_kappa_is_zero_at_chance():
    """Both annotators split 50/50 with no correspondence between them."""
    a = table("Quality", "Quality", "Manner", "Manner")
    b = table("Quality", "Manner", "Quality", "Manner")
    assert cohens_kappa(a, b) == pytest.approx(0.0)


def test_cohens_kappa_ignores_unshared_items():
    a = table("Quality", "Manner", "Relation")
    b = table("Quality", "Manner", None)
    assert cohens_kappa(a, b) == pytest.approx(1.0)


def test_fleiss_kappa_is_one_for_identical_raters():
    a = table("Quality", "Manner", "Relation", "Cooperative", "Quantity")
    k, n = fleiss_kappa({"a": a, "b": dict(a), "c": dict(a)})
    assert k == pytest.approx(1.0)
    assert n == 5


def test_fleiss_kappa_uses_only_complete_items():
    a = table("Quality", "Manner", "Relation")
    b = table("Quality", "Manner", "Relation")
    c = table("Quality", "Manner", None)
    _, n = fleiss_kappa({"a": a, "b": b, "c": c})
    assert n == 2


# --- helpers ---------------------------------------------------------------

def test_percent_agreement_counts_only_shared_items():
    a = table("Quality", "Manner", "Relation")
    b = table("Quality", "Relation", None)
    rate, n = percent_agreement(a, b)
    assert n == 2
    assert rate == pytest.approx(0.5)


def test_percent_agreement_exceeds_kappa_when_one_label_dominates():
    """The reason percent agreement is never reported on its own."""
    a = table(*(["Cooperative"] * 18 + ["Manner", "Quality"]))
    b = table(*(["Cooperative"] * 18 + ["Quality", "Manner"]))
    rate, _ = percent_agreement(a, b)
    assert rate == pytest.approx(0.9)
    assert cohens_kappa(a, b) < 0.5


def test_per_label_agreement_localises_the_weak_category():
    """Annotators agree on Quality, never on Manner."""
    a = table("Quality", "Quality", "Manner", "Manner")
    b = table("Quality", "Quality", "Relation", "Relation")
    stats = per_label_agreement({"a": a, "b": b}, ["Quality", "Manner", "Relation"])
    assert stats["Quality"][0] == pytest.approx(1.0)
    assert stats["Manner"][0] == pytest.approx(0.0)

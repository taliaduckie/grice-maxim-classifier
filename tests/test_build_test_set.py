"""Sampling and gold-loading for the frozen test set.

test_test_set.py checks the files on disk; this checks the logic that produced
them — the part that decides what is held out.
"""

import pytest

from src.build_test_set import load_gold, stratified_sample
from src.provenance import row_id


def rows(spec):
    """spec: {"Manner": 4, ...} -> rows with row_ids."""
    out = []
    for maxim, n in spec.items():
        for i in range(n):
            r = {"utterance": f"{maxim}-{i}", "context": "c", "maxim": maxim}
            r["row_id"] = row_id(r)
            out.append(r)
    return out


def counts(sample):
    from collections import Counter
    return dict(Counter(r["maxim"] for r in sample))


def test_fraction_sampling_is_proportional_within_each_class():
    got = counts(stratified_sample(rows({"A": 20, "B": 10}), fraction=0.2))
    assert got == {"A": 4, "B": 2}


def test_size_sampling_hits_roughly_the_requested_total():
    sample = stratified_sample(rows({"A": 60, "B": 40}), size=20)
    assert 18 <= len(sample) <= 22


def test_every_class_survives_even_when_rounding_would_drop_it():
    """A class with one member must not vanish from the test split."""
    got = counts(stratified_sample(rows({"A": 100, "B": 1}), fraction=0.1))
    assert got["B"] >= 1


def test_sampling_is_deterministic_for_a_seed():
    pool = rows({"A": 30, "B": 30})
    a = [r["row_id"] for r in stratified_sample(pool, fraction=0.3, seed=7)]
    b = [r["row_id"] for r in stratified_sample(pool, fraction=0.3, seed=7)]
    assert a == b


def test_different_seeds_select_differently():
    pool = rows({"A": 40, "B": 40})
    a = {r["row_id"] for r in stratified_sample(pool, fraction=0.25, seed=1)}
    b = {r["row_id"] for r in stratified_sample(pool, fraction=0.25, seed=2)}
    assert a != b


def test_sampling_does_not_depend_on_input_order():
    """Otherwise the split would silently change when the corpus is re-sorted."""
    pool = rows({"A": 30, "B": 30})
    a = {r["row_id"] for r in stratified_sample(pool, fraction=0.2, seed=5)}
    b = {r["row_id"] for r in stratified_sample(list(reversed(pool)), fraction=0.2, seed=5)}
    assert a == b


# --- gold loading ----------------------------------------------------------

HEADER = ("utterance,context,predicted_maxim,predicted_violation_type,"
          "confidence,gold_maxim,gold_violation_type,subreddit,post_title\n")


def write(tmp_path, body):
    p = tmp_path / "gold.csv"
    p.write_text(HEADER + body, encoding="utf-8")
    return p


def test_load_gold_normalises_short_violation_forms(tmp_path):
    p = write(tmp_path, "u,c,Quality,unknown,0.9,Manner,flout,askreddit,title\n")
    clean, rejected = load_gold(p)
    assert not rejected
    assert clean[0]["maxim"] == "Manner"
    assert clean[0]["violation_type"] == "flouting"


def test_load_gold_rejects_a_shifted_row_instead_of_guessing(tmp_path):
    """A column shift would otherwise become a silent relabel in the test set."""
    p = write(tmp_path, "u,c,Quality,unknown,0.9,0.979,flout,askreddit,title\n")
    clean, rejected = load_gold(p)
    assert clean == []
    assert len(rejected) == 1


def test_load_gold_repairs_a_context_split_by_an_unquoted_comma(tmp_path):
    p = write(tmp_path, "u,part one, and part two,Quality,unknown,0.9,Manner,flout,sub,title\n")
    clean, rejected = load_gold(p)
    assert not rejected
    assert clean[0]["context"] == "part one, and part two"


def test_load_gold_marks_every_row_natural(tmp_path):
    p = write(tmp_path, "u,c,Quality,unknown,0.9,Manner,flout,askreddit,title\n")
    clean, _ = load_gold(p)
    assert clean[0]["source"] == "natural"

import csv
from collections import Counter
from src.labels import MAXIMS, VIOLATION_TYPES


CORPUS_PATH = "data/annotated/corpus.csv"


def _load_corpus():
    with open(CORPUS_PATH) as f:
        return list(csv.DictReader(f))


def test_all_maxim_labels_are_valid():
    rows = _load_corpus()
    for i, r in enumerate(rows):
        assert r["maxim"] in MAXIMS, (
            f"Row {i+2}: unknown maxim '{r['maxim']}'. "
            f"Valid: {MAXIMS}"
        )


def test_all_violation_types_are_valid():
    rows = _load_corpus()
    for i, r in enumerate(rows):
        assert r["violation_type"] in VIOLATION_TYPES, (
            f"Row {i+2}: unknown violation_type '{r['violation_type']}'. "
            f"Valid: {VIOLATION_TYPES}"
        )


def test_no_duplicate_pairs():
    rows = _load_corpus()
    seen = set()
    dupes = []
    for i, r in enumerate(rows):
        key = (r["utterance"], r["context"])
        if key in seen:
            dupes.append((i+2, r["utterance"][:50]))
        seen.add(key)
    assert len(dupes) == 0, (
        f"Found {len(dupes)} duplicate (utterance, context) pairs: {dupes[:5]}"
    )


def test_corpus_not_empty():
    rows = _load_corpus()
    assert len(rows) > 0


def test_all_maxims_represented():
    rows = _load_corpus()
    maxims_in_corpus = set(r["maxim"] for r in rows)
    for m in MAXIMS:
        assert m in maxims_in_corpus, f"Maxim '{m}' has zero examples in corpus"


def test_minimum_examples_per_class():
    rows = _load_corpus()
    counts = Counter(r["maxim"] for r in rows)
    for m in MAXIMS:
        assert counts[m] >= 30, (
            f"Maxim '{m}' has only {counts[m]} examples, need at least 30"
        )


def test_cooperative_always_none():
    rows = _load_corpus()
    bad = []
    for i, r in enumerate(rows):
        if r["maxim"] == "Cooperative" and r["violation_type"] != "none":
            bad.append((i+2, r["violation_type"]))
    assert len(bad) == 0, (
        f"Cooperative examples with violation_type != 'none': {bad[:5]}"
    )


def test_non_cooperative_never_none():
    rows = _load_corpus()
    bad = []
    for i, r in enumerate(rows):
        if r["maxim"] != "Cooperative" and r["violation_type"] == "none":
            bad.append((i+2, r["maxim"]))
    assert len(bad) == 0, (
        f"Non-Cooperative examples with violation_type == 'none': {bad[:5]}"
    )

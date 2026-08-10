"""Guards on the frozen test set.

The whole value of a held-out set is that it stays held out. These tests fail
loudly if a training row and a test row ever meet again.
"""

import csv
import json
from pathlib import Path

import pytest

from src.labels import MAXIMS, VIOLATION_TYPES
from src.provenance import key, norm, read_csv_tolerant, repair_row, row_id

DATA = Path("data")
TEST_DIR = DATA / "test"


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def splits():
    return {
        "train": load(DATA / "annotated" / "corpus_train.csv"),
        "natural": load(TEST_DIR / "test_natural.csv"),
        "synthetic": load(TEST_DIR / "test_synthetic.csv"),
        "pending": load(TEST_DIR / "test_natural_pending.csv"),
    }


# --- contamination ---------------------------------------------------------

def test_no_test_row_appears_in_training(splits):
    train_ids = {r["row_id"] for r in splits["train"]}
    for name in ("natural", "synthetic", "pending"):
        overlap = train_ids & {r["row_id"] for r in splits[name]}
        assert not overlap, f"{len(overlap)} rows leaked between train and {name}"


def test_no_test_row_appears_in_training_by_text(splits):
    """row_id is a hash of the text, but check the text directly too.

    A hash collision or an id column edited by hand would slip past the
    id-based check. This one cannot.
    """
    train_keys = {key(r) for r in splits["train"]}
    for name in ("natural", "synthetic"):
        overlap = train_keys & {key(r) for r in splits[name]}
        assert not overlap, f"{len(overlap)} texts shared between train and {name}"


def test_test_splits_do_not_overlap_each_other(splits):
    ids = [{r["row_id"] for r in splits[n]} for n in ("natural", "synthetic", "pending")]
    assert not (ids[0] & ids[1])
    assert not (ids[0] & ids[2])
    assert not (ids[1] & ids[2])


def test_no_duplicates_within_a_split(splits):
    for name, rows in splits.items():
        ids = [r["row_id"] for r in rows]
        assert len(ids) == len(set(ids)), f"duplicate row_ids in {name}"


# --- the manifest still describes the files on disk ------------------------

def test_manifest_matches_files(splits):
    manifest = json.loads((TEST_DIR / "manifest.json").read_text())
    for name, split_key in [("natural", "test_natural"), ("synthetic", "test_synthetic"),
                            ("pending", "test_natural_pending")]:
        recorded = manifest["splits"][split_key]
        assert recorded["rows"] == len(splits[name])
        assert sorted(r["row_id"] for r in splits[name]) == recorded["row_ids"], (
            f"{split_key} has drifted from the manifest — the test set changed "
            "underneath whatever results were computed on it"
        )


# --- label hygiene ---------------------------------------------------------

def test_labelled_splits_use_the_schema(splits):
    for name in ("train", "natural", "synthetic"):
        for r in splits[name]:
            assert r["maxim"] in MAXIMS, f"{name}: bad maxim {r['maxim']!r}"
            assert r["violation_type"] in VIOLATION_TYPES, (
                f"{name}: bad violation_type {r['violation_type']!r}")


def test_pending_sheet_is_blank(splits):
    """Annotators must not be shown a pre-filled label — see CHANGELOG ed9a122."""
    for r in splits["pending"]:
        assert r["maxim"] == ""
        assert r["violation_type"] == ""


def test_every_row_has_a_source(splits):
    for name in ("train", "natural", "synthetic"):
        for r in splits[name]:
            assert r["source"] in {"synthetic", "natural"}


def test_synthetic_test_split_covers_every_class(splits):
    present = {r["maxim"] for r in splits["synthetic"]}
    assert present == set(MAXIMS)


# --- the CSV repair path ---------------------------------------------------

def test_repair_row_rejoins_a_split_context():
    fields = ["utt", "context part one", " and part two", "Quality", "unknown",
              "0.9", "Manner", "flout", "askreddit", "title"]
    fixed = repair_row(fields, 9, context_index=1)
    assert len(fixed) == 9
    assert fixed[1] == "context part one, and part two"
    assert fixed[-1] == "title"


def test_repair_row_leaves_well_formed_rows_alone():
    fields = [f"f{i}" for i in range(9)]
    assert repair_row(fields, 9, 1) == fields


def test_repair_row_refuses_short_rows():
    assert repair_row(["a", "b"], 9, 1) is None


def test_read_csv_tolerant_recovers_the_gold_file_rows(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text(
        "utterance,context,c,d,e,f,g,h,i\n"
        "u1,plain context,1,2,3,4,5,6,7\n"
        'u2,"broken, context",1,2,3,4,5,6,7\n'.replace('"', ""),
        encoding="utf-8",
    )
    rows, repaired, dropped = read_csv_tolerant(path)
    assert len(rows) == 2
    assert repaired == 1
    assert dropped == 0
    assert rows[1]["context"] == "broken, context"


# --- join-key normalisation ------------------------------------------------

def test_norm_collapses_whitespace_and_outer_quotes():
    assert norm('  "hello   there" ') == "hello   there".replace("   ", " ")
    assert norm("a\nb") == "a b"


def test_row_id_is_stable_across_formatting_differences():
    a = {"utterance": "Hello  there", "context": "Hi"}
    b = {"utterance": ' "Hello there" ', "context": "Hi\n"}
    assert row_id(a) == row_id(b)


def test_row_id_distinguishes_different_contexts():
    a = {"utterance": "Sure", "context": "Did you finish?"}
    b = {"utterance": "Sure", "context": "Are you coming?"}
    assert row_id(a) != row_id(b)

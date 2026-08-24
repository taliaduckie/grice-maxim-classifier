"""Correction recording.

app.py wrote four columns and api.py wrote five to the same file, so whichever
ran first defined the header and the other appended rows of the wrong width.
"""

import csv

import pytest

from src.feedback import COLUMNS, record_correction


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_a_header_then_one_row(tmp_path):
    p = tmp_path / "corrections.csv"
    record_correction("u", "c", "Manner", "note", path=p)
    rows = read(p)
    assert list(rows[0]) == COLUMNS
    assert rows[0]["corrected_maxim"] == "Manner"


def test_appends_without_repeating_the_header(tmp_path):
    p = tmp_path / "corrections.csv"
    record_correction("u1", "c", "Manner", path=p)
    record_correction("u2", "c", "Quality", path=p)
    rows = read(p)
    assert len(rows) == 2
    assert [r["utterance"] for r in rows] == ["u1", "u2"]


def test_both_front_ends_produce_the_same_width(tmp_path):
    """The bug: two writers, two schemas, one file."""
    p = tmp_path / "corrections.csv"
    record_correction("from the app", "c", "Manner", path=p)          # no notes
    record_correction("from the api", "c", "Quality", "note", path=p)  # with notes
    with open(p, newline="", encoding="utf-8") as f:
        widths = {len(row) for row in csv.reader(f)}
    assert widths == {len(COLUMNS)}


def test_every_row_gets_a_timestamp(tmp_path):
    p = tmp_path / "corrections.csv"
    record_correction("u", "c", "Manner", path=p)
    assert read(p)[0]["timestamp"]


def test_rejects_an_empty_utterance(tmp_path):
    with pytest.raises(ValueError, match="utterance"):
        record_correction("   ", "c", "Manner", path=tmp_path / "c.csv")


def test_rejects_a_maxim_outside_the_schema(tmp_path):
    with pytest.raises(ValueError, match="corrected_maxim"):
        record_correction("u", "c", "Sarcasm", path=tmp_path / "c.csv")


def test_creates_the_directory_if_missing(tmp_path):
    p = tmp_path / "nested" / "dir" / "corrections.csv"
    record_correction("u", "c", "Manner", path=p)
    assert p.exists()


def test_missing_context_and_notes_become_empty_strings(tmp_path):
    p = tmp_path / "c.csv"
    record_correction("u", corrected_maxim="Manner", path=p)
    row = read(p)[0]
    assert row["context"] == "" and row["notes"] == ""

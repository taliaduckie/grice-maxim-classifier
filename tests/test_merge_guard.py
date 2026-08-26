"""merge_corpus must refuse frozen test items.

The frozen splits are only held out for as long as nothing merges their text
back into the source corpus under a new label.
"""

import csv

import pytest

from src import merge_corpus
from src.paths import TEST_NATURAL
from src.provenance import row_id


@pytest.fixture
def corpus_in_tmp(tmp_path, monkeypatch):
    """Point merge_corpus at a scratch copy of the corpus."""
    scratch = tmp_path / "corpus.csv"
    with open(merge_corpus.CORPUS_PATH, newline="", encoding="utf-8") as f:
        scratch.write_text(f.read(), encoding="utf-8")
    monkeypatch.setattr(merge_corpus, "CORPUS_PATH", scratch)
    return scratch


def corpus_size(path):
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def frozen_example():
    """A real row from the frozen natural test set."""
    with open(TEST_NATURAL, newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def test_manifest_ids_are_loaded():
    ids = merge_corpus._frozen_test_ids()
    assert len(ids) == 50 + 73 + 100      # natural + synthetic + pending


def test_a_frozen_item_is_refused(corpus_in_tmp, capsys):
    row = frozen_example()
    before = corpus_size(corpus_in_tmp)
    # relabel it Cooperative and try to merge it back in
    pipe = f"{row['utterance']}|{row['context']}|Cooperative|none"
    merge_corpus.merge_pipe_data(pipe)
    assert corpus_size(corpus_in_tmp) == before
    assert "REFUSED (frozen test item)" in capsys.readouterr().out


def test_a_novel_item_still_merges(corpus_in_tmp):
    before = corpus_size(corpus_in_tmp)
    pipe = ("A completely novel utterance for the merge guard test."
            "|A novel context?|Manner|violating")
    merge_corpus.merge_pipe_data(pipe)
    assert corpus_size(corpus_in_tmp) == before + 1


def test_refusal_is_by_text_not_by_id_column():
    """The guard recomputes row_id from the text, so a pasted row with no id
    column — or a doctored one — is still caught."""
    row = frozen_example()
    recomputed = row_id({"utterance": row["utterance"], "context": row["context"]})
    assert recomputed in merge_corpus._frozen_test_ids()

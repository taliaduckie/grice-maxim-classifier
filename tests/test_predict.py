import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import predict, predict_batch
from labels import MAXIMS

EXPECTED_KEYS = {"utterance", "context", "predicted_maxim", "violation_type", "confidence", "all_scores"}


def test_predict_returns_expected_keys():
    result = predict("The weather is nice today.", "Why were you late?")
    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Missing keys: {EXPECTED_KEYS - set(result.keys())}. "
        f"Extra keys: {set(result.keys()) - EXPECTED_KEYS}"
    )


def test_predict_maxim_is_valid():
    result = predict("Hello.", "Hi there.")
    assert result["predicted_maxim"] in MAXIMS


def test_predict_confidence_in_range():
    result = predict("Some students passed.", "Did everyone pass?")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_all_scores_sum_to_one():
    result = predict("The meeting is at 3pm.", "When is the meeting?")
    total = sum(result["all_scores"].values())
    assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"


def test_predict_all_scores_keys_match_maxims():
    result = predict("Fine.", "How was your day?")
    assert set(result["all_scores"].keys()) == set(MAXIMS), (
        f"all_scores keys {set(result['all_scores'].keys())} don't match MAXIMS {set(MAXIMS)}"
    )


def test_predict_empty_context():
    result = predict("The meeting is at 3pm.", "")
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["context"] == ""


def test_predict_empty_utterance():
    result = predict("", "What happened?")
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["predicted_maxim"] in MAXIMS


def test_predict_very_long_utterance():
    long_text = "the thing " * 200
    result = predict(long_text, "What happened?")
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["predicted_maxim"] in MAXIMS


def test_predict_deterministic():
    r1 = predict("The weather is nice today.", "Why were you late?")
    r2 = predict("The weather is nice today.", "Why were you late?")
    assert r1["predicted_maxim"] == r2["predicted_maxim"]
    assert r1["confidence"] == r2["confidence"]


def test_batch_single_row():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["utterance", "context"])
        writer.writeheader()
        writer.writerow({"utterance": "The meeting is at 3pm.", "context": "When is the meeting?"})
        path = f.name

    results = predict_batch(path)
    assert len(results) == 1
    assert "predicted_maxim" in results[0]

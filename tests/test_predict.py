import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from predict import predict

EXPECTED_KEYS = {"utterance", "context", "predicted_maxim", "violation_type", "confidence", "all_scores"}


def test_predict_returns_expected_keys():
    result = predict("The weather is nice today.", "Why were you late?")
    assert set(result.keys()) == EXPECTED_KEYS, (
        f"Missing keys: {EXPECTED_KEYS - set(result.keys())}. "
        f"Extra keys: {set(result.keys()) - EXPECTED_KEYS}"
    )


def test_predict_maxim_is_valid():
    from labels import MAXIMS
    result = predict("Hello.", "Hi there.")
    assert result["predicted_maxim"] in MAXIMS


def test_predict_confidence_in_range():
    result = predict("Some students passed.", "Did everyone pass?")
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_all_scores_sum_to_one():
    result = predict("The meeting is at 3pm.", "When is the meeting?")
    total = sum(result["all_scores"].values())
    assert abs(total - 1.0) < 0.01, f"Scores sum to {total}, expected ~1.0"

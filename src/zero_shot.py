"""
zero_shot.py
Zero-shot maxim classification using BART-MNLI.
"""

from transformers import pipeline
from labels import MAXIMS, ZS_HYPOTHESES, MaximPrediction

MODEL = "facebook/bart-large-mnli"
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        print(f"Loading {MODEL}...")
        _classifier = pipeline("zero-shot-classification", model=MODEL)
    return _classifier


def classify(utterance: str, context: str = "") -> MaximPrediction:
    clf = get_classifier()
    input_text = f"[Context: {context}] {utterance}" if context else utterance
    hypotheses = list(ZS_HYPOTHESES.values())

    result = clf(input_text, candidate_labels=hypotheses, multi_label=False)

    hyp_to_maxim = {v: k for k, v in ZS_HYPOTHESES.items()}
    scores = {
        hyp_to_maxim[label]: score
        for label, score in zip(result["labels"], result["scores"])
    }

    top_hyp = result["labels"][0]
    top_maxim = hyp_to_maxim[top_hyp]
    confidence = result["scores"][0]
    violation_type = "none" if top_maxim == "Cooperative" else "flouting"

    return MaximPrediction(
        utterance=utterance,
        context=context,
        predicted_maxim=top_maxim,
        violation_type=violation_type,
        confidence=confidence,
        all_scores=scores,
    )

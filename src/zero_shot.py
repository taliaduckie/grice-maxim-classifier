from transformers import pipeline
from labels import MAXIMS, ZS_HYPOTHESES, MaximPrediction

MODEL = "facebook/bart-large-mnli"

# Lazy loading — don't download until we actually need it.
# This is both good practice and a way of deferring the download
# to the moment you've actually committed to running this.
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        print(f"Loading {MODEL}...")
        print("(If this is your first run, this will take a while. It's 1.6GB. That's fine.)")
        _classifier = pipeline("zero-shot-classification", model=MODEL)
    return _classifier


def classify(utterance: str, context: str = "") -> MaximPrediction:
    clf = get_classifier()

    # context format hasn't been formally ablated
    input_text = f"[Context: {context}] {utterance}" if context else utterance

    hypotheses = list(ZS_HYPOTHESES.values())

    # multi_label=False forces winner-take-all: exactly one maxim wins.
    # this means "clash" (two maxims violated simultaneously) is architecturally
    # unreachable in this path. clash is defined in VIOLATION_TYPES but the
    # zero-shot setup literally cannot produce it — you'd need multi_label=True
    # with a threshold and logic to detect when two maxims score high.
    # that's a TODO. for now, clash is reserved for fine-tuned + human annotation.
    result = clf(input_text, candidate_labels=hypotheses, multi_label=False)

    # Map hypothesis strings back to maxim names.
    # We passed the hypothesis TEXT to the model (because it doesn't know
    # what 'Relation' means), so now we need to reverse that mapping.
    # the indignity of having to reverse your own dictionary
    hyp_to_maxim = {v: k for k, v in ZS_HYPOTHESES.items()}
    scores = {
        hyp_to_maxim[label]: score
        for label, score in zip(result["labels"], result["scores"])
    }

    top_hyp   = result["labels"][0]
    top_maxim = hyp_to_maxim[top_hyp]
    confidence = result["scores"][0]

    # For zero-shot, we genuinely don't know if it's flouting or violating.
    # the old default was "flouting" which sounds reasonable until you realize
    # these predictions seed the annotation pipeline and annotators anchor
    # on whatever they see first. "flouting" as a default = flouting-heavy
    # corpus = biased fine-tuning. the model doesn't know. say so.
    violation_type = "none" if top_maxim == "Cooperative" else "unknown"

    # ship it. this dataclass is doing more theoretical work than my undergrad thesis did.
    return MaximPrediction(
        utterance=utterance,
        context=context,
        predicted_maxim=top_maxim,
        violation_type=violation_type,
        confidence=confidence,
        all_scores=scores,
    )

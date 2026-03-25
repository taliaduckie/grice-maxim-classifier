"""
zero_shot.py

Zero-shot maxim classification using BART-MNLI.

The setup: we have five hypotheses, (one per maxim) and we ask the NLI model
to score how well each hypothesis entails the input utterance. The highest
score wins.

UGH. is good enough to bootstrap annotation of new data, which is the
actual use case here. am not going to claim this is solving pragmatics.
pragmatics is famously not solved.

the model we're using is facebook/bart-large-mnli, which was trained on
MultiNLI and SNLI. It has no idea what a Gricean maxim is. (join the club boi.)
It knows what "irrelevant to what was asked" means in the loose distributional
sense that large language models know things. proceed to debate the stochastic
parrot argument. yay fun.

On download: the first run will download ~1.6GB. This is normal.
Get a coffee. Maybe two.
"""

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
    """
    Classify an utterance in optional context.

    Context is prepended as: "[Context: {context}] {utterance}"
    This is a simple but surprisingly effective way to give the model
    the discourse situation without fine-tuning. The brackets are doing
    real work here — they signal "this is metadata, not the utterance."

    Args:
        utterance: the thing someone said
        context:   what they were responding to, if known.
                   The classifier is much better WITH context.
                   'The weather is nice today' without context is just
                   a statement about weather. With 'Why were you late?'
                   it becomes something more interesting.

    Returns:
        MaximPrediction with the top label and full score distribution.

    Notes on failure modes:
        - Manner is chronically underdetected. The model has no good
          theory of what 'unnecessarily long' means for a given context.
        - Quality flouting (irony, hyperbole) is hard. 'I've told you
          a million times' often gets labeled Quality-violating, which
          misses the point entirely. This is a known limitation.
        - Very short utterances ('Fine.', 'Sure.', 'Whatever.') are
          ambiguous even to humans. The model's uncertainty there is
          appropriate, actually.
    """
    clf = get_classifier()

    # Combine context and utterance into a single input string.
    # Tested several formats; this one performed best on manual eval.
    input_text = f"[Context: {context}] {utterance}" if context else utterance

    # feeding the model plain english descriptions of abstract pragmatic categories
    # and hoping for the best. this is either clever or unhinged. possibly both.
    hypotheses = list(ZS_HYPOTHESES.values())

    # multi_label=False because an utterance violates at most one maxim at a time.
    # this is a simplification. grice would not approve. but grice is not reviewing
    # this code so we're fine.
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

    # For zero-shot, we can only reliably distinguish cooperative from
    # non-cooperative. The flouting/violating distinction requires either
    # a fine-tuned model or more context than we typically have.
    # So: if the top label is Cooperative, mark as none.
    # Otherwise, mark as flouting — the optimistic default.
    # (Most interesting cases are flouting anyway.)
    violation_type = "none" if top_maxim == "Cooperative" else "flouting"

    # ship it. this dataclass is doing more theoretical work than my undergrad thesis did.
    return MaximPrediction(
        utterance=utterance,
        context=context,
        predicted_maxim=top_maxim,
        violation_type=violation_type,
        confidence=confidence,
        all_scores=scores,
    )

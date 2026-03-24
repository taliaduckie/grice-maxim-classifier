"""
predict.py

The part you actually run WOOWOOWOOOOOOO

Usage:
    python predict.py --text "The weather is nice today." --context "Why were you late?"
    python predict.py --text "Some students passed." --context "Did everyone pass?"
    python predict.py --text "Fine." --context "Are you okay with this?"

If you have a fine-tuned model in ../models/roberta-grice/, it'll use that.
If you don't (you probably don't, the annotated corpus has eight examples),
it falls back to the zero-shot BART-MNLI baseline.

The zero-shot version is surprisingly usable! It gets Relation flouting
reliably, Quantity most of the time, Quality not as much (when it's obvious).
Manner is a mess. But Manner is a mess for humans too, so. Womp womp.

One thing I want to add eventually: a --batch flag that reads from a CSV
and outputs predictions alongside gold labels for eval. For now this is
one-shot only. Use a shell loop if you need to run it on a file.
"""

import argparse
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "roberta-grice"


def predict(text: str, context: str = "") -> dict:
    """
    Run inference on a single utterance.

    Routing logic:
        - If ../models/roberta-grice/ exists: use fine-tuned RoBERTa
        - Otherwise: use zero-shot BART-MNLI

    The fine-tuned model will obviously be better once you have enough
    annotated data to actually fine-tune on. Eight examples is not
    enough. I know. I'm working on it.
    """
    if MODEL_DIR.exists():
        # Fine-tuned model path.
        # If you're seeing this message and you fine-tuned on the eight
        # seed examples in corpus.csv: don't. Add more first.
        from transformers import pipeline
        clf = pipeline("text-classification", model=str(MODEL_DIR))
        input_text = f"[Context: {context}] {text}" if context else text
        result = clf(input_text, return_all_scores=True)[0]
        scores = {r["label"]: r["score"] for r in result}
        top = max(scores, key=scores.get)
        return {
            "utterance": text,
            "context": context,
            "predicted_maxim": top,
            "confidence": scores[top],
            "all_scores": scores,
        }
    else:
        # Zero-shot baseline. This is what runs until you annotate
        # enough data to fine-tune. See data/annotated/ for the format.
        print("No fine-tuned model found — using zero-shot baseline.")
        print("(Add more labeled examples to data/annotated/corpus.csv to train one.)")
        from src.zero_shot import classify
        pred = classify(text, context)
        return {
            "utterance": pred.utterance,
            "context": pred.context,
            "predicted_maxim": pred.predicted_maxim,
            "violation_type": pred.violation_type,
            "confidence": pred.confidence,
            "all_scores": pred.all_scores,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a single utterance by Gricean maxim violation.",
        epilog="Example: python predict.py --text 'Fine.' --context 'Are you happy about this?'"
    )
    parser.add_argument("--text",    required=True,  help="The utterance to classify.")
    parser.add_argument("--context", default="",     help="What the utterance was responding to.")
    args = parser.parse_args()

    result = predict(args.text, args.context)
    print(json.dumps(result, indent=2))

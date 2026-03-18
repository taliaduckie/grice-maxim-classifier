"""
predict.py
Inference CLI. Uses fine-tuned model if available, else zero-shot baseline.

Usage:
    python predict.py --text "Nice weather." --context "Why were you late?"
"""

import argparse
import json
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "models" / "roberta-grice"


def predict(text: str, context: str = "") -> dict:
    if MODEL_DIR.exists():
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
        print("No fine-tuned model found — using zero-shot baseline.")
        from zero_shot import classify
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--context", default="")
    args = parser.parse_args()
    result = predict(args.text, args.context)
    print(json.dumps(result, indent=2))

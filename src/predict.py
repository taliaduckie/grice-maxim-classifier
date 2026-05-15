"""
predict.py

The part you actually run WOOWOOWOOOOOOO

Usage:
    # single utterance
    python predict.py --text "The weather is nice today." --context "Why were you late?"

    # batch mode — run on a CSV, compare against gold labels if present
    python predict.py --batch data/annotated/corpus.csv
    python predict.py --batch data/annotated/corpus.csv --output results.csv

If you have a fine-tuned model in ../models/roberta-grice/, it'll use that.
Otherwise it falls back to zero-shot BART-MNLI.

The fine-tuned model hit 0.84 macro F1 on 229 examples with stratified split.
Quality: 0.95, Quantity: 0.86, Relation: 0.84, Manner: 0.82, Cooperative: 0.75.
still can't reliably detect sarcasm but honestly neither can most humans so.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# make imports work from src/ or project root
sys.path.insert(0, str(Path(__file__).parent))

MODEL_DIR = Path(__file__).parent.parent / "models" / "roberta-grice"

# lazy-loaded pipeline cache. batch mode was reloading the model on
# every prediction before this — slow.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        _pipeline = pipeline("text-classification", model=str(MODEL_DIR))
    return _pipeline


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
        clf = _get_pipeline()
        # same bracketed format as zero_shot.py
        input_text = f"[Context: {context}] {text}" if context else text
        # top_k=None replaces deprecated return_all_scores
        result = clf(input_text, top_k=None)
        scores = {r["label"]: r["score"] for r in result}
        top = max(scores, key=scores.get)
        # the model only predicts maxim, not violation type
        # cooperative gets "none", everything else is "unknown" until we
        # have a second head or separate model for it
        violation_type = "none" if top == "Cooperative" else "unknown"
        return {
            "utterance": text,
            "context": context,
            "predicted_maxim": top,
            "violation_type": violation_type,
            "confidence": scores[top],
            "all_scores": scores,
        }
    else:
        # zero-shot fallback when no fine-tuned model is saved
        print("No fine-tuned model found — using zero-shot baseline.")
        print("(Add more labeled examples to data/annotated/corpus.csv to train one.)")
        from zero_shot import classify
        pred = classify(text, context)
        # manually unpacking so keys match the fine-tuned path
        return {
            "utterance": pred.utterance,
            "context": pred.context,
            "predicted_maxim": pred.predicted_maxim,
            "violation_type": pred.violation_type,
            "confidence": pred.confidence,
            "all_scores": pred.all_scores,
        }


def predict_batch(csv_path: str, output_path: str = None) -> list:
    """
    Run predictions on a CSV file. If the CSV has 'maxim' and/or
    'violation_type' columns, treat them as gold labels and report
    accuracy. because what good is a model if you can't measure
    how wrong it is.

    Expected CSV columns: utterance, context (optional), maxim (optional)
    Outputs: the input columns plus predicted_maxim, confidence, correct (if gold exists)
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "utterance" not in df.columns:
        raise ValueError("CSV must have an 'utterance' column")

    has_gold = "maxim" in df.columns
    results = []
    n = len(df)

    for i, row in df.iterrows():
        utterance = str(row["utterance"])
        context = str(row.get("context", "")) if "context" in df.columns else ""
        # pandas gives "nan" for empty cells
        if context == "nan":
            context = ""

        pred = predict(utterance, context)

        result = {
            "utterance": utterance,
            "context": context,
            "predicted_maxim": pred["predicted_maxim"],
            "predicted_violation_type": pred["violation_type"],
            "confidence": f"{pred['confidence']:.3f}",
        }

        if has_gold:
            gold = str(row["maxim"])
            result["gold_maxim"] = gold
            result["correct"] = pred["predicted_maxim"] == gold

        results.append(result)

        status = ""
        if has_gold:
            status = " ✓" if result["correct"] else " ✗"
        print(f"  [{i+1}/{n}] {utterance[:50]:<50} → {pred['predicted_maxim']} ({pred['confidence']:.0%}){status}")

    # summary stats if we have gold labels
    if has_gold:
        correct = sum(1 for r in results if r["correct"])
        print(f"\nAccuracy: {correct}/{n} ({correct/n:.1%})")

        # per-class breakdown — aggregate accuracy hides class-level failures
        from collections import Counter
        class_correct = Counter()
        class_total = Counter()
        for r in results:
            class_total[r["gold_maxim"]] += 1
            if r["correct"]:
                class_correct[r["gold_maxim"]] += 1
        print("\nPer-class accuracy:")
        for maxim in sorted(class_total):
            c, t = class_correct[maxim], class_total[maxim]
            print(f"  {maxim:<12} {c}/{t} ({c/t:.0%})")

        # confusion matrix shows which classes get confused with which
        from labels import MAXIMS
        all_labels = sorted(set(MAXIMS) & (set(class_total) | set(r["predicted_maxim"] for r in results)))
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        matrix = [[0] * len(all_labels) for _ in all_labels]
        for r in results:
            gold_idx = label_to_idx.get(r["gold_maxim"])
            pred_idx = label_to_idx.get(r["predicted_maxim"])
            if gold_idx is not None and pred_idx is not None:
                matrix[gold_idx][pred_idx] += 1

        col_width = max(len(l) for l in all_labels) + 2
        header = " " * col_width + "".join(l[:6].rjust(7) for l in all_labels)
        print(f"\nConfusion matrix (rows=gold, cols=predicted):\n{header}")
        for i, label in enumerate(all_labels):
            row = label.ljust(col_width) + "".join(str(matrix[i][j]).rjust(7) for j in range(len(all_labels)))
            print(row)

    # write output CSV if requested
    if output_path:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify utterances by Gricean maxim violation.",
        epilog="Example: python predict.py --text 'Fine.' --context 'Are you happy about this?'"
    )
    # single mode
    parser.add_argument("--text",    default=None,   help="The utterance to classify.")
    parser.add_argument("--context", default="",     help="What the utterance was responding to.")
    # batch mode
    parser.add_argument("--batch",   default=None,   help="Path to CSV file for batch prediction.")
    parser.add_argument("--output",  default=None,   help="Path to write batch results CSV.")
    args = parser.parse_args()

    if args.batch:
        predict_batch(args.batch, args.output)
    elif args.text:
        result = predict(args.text, args.context)
        print(json.dumps(result, indent=2))
    else:
        parser.error("either --text or --batch is required")

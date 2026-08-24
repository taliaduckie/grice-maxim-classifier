import argparse
import csv
import json
import sys
from pathlib import Path

# make imports work from src/ or project root

from paths import MODEL_DIR

# cache so batch mode doesn't reload the model every iteration
_pipeline = None


def predict(text: str, context: str = "") -> dict:
    """Run inference. Uses fine-tuned model if available, else zero-shot BART-MNLI."""
    global _pipeline
    if MODEL_DIR.exists():
        if _pipeline is None:
            from transformers import pipeline
            _pipeline = pipeline("text-classification", model=str(MODEL_DIR))
        clf = _pipeline
        # Feed the model a sentence PAIR, matching training. RoBERTa is
        # pretrained on <s> A </s></s> B </s> and carries positional/segment
        # structure for that format; GriceDataset trains on
        # tokenizer(utterance, context) as a pair. A flat "[Context: ..] .."
        # string asks the model to recover pair structure through an interface
        # it was never shaped for — it diverged from the pair encoding on ~1/3
        # of inputs. Always pass text_pair (empty string when no context) so
        # inference is byte-for-byte the training encoding. max_length=128
        # matches dataset.py so truncation happens on the same side/length.
        pair = {"text": text, "text_pair": context or ""}
        # top_k=None gets all scores (return_all_scores is deprecated)
        result = clf(pair, top_k=None, truncation=True, max_length=128)
        scores = {r["label"]: r["score"] for r in result}
        top = max(scores, key=scores.get)
        # model only predicts maxim, not violation type
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
        print("no fine-tuned model — falling back to zero-shot")
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


def predict_batch(csv_path: str, output_path: str = None) -> list:
    """Run predictions on a CSV. Reports accuracy if 'maxim' column exists."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "utterance" not in df.columns:
        raise ValueError("need an 'utterance' column")

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

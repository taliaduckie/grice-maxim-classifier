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

# make imports work whether you run from src/ or the project root.
# this is the kind of thing that shouldn't be hard but here we are.
sys.path.insert(0, str(Path(__file__).parent))

# if this path exists you are living in the future where i annotated enough data.
# congratulations future me. or condolences. depending on how the f1 looks.
MODEL_DIR = Path(__file__).parent.parent / "models" / "roberta-grice"

# lazy-loaded pipeline cache. loading the model 229 times in batch mode
# was a manner violation of the highest order. now it loads once and
# stays loaded like a responsible adult.
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
        # Fine-tuned model path. it WORKS now. 229 examples and a tokenizer
        # save later, we have a model that hits 0.84 macro F1. take THAT,
        # manner. (manner is still hard but we don't talk about manner.)
        clf = _get_pipeline()
        # same bracketed format as zero_shot.py. consistency! the manner maxim
        # would be proud of me. (it wouldn't. nothing satisfies manner.)
        input_text = f"[Context: {context}] {text}" if context else text
        # top_k=None gets scores for all labels. return_all_scores is deprecated
        # because transformers loves renaming things. keep up or get left behind.
        result = clf(input_text, top_k=None)
        scores = {r["label"]: r["score"] for r in result}
        top = max(scores, key=scores.get)
        # the model only predicts the maxim, not the violation type.
        # we don't know if it's flouting or violating and we're not going
        # to pretend we do. "unknown" until there's a second head or a
        # separate model. cooperative gets "none" because that one's obvious.
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
        # Zero-shot baseline. This is what runs until you annotate
        # enough data to fine-tune. See data/annotated/ for the format.
        print("No fine-tuned model found — using zero-shot baseline.")
        print("(Add more labeled examples to data/annotated/corpus.csv to train one.)")
        from zero_shot import classify
        pred = classify(text, context)
        # manually unpacking instead of asdict() because i want the keys
        # to match the fine-tuned path above. yes this is annoying.
        # yes i could use a shared serializer. no i'm not doing that today.
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
        raise ValueError("CSV must have an 'utterance' column. c'mon.")

    has_gold = "maxim" in df.columns
    results = []
    n = len(df)

    for i, row in df.iterrows():
        utterance = str(row["utterance"])
        context = str(row.get("context", "")) if "context" in df.columns else ""
        # nan check — pandas loves giving you nan for empty cells
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

        # progress because running 229 predictions in silence is a manner violation
        status = ""
        if has_gold:
            status = " ✓" if result["correct"] else " ✗"
        print(f"  [{i+1}/{n}] {utterance[:50]:<50} → {pred['predicted_maxim']} ({pred['confidence']:.0%}){status}")

    # summary stats if we have gold labels
    if has_gold:
        correct = sum(1 for r in results if r["correct"])
        print(f"\nAccuracy: {correct}/{n} ({correct/n:.1%})")

        # per-class breakdown because the aggregate number lies.
        # a model that gets 90% accuracy by always predicting Cooperative
        # is not a good model. it's a cooperative model. which is ironic.
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

        # confusion matrix — the part where you find out your model thinks
        # all sarcasm is Quantity and all Cooperative responses are Quality.
        # or at least it used to. hopefully the 20 sarcasm examples fixed that.
        from labels import MAXIMS
        all_labels = sorted(set(MAXIMS) & (set(class_total) | set(r["predicted_maxim"] for r in results)))
        label_to_idx = {l: i for i, l in enumerate(all_labels)}
        matrix = [[0] * len(all_labels) for _ in all_labels]
        for r in results:
            gold_idx = label_to_idx.get(r["gold_maxim"])
            pred_idx = label_to_idx.get(r["predicted_maxim"])
            if gold_idx is not None and pred_idx is not None:
                matrix[gold_idx][pred_idx] += 1

        # print it. it's not pretty but it's informative. manner would
        # have opinions about the formatting. manner always has opinions.
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
        # json.dumps because pretty-printing dicts is a quantity violation
        print(json.dumps(result, indent=2))
    else:
        parser.error("either --text or --batch is required. pick one. or both. no wait, just one.")

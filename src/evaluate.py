"""Evaluate a saved checkpoint against the frozen test sets.

    python3 src/evaluate.py                       # models/roberta-grice
    python3 src/evaluate.py --model models/roberta-grice-1197
    python3 src/evaluate.py --split data/test/test_natural_v2.csv

Reports macro F1 over the classes present in each split, a 95% bootstrap CI,
and — where the split has a `thread` or `post_title` column — a cluster
bootstrap over threads, which is the right interval for data drawn from a
handful of discussions. Writes per-row predictions next to the results so
error analysis can run without re-inference.
"""

import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

from labels import MAXIMS
from metrics import bootstrap_ci, cluster_bootstrap_ci, macro_f1, present_labels
from paths import MODEL_DIR, RESULTS_DIR, TEST_NATURAL, TEST_SYNTHETIC, rel
from provenance import load_labelled
from training_utils import HPARAMS

warnings.filterwarnings("ignore")


def predict_split(pipe, rows, max_length):
    enc = [{"text": r["utterance"], "text_pair": r["context"]} for r in rows]
    out = pipe(enc, truncation=True, max_length=max_length, batch_size=16)
    return [max(p, key=lambda d: d["score"])["label"] for p in out]


def thread_of(row):
    return row.get("thread") or row.get("post_title") or None


def evaluate(pipe, name, path, n_boot):
    rows = load_labelled(path)
    y_true = [r["maxim"] for r in rows]
    y_pred = predict_split(pipe, rows, HPARAMS["max_length"])
    clusters = [thread_of(r) for r in rows]
    has_clusters = all(clusters) and len(set(clusters)) > 1

    result = {
        "split": name,
        "path": rel(path),
        "n": len(rows),
        "classes_present": present_labels(y_true),
        "macro_f1": macro_f1(y_true, y_pred),
        "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
        "ci95_item": bootstrap_ci(y_true, y_pred, n_boot),
        "ci95_cluster": cluster_bootstrap_ci(y_true, y_pred, clusters, n_boot) if has_clusters else None,
        "n_threads": len(set(clusters)) if has_clusters else None,
        "predicted_distribution": dict(Counter(y_pred)),
        "pct_predicted_cooperative": float(np.mean([p == "Cooperative" for p in y_pred])),
        "predictions": [{"row_id": r.get("row_id"), "gold": t, "pred": p}
                        for r, t, p in zip(rows, y_true, y_pred)],
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(MODEL_DIR))
    ap.add_argument("--split", action="append",
                    help="Test CSV to evaluate; repeatable. Default: both frozen sets.")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--out", default=None, help="Results JSON path.")
    args = ap.parse_args()

    from transformers import pipeline
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"No model at {model_dir}", file=sys.stderr)
        return 1
    pipe = pipeline("text-classification", model=str(model_dir), top_k=None)

    splits = ([(Path(s).stem, Path(s)) for s in args.split] if args.split
              else [("test_synthetic", TEST_SYNTHETIC), ("test_natural", TEST_NATURAL)])

    print(f"Model: {rel(model_dir)}\n")
    print(f"{'split':<20}{'n':>5}{'macroF1':>9}{'item CI':>16}{'thread CI':>16}"
          f"{'thr':>5}{'%coop':>7}")
    print("-" * 78)
    results = []
    for name, path in splits:
        r = evaluate(pipe, name, path, args.bootstrap)
        results.append(r)
        lo, hi = r["ci95_item"]
        cl = (f"[{r['ci95_cluster'][0]:.2f},{r['ci95_cluster'][1]:.2f}]"
              if r["ci95_cluster"] else "—")
        print(f"{name:<20}{r['n']:>5}{r['macro_f1']:>9.3f}{f'[{lo:.2f},{hi:.2f}]':>16}"
              f"{cl:>16}{(r['n_threads'] or '—'):>5}{r['pct_predicted_cooperative']:>7.0%}")

    out = Path(args.out) if args.out else RESULTS_DIR / f"eval_{model_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": rel(model_dir), "bootstrap": args.bootstrap,
                               "results": results}, indent=2) + "\n")
    print(f"\nWrote {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

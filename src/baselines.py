"""Non-neural baselines with uncertainty (checklist items 7 and 10).

The question this answers: how much of the classifier's score is the classifier?

If TF-IDF + logistic regression lands near RoBERTa, the transformer is not
buying pragmatic understanding. If a model given *nothing but surface
statistics* — utterance length, context length, question marks, word overlap —
lands near TF-IDF, then neither is, and the task as posed is mostly register
detection. That last baseline is the diagnostic one and it is deliberately
stupid: it cannot read.

Baselines, weakest first:

  majority        always the most frequent training class
  stratified      random, drawn from the training class priors (chance)
  surface         13 hand-built numeric features, no lexical content at all
  tfidf-utt       word 1-2 grams over the utterance only
  tfidf-ctx       word 1-2 grams over utterance + context (separate blocks)
  tfidf-char      char 3-5 grams over the utterance, catches style not words

Every model is evaluated on both frozen test sets, under the four training
regimes of the item-8 ablation grid (synthetic / natural / both). Hyperparameters
are fixed in this file and were never tuned against the test sets.

Uncertainty:
  - macro F1 as mean +/- sd over N bootstrap resamples of the *training* set,
    which is the spread attributable to which examples you happened to collect
  - 95% percentile CI from bootstrap resamples of the *test* set, which is the
    spread attributable to which examples you happened to test on
  - McNemar's exact test for the paired model comparisons that matter

Usage:
    python3 src/baselines.py
    python3 src/baselines.py --seeds 20 --bootstrap 2000
"""

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from labels import MAXIMS

DATA_DIR = Path(__file__).parent.parent / "data"
TRAIN_PATH = DATA_DIR / "annotated" / "corpus_train.csv"
TEST_NATURAL = DATA_DIR / "test" / "test_natural.csv"
TEST_SYNTHETIC = DATA_DIR / "test" / "test_synthetic.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Fixed. Not tuned on the test sets — see module docstring.
LOGREG_KWARGS = dict(max_iter=2000, class_weight="balanced", C=1.0)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("maxim") in MAXIMS]
    return rows


def texts(rows, use_context):
    """Feature text. Context is fenced so the vectorizer can't blend the two."""
    if use_context:
        return [f"{r['utterance']} [SEP] {r['context']}" for r in rows]
    return [r["utterance"] for r in rows]


def labels_of(rows):
    return np.array([r["maxim"] for r in rows])


# --------------------------------------------------------------------------
# the deliberately illiterate baseline
# --------------------------------------------------------------------------

WORD = re.compile(r"[a-z']+")


def surface_features(rows):
    """Numeric features only. No vocabulary, no content words, no topic.

    Anything this model gets right is available from shape alone: how long the
    turn is, how much it echoes the question, how it's punctuated. If it scores
    well, the labels are correlated with register rather than with pragmatics.
    """
    out = []
    for r in rows:
        u, c = r["utterance"], r["context"]
        uw = WORD.findall(u.lower())
        cw = WORD.findall(c.lower())
        uset, cset = set(uw), set(cw)
        overlap = len(uset & cset) / (len(uset | cset) or 1)
        out.append([
            len(u),                                     # utterance chars
            len(uw),                                    # utterance words
            len(cw),                                    # context words
            len(uw) / (len(cw) + 1),                    # length ratio
            math.log1p(len(uw)),
            overlap,                                    # jaccard with context
            len(uset & cset) / (len(uset) or 1),        # coverage of utterance
            u.count("?"),
            u.count("!"),
            c.count("?"),                               # was a question asked
            sum(ch.isupper() for ch in u) / (len(u) or 1),
            u.count(",") / (len(uw) + 1),               # clause density
            len(set(uw)) / (len(uw) or 1),              # type-token ratio
        ])
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def build_model(name, seed):
    """Return (fit_fn, predict_fn) closing over a fresh estimator."""
    if name == "majority":
        state = {}

        def fit(rows):
            state["label"] = Counter(labels_of(rows)).most_common(1)[0][0]

        def predict(rows):
            return np.array([state["label"]] * len(rows))

    elif name == "stratified":
        state = {}

        def fit(rows):
            counts = Counter(labels_of(rows))
            total = sum(counts.values())
            state["labels"] = sorted(counts)
            state["probs"] = [counts[l] / total for l in state["labels"]]

        def predict(rows):
            rng = np.random.default_rng(seed)
            return rng.choice(state["labels"], size=len(rows), p=state["probs"])

    elif name == "surface":
        clf = Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(random_state=seed, **LOGREG_KWARGS)),
        ])

        def fit(rows):
            clf.fit(surface_features(rows), labels_of(rows))

        def predict(rows):
            return clf.predict(surface_features(rows))

    elif name in ("tfidf-utt", "tfidf-ctx", "tfidf-char"):
        use_context = name == "tfidf-ctx"
        if name == "tfidf-char":
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True, max_features=50000)
        else:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, strip_accents="unicode")
        clf = Pipeline([
            ("vec", vec),
            ("lr", LogisticRegression(random_state=seed, **LOGREG_KWARGS)),
        ])

        def fit(rows):
            clf.fit(texts(rows, use_context), labels_of(rows))

        def predict(rows):
            return clf.predict(texts(rows, use_context))

    else:
        raise ValueError(f"unknown model: {name}")

    return fit, predict


MODELS = ["majority", "stratified", "surface", "tfidf-utt", "tfidf-ctx", "tfidf-char"]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def present_labels(y_true):
    """Score only over classes the gold set actually contains.

    test_natural has no Cooperative examples, so a 5-class macro average would
    silently fold in an undefined class and drag the number toward zero for
    reasons that have nothing to do with the model.
    """
    return sorted(set(y_true))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=present_labels(y_true),
                    average="macro", zero_division=0)


def bootstrap_ci(y_true, y_pred, n_boot, seed=0):
    """95% percentile CI for macro F1 over resamples of the test set."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(set(yt)) < 2:
            continue
        scores.append(macro_f1(yt, yp))
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar on the two models' disagreements. Returns (b, c, p)."""
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int(np.sum(a_right & ~b_right))   # a right, b wrong
    c = int(np.sum(~a_right & b_right))   # b right, a wrong
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(model_name, train_rows, test_sets, seeds, n_boot):
    """Point estimate at seed 0 plus spread over training-set resamples."""
    fit, predict = build_model(model_name, 0)
    fit(train_rows)

    result = {"model": model_name, "n_train": len(train_rows), "splits": {}}
    for split_name, test_rows in test_sets.items():
        y_true = labels_of(test_rows)
        y_pred = predict(test_rows)

        lo, hi = bootstrap_ci(y_true, y_pred, n_boot)
        report = classification_report(
            y_true, y_pred, labels=present_labels(y_true),
            output_dict=True, zero_division=0,
        )
        result["splits"][split_name] = {
            "macro_f1": macro_f1(y_true, y_pred),
            "accuracy": float(np.mean(y_pred == y_true)),
            "ci95": [lo, hi],
            "per_class": {
                lab: {k: report[lab][k] for k in ("precision", "recall", "f1-score", "support")}
                for lab in present_labels(y_true) if lab in report
            },
            "predicted_distribution": dict(Counter(y_pred.tolist())),
            "pct_predicted_cooperative": float(np.mean(y_pred == "Cooperative")),
            "predictions": y_pred.tolist(),
        }

    # spread over which training examples you happened to collect
    rng = np.random.default_rng(12345)
    seed_scores = {s: [] for s in test_sets}
    for seed in range(seeds):
        idx = rng.integers(0, len(train_rows), len(train_rows))
        resampled = [train_rows[i] for i in idx]
        if len(set(labels_of(resampled))) < 2:
            continue
        f, p = build_model(model_name, seed)
        f(resampled)
        for split_name, test_rows in test_sets.items():
            seed_scores[split_name].append(macro_f1(labels_of(test_rows), p(test_rows)))
    for split_name, scores in seed_scores.items():
        result["splits"][split_name]["train_resample_mean"] = float(np.mean(scores))
        result["splits"][split_name]["train_resample_sd"] = float(np.std(scores))

    return result


def print_confusion(y_true, y_pred, title):
    labs = present_labels(y_true)
    pred_labs = sorted(set(y_pred.tolist()) | set(labs))
    cm = confusion_matrix(y_true, y_pred, labels=pred_labs)
    print(f"\n  {title} (rows = gold, cols = predicted)")
    width = max(len(l) for l in pred_labs) + 1
    print("    " + " " * width + "".join(f"{l[:6]:>8}" for l in pred_labs))
    for i, lab in enumerate(pred_labs):
        if lab not in labs:
            continue
        print(f"    {lab:<{width}}" + "".join(f"{cm[i][j]:>8}" for j in range(len(pred_labs))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    train_all = load(TRAIN_PATH)
    test_sets = {"natural": load(TEST_NATURAL), "synthetic": load(TEST_SYNTHETIC)}

    regimes = {
        "synthetic-only": [r for r in train_all if r["source"] == "synthetic"],
        "natural-only": [r for r in train_all if r["source"] == "natural"],
        "both": train_all,
    }

    print(f"Train pool: {len(train_all)} rows "
          f"({len(regimes['synthetic-only'])} synthetic, {len(regimes['natural-only'])} natural)")
    for name, rows in test_sets.items():
        print(f"Test {name}: {len(rows)} rows, classes present: "
              f"{dict(Counter(labels_of(rows)).most_common())}")
    print(f"Seeds: {args.seeds}   Bootstrap resamples: {args.bootstrap}")

    all_results = {}
    for regime, train_rows in regimes.items():
        print(f"\n\n{'#'*74}\n# TRAIN: {regime}  ({len(train_rows)} rows)\n{'#'*74}")
        all_results[regime] = {}
        for model_name in MODELS:
            res = evaluate(model_name, train_rows, test_sets, args.seeds, args.bootstrap)
            all_results[regime][model_name] = res

        header = f"{'model':<14}" + "".join(
            f"{'test ' + s:>34}" for s in test_sets)
        print("\n" + header)
        print(f"{'':<14}" + "".join(f"{'macroF1 [95% CI]      %coop':>34}" for _ in test_sets))
        print("-" * (14 + 34 * len(test_sets)))
        for model_name in MODELS:
            line = f"{model_name:<14}"
            for split in test_sets:
                s = all_results[regime][model_name]["splits"][split]
                lo, hi = s["ci95"]
                line += (f"{s['macro_f1']:>10.3f} [{lo:.2f},{hi:.2f}]"
                         f" +/-{s['train_resample_sd']:.3f}{s['pct_predicted_cooperative']:>7.0%}")
            print(line)

        # confusion matrices for the strongest lexical model
        best = max(("tfidf-utt", "tfidf-ctx", "tfidf-char"),
                   key=lambda m: all_results[regime][m]["splits"]["natural"]["macro_f1"])
        for split, rows in test_sets.items():
            print_confusion(labels_of(rows),
                            np.array(all_results[regime][best]["splits"][split]["predictions"]),
                            f"{best} on test-{split}")

    # ---- the comparisons worth a significance test ------------------------
    print(f"\n\n{'='*74}\nPAIRED COMPARISONS (exact McNemar)\n{'='*74}")
    comparisons = [
        ("both", "tfidf-ctx", "both", "majority", "does TF-IDF beat always-guessing?"),
        ("both", "tfidf-ctx", "both", "tfidf-utt", "does adding context help? (item 3)"),
        ("both", "tfidf-ctx", "both", "surface", "does reading the words beat counting them?"),
        ("synthetic-only", "tfidf-char", "synthetic-only", "surface",
         "...and on the synthetic corpus specifically? (item 6)"),
        ("synthetic-only", "tfidf-ctx", "natural-only", "tfidf-ctx", "does training domain matter? (item 4)"),
    ]
    for split in test_sets:
        y_true = labels_of(test_sets[split])
        print(f"\ntest-{split}:")
        for reg_a, mod_a, reg_b, mod_b, question in comparisons:
            pa = np.array(all_results[reg_a][mod_a]["splits"][split]["predictions"])
            pb = np.array(all_results[reg_b][mod_b]["splits"][split]["predictions"])
            b, c, p = mcnemar(y_true, pa, pb)
            label_a = f"{mod_a}/{reg_a}" if reg_a != reg_b else mod_a
            label_b = f"{mod_b}/{reg_b}" if reg_a != reg_b else mod_b
            sig = "significant" if p < 0.05 else "not significant"
            print(f"  {label_a} vs {label_b:<24} b={b:<3} c={c:<3} p={p:.4f}  {sig}")
            print(f"      ({question})")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "baselines.json"
    slim = {
        reg: {m: {**r, "splits": {s: {k: v for k, v in d.items() if k != "predictions"}
                                  for s, d in r["splits"].items()}}
              for m, r in models.items()}
        for reg, models in all_results.items()
    }
    out.write_text(json.dumps({"config": vars(args), "results": slim}, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(Path(__file__).parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

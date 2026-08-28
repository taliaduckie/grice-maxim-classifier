"""Evaluation metrics shared by baselines.py and ablations.py.

These two produce adjacent tables in results/foundation_report.md, so they have
to compute macro F1 and its confidence interval the same way. They previously
had separate copies that happened to agree but weren't forced to.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score


def present_labels(y_true):
    """The classes the gold set actually contains.

    test_natural has no Cooperative examples, so averaging over all five would
    fold in an undefined class.
    """
    return sorted(set(np.asarray(y_true).tolist()))


def macro_f1(y_true, y_pred):
    """Macro F1 over classes present in the gold set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(f1_score(y_true, y_pred, labels=present_labels(y_true),
                          average="macro", zero_division=0))


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    """95% percentile CI for macro F1 over resamples of the test set.

    Resamples items independently. The natural test set is drawn from only a
    handful of threads, so its true interval is wider than this — see
    results/foundation_report.md §11.4.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(set(y_true[idx].tolist())) < 2:
            continue
        scores.append(macro_f1(y_true[idx], y_pred[idx]))
    if not scores:
        return [float("nan"), float("nan")]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar over the two models' disagreements. Returns (b, c, p).

    b = a right where b is wrong; c = the reverse.
    """
    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true
    b = int(np.sum(a_right & ~b_right))
    c = int(np.sum(~a_right & b_right))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(binomtest(b, b + c, 0.5).pvalue)


def cluster_bootstrap_ci(y_true, y_pred, clusters, n_boot=1000, seed=0):
    """95% CI for macro F1, resampling whole clusters (threads) with replacement.

    Items from one discussion share topic, register and the annotator's
    attention, so they are not independent draws. Resampling items treats them
    as if they were and understates the interval; resampling clusters does not.
    With few clusters the interval is wide — that is the honest answer.
    """
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    clusters = np.asarray(clusters)
    ids = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in ids}
    scores = []
    for _ in range(n_boot):
        picked = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([members[c] for c in picked])
        if len(set(y_true[idx].tolist())) < 2:
            continue
        scores.append(macro_f1(y_true[idx], y_pred[idx]))
    if not scores:
        return [float("nan"), float("nan")]
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]

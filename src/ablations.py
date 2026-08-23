"""RoBERTa ablation grid.

    config  utterance  context  synthetic  natural
    A          x          -         x         -
    B          x          x         x         -
    C          x          x         -         x
    D          x          x         x         x

A vs B isolates context; B/C/D isolate training domain. Each run is evaluated on
both frozen test sets.

Each run carves its own stratified dev split from its own training pool and
picks the best epoch on that. The frozen sets are used once per run, after
selection is finished. Hyperparameters are identical across configs and match
train.py, since the grid is an experiment about data.

Per-run predictions are saved for later error analysis.

Usage:
    python3 src/ablations.py --pilot
    python3 src/ablations.py --seeds 3
    python3 src/ablations.py --configs A B
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from labels import MAXIMS
from metrics import bootstrap_ci, macro_f1, present_labels
from paths import RESULTS_DIR, TEST_NATURAL, TEST_SYNTHETIC, TRAIN_PATH, rel
from training_utils import (
    HPARAMS, KeepBestState, WeightedTrainer, class_weights,
    macro_f1_metrics, training_args,
)

MODEL_NAME = "roberta-base"
LABEL2ID = {m: i for i, m in enumerate(MAXIMS)}
ID2LABEL = {i: m for m, i in LABEL2ID.items()}

DEV_FRACTION = 0.15

CONFIGS = {
    "A": {"context": False, "sources": ["synthetic"]},
    "B": {"context": True, "sources": ["synthetic"]},
    "C": {"context": True, "sources": ["natural"]},
    "D": {"context": True, "sources": ["synthetic", "natural"]},
}

SCRATCH = Path(os.environ.get(
    "GRICE_SCRATCH",
    "/private/tmp/claude-501/-Users-taliahonikman-grice-maxim-classifier/"
    "14d2c303-9dff-41ee-8719-cdcf0316d29a/scratchpad/ablations",
))


class PairDataset(Dataset):
    """Encodes (utterance, context) as a sequence pair, or utterance alone.

    Pair encoding gives RoBERTa two segments rather than one concatenated
    string, which is what config B is testing.
    """

    def __init__(self, rows, tokenizer, use_context, max_length):
        utterances = [r["utterance"] for r in rows]
        if use_context:
            contexts = [r["context"] for r in rows]
            self.encodings = tokenizer(utterances, contexts, truncation=True,
                                       padding="max_length", max_length=max_length)
        else:
            self.encodings = tokenizer(utterances, truncation=True,
                                       padding="max_length", max_length=max_length)
        self.labels = [LABEL2ID[r["maxim"]] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("maxim") in MAXIMS]


def run_one(config_name, seed, train_pool, test_sets, tokenizer, quiet=True):
    cfg = CONFIGS[config_name]
    rows = [r for r in train_pool if r["source"] in cfg["sources"]]

    # dev split for epoch selection — carved from training data, never from test
    labels = [r["maxim"] for r in rows]
    train_rows, dev_rows = train_test_split(
        rows, test_size=DEV_FRACTION, stratify=labels, random_state=seed)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(MAXIMS), id2label=ID2LABEL, label2id=LABEL2ID)

    max_length = HPARAMS["max_length"]
    train_ds = PairDataset(train_rows, tokenizer, cfg["context"], max_length)
    dev_ds = PairDataset(dev_rows, tokenizer, cfg["context"], max_length)

    weights = class_weights(train_ds.labels, len(MAXIMS))
    compute_metrics = macro_f1_metrics(MAXIMS)
    args = training_args(SCRATCH / f"{config_name}_seed{seed}", seed=seed,
                         data_seed=seed, disable_tqdm=quiet)

    keep_best = KeepBestState(model)
    trainer = WeightedTrainer(
        weights=weights, model=model, args=args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        compute_metrics=compute_metrics, callbacks=[keep_best],
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    if not keep_best.restore():
        raise RuntimeError("no evaluation ran, so no best epoch was recorded")

    # the restored model must reproduce the score that selected it
    dev_f1 = trainer.evaluate()["eval_macro_f1"]
    if abs(dev_f1 - keep_best.best_score) > 1e-6:
        raise RuntimeError(
            f"restored model scores {dev_f1:.6f} but was selected at "
            f"{keep_best.best_score:.6f} — weight restoration is broken")

    result = {
        "config": config_name,
        "seed": seed,
        "context": cfg["context"],
        "sources": cfg["sources"],
        "n_train": len(train_rows),
        "n_dev": len(dev_rows),
        "dev_macro_f1": float(dev_f1),
        "best_epoch": keep_best.best_epoch,
        "train_seconds": round(train_seconds, 1),
        "splits": {},
    }

    # the frozen sets are touched here, once, after selection is finished
    for split_name, test_rows in test_sets.items():
        ds = PairDataset(test_rows, tokenizer, cfg["context"], max_length)
        pred_ids = trainer.predict(ds).predictions.argmax(axis=-1)
        y_pred = [ID2LABEL[i] for i in pred_ids]
        y_true = [r["maxim"] for r in test_rows]
        present = sorted(set(y_true))
        report = classification_report(y_true, y_pred, labels=present,
                                       output_dict=True, zero_division=0)
        result["splits"][split_name] = {
            "macro_f1": macro_f1(y_true, y_pred),
            "accuracy": float(np.mean(np.array(y_pred) == np.array(y_true))),
            "ci95": bootstrap_ci(y_true, y_pred),
            "per_class": {lab: report[lab] for lab in present if lab in report},
            "predicted_distribution": dict(Counter(y_pred)),
            "pct_predicted_cooperative": float(np.mean(np.array(y_pred) == "Cooperative")),
            "predictions": y_pred,
            "row_ids": [r["row_id"] for r in test_rows],
        }

    del keep_best.best_state
    return result


def summarise(results):
    """Group runs by config and report mean +/- sd across seeds."""
    by_config = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)

    print(f"\n{'='*88}")
    print("ABLATION GRID — macro F1 (mean +/- sd over seeds)")
    print(f"{'='*88}")
    print(f"{'cfg':<5}{'context':<9}{'train data':<22}{'n':>6}"
          f"{'test_natural':>24}{'test_synthetic':>24}")
    print("-" * 88)
    for name in sorted(by_config):
        runs = by_config[name]
        cfg = CONFIGS[name]
        line = (f"{name:<5}{'yes' if cfg['context'] else 'no':<9}"
                f"{'+'.join(cfg['sources']):<22}{runs[0]['n_train']:>6}")
        for split in ("natural", "synthetic"):
            scores = [r["splits"][split]["macro_f1"] for r in runs]
            coop = np.mean([r["splits"][split]["pct_predicted_cooperative"] for r in runs])
            line += f"{np.mean(scores):>12.3f} +/-{np.std(scores):<5.3f}{coop:>6.0%}"
        print(line)
    print("\n(%coop = share of that test set predicted Cooperative; "
          "true rate is 0% on test_natural)")
    return by_config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--pilot", action="store_true",
                    help="One config, one seed — validates the harness cheaply.")
    ap.add_argument("--out", default=str(RESULTS_DIR / "ablations.json"))
    args = ap.parse_args()

    if args.pilot:
        args.configs, args.seeds = ["A"], 1

    train_pool = load(TRAIN_PATH)
    test_sets = {"natural": load(TEST_NATURAL), "synthetic": load(TEST_SYNTHETIC)}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    print(f"Train pool: {len(train_pool)} rows "
          f"({sum(r['source'] == 'synthetic' for r in train_pool)} synthetic, "
          f"{sum(r['source'] == 'natural' for r in train_pool)} natural)")
    print(f"Test: natural={len(test_sets['natural'])}, "
          f"synthetic={len(test_sets['synthetic'])}")
    print(f"Configs: {', '.join(args.configs)}   Seeds: {args.seeds}")
    print(f"Epoch selection on a {DEV_FRACTION:.0%} dev split of the training "
          "pool. Test sets are used once per run, after selection.\n")

    results = []
    total = len(args.configs) * args.seeds
    for i, config_name in enumerate(args.configs):
        for seed in range(args.seeds):
            n = i * args.seeds + seed + 1
            print(f"[{n}/{total}] config {config_name}, seed {seed} ...", flush=True)
            r = run_one(config_name, seed, train_pool, test_sets, tokenizer)
            results.append(r)
            print(f"      {r['train_seconds']:.0f}s  dev={r['dev_macro_f1']:.3f}  "
                  f"natural={r['splits']['natural']['macro_f1']:.3f}  "
                  f"synthetic={r['splits']['synthetic']['macro_f1']:.3f}", flush=True)

            out = Path(args.out)
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(
                {"hparams": HPARAMS, "dev_fraction": DEV_FRACTION, "runs": results},
                indent=2) + "\n")

    summarise(results)
    out = Path(args.out)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nWrote {shown} (includes per-row predictions for error analysis)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

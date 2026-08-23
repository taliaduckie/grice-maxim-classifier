import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.utils.data import Subset
from transformers import AutoModelForSequenceClassification

from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS
from paths import TRAIN_PATH, rel
from training_utils import (
    HPARAMS, KeepBestState, WeightedTrainer, class_weights,
    macro_f1_metrics, training_args,
)


def run_kfold(data_path: str, n_folds: int = 5, seed: int = 42):
    dataset = GriceDataset(data_path, max_length=HPARAMS["max_length"])
    n = len(dataset)
    print(f"Loaded {n} examples from {rel(data_path)}. "
          f"Running {n_folds}-fold cross-validation.")
    print(f"Hyperparameters are shared with train.py: "
          f"{HPARAMS['num_train_epochs']} epochs, lr={HPARAMS['learning_rate']}, "
          f"batch={HPARAMS['per_device_train_batch_size']}, fully unfrozen.\n")

    compute_metrics = macro_f1_metrics(MAXIMS)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_results = []
    per_class_results = {m: [] for m in MAXIMS}

    for fold, (train_idx, eval_idx) in enumerate(skf.split(range(n), dataset.labels)):
        print(f"{'='*60}\nFOLD {fold + 1}/{n_folds}\n{'='*60}")
        print(f"Train: {len(train_idx)}, Eval: {len(eval_idx)}")

        # fresh model each fold — no leakage between folds
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(MAXIMS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        train_ds = Subset(dataset, train_idx.tolist())
        eval_ds = Subset(dataset, eval_idx.tolist())
        weights = class_weights([dataset.labels[i] for i in train_idx], len(MAXIMS))

        keep_best = KeepBestState(model)
        trainer = WeightedTrainer(
            weights=weights,
            model=model,
            args=training_args(f"/tmp/kfold_fold_{fold}", seed=seed),
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
            callbacks=[keep_best],
        )

        trainer.train()
        keep_best.restore()

        preds = trainer.predict(eval_ds)
        pred_labels = preds.predictions.argmax(axis=-1)
        report = classification_report(
            preds.label_ids, pred_labels,
            labels=list(range(len(MAXIMS))),
            target_names=MAXIMS,
            output_dict=True,
            zero_division=0,
        )
        macro_f1 = report["macro avg"]["f1-score"]
        fold_results.append(macro_f1)

        print(f"\nFold {fold + 1} macro F1: {macro_f1:.4f}")
        for m in MAXIMS:
            f1 = report[m]["f1-score"]
            per_class_results[m].append(f1)
            print(f"  {m:<12} F1={f1:.3f}")
        keep_best.release()

    print(f"\n{'='*60}\nSUMMARY ({n_folds}-fold cross-validation)\n{'='*60}")
    print(f"Macro F1: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    print(f"Per fold: {[f'{f:.3f}' for f in fold_results]}")
    print("\nPer-class averages:")
    for m in MAXIMS:
        scores = per_class_results[m]
        print(f"  {m:<12} F1={np.mean(scores):.3f} +/- {np.std(scores):.3f}  "
              f"({[f'{s:.2f}' for s in scores]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified k-fold cross-validation for the Grice classifier.",
    )
    # corpus_train.csv, not corpus.csv — the latter still contains every row
    # held out in data/test/, so cross-validating over it trains on test data.
    parser.add_argument("--data", default=str(TRAIN_PATH))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_kfold(args.data, args.folds, args.seed)

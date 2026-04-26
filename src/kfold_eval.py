"""
kfold_eval.py

Stratified 5-fold cross-validation to get a real macro F1 number
instead of trusting one lucky/unlucky 80/20 split.

Trains 5 models, each on a different 80% of the data, evaluates on
the held-out 20%. Reports per-fold and average macro F1.

This doesn't save a model — it's purely for evaluation. Use train.py
to train the actual production model after you know the real score.

Usage:
    python kfold_eval.py
    python kfold_eval.py --data ../data/annotated/corpus.csv
    python kfold_eval.py --folds 5
"""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.utils.data import Subset
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS


def freeze_model(model):
    """same freezing strategy as train.py"""
    for name, param in model.roberta.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < 10:
                param.requires_grad = False
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    report = classification_report(
        labels, preds,
        target_names=MAXIMS,
        output_dict=True,
        zero_division=0,
    )
    return {"macro_f1": report["macro avg"]["f1-score"]}


def run_kfold(data_path: str, n_folds: int = 5):
    dataset = GriceDataset(data_path, max_length=128)
    n = len(dataset)
    print(f"Loaded {n} examples. Running {n_folds}-fold cross-validation.\n")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    per_class_results = {m: [] for m in MAXIMS}

    for fold, (train_idx, eval_idx) in enumerate(skf.split(range(n), dataset.labels)):
        print(f"{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        print(f"Train: {len(train_idx)}, Eval: {len(eval_idx)}")

        # fresh model each fold — no leakage between folds
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(MAXIMS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        freeze_model(model)

        train_ds = Subset(dataset, train_idx.tolist())
        eval_ds = Subset(dataset, eval_idx.tolist())

        # same hyperparameters as train.py
        args = TrainingArguments(
            output_dir=f"/tmp/kfold_fold_{fold}",
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="no",  # don't save checkpoints — disk fills up fast
                                 # with 5 folds x 20 epochs x 500MB each
            report_to="none",
            use_cpu=True,
            logging_steps=9999,  # suppress per-step logging
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        # evaluate on held-out fold
        eval_result = trainer.evaluate()
        macro_f1 = eval_result["eval_macro_f1"]
        fold_results.append(macro_f1)

        # per-class breakdown
        preds = trainer.predict(eval_ds)
        pred_labels = preds.predictions.argmax(axis=-1)
        true_labels = preds.label_ids
        report = classification_report(
            true_labels, pred_labels,
            target_names=MAXIMS,
            output_dict=True,
            zero_division=0,
        )
        print(f"\nFold {fold + 1} macro F1: {macro_f1:.4f}")
        for m in MAXIMS:
            f1 = report[m]["f1-score"]
            per_class_results[m].append(f1)
            print(f"  {m:<12} F1={f1:.3f}")

    # summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ({n_folds}-fold cross-validation)")
    print(f"{'='*60}")
    print(f"Macro F1: {np.mean(fold_results):.4f} +/- {np.std(fold_results):.4f}")
    print(f"Per fold: {[f'{f:.3f}' for f in fold_results]}")
    print(f"\nPer-class averages:")
    for m in MAXIMS:
        scores = per_class_results[m]
        print(f"  {m:<12} F1={np.mean(scores):.3f} +/- {np.std(scores):.3f}  ({[f'{s:.2f}' for s in scores]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stratified k-fold cross-validation for the Grice classifier.",
    )
    parser.add_argument(
        "--data",
        default=str(Path(__file__).parent.parent / "data" / "annotated" / "corpus.csv"),
    )
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    run_kfold(args.data, args.folds)

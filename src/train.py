"""
train.py
Fine-tune RoBERTa on the annotated Grice corpus.

Usage:
    python train.py --data ../data/annotated/corpus.csv
"""

import argparse
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
import numpy as np
from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS

OUTPUT_DIR = "../models/roberta-grice"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    report = classification_report(labels, preds, target_names=MAXIMS, output_dict=True)
    return {"macro_f1": report["macro avg"]["f1-score"]}


def train(data_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(MAXIMS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset = GriceDataset(data_path)
    n = len(dataset)
    train_size = int(0.8 * n)
    train_ds = dataset[:train_size]
    eval_ds  = dataset[train_size:]

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_dir="../models/logs",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    train(args.data)

"""
train.py

Fine-tune RoBERTa on the annotated Grice corpus.

Usage:
    python train.py --data ../data/annotated/corpus.csv

Prerequisites:
    - At least ~50 labeled examples per maxim before this is worth running.
      The seed corpus has 8 total. That's not enough. I know.
      It's a scaffold. Annotate more data first.
    - A GPU helps a lot. On CPU this will take a while.
      'A while' meaning: long enough to question your choices.

The model: roberta-base fine-tuned for 5-way sequence classification.
The task: given utterance + context (concatenated), predict which maxim
is being violated, if any.

Training setup is fairly standard. Nothing clever here cuz i am not clever.
the cleverness is in the data, which is where it should be for a task this
dependent on annotation quality. 

One decision I made is 80/20 train/eval split w no cross-validation.
For a real publication you'd want k-fold. For a GitHub project scaffolded
in an afternoon: 80/20. i'm going to hopefully come back and work on this.
"""

import argparse
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS

OUTPUT_DIR = "../models/roberta-grice"


def compute_metrics(eval_pred):
    """
    Compute macro F1 for the evaluation set.

    Macro F1 rather than accuracy because the class distribution
    is uneven — Cooperative examples are easy to find, Manner examples
    less so. Accuracy would flatter a model that just predicts the
    majority class. Macro F1 doesn't let you get away with that.
    """
    logits, labels = eval_pred
    preds  = logits.argmax(axis=-1)
    report = classification_report(
        labels, preds,
        target_names=MAXIMS,
        output_dict=True,
        zero_division=0,  # don't crash if a class has no predictions
    )

    # Log the per-class breakdown too — the aggregate macro F1
    # can hide that the model has learned nothing about Manner
    # while being confident about everything else.
    for maxim in MAXIMS:
        if maxim in report:
            f1 = report[maxim]["f1-score"]
            print(f"  {maxim}: F1={f1:.3f}")

    return {"macro_f1": report["macro avg"]["f1-score"]}


def train(data_path: str):
    """
    Fine-tune and save the model.

    The saved model will be loaded automatically by predict.py
    once it exists in OUTPUT_DIR.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(MAXIMS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    dataset = GriceDataset(data_path)
    n = len(dataset)
    print(f"Loaded {n} examples from {data_path}.")

    if n < 40:
        print(
            f"Warning: {n} examples is probably not enough to fine-tune well. "
            "The model will overfit. Consider annotating more data before training, "
            "or use the zero-shot baseline (predict.py without a saved model) "
            "until you have a bigger corpus."
        )

    train_size = int(0.8 * n)
    train_ds   = dataset[:train_size]
    eval_ds    = dataset[train_size:]

    print(f"Training on {train_size} examples, evaluating on {n - train_size}.")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir="../models/logs",
        report_to="none",  # disable wandb / other experiment trackers
                           # unless you've set them up and want them
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
    print(f"Model saved to {OUTPUT_DIR}.")
    print("predict.py will use this model automatically from now on.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RoBERTa on the Grice maxim corpus.",
    )
    parser.add_argument("--data", required=True, help="Path to annotated CSV file.")
    args = parser.parse_args()
    train(args.data)

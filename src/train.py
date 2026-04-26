"""
train.py

Fine-tune RoBERTa on the annotated Grice corpus.

Usage:
    python train.py --data ../data/annotated/corpus.csv

Prerequisites:
    - At least ~50 labeled examples per maxim before this is worth running.
      The corpus has 367 examples across 5 classes now so we're well past that.
    - A GPU helps but CPU works fine. ~9 minutes for 10 epochs on an M3.

The model: roberta-base fine-tuned for 5-way sequence classification.
The task: given utterance + context (concatenated), predict which maxim
is being violated, if any.

Training setup: stratified 80/20 split, lr=2e-5, warmup 10%, 10 epochs.
the cleverness is in the data, which is where it should be for a task this
dependent on annotation quality. macro F1 = 0.86 as of the last run.

For a real publication you'd want k-fold. that's on the TODO list.
"""

import argparse
import sys
import numpy as np
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# same sys.path dance as predict.py. i refuse to write a setup.py for this.
sys.path.insert(0, str(Path(__file__).parent))
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS

# resolve relative to this file so it works from anywhere.
# learned this the hard way with predict.py's MODEL_DIR.
OUTPUT_DIR = str(Path(__file__).parent.parent / "models" / "roberta-grice")


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
    # pass explicit labels so sklearn doesn't freak out when the eval set
    # is missing a class. which it will be. because 13 examples across 5 classes
    # is not a number that guarantees coverage. ask me how i know.
    report = classification_report(
        labels, preds,
        labels=list(range(len(MAXIMS))),
        target_names=MAXIMS,
        output_dict=True,
        zero_division=0,
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
    # roberta has no idea what pragmatics is but it's about to learn.
    # or overfit trying. probably the second one with 8 examples.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(MAXIMS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # freeze the first 10 of 12 transformer layers. tested unfrozen at
    # 547 examples with lr=1e-5 and weight_decay=0.01 — got 0.86 macro F1.
    # frozen got 0.91. you'd probably need 2000+ examples before unfreezing
    # beats freezing. the bottom layers already know english; we're just
    # teaching the top layers what pragmatics is.
    for name, param in model.roberta.named_parameters():
        if "encoder.layer" in name:
            layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
            if layer_num < 10:
                param.requires_grad = False
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen: {total - trainable:,} / {total:,} parameters ({(total-trainable)/total:.0%})")
    print(f"Training: {trainable:,} parameters ({trainable/total:.0%})")

    # max_length=128 because most utterance pairs are under 50 tokens
    # and 256 was just burning memory for padding. the manner examples
    # are the longest and even those fit in 128 comfortably.
    dataset = GriceDataset(data_path, max_length=128)
    n = len(dataset)
    print(f"Loaded {n} examples from {data_path}.")

    if n < 40:
        print(
            f"Warning: {n} examples is probably not enough to fine-tune well. "
            "The model will overfit. Consider annotating more data before training, "
            "or use the zero-shot baseline (predict.py without a saved model) "
            "until you have a bigger corpus."
        )

    # stratified 80/20 split so every class actually shows up in eval.
    # the previous naive split gave Relation F1=0.00 because the eval set
    # had zero Relation examples. which is technically a valid split but
    # also technically useless. sklearn to the rescue.
    indices = list(range(n))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,  # reproducibility is a cooperative maxim
    )
    train_ds = Subset(dataset, train_idx)
    eval_ds  = Subset(dataset, eval_idx)

    print(f"Training on {len(train_idx)} examples, evaluating on {len(eval_idx)}.")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=20,
        # bumped to 8 for less noisy gradients. CPU doesn't OOM like GPU did.
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # 5e-6 because unfrozen 125M parameters on 547 examples needs
        # a very gentle touch. 2e-5 was overfitting. 5e-6 is the
        # "i've been hurt before and i learned from it" setting.
        learning_rate=2e-5,
        weight_decay=0.01,  # regularization. fights overfitting directly.
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_dir=str(Path(__file__).parent.parent / "models" / "logs"),
        report_to="none",  # disable wandb / other experiment trackers
                           # unless you've set them up and want them
        use_cpu=True,  # MPS on apple silicon + transformers = pain.
                       # CPU is slower but at least it finishes.
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
    )

    # moment of truth. or moment of overfitting. same thing at this sample size.
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    # save the tokenizer too or pipeline can't find it and produces
    # identical scores for every input. the model was learning fine —
    # 0.56 macro F1 at epoch 4!! — but at inference time it couldn't
    # understand its own inputs. a model that can't read its own
    # tokenization is a manner violation if i ever saw one.
    # ask me how long i debugged this. (too long. the answer is too long.)
    dataset.tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}.")
    print("predict.py will use this model automatically from now on.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RoBERTa on the Grice maxim corpus.",
    )
    parser.add_argument("--data", required=True, help="Path to annotated CSV file.")
    args = parser.parse_args()
    train(args.data)

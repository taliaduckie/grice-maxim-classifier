import argparse
import sys
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from collections import Counter

# i refuse to write a setup py for this
sys.path.insert(0, str(Path(__file__).parent))
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import classification_report
from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS
from training_utils import KeepBestState

# resolve MODEL DIR situation
OUTPUT_DIR = str(Path(__file__).parent.parent / "models" / "roberta-grice")


def train(data_path: str):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        # explicit labels arg prevents sklearn from crashing when a class is missing
        report = classification_report(
            labels, preds,
            labels=list(range(len(MAXIMS))),
            target_names=MAXIMS,
            output_dict=True,
            zero_division=0,
        )
        for maxim in MAXIMS:
            if maxim in report:
                print(f"  {maxim}: F1={report[maxim]['f1-score']:.3f}")
        return {"macro_f1": report["macro avg"]["f1-score"]}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(MAXIMS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # unfrozen at ~1200 examples
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Training: {trainable:,} / {total:,} parameters ({trainable/total:.0%})")

    # max_length=128 because most utterance pairs are under 50 tokens
    dataset = GriceDataset(data_path, max_length=128)
    n = len(dataset)
    print(f"Loaded {n} examples from {data_path}.")

    if n < 40:
        print(
            f"Bro. {n} examples is not enough to fine tune well. "
            "Consider annotating more data before training, "
            "or use the zero-shot baseline (predict.py without a saved model) "
            "until there's a bigger corpus."
        )

    # stratified 80/20 split so every class actually shows up in eval
    indices = list(range(n))
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=dataset.labels,
        random_state=42,
    )
    train_ds = Subset(dataset, train_idx)
    eval_ds  = Subset(dataset, eval_idx)

    # inverse-frequency class weights
    counts = Counter(dataset.labels)
    n_total = len(dataset.labels)
    n_cls = len(MAXIMS)
    class_weights = torch.tensor([
        n_total / (n_cls * counts[i]) for i in range(n_cls)
    ], dtype=torch.float32)
    print(f"Class weights: {', '.join(f'{MAXIMS[i]}={class_weights[i]:.2f}' for i in range(n_cls))}")

    print(f"Training on {len(train_idx)} examples, evaluating on {len(eval_idx)}.")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        # bumped to 8 for less noisy gradients
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        # 1e-5 for unfrozen. 
        learning_rate=1e-5,
        weight_decay=0.01,  
        warmup_ratio=0.1,
        eval_strategy="epoch",
        # Best-epoch weights are kept in RAM by KeepBestState rather than
        # restored from a checkpoint. Trainer's load_best_model_at_end silently
        # fails to restore LayerNorm parameters in this transformers version and
        # saves a model that was never evaluated — see src/training_utils.py.
        save_strategy="no",
        load_best_model_at_end=False,
        logging_dir=str(Path(__file__).parent.parent / "models" / "logs"),
        report_to="none",  
        use_cpu=True,  # MPS on apple silicon + transformers = pain
                       # CPU is slower but at least it finishes
    )

    # weighted CE so Cooperative doesn't dominate
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    keep_best = KeepBestState(model)
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[keep_best],
    )

    trainer.train()

    # roll back to the best epoch, then check the restored model reproduces the
    # score it was picked for — otherwise the saved model isn't the one measured.
    if keep_best.restore():
        recheck = trainer.evaluate()["eval_macro_f1"]
        if abs(recheck - keep_best.best_score) > 1e-6:
            raise RuntimeError(
                f"restored model scores {recheck:.4f} but was selected at "
                f"{keep_best.best_score:.4f} — refusing to save it")
        print(f"Restored best epoch ({keep_best.best_epoch:.0f}), "
              f"macro F1 = {keep_best.best_score:.4f}")

    trainer.save_model(OUTPUT_DIR)
    # save the tokenizer too or pipeline can't find it and produces
    # identical scores for every input. the model was learning fine 
    # w 0.56 macro F1 at epoch 4(!!) but at inference time it couldn't
    # understand its own inputs
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

import argparse
import sys
from pathlib import Path
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

# i refuse to write a setup py for this
sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoModelForSequenceClassification
from dataset import GriceDataset, LABEL2ID, ID2LABEL, MODEL_NAME
from labels import MAXIMS
from paths import MODEL_DIR, MODELS_DIR
from training_utils import (
    HPARAMS, KeepBestState, WeightedTrainer, class_weights,
    macro_f1_metrics, training_args,
)

OUTPUT_DIR = str(MODEL_DIR)


def train(data_path: str):
    compute_metrics = macro_f1_metrics(MAXIMS, verbose=True)

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
    dataset = GriceDataset(data_path, max_length=HPARAMS["max_length"])
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

    weights = class_weights(dataset.labels, len(MAXIMS))
    print(f"Class weights: {', '.join(f'{MAXIMS[i]}={weights[i]:.2f}' for i in range(len(MAXIMS)))}")

    print(f"Training on {len(train_idx)} examples, evaluating on {len(eval_idx)}.")

    args = training_args(
        OUTPUT_DIR,
        logging_dir=str(MODELS_DIR / "logs"),
    )

    keep_best = KeepBestState(model)
    trainer = WeightedTrainer(
        weights=weights,
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

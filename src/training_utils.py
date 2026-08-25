"""Shared fine-tuning machinery.

train.py, kfold_eval.py and ablations.py each had their own trainer, metrics
function and hyperparameters. They drifted: kfold_eval ran 20 epochs at 2e-5
with the lower encoder frozen while the other two ran 10 epochs at 1e-5 fully
unfrozen, and its docstring claimed they matched. Anything that must be the
same across those three lives here.
"""

from collections import Counter

import torch
from sklearn.metrics import classification_report
from transformers import Trainer, TrainerCallback, TrainingArguments

# Shared hyperparameters — changing a number here changes it for train.py,
# kfold_eval.py and the ablation grid at once.
HPARAMS = dict(
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    max_length=128,
)


def class_weights(label_ids, n_classes):
    """Inverse-frequency weights so Cooperative doesn't dominate the loss."""
    counts = Counter(label_ids)
    n_total = len(label_ids)
    return torch.tensor(
        [n_total / (n_classes * counts[i]) if counts[i] else 0.0
         for i in range(n_classes)],
        dtype=torch.float32,
    )


def macro_f1_metrics(label_names, verbose=False):
    """compute_metrics returning macro F1 over all label ids.

    The explicit `labels` argument matters: without it sklearn infers the label
    set from the data and raises when a fold contains no examples of a class.
    """
    def compute(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        report = classification_report(
            labels, preds,
            labels=list(range(len(label_names))),
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        if verbose:
            for name in label_names:
                if name in report:
                    print(f"  {name}: F1={report[name]['f1-score']:.3f}")
        return {"macro_f1": report["macro avg"]["f1-score"]}
    return compute


def training_args(output_dir, seed=42, **overrides):
    """TrainingArguments with the shared hyperparameters applied.

    Checkpointing is off and best-epoch restore is handled by KeepBestState —
    see its docstring for why Trainer's own restore can't be used.
    """
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=HPARAMS["num_train_epochs"],
        per_device_train_batch_size=HPARAMS["per_device_train_batch_size"],
        per_device_eval_batch_size=HPARAMS["per_device_eval_batch_size"],
        learning_rate=HPARAMS["learning_rate"],
        weight_decay=HPARAMS["weight_decay"],
        warmup_ratio=HPARAMS["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        seed=seed,
        report_to="none",
        use_cpu=True,   # MPS on apple silicon + transformers = pain
        logging_strategy="no",
    )
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


class WeightedTrainer(Trainer):
    """Trainer with weighted cross-entropy."""

    def __init__(self, weights=None, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.weights.to(logits.device) if self.weights is not None else None
        loss = torch.nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


class KeepBestState(TrainerCallback):
    """Snapshot the best epoch's weights in memory, replacing `load_best_model_at_end`.

    Trainer's own best-model restore is not safe in this transformers version.
    The checkpoint writer stores LayerNorm parameters under the legacy
    `gamma`/`beta` names, while the reload path looks for `weight`/`bias`. The
    mismatch is reported as "missing keys" and then ignored, so every LayerNorm
    in the model is silently left at whatever the *final* epoch produced while
    the rest of the weights come from the best epoch. The result is a hybrid
    model that was never evaluated, and it is what gets saved to disk.

    `from_pretrained` applies the rename correctly, so a plain save/load
    round-trip is unaffected; only the Trainer checkpoint path is broken. The
    metrics printed during training are computed on the in-memory model and are
    correct, so nothing looks wrong at the time.

    Holding the state dict in RAM avoids serialisation altogether. Costs about
    0.5 GB for roberta-base and no disk.

    Usage:
        keep_best = KeepBestState(model)
        trainer = Trainer(..., callbacks=[keep_best])
        trainer.train()
        keep_best.restore()
        assert abs(trainer.evaluate()[keep_best.metric] - keep_best.best_score) < 1e-6
    """

    def __init__(self, model, metric="eval_macro_f1", greater_is_better=True):
        self.model = model
        self.metric = metric
        self.greater_is_better = greater_is_better
        self.best_score = -float("inf") if greater_is_better else float("inf")
        self.best_epoch = None
        self.best_state = None

    def _is_better(self, score):
        return score > self.best_score if self.greater_is_better else score < self.best_score

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return
        if self._is_better(metrics[self.metric]):
            self.best_score = metrics[self.metric]
            self.best_epoch = state.epoch
            self.best_state = {
                k: v.detach().to("cpu", copy=True)
                for k, v in self.model.state_dict().items()
            }

    def restore(self):
        """Load the best snapshot back into the model. False if none was taken."""
        if self.best_state is None:
            return False
        self.model.load_state_dict(self.best_state)
        return True

    def release(self):
        self.best_state = None

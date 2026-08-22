"""Shared training helpers.

Exists for one transformers workaround that train.py and ablations.py both need.
See KeepBestState.
"""

import copy

from transformers import TrainerCallback


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

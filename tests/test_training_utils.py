"""KeepBestState — the replacement for Trainer's load_best_model_at_end.

Uses a toy module rather than roberta-base so the suite stays fast. The
property under test is the one the transformers bug broke: that *every*
parameter rolls back together, LayerNorm included.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from src.training_utils import KeepBestState


class Toy(torch.nn.Module):
    """Has a LayerNorm specifically because that is what failed to restore."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)


def evaluate(cb, score, epoch):
    cb.on_evaluate(None, SimpleNamespace(epoch=epoch), None,
                   metrics={"eval_macro_f1": score})


def bump(model, amount):
    """Move every parameter, so a partial restore is detectable."""
    with torch.no_grad():
        for p in model.parameters():
            p.add_(amount)


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_restores_the_best_epoch_not_the_last():
    model = Toy()
    cb = KeepBestState(model)

    bump(model, 1.0)
    evaluate(cb, 0.20, epoch=1)
    bump(model, 1.0)
    evaluate(cb, 0.90, epoch=2)      # best
    best = snapshot(model)
    bump(model, 1.0)
    evaluate(cb, 0.40, epoch=3)

    assert cb.best_epoch == 2
    assert cb.best_score == pytest.approx(0.90)
    assert cb.restore()
    for k, v in model.state_dict().items():
        assert torch.allclose(v, best[k]), f"{k} was not restored"


def test_layernorm_rolls_back_with_everything_else():
    """The exact failure mode of Trainer's checkpoint restore."""
    model = Toy()
    cb = KeepBestState(model)
    evaluate(cb, 0.9, epoch=1)
    best_norm = model.norm.weight.detach().clone()
    best_linear = model.linear.weight.detach().clone()

    bump(model, 5.0)
    evaluate(cb, 0.1, epoch=2)
    assert not torch.allclose(model.norm.weight, best_norm)

    cb.restore()
    assert torch.allclose(model.norm.weight, best_norm)
    assert torch.allclose(model.linear.weight, best_linear)


def test_snapshot_is_a_copy_not_a_view():
    """A view would track later training and silently restore nothing."""
    model = Toy()
    cb = KeepBestState(model)
    evaluate(cb, 0.9, epoch=1)
    before = cb.best_state["norm.weight"].clone()
    bump(model, 3.0)
    assert torch.allclose(cb.best_state["norm.weight"], before)


def test_restore_is_a_noop_when_nothing_was_evaluated():
    cb = KeepBestState(Toy())
    assert cb.restore() is False


def test_ignores_evaluations_missing_the_metric():
    model = Toy()
    cb = KeepBestState(model)
    cb.on_evaluate(None, SimpleNamespace(epoch=1), None, metrics={"eval_loss": 0.5})
    cb.on_evaluate(None, SimpleNamespace(epoch=1), None, metrics=None)
    assert cb.best_state is None


def test_ties_keep_the_earlier_epoch():
    model = Toy()
    cb = KeepBestState(model)
    evaluate(cb, 0.5, epoch=1)
    first = snapshot(model)
    bump(model, 1.0)
    evaluate(cb, 0.5, epoch=2)
    cb.restore()
    assert cb.best_epoch == 1
    for k, v in model.state_dict().items():
        assert torch.allclose(v, first[k])


def test_lower_is_better_mode():
    model = Toy()
    cb = KeepBestState(model, metric="eval_loss", greater_is_better=False)
    cb.on_evaluate(None, SimpleNamespace(epoch=1), None, metrics={"eval_loss": 0.9})
    bump(model, 1.0)
    cb.on_evaluate(None, SimpleNamespace(epoch=2), None, metrics={"eval_loss": 0.2})
    best = snapshot(model)
    bump(model, 1.0)
    cb.on_evaluate(None, SimpleNamespace(epoch=3), None, metrics={"eval_loss": 0.7})
    cb.restore()
    assert cb.best_epoch == 2
    for k, v in model.state_dict().items():
        assert torch.allclose(v, best[k])


def test_release_frees_the_snapshot():
    model = Toy()
    cb = KeepBestState(model)
    evaluate(cb, 0.9, epoch=1)
    cb.release()
    assert cb.restore() is False

"""Integration: a real `pl.Trainer.fit()`, recorded with one callback.

Measured against lightning 2.6.6 / torch 2.15.0. The hook names are misleading, and two of
the tests below exist because believing them produced a recording that was silently one step
behind:

* at `on_before_optimizer_step` the weights are still **old** - the step has not happened;
* by `on_train_batch_end` they are new, and that is where a store point belongs;
* `trainer.global_step` has advanced by exactly one when a batch end completed an update,
  and not at all on an accumulation micro-batch - which is how a micro-batch is told from an
  update without counting anything.

The loss is the one thing this adapter gets for free: Lightning passes `outputs` to the batch
hook, so tier 1 is reached with no user code at all.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")
pytest.importorskip("lightning", reason="lightning is not installed on this runner")

from lightning.pytorch import Callback, LightningModule, Trainer       # noqa: E402
from torch.utils.data import DataLoader, TensorDataset                # noqa: E402

from archx.adapters.lightning import ArchxCallback                    # noqa: E402
from archx.container.reader import Reader                             # noqa: E402
from archx.container.schema import BlockType                          # noqa: E402
from archx.container.values import Timeline                           # noqa: E402
from archx.policy import PhasePolicy                                  # noqa: E402


class TinyNet(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        self.frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.body(x)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        return torch.nn.functional.mse_loss(self(x), y)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.02)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.randn(16, 4), torch.randn(16, 2)),
                          batch_size=4)


class Spy(Callback):
    """Records the live weights after every batch so the file can be checked against truth."""

    def __init__(self) -> None:
        self.truth: dict[int, dict[str, np.ndarray]] = {}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.truth[trainer.global_step] = {
            name: parameter.detach().numpy().copy()
            for name, parameter in pl_module.named_parameters(remove_duplicate=False)}


def fit(path, *, callbacks=(), policy=None, extra=None, max_epochs=1):
    torch.manual_seed(0)
    model = TinyNet()
    spy = Spy()
    callback = ArchxCallback(
        str(path), policy=policy,
        regions={"body": "body", "head": "head"},
        components={"body": "net", "head": "net"})
    trainer = Trainer(
        max_epochs=max_epochs, accelerator="cpu", logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        callbacks=[*callbacks, spy, callback], **(extra or {}))
    trainer.fit(model)
    return model, spy, callback, trainer


def events(path):
    return [json.loads(block.payload) for block in Reader(path).blocks(BlockType.EVENT)]


def capabilities(path):
    return next(entry for entry in
                (json.loads(b.payload) for b in Reader(path).blocks(BlockType.INFO))
                if entry["event"] == "capabilities")


def test_the_file_holds_the_weights_after_each_update(tmp_path):
    """Step N must reconstruct the weights as they were right after update N.

    The first version of this adapter stored at `on_before_optimizer_step`, where the weights
    are still old, so every step was one update behind - and it looked plausible, because the
    counts and the losses were all correct.
    """
    path = tmp_path / "pl.archx.mdl"
    _model, spy, callback, _trainer = fit(path)

    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1, 2, 3]
    assert callback.stats.steps_seen == 4

    for step in timeline.steps:
        rebuilt = timeline.reconstruct(step)
        after = spy.truth[step + 1]          # truth is keyed by the incremented global_step
        for name, value in after.items():
            # Deltas are lossless at the default budget; the residual is float32 summation.
            assert np.allclose(rebuilt[name], value, atol=1e-5), f"{name} at step {step}"


def test_the_loss_is_recorded_without_any_user_code(tmp_path):
    """Lightning hands `outputs` to the batch hook: tier 1 costs the user nothing."""
    path = tmp_path / "loss.archx.mdl"
    _model, _spy, _callback, trainer = fit(path)

    observations = events(path)
    losses = [entry["loss"] for entry in observations]
    assert all(loss is not None for loss in losses), "every step must carry its loss"
    assert len(set(losses)) > 1, "a constant loss means the wrong value was read"

    capability = capabilities(path)
    assert capability["capability_tier"] == 1
    assert "loss" not in capability["capabilities_missing"]


def test_gradient_norms_are_recorded_and_are_not_clipped(tmp_path):
    """The norm is read before Lightning's clipping component runs."""
    path = tmp_path / "grads.archx.mdl"
    fit(path)

    norms = [entry["grad_norm"] for entry in events(path)]
    assert all(norm is not None and norm > 0 for norm in norms)
    assert len({round(norm, 6) for norm in norms}) > 1
    meta = json.loads(next(iter(Reader(path).blocks(BlockType.META))).payload)
    assert meta["grad_norm_clipped"] is False


def test_a_scheduler_that_steps_every_epoch_is_shown_as_it_changes(tmp_path):
    """The learning rate in the file must be the one the update used."""
    path = tmp_path / "lr.archx.mdl"
    fit(path, max_epochs=2)                 # StepLR halves the rate once per epoch

    rates = [entry["lr"] for entry in events(path)]
    assert len(set(rates)) == 2, f"expected two distinct rates across two epochs, got {rates}"
    assert rates[0] > rates[-1]


def test_gradient_accumulation_gives_one_store_point_per_update(tmp_path):
    """Measured: two batch ends at gs=1, then one real update. One store point, not two."""
    path = tmp_path / "accum.archx.mdl"
    # 16 samples / batch 4 = 4 micro-batches = 2 updates at accumulate_grad_batches=2.
    _model, _spy, callback, _trainer = fit(path, extra={"accumulate_grad_batches": 2})

    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1], "one store point per optimizer update"
    assert callback.stats.steps_seen == 2


def test_an_explicit_policy_reaches_the_recorder(tmp_path):
    """A sparse policy must produce a sparse file - with its event triggers out of the way.

    The first version of this test failed for a *good* reason: with a real optimizer on an
    untrained net, region movement exceeds the policy's 2% threshold on every step, so the
    event trigger stored every step. That is the policy working, not the adapter ignoring it.
    The threshold is raised here so the phase rule is what decides.
    """
    from archx.policy import Triggers

    path = tmp_path / "policy.archx.mdl"
    _model, _spy, callback, _trainer = fit(
        path, policy=PhasePolicy(total_steps=4, min_warmup_steps=2, heartbeat_steps=100,
                                 triggers=Triggers(region_move_fraction=1e9)))

    stored = Timeline(Reader(path)).steps
    assert stored[0] == 0
    assert len(stored) < 4, "a sparse policy must produce a sparse file"
    assert callback.stats.steps_seen == 4


def test_the_callback_is_a_real_lightning_callback(tmp_path):
    from lightning.pytorch import Callback as RealCallback

    assert isinstance(ArchxCallback(str(tmp_path / "x.mdl")), RealCallback)


def test_the_adapter_imports_without_lightning(tmp_path):
    """`import archx.adapters.lightning` must not need lightning; only instantiation does."""
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['lightning'] = None; sys.modules['lightning.pytorch'] = None;"
        "from archx.adapters import lightning as adapter;"
        "assert adapter.ArchxCallback.__name__ == 'ArchxCallback';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_the_metadata_says_which_adapter_and_model(tmp_path):
    path = tmp_path / "meta.archx.mdl"
    fit(path)
    meta = json.loads(next(iter(Reader(path).blocks(BlockType.META))).payload)
    assert meta["adapter"] == "lightning"
    assert meta["model_class"] == "TinyNet"
    assert "lightning" in meta and meta["lightning"] != "unknown"

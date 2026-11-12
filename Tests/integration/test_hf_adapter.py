"""Integration: a real Hugging Face `Trainer`, recorded with one callback, read back alone.

The promises tested here are the ones the roadmap sets for Phase 6, and each was checked
against a live `Trainer` on torch 2.15 / transformers 5.22 / accelerate 1.15 rather than
assumed from the docs:

1. one callback records every optimizer step, and the file rebuilds the live weights;
2. recording changes no number in the run (same seed, same losses, same weights);
3. the loss, gradient norm and learning rate join to the right step - and when the loss
   genuinely cannot be seen, the file says so instead of inventing one;
4. gradient accumulation gives one store point per *update*, not one per micro-batch;
5. TRL-style subclasses take the callback unchanged (no fork, no special case);
6. a resumed run states where it resumed.

The module skips cleanly when transformers is not installed, so CI stays honest on a runner
without it. `importorskip` is at module level on purpose: a partial skip would hide a broken
adapter behind a green run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")
pytest.importorskip("transformers", reason="transformers is not installed on this runner")
pytest.importorskip("accelerate", reason="accelerate is not installed on this runner")

from transformers import Trainer, TrainerCallback, TrainingArguments    # noqa: E402

from archx.adapters.hf import ArchxCallback, loss_hook                  # noqa: E402
from archx.container.reader import Reader                              # noqa: E402
from archx.container.schema import BlockType                           # noqa: E402
from archx.container.values import Timeline                            # noqa: E402
from archx.policy import PhasePolicy                                   # noqa: E402


class TinyNet(torch.nn.Module):
    """Two named regions plus a frozen tensor, which the registry must still record."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Linear(4, 4)
        self.frozen = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden = torch.tanh(self.body(input_ids))
        logits = self.head(hidden)
        loss = (torch.nn.functional.mse_loss(logits, labels)
                if labels is not None else None)
        return {"loss": loss, "logits": logits}


def make_dataset(steps: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return [{"input_ids": torch.randn(4, generator=generator),
             "labels": torch.randn(2, generator=generator)} for _ in range(steps)]


def train(path, *, steps=4, seed=0, hook=False, callback=None, **arg_overrides):
    """Run a real Trainer. Returns the trainer, the live weights and the callback."""
    torch.manual_seed(seed)
    model = TinyNet()
    callback = callback or ArchxCallback(
        str(path),
        regions={"body": "body", "head": "head"},
        components={"body": "net", "head": "net"},
    )
    settings = dict(
        output_dir=str(path) + ".hf", per_device_train_batch_size=1, max_steps=steps,
        logging_steps=100, report_to=[], use_cpu=True, remove_unused_columns=False,
        learning_rate=0.01, max_grad_norm=0.0, seed=seed,
    )
    settings.update(arg_overrides)
    trainer = Trainer(model=model, args=TrainingArguments(**settings),
                      train_dataset=make_dataset(steps, seed), callbacks=[callback])
    if hook:
        loss_hook(trainer)
    trainer.train()
    weights = {name: parameter.detach().clone()
               for name, parameter in model.named_parameters(remove_duplicate=False)}
    return trainer, weights, callback


def blocks(path, kind):
    return [json.loads(block.payload) for block in Reader(path).blocks(kind)]


def capabilities(path):
    return next(entry for entry in blocks(path, BlockType.INFO)
                if entry["event"] == "capabilities")


def test_one_callback_records_every_step_and_rebuilds_the_weights(tmp_path):
    path = tmp_path / "hf.archx.mdl"
    _trainer, live, callback = train(path, steps=4, hook=True)

    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1, 2, 3], "one store point per optimizer step, from zero"

    rebuilt = timeline.reconstruct(3)
    for name, weight in live.items():
        assert np.allclose(rebuilt[name], weight.numpy(), atol=1e-5), name

    # The frozen tensor is in the file and marked as not trained, not silently dropped.
    by_name = {info.key: info for info in timeline.tensors}
    assert by_name["frozen"].frozen is True
    assert by_name["body.weight"].frozen is False
    assert callback.stats.steps_stored == 4


def test_recording_does_not_change_a_single_number(tmp_path):
    """Same seed, same data: the recorded run must land on the same weights and losses."""
    plain, plain_weights, _ = train(tmp_path / "plain.hf", steps=4, seed=11)
    recorded, recorded_weights, _ = train(tmp_path / "rec.hf", steps=4, seed=11, hook=True)

    plain_losses = [entry["loss"] for entry in plain.state.log_history if "loss" in entry]
    recorded_losses = [entry["loss"] for entry in recorded.state.log_history
                       if "loss" in entry]
    assert plain_losses == recorded_losses, "the loss curve must be identical"
    for name, weight in plain_weights.items():
        assert torch.equal(weight, recorded_weights[name]), (
            f"{name} differs: the callback touched the training path")


def test_the_loss_joins_to_the_step_the_trainer_reported(tmp_path):
    """`loss_hook` must record exactly the number the Trainer itself computed."""
    path = tmp_path / "join.hf"
    # logging_steps=1 so the Trainer itself reports a loss for every step to compare against.
    trainer, _live, _callback = train(path, steps=5, hook=True, logging_steps=1)

    reported = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    observed = [entry["loss"] for entry in blocks(path, BlockType.EVENT)]

    assert len(reported) == len(observed) == 5
    assert observed == pytest.approx(reported, rel=1e-6), (
        "the recorded loss must be the Trainer's own, not a re-computation")


def test_gradient_norms_and_learning_rate_are_real_numbers(tmp_path):
    path = tmp_path / "grads.hf"
    train(path, steps=4, hook=True)

    observations = blocks(path, BlockType.EVENT)
    norms = [entry["grad_norm"] for entry in observations]
    rates = [entry["lr"] for entry in observations]

    assert all(norm is not None and norm > 0 for norm in norms)
    assert len({round(norm, 6) for norm in norms}) > 1, (
        "a constant gradient norm means the value was read after clipping, or not at all")
    # The scheduler steps after the update, so the rate falls by the same amount each step.
    assert rates[0] > rates[-1]
    assert len({round(rate, 9) for rate in rates}) == len(rates)


def test_without_a_way_to_see_the_loss_the_file_says_so(tmp_path):
    """The honest failure: tier 0, the reason named, and no invented number."""
    path = tmp_path / "noloss.hf"
    train(path, steps=3, hook=False)

    observations = blocks(path, BlockType.EVENT)
    assert all(entry["loss"] is None for entry in observations), "no loss was invented"

    capability = capabilities(path)
    assert capability["capability_tier"] == 0
    reason = capability["capabilities_missing"]["loss"]
    assert "loss_hook" in reason, "the reason must name the way out, not just the gap"
    # Gradient norms were still reached, and the file must not understate that.
    assert "grad_norms" not in capability["capabilities_missing"]


def test_a_throttled_on_log_is_never_used_to_fill_in_a_loss(tmp_path):
    """`on_log` reports a window mean, after the step: it must not become a step's loss."""
    path = tmp_path / "nolog.hf"
    train(path, steps=6, hook=False, logging_steps=1)
    assert all(entry["loss"] is None for entry in blocks(path, BlockType.EVENT))


def test_gradient_accumulation_gives_one_store_point_per_update(tmp_path):
    """Two micro-batches per update must not become two store points."""
    path = tmp_path / "accum.hf"
    _trainer, _live, callback = train(path, steps=3, hook=True, gradient_accumulation_steps=2)

    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1, 2]
    assert callback.stats.steps_seen == 3, "one call per optimizer step, not per micro-batch"


def test_the_metadata_says_which_gradient_norm_is_in_the_file(tmp_path):
    """The Trainer clips before any callback can look, so the file must label its norms."""
    clipped = tmp_path / "clipped.hf"
    train(clipped, steps=2, hook=True, max_grad_norm=1.0)   # the Trainer's own default
    assert json.loads(_first_meta(clipped))["grad_norm_clipped"] is True

    unclipped = tmp_path / "unclipped.hf"
    train(unclipped, steps=2, hook=True, max_grad_norm=0.0)
    assert json.loads(_first_meta(unclipped))["grad_norm_clipped"] is False


def _first_meta(path):
    return next(iter(Reader(path).blocks(BlockType.META))).payload


def test_a_trl_style_subclass_takes_the_callback_unchanged(tmp_path):
    """TRL and Axolotl subclass `Trainer`; the callback must need no fork for them."""

    class SFTLikeTrainer(Trainer):
        """Shaped like TRL's SFTTrainer: a Trainer subclass with its own step bookkeeping."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, _ = super().compute_loss(model, inputs, return_outputs=True,
                                           num_items_in_batch=num_items_in_batch)
            self.sft_calls = getattr(self, "sft_calls", 0) + 1
            return (loss, None) if return_outputs else loss

    path = tmp_path / "sft.hf"
    callback = ArchxCallback(str(path))
    torch.manual_seed(0)
    model = TinyNet()
    trainer = SFTLikeTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(path) + ".hf", per_device_train_batch_size=1, max_steps=3,
            logging_steps=100, report_to=[], use_cpu=True, remove_unused_columns=False,
            max_grad_norm=0.0),
        train_dataset=make_dataset(3), callbacks=[callback])
    loss_hook(trainer)
    trainer.train()

    assert trainer.sft_calls == 3, "the subclass's own compute_loss must still run"
    assert Timeline(Reader(path)).steps == [0, 1, 2]
    assert all(entry["loss"] is not None for entry in blocks(path, BlockType.EVENT))


def test_a_resumed_run_says_where_it_resumed(tmp_path):
    """A non-zero starting global_step is recorded as an annotation, not hidden.

    `Trainer.train()` rebuilds `TrainerState`, so a resumed run is reproduced the way it
    really arrives: the callback's `on_train_begin` sees a state that is already past zero.
    """
    from types import SimpleNamespace

    path = tmp_path / "resumed.hf"
    callback = ArchxCallback(str(path))
    state = SimpleNamespace(global_step=40)
    args = SimpleNamespace(learning_rate=0.01, max_grad_norm=0.0)
    callback.on_train_begin(args, state, None, model=TinyNet())
    callback.close()

    annotations = blocks(path, BlockType.ANNOT)
    resumed = [entry for entry in annotations if entry["event"] == "resumed"]
    assert resumed, "a resumed run must say so"
    assert "40" in resumed[0]["detail"]


def test_the_callback_does_not_need_transformers_at_import_time(tmp_path):
    """The library must import without transformers; only instantiating needs it."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "sys.modules['transformers'] = None;"        # make the import fail, hard
        "import archx;"
        "from archx.adapters import hf;"
        "assert hf.ArchxCallback.__name__ == 'ArchxCallback';"
        "import archx.container.reader;"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_an_impossible_data_level_is_refused_not_ignored(tmp_path):
    """Asking for segments the Trainer cannot provide must fail loudly at construction."""
    with pytest.raises(ValueError, match="data_level"):
        ArchxCallback(str(tmp_path / "x.mdl"), data_level="segments")


def test_the_callback_is_a_real_trainer_callback(tmp_path):
    """It must be accepted by transformers' own type check, or `callbacks=[...]` lies."""
    callback = ArchxCallback(str(tmp_path / "y.mdl"))
    assert isinstance(callback, TrainerCallback)


def test_an_explicit_policy_controls_what_is_stored(tmp_path):
    """A caller-supplied policy must reach the recorder, not be overridden by the adapter."""
    path = tmp_path / "policy.hf"
    callback = ArchxCallback(
        str(path), policy=PhasePolicy(total_steps=6, min_warmup_steps=2, heartbeat_steps=100))
    _trainer, _live, callback = train(path, steps=6, callback=callback, hook=True)

    stored = Timeline(Reader(path)).steps
    assert stored[0] == 0
    assert len(stored) < 6, "a sparse policy must produce a sparse file"
    assert callback.stats.steps_seen == 6


def test_only_the_main_rank_writes_a_file(tmp_path):
    """Under DDP every rank holds identical weights; N ranks must not append to one file.

    A non-main rank is simulated by overriding the process check, because a real multi-GPU
    runner is not available on the CI matrix. The behaviour being pinned is the guard
    itself: no recorder, no file, and the training run still completes.
    """
    from unittest import mock

    path = tmp_path / "rank.hf"
    callback = ArchxCallback(str(path))
    torch.manual_seed(0)
    trainer = Trainer(
        model=TinyNet(),
        args=TrainingArguments(
            output_dir=str(path) + ".hf", per_device_train_batch_size=1, max_steps=2,
            logging_steps=100, report_to=[], use_cpu=True, remove_unused_columns=False,
            max_grad_norm=0.0),
        train_dataset=make_dataset(2), callbacks=[callback])

    with mock.patch.object(ArchxCallback, "_is_main_process", staticmethod(lambda args: False)):
        trainer.train()

    assert callback.recorder is None, "a non-main rank must not open a recorder"
    assert not path.exists(), "a non-main rank must not write the file"
    assert trainer.state.global_step == 2, "training must still run to completion"


def test_the_main_rank_marker_is_in_the_file(tmp_path):
    """The file says which rank recorded it, so a sharded-looking file can be explained."""
    path = tmp_path / "meta.hf"
    train(path, steps=2, hook=True)
    assert json.loads(_first_meta(path))["recording_rank"] == 0

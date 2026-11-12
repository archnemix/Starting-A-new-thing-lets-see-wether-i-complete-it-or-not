"""Unit tests for the recorder: it observes, it never mutates, and it degrades instead of raising."""

from __future__ import annotations

import json

import numpy as np
import pytest

from archx.container.reader import Reader
from archx.container.values import Timeline
from archx.container.schema import BlockType
from archx.formats.registry import TensorRegistry
from archx.policy import Decision, PhasePolicy, Triggers
from archx.recorder import Recorder


#: Switches every event trigger off, so a test can state exactly when a store happens.
_NO_TRIGGERS = Triggers(grad_norm_zscore=float("inf"), loss_drop_fraction=float("inf"),
                        region_move_fraction=float("inf"))


class FakeModel:
    """Minimal stand-in for a model's parameters, with a step that moves them."""

    def __init__(self, seed: int = 0, size: int = 64) -> None:
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, 0.1, (size, size)).astype(np.float32),
            "b1": np.zeros(size, dtype=np.float32),
        }

    def read(self, key: str) -> np.ndarray:
        return self.params[key]

    def registry(self) -> TensorRegistry:
        reg = TensorRegistry()
        for key, array in self.params.items():
            reg.add(key, array, pointer=array.ctypes.data)
        return reg

    def step(self, rng: np.random.Generator, scale: float = 1e-3) -> None:
        for key, array in self.params.items():
            array += rng.normal(0, scale, array.shape).astype(np.float32)


def run_training(tmp_path, steps=30, policy=None, seed=0):
    model = FakeModel(seed=seed)
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "run.archx.mdl"),
                        policy=policy or PhasePolicy(total_steps=steps, min_warmup_steps=2))
    rng = np.random.default_rng(seed)
    for step in range(steps):
        model.step(rng)
        recorder.step(step, loss=1.0 / (step + 1))
    recorder.close()
    return model, recorder, tmp_path / "run.archx.mdl"


def test_recording_writes_a_readable_file_with_meta_and_registry(tmp_path):
    _, _, path = run_training(tmp_path)
    reader = Reader(path)
    assert reader.container.footer is not None
    meta = reader.meta()
    assert meta["attribution"] == "observational"
    assert meta["tensors"] == 2
    assert list(reader.blocks(BlockType.TENSOR_REGISTRY))


def test_recording_never_changes_the_weights(tmp_path):
    model = FakeModel(seed=5)
    rng_a = np.random.default_rng(9)
    for _ in range(20):
        model.step(rng_a)
    without = {k: v.copy() for k, v in model.params.items()}

    model_b = FakeModel(seed=5)
    recorder = Recorder(model_b.registry(), model_b.read, str(tmp_path / "b.archx.mdl"),
                        policy=PhasePolicy(total_steps=20))
    rng_b = np.random.default_rng(9)
    for step in range(20):
        model_b.step(rng_b)
        recorder.step(step, loss=1.0)
    recorder.close()

    for key in without:
        assert np.array_equal(without[key], model_b.params[key]), \
            "recording must not alter a single weight"


def test_a_recorder_fault_degrades_instead_of_breaking_training(tmp_path):
    model = FakeModel()
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "faulty.archx.mdl"),
                        policy=PhasePolicy(total_steps=5))

    def broken_read(key):
        raise RuntimeError("simulated read failure")

    recorder._read = broken_read
    decision = recorder.step(0)
    assert decision is Decision.SKIP
    assert recorder.stats.failed >= 1
    with pytest.raises(RuntimeError):  # the fault is remembered, not hidden
        recorder.close()


def test_a_tiny_queue_records_every_step_in_order(tmp_path):
    """Backpressure must cost the trainer time, not the history.

    This used to assert the opposite: that a full queue was counted in `stats.failed` and
    the step was thrown away. That was the recorder silently losing a third of a run's
    stored steps at the default queue size, which is the one thing it must never do.
    """
    model = FakeModel()
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "busy.archx.mdl"),
                        policy=PhasePolicy(total_steps=10, min_warmup_steps=10), queue_size=1)
    for step in range(10):
        recorder.step(step, loss=1.0)  # every step stores; queue is size 1
    recorder.close()
    assert recorder.stats.steps_seen == 10
    assert recorder.stats.steps_stored == 10
    assert recorder.stats.failed == 0

    reader = Reader(tmp_path / "busy.archx.mdl")
    stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                    | {b.step for b in reader.blocks(BlockType.ANCHOR)})
    assert stored == list(range(10)), "stored steps must be complete and in order"


def test_annotations_stay_out_of_the_step_observations(tmp_path):
    """ANNOT is the user's word; EVENT is what the run measured. Mixing them loses the join."""
    model = FakeModel()
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "notes.archx.mdl"),
                        policy=PhasePolicy(total_steps=3))
    recorder.step(0, loss=1.0)
    recorder.annotate(1, "stage2_start", "unfroze the language model")
    recorder.close()

    reader = Reader(tmp_path / "notes.archx.mdl")
    notes = [json.loads(b.payload) for b in reader.blocks(BlockType.ANNOT)]
    assert notes == [{"step": 1, "event": "stage2_start",
                      "detail": "unfroze the language model"}]
    observations = [json.loads(b.payload) for b in reader.blocks(BlockType.EVENT)]
    assert observations and not any("event" in o for o in observations)


def test_a_stored_step_keeps_the_loss_the_gradient_norm_and_the_movement(tmp_path):
    """Tier 1 is data the loop already had. Dropping it would make the file poorer than the run."""
    model = FakeModel(seed=2, size=16)
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "obs.archx.mdl"),
                        policy=PhasePolicy(total_steps=3, min_warmup_steps=3))
    rng = np.random.default_rng(7)
    for step in range(2):
        model.step(rng)
        recorder.step(step, loss=1.0 / (step + 1), grad_norm=0.5)
    recorder.close()

    observations = [json.loads(b.payload) for b in Reader(tmp_path / "obs.archx.mdl")
                    .blocks(BlockType.EVENT)]
    assert [o["step"] for o in observations] == [0, 1]
    assert observations[0]["loss"] == 1.0 and observations[1]["loss"] == 0.5
    assert all(o["grad_norm"] == 0.5 for o in observations)
    assert all(o["regions"] for o in observations), "movement per region must be recorded"


def test_log_carries_the_loss_and_is_not_reused_by_the_next_step(tmp_path):
    model = FakeModel(seed=3, size=8)
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "logged.archx.mdl"),
                        policy=PhasePolicy(total_steps=3, min_warmup_steps=3))
    rng = np.random.default_rng(11)
    model.step(rng)
    recorder.log(loss=2.0, data=b"a batch of text\n")     # the adapter's one optional line
    recorder.step(0, grad_norm=0.1)
    model.step(rng)
    recorder.step(1, grad_norm=0.1)                        # nothing logged for this step
    recorder.close()

    reader = Reader(tmp_path / "logged.archx.mdl")
    observations = [json.loads(b.payload) for b in reader.blocks(BlockType.EVENT)]
    assert observations[0]["loss"] == 2.0
    assert observations[1]["loss"] is None, "a logged value belongs to one step only"
    assert [json.loads(b.payload) for b in reader.blocks(BlockType.DATA_SEGMENT)]


def test_the_file_states_the_tier_it_reached_and_what_it_could_not_see(tmp_path):
    model = FakeModel(seed=4, size=8)
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "tier.archx.mdl"),
                        policy=PhasePolicy(total_steps=2, min_warmup_steps=2))
    model.step(np.random.default_rng(5))
    recorder.step(0, loss=1.0, grad_norm=1.0, data=b"text\n")
    recorder.close()

    info = [json.loads(b.payload) for b in Reader(tmp_path / "tier.archx.mdl")
            .blocks(BlockType.INFO)]
    capability = next(block for block in info if block["event"] == "capabilities")
    assert capability["capability_tier"] == 2, "loss, gradients and data were all recorded"
    assert capability["capabilities_missing"].get("activations"), (
        "tier 3 must be reported missing with a reason, not silently absent")


def test_stored_delta_covers_every_skipped_step(tmp_path):
    """A skipped step must be absorbed by the next stored delta, or history loses motion."""
    model = FakeModel(seed=1, size=256)
    # Triggers off: this test is about absorption, not about what counts as an event. The
    # default movement trigger would fire here, because the test moves the weights by ~10%
    # of their magnitude per step, which is a large move by any measure.
    policy = PhasePolicy(total_steps=1_000_000, warmup_fraction=0.0, min_warmup_steps=0,
                         heartbeat_steps=1_000_000, triggers=_NO_TRIGGERS)
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "acc.archx.mdl"),
                        policy=policy)
    rng = np.random.default_rng(4)

    model.step(rng, scale=1e-2)
    recorder.step(0, loss=1.0)                      # step 0 is an anchor
    before = {k: v.copy() for k, v in model.params.items()}

    decisions = []
    for step in range(1, 6):                        # middle phase, no triggers -> skipped
        model.step(rng, scale=1e-2)
        decisions.append(recorder.step(step, loss=1.0))
    decision = recorder.step(6, loss=1.0, event=True)   # a stage change forces a store
    recorder.close()

    assert decisions == [Decision.SKIP] * 5, "the middle phase must skip uneventful steps"

    # Reconstruct step 6 from the file alone and compare with the live weights: the delta
    # must equal the sum of six steps of movement, not the last one.
    reader = Reader(tmp_path / "acc.archx.mdl")
    anchor = reader.blocks(BlockType.ANCHOR).__next__().payload
    deltas = [b for b in reader.blocks(BlockType.DELTA)]
    assert len(anchor) > 0 and len(deltas) >= 1
    stored_movement = 0.0
    for block in deltas:
        body = block.payload
        count = int(np.frombuffer(body[8:12], dtype=np.uint32)[0])
        offset = 12
        for _ in range(count):
            index = int(np.frombuffer(body[offset:offset + 4], dtype=np.uint32)[0])
            encoding = int(body[offset + 4])
            dtype_id = int(body[offset + 5])
            length = int(np.frombuffer(body[offset + 6:offset + 10], dtype=np.uint32)[0])
            payload = body[offset + 10:offset + 10 + length]
            offset += 10 + length
            if index == 0:  # w1
                decoder = {1: _Decoders.sparse, 2: _Decoders.dense, 3: _Decoders.lowrank,
                           4: _Decoders.raw}.get(int(encoding))
                if decoder is not None:
                    stored_movement += float(np.linalg.norm(
                        decoder(payload, model.params["w1"].shape)))
    live_movement = float(np.linalg.norm(model.params["w1"] - before["w1"]))
    assert stored_movement > 0
    assert live_movement / 2 < stored_movement < live_movement * 2, (
        "the stored delta must cover the whole skipped run, not one step: "
        f"stored={stored_movement:.4f} live={live_movement:.4f}")


class _Decoders:
    """Decode a delta body for the accumulation test."""

    @staticmethod
    def sparse(body, shape):
        from archx.encoders import Sparse

        return Sparse().decode(body, shape, "f32", 0)

    @staticmethod
    def dense(body, shape):
        from archx.encoders import DenseQuant

        return DenseQuant(bits=8).decode(body, shape, "i8", 0)

    @staticmethod
    def lowrank(body, shape):
        from archx.encoders import LowRank

        return LowRank().decode(body, shape, "f32", 0)

    @staticmethod
    def raw(body, shape):
        from archx.encoders import DenseRaw

        return DenseRaw().decode(body, shape, "f32", 0)


def test_close_reports_the_failure_it_swallowed(tmp_path):
    model = FakeModel()
    recorder = Recorder(model.registry(), model.read, str(tmp_path / "x.archx.mdl"),
                        policy=PhasePolicy(total_steps=2))

    def broken(key):
        raise ValueError("boom")

    recorder._read = broken
    recorder.step(0)
    with pytest.raises(RuntimeError):
        recorder.close()


def test_a_full_queue_loses_no_step(tmp_path):
    """A small queue must slow the trainer, never drop history.

    At the default queue size a 300-step dense run used to lose 13 of its stored steps:
    the writer could not keep up, `_enqueue` counted a failure and threw the item away.
    Silently losing history breaks the one promise the product makes.
    """
    steps = 300
    model = FakeModel(seed=1)
    path = tmp_path / "tight.archx.mdl"
    recorder = Recorder(model.registry(), model.read, str(path),
                        policy=PhasePolicy(total_steps=steps, min_warmup_steps=steps),
                        queue_size=2)
    rng = np.random.default_rng(1)
    truth = {}
    for step in range(steps):
        model.step(rng)
        recorder.step(step, loss=1.0 / (step + 1))
        truth[step] = {key: value.copy() for key, value in model.params.items()}
    recorder.close()

    assert recorder.stats.failed == 0
    assert recorder.stats.steps_stored == steps

    reader = Reader(path)
    stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                    | {b.step for b in reader.blocks(BlockType.ANCHOR)})
    assert stored == list(range(steps)), "every step must be present exactly once"

    timeline = Timeline(reader)
    for step in range(steps):
        rebuilt = timeline.reconstruct(step)
        for key, expected in truth[step].items():
            assert np.array_equal(rebuilt[key], expected), f"step {step} {key} differs"


def test_the_first_stored_step_is_an_anchor_at_step_zero(tmp_path):
    """Reconstruction starts somewhere; if it starts late the file's first steps are fiction."""
    model = FakeModel(seed=2)
    path = tmp_path / "first.archx.mdl"
    recorder = Recorder(model.registry(), model.read, str(path),
                        policy=PhasePolicy(total_steps=500), queue_size=8)
    rng = np.random.default_rng(2)
    for step in range(120):
        model.step(rng)
        recorder.step(step, loss=1.0)
    recorder.close()

    reader = Reader(path)
    anchors = [b.step for b in reader.blocks(BlockType.ANCHOR)]
    deltas = [b.step for b in reader.blocks(BlockType.DELTA)]
    assert anchors and anchors[0] == 0
    assert deltas and deltas[0] == 1
    assert min(anchors + deltas) == 0

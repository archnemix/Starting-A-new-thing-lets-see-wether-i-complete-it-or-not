"""Unit tests for reading values: registry, anchors, deltas and reconstruction."""

from __future__ import annotations

import numpy as np
import pytest

from archx.container.reader import Reader
from archx.container.values import Timeline, parse_delta, parse_registry
from archx.errors import FormatError
from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy
from archx.recorder import Recorder


class Model:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {"w": rng.normal(0, 0.1, (16, 16)).astype(np.float32),
                       "b": np.zeros(16, dtype=np.float32)}

    def registry(self):
        reg = TensorRegistry()
        for key, array in self.params.items():
            reg.add(key, array, pointer=array.ctypes.data)
        return reg


def recording(tmp_path, steps=3, seed=0):
    model = Model(seed)
    path = tmp_path / "run.archx.mdl"
    recorder = Recorder(model.registry(), model.params.__getitem__, str(path),
                        policy=PhasePolicy(total_steps=steps, min_warmup_steps=steps))
    rng = np.random.default_rng(seed)
    history = []
    for step in range(steps):
        model.params["w"] += rng.normal(0, 1e-3, (16, 16)).astype(np.float32)
        recorder.step(step, loss=1.0 / (step + 1))
        history.append({k: v.copy() for k, v in model.params.items()})
    recorder.close()
    return path, history


def test_registry_round_trips_names_shapes_and_regions(tmp_path):
    model = Model()
    path = tmp_path / "reg.archx.mdl"
    reg = TensorRegistry()
    reg.add("attention.q_proj.weight", model.params["w"], region="attention",
            component="language_model")
    reader_fn = {"attention.q_proj.weight": model.params["w"]}.__getitem__
    recorder = Recorder(reg, reader_fn, str(path), policy=PhasePolicy(total_steps=1))
    recorder.step(0, loss=1.0)
    recorder.close()

    reader = Reader(path)
    tensors = parse_registry(list(reader.blocks())[1].payload)
    assert tensors[0].key == "attention.q_proj.weight"
    assert tensors[0].shape == (16, 16)
    assert tensors[0].region == "attention"
    assert tensors[0].component == "language_model"


def test_reconstruction_at_the_anchor_is_exact(tmp_path):
    path, history = recording(tmp_path)
    timeline = Timeline(Reader(path))
    restored = timeline.reconstruct(0)
    for key, expected in history[0].items():
        assert np.array_equal(restored[key], expected), f"{key} must be bit exact at an anchor"


def test_reconstruction_after_deltas_reproduces_the_weights(tmp_path):
    path, history = recording(tmp_path, steps=4)
    timeline = Timeline(Reader(path))
    restored = timeline.reconstruct(3)
    for key, expected in history[3].items():
        error = np.max(np.abs(restored[key] - expected))
        assert error < 1e-5, f"{key} off by {error} after chained deltas"


def test_timeline_lists_steps_in_order(tmp_path):
    path, _ = recording(tmp_path, steps=3)
    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1, 2]


def test_reconstruct_before_any_anchor_is_refused(tmp_path):
    path, _ = recording(tmp_path)
    timeline = Timeline(Reader(path))
    with pytest.raises(FormatError):
        timeline.reconstruct(-1)


def test_delta_with_an_unknown_encoding_is_an_error_not_a_guess(tmp_path):
    path, _ = recording(tmp_path, steps=2)
    reader = Reader(path)
    delta_block = [b for b in reader.blocks() if b.block_type == 4][0]
    payload = bytearray(delta_block.payload)
    payload[12 + 4] = 99  # encoding byte of the first entry
    tensors = parse_registry(list(reader.blocks())[1].payload)
    with pytest.raises(FormatError):
        parse_delta(bytes(payload), tensors)


def test_tied_tensors_come_back_identical_after_deltas_are_applied(tmp_path):
    """Two names over one storage must not drift apart once a delta is added.

    The delta is stored under the primary name only, so a reader that applies it to that name
    and forgets the alias would report a model where the shared weight has two values.
    """
    from archx.formats.registry import TensorRegistry
    from archx.policy import PhasePolicy
    from archx.recorder import Recorder

    shared = np.random.default_rng(0).normal(0, 0.1, (4, 4)).astype(np.float32)
    other = np.random.default_rng(1).normal(0, 0.1, (4, 4)).astype(np.float32)
    params = {"encoder.weight": shared, "decoder.weight": shared, "head.weight": other}
    registry = TensorRegistry()
    for key, array in params.items():
        registry.add(key, array, pointer=array.ctypes.data)

    path = tmp_path / "tied.archx.mdl"
    recorder = Recorder(registry, params.__getitem__, str(path),
                        policy=PhasePolicy(total_steps=3, min_warmup_steps=3))
    rng = np.random.default_rng(2)
    for step in range(3):
        shared += rng.normal(0, 1e-2, shared.shape).astype(np.float32)
        other += rng.normal(0, 1e-2, other.shape).astype(np.float32)
        recorder.step(step, loss=1.0)
    recorder.close()

    state = Timeline(Reader(path)).reconstruct(2)
    assert np.array_equal(state["decoder.weight"], state["encoder.weight"])
    assert np.allclose(state["encoder.weight"], shared, atol=1e-6)
    assert np.allclose(state["head.weight"], other, atol=1e-6)

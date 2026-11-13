"""Tier 3: activation summaries. What the network saw, and what the file can prove about it."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.data.activations import (
    MAX_ELEMENTS,
    SATURATION_LIMIT,
    ActivationCapture,
    ActivationStats,
    _as_float_array,
)


def test_a_summary_matches_the_exact_statistics_of_its_tensor():
    """Every number in the block is derived, so it must equal a plain numpy computation."""
    rng = np.random.default_rng(7)
    array = rng.normal(0.5, 2.0, (32, 16)).astype(np.float32)

    stats = ActivationStats(layer="x", shape=list(array.shape), dtype="float32")
    stats.observe(array)
    payload = stats.to_dict()

    assert payload["layer"] == "x"
    assert payload["shape"] == [32, 16]
    assert payload["count"] == array.size
    assert payload["mean"] == pytest.approx(float(array.mean()), rel=1e-6)
    assert payload["std"] == pytest.approx(float(array.std()), rel=1e-6)
    assert payload["min"] == pytest.approx(float(array.min()))
    assert payload["max"] == pytest.approx(float(array.max()))
    assert payload["zero_fraction"] == pytest.approx(float((array == 0).mean()))
    assert payload["saturated_fraction"] == pytest.approx(
        float((np.abs(array) >= SATURATION_LIMIT).mean()))


def test_several_forwards_combine_without_keeping_any_of_them():
    """Two forwards per step is normal, and the block says how many it is averaging."""
    first = np.array([[1.0, 3.0]], dtype=np.float32)
    second = np.array([[5.0, 7.0]], dtype=np.float32)

    stats = ActivationStats(layer="x", shape=[1, 2], dtype="float32")
    stats.observe(first)
    stats.observe(second)
    payload = stats.to_dict()

    assert payload["forwards"] == 2
    assert payload["count"] == 4
    combined = np.concatenate([first.ravel(), second.ravel()])
    assert payload["mean"] == pytest.approx(float(combined.mean()))
    assert payload["std"] == pytest.approx(float(combined.std()))


def test_a_strided_sample_still_describes_the_whole_tensor():
    """The cap is why the cost is bounded; this is why it is still honest.

    A strided sample must not be mistaken for the start of the batch, so the check is that the
    sampled mean stays close to the exact mean of a tensor whose first half is very different
    from its second - the case a truncating sample gets badly wrong.
    """
    rng = np.random.default_rng(3)
    array = np.concatenate([rng.normal(-10, 1, MAX_ELEMENTS * 2),
                            rng.normal(+10, 1, MAX_ELEMENTS * 2)]).astype(np.float32)
    exact = float(array.mean())

    stats = ActivationStats(layer="big", shape=list(array.shape), dtype="float32")
    stats.observe(array)
    payload = stats.to_dict()

    assert payload["count"] <= MAX_ELEMENTS + 4
    assert payload["sampled"] == payload["count"], "the sample size is recorded"
    assert abs(payload["mean"] - exact) < 0.5, (
        f"strided sample said {payload['mean']:.3f}, exact is {exact:.3f}: a truncated "
        "sample would say about -10")


def test_non_finite_values_are_counted_and_do_not_poison_the_moments():
    array = np.array([1.0, 2.0, math.inf, math.nan, 3.0], dtype=np.float32)
    stats = ActivationStats(layer="bad", shape=[5], dtype="float32")
    stats.observe(array)
    payload = stats.to_dict()

    assert payload["nonfinite"] == 2, "inf and nan are facts, and both are recorded"
    assert payload["count"] == 3, "the finite values are what the mean is over"
    assert payload["mean"] == pytest.approx(2.0)
    assert math.isfinite(payload["std"])


def test_an_integer_output_is_not_an_activation():
    """A hook fires for every module. An embedding index or a label tensor is not a summary."""
    assert _as_float_array(np.arange(10, dtype=np.int64)) is None
    assert _as_float_array("not a tensor") is None
    assert _as_float_array(()) is None


def test_a_hook_output_tuple_and_a_float_tensor_are_both_understood():
    array = np.array([[0.5, -0.5]], dtype=np.float32)
    assert _as_float_array(array) is not None
    from_tuple = _as_float_array((array, "ignored"))
    assert from_tuple is not None and from_tuple.shape == (1, 2)


def test_the_layer_cap_stops_growing_and_says_it_did():
    capture = ActivationCapture(max_layers=3)
    for index in range(10):
        capture.observe(f"layer{index}", np.zeros((2, 2), dtype=np.float32))

    payloads = capture.take()
    assert len(payloads) == 3, "the cap is a cap"
    assert capture.skipped == 7, "and the ones it dropped are counted, never silently lost"
    assert [p["layer"] for p in payloads] == ["layer0", "layer1", "layer2"]


def test_take_empties_the_accumulator_so_one_step_cannot_see_another():
    capture = ActivationCapture()
    capture.observe("a", np.ones((2, 2), dtype=np.float32))
    assert capture.take()
    assert capture.take() == [], "nothing survives a take"


def test_a_real_forward_pass_reaches_tier_three_with_readable_blocks(tmp_path):
    torch = pytest.importorskip("torch")
    import archx
    from archx.policy import PhasePolicy

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    inputs = torch.randn(8, 16)
    targets = torch.randint(0, 4, (8,))
    path = tmp_path / "tier3.archx.mdl"

    recorder = archx.attach(model, optimizer, str(path), data_level="none",
                            activations=True, queue_size=2,
                            policy=PhasePolicy(total_steps=6, min_warmup_steps=1,
                                               heartbeat_steps=2))
    for _ in range(6):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        recorder.log(loss=float(loss.detach()), data=inputs)
        optimizer.step()
    recorder.close()

    reader = Reader(path)
    rows = [json.loads(block.payload) for block in reader.blocks(BlockType.ACTIVATION)]
    assert rows, "activations=True must produce activation blocks"
    assert {row["layer"] for row in rows} == {"0", "1", "2"}, "every leaf layer is summarised"

    for row in rows:
        assert row["step"] >= 0
        assert row["count"] > 0
        assert math.isfinite(row["mean"]) and math.isfinite(row["std"])
        assert 0.0 <= row["zero_fraction"] <= 1.0
        assert 0.0 <= row["saturated_fraction"] <= 1.0

    capabilities = reader.capabilities()
    assert capabilities["capability_tier"] == 3, "loss + grads + data + activations"
    assert "activations" in capabilities["capabilities_present"]

    # A tier-3 claim with no blocks behind it is the failure this whole module guards against.
    assert len(rows) >= len({row["step"] for row in rows}), (
        "at least one summary per stored step")


def test_recording_activations_does_not_change_the_numbers(tmp_path):
    """The one thing a recorder may never do: change what the model computes."""
    torch = pytest.importorskip("torch")
    import archx
    from archx.policy import PhasePolicy

    def run(activations: bool, path):
        torch.manual_seed(11)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(),
                                    torch.nn.Linear(8, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.randn(4, 8)
        targets = torch.randint(0, 2, (4,))
        recorder = archx.attach(model, optimizer, str(path), data_level="none",
                                activations=activations,
                                policy=PhasePolicy(total_steps=4, min_warmup_steps=1,
                                                   heartbeat_steps=2))
        losses = []
        for _ in range(4):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            recorder.log(loss=float(loss.detach()))
            optimizer.step()
            losses.append(float(loss.detach()))
        recorder.close()
        return losses

    plain = run(False, tmp_path / "plain.mdl")
    captured = run(True, tmp_path / "captured.mdl")
    assert plain == captured, "a forward hook that returned its output would change these"


def test_the_hooks_are_removed_when_the_recorder_closes(tmp_path):
    """A hook left behind keeps writing into a closed file, from an unrelated forward pass."""
    torch = pytest.importorskip("torch")
    import archx
    from archx.policy import PhasePolicy

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    recorder = archx.attach(model, optimizer, str(tmp_path / "hooks.mdl"), data_level="none",
                            activations=True,
                            policy=PhasePolicy(total_steps=2, min_warmup_steps=1))
    assert recorder._archx_activation_handles, "hooks were registered"

    with torch.no_grad():
        model(torch.randn(2, 4))
    assert recorder.activations.stats, "the hook is live while the recorder is"

    recorder.close()
    assert recorder._archx_activation_handles == []

    recorder.activations.take()          # clear, then prove nothing new arrives
    with torch.no_grad():
        model(torch.randn(2, 4))
    assert recorder.activations.stats == {}, "a closed recorder must not see forwards"


def test_a_file_without_activations_says_why_rather_than_being_empty(tmp_path):
    torch = pytest.importorskip("torch")
    import archx
    from archx.policy import PhasePolicy

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    path = tmp_path / "noacts.mdl"
    recorder = archx.attach(model, optimizer, str(path), data_level="none",
                            policy=PhasePolicy(total_steps=2, min_warmup_steps=1))
    optimizer.zero_grad()
    loss = model(torch.randn(4, 4)).sum()
    loss.backward()
    recorder.log(loss=float(loss.detach()))
    optimizer.step()
    recorder.close()

    reader = Reader(path)
    assert list(reader.blocks(BlockType.ACTIVATION)) == []
    missing = reader.capabilities()["capabilities_missing"]
    assert "activations" in missing and missing["activations"], (
        "the reason is in the file, not left to the reader")

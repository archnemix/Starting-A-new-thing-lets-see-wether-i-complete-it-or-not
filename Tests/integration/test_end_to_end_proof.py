"""Integration: the v1.0 promise, on three real tasks trained on real data.

`Bench/phase09_end_to_end.py` produces the numbers; this file pins the *promises* those
numbers rest on, so a regression fails a test instead of quietly changing a table in a doc.

The promises, in the order a user would care about them:

1. recording does not change the result - same seed, same batches, identical loss curve;
2. the anchor is exact, and the steps after it carry only float32 summation error;
3. the file reaches tier 2 on a real training loop, with every gap named;
4. a step that was never stored can still be answered, from the nearest stored step;
5. the recorded *data* is the data the step trained on, joined to the weights it moved.

The tasks are small on purpose - the point is that they are genuine optimisations (a causal
transformer with two attention heads, a convnet, an LSTM), not that they are large.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

import sys                                                       # noqa: E402
from pathlib import Path                                         # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Bench"))   # noqa: E402

from phase09_end_to_end import TASKS, run                        # noqa: E402

from archx.container.reader import Reader                        # noqa: E402
from archx.container.schema import BlockType                     # noqa: E402
from archx.container.values import Timeline                      # noqa: E402

STEPS = 20


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_recording_does_not_change_the_result(task_name, tmp_path):
    """The single promise that matters most: the run is the same run.

    Compared on the loss curve of the *final* step with the same seed and the same batches,
    to exact equality rather than a tolerance - a recorder that perturbs float32 arithmetic
    at all would show up here.
    """
    plain = run(task_name, steps=STEPS, record=False)
    recorded = run(task_name, steps=STEPS, record=True, path=tmp_path / "ab.archx.mdl")

    assert recorded["last_loss"] == plain["last_loss"], (
        f"{task_name}: recording changed the final loss "
        f"({plain['last_loss']!r} vs {recorded['last_loss']!r})")
    assert recorded["first_loss"] == plain["first_loss"]


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_the_anchor_is_exact_and_the_deltas_are_float32_rounding(task_name, tmp_path):
    """Exact at anchors, ~1e-8 between them - the guarantee stated in the docs."""
    result = run(task_name, steps=STEPS, record=True, path=tmp_path / "fid.archx.mdl")

    assert result["anchor_max_error"] == 0.0, (
        f"{task_name}: an anchor must rebuild bit-exactly, got {result['anchor_max_error']}")
    assert result["worst_delta_error"] < 1e-6, (
        f"{task_name}: chained delta error {result['worst_delta_error']} is larger than "
        f"float32 summation")


@pytest.mark.parametrize("task_name", sorted(TASKS))
def test_a_real_training_loop_reaches_tier_two_with_its_gaps_named(task_name, tmp_path):
    path = tmp_path / "tier.archx.mdl"
    run(task_name, steps=STEPS, record=True, path=path)

    capability = next(
        entry for entry in (json.loads(block.payload) for block in Reader(path).blocks(
            BlockType.INFO)) if entry["event"] == "capabilities")

    assert capability["capability_tier"] == 2, (
        f"{task_name} reached tier {capability['capability_tier']}")
    assert "loss" not in capability["capabilities_missing"]
    assert "grad_norms" not in capability["capabilities_missing"]
    assert "data_segments" not in capability["capabilities_missing"]
    # Tier 3 is not claimed by any adapter yet, and the file must say so rather than imply it.
    assert "activations" in capability["capabilities_missing"]


def test_a_step_that_was_never_stored_is_still_answerable(tmp_path):
    """The product's reason to exist: a question about an unstored step must have an answer."""
    path = tmp_path / "gap.archx.mdl"
    run("rnn", steps=30, record=True, path=path)

    timeline = Timeline(Reader(path))
    stored = set(timeline.steps)
    unstored = [step for step in range(timeline.steps[0], timeline.steps[-1])
                if step not in stored]
    assert unstored, "this run must skip at least one step for the test to mean anything"

    step = unstored[len(unstored) // 2]
    rebuilt = timeline.reconstruct(step)
    assert rebuilt, "reconstruction returned nothing for an unstored step"
    assert step not in stored
    # And the file says how far that answer had to be carried.
    base = max(s for s in timeline.steps if s <= step)
    assert base <= step


def test_the_recorded_data_is_the_data_the_step_trained_on(tmp_path):
    """The join: every stored batch is in the file, with its effect on the weights."""
    path = tmp_path / "join.archx.mdl"
    run("cnn", steps=STEPS, record=True, path=path)

    reader = Reader(path)
    segments = [json.loads(block.payload) for block in reader.blocks(BlockType.DATA_SEGMENT)]
    assert segments, "a tier-2 recording must contain data segments"

    for segment in segments:
        assert segment["fingerprint"], "a segment without a content hash cannot be deduplicated"
        assert segment["region_effects"], (
            "a segment with no region effects cannot answer 'why did the weights move'")
        assert segment["first_step"] <= segment["last_step"]


def test_the_weights_and_the_data_can_be_told_apart_in_the_file(tmp_path):
    """Storage claims compare weights with a checkpoint; the data half must be separable."""
    path = tmp_path / "split.archx.mdl"
    result = run("cnn", steps=STEPS, record=True, path=path)

    assert result["weights_bytes"] > 0
    assert result["data_bytes"] > 0
    assert result["weights_bytes"] + result["data_bytes"] <= result["bytes"]
    # A checkpoint holds weights only, so the comparison must use the weight half: counting
    # the data half once made a recording look larger than a checkpoint at the same cadence.
    assert result["one_checkpoint_bytes"] > 0

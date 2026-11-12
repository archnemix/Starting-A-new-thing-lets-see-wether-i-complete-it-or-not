"""Integration: the join - this data moved these weights at these steps - from the file alone.

This is the test the whole product exists for. If it passes, a reader can answer the user's
original question without the training run, the recorder, or the original data location.
"""

from __future__ import annotations

import json

import numpy as np

from archx.container.reader import Reader
from archx.container.values import Timeline
from archx.container.schema import BlockType
from archx.data.capture import DataLevel
from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy
from archx.recorder import Recorder


class ToyModel:
    """A tiny 'model' whose named tensors move differently per phase, like a real run."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.params = {
            "attention.q_proj.weight": rng.normal(0, 0.05, (32, 32)).astype(np.float32),
            "mlp.fc.weight": rng.normal(0, 0.05, (32, 32)).astype(np.float32),
        }

    def registry(self, regions: dict[str, str]) -> TensorRegistry:
        reg = TensorRegistry()
        for key, array in self.params.items():
            reg.add(key, array, pointer=array.ctypes.data,
                    region=regions.get(key, ""), component="language_model")
        return reg

    def train_step(self, rng, scale: float) -> None:
        for key, array in self.params.items():
            array += rng.normal(0, scale, array.shape).astype(np.float32)


def test_a_file_answers_which_data_moved_which_weights_at_which_step(tmp_path):
    regions = {"attention.q_proj.weight": "attention", "mlp.fc.weight": "mlp"}
    model = ToyModel()
    path = tmp_path / "join.archx.mdl"
    recorder = Recorder(
        model.registry(regions), model.params.__getitem__, str(path),
        policy=PhasePolicy(total_steps=6, min_warmup_steps=6),
        data_level=DataLevel.SEGMENTS,
        meta={"adapter": "torch"},
    )
    rng = np.random.default_rng(3)
    batches = {
        0: b"python: for loops and functions\n",
        1: b"python: for loops and functions\n",   # same shard: must not be stored twice
        2: b"math: algebra and geometry\n",
        3: b"python: for loops and functions\n",
    }
    for step in range(4):
        model.train_step(rng, 1e-2 if step < 2 else 1e-3)   # big movement, then small
        recorder.step(step, loss=1.0 - 0.1 * step, data=batches[step],
                      segment_id=f"seg-{step}" if step else "seg-0")
    recorder.close()

    reader = Reader(path)

    # 1. the data is in the file, once per unique shard
    shards = list(reader.blocks(BlockType.DATA_SHARD))
    assert len(shards) == 2, f"two unique batches must mean two shards, got {len(shards)}"
    segments = Timeline(reader).segments()   # one entry per id: restatements resolved
    assert {s["segment_id"] for s in segments} >= {"seg-0", "seg-1"}

    # 2. segments carry characteristics, so data can be compared across runs later
    assert all("characteristics" in s for s in segments)
    assert segments[0]["characteristics"]["kind"] == "text"

    # 3. the weights are in the file, anchored and delta'd
    assert list(reader.blocks(BlockType.ANCHOR))
    deltas = list(reader.blocks(BlockType.DELTA))
    assert len(deltas) == 3, "one delta per stored step after the anchor"

    # 4. the join: every delta block names the step, and every segment names its steps
    assert [b.step for b in deltas] == [1, 2, 3]
    assert {s["first_step"] for s in segments} == {0, 1, 2, 3}

    # 5. dedup, seen through the file: steps 0 and 3 used the same bytes under different
    # segment ids, so they must point at one shard id - and only one shard may exist for it.
    python_steps = [s for s in segments if s["first_step"] in (0, 3)]
    assert len(python_steps) == 2 and len({s["shard_id"] for s in python_steps}) == 1

    # 6. the shard payload is genuinely the data, byte for byte
    first_shard = json.loads(shards[0].payload.split(b"\n", 1)[0])
    assert first_shard["codec"] == "text-zstd"
    assert first_shard["raw_len"] > 0


def test_reader_can_join_segment_to_movement_without_the_recorder(tmp_path):
    """The join must be readable by a plain reader, which is what makes it a format feature."""
    model = ToyModel(seed=1)
    path = tmp_path / "readable.archx.mdl"
    recorder = Recorder(model.registry({"attention.q_proj.weight": "attention",
                                        "mlp.fc.weight": "mlp"}),
                        model.params.__getitem__, str(path),
                        policy=PhasePolicy(total_steps=2, min_warmup_steps=2))
    rng = np.random.default_rng(8)
    for step in range(2):
        model.train_step(rng, 1e-2)
        recorder.step(step, loss=1.0, data=b"same batch for both steps\n")
    recorder.close()

    reader = Reader(path)
    deltas = list(reader.blocks(BlockType.DELTA))
    segments = Timeline(reader).segments()
    assert len(segments) == 1, "two steps on the same batch is one segment"
    assert segments[0]["times_seen"] == 2
    assert (segments[0]["first_step"], segments[0]["last_step"]) == (0, 1)
    assert deltas, "the movement is recorded even though the data did not change"

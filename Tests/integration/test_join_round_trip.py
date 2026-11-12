"""Claim 14: the file alone must answer "which data moved which weights at which step".

The test is written as a reader with no access to the run: it opens the path, decodes what
is there, and checks the three answers against the bytes rather than against the recorder's
own bookkeeping.
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pytest

from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.data.codecs import EncodedShard, content_id, decode_shard
from archx.data.capture import DataLevel
from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy
from archx.recorder import Recorder


class ToyModel:
    def __init__(self, width: int = 16):
        self.params = {
            "attention.q_proj.weight": np.zeros((width, width), dtype=np.float32),
            "mlp.fc.weight": np.zeros((width, 4), dtype=np.float32),
        }

    def registry(self) -> TensorRegistry:
        reg = TensorRegistry()
        regions = {"attention.q_proj.weight": "attention", "mlp.fc.weight": "mlp"}
        for key, array in self.params.items():
            reg.add(key, array, pointer=array.ctypes.data, region=regions[key],
                    component="language_model")
        return reg


def _record(path, *, level=DataLevel.SEGMENTS, steps=8):
    model = ToyModel()
    recorder = Recorder(model.registry(), model.params.__getitem__, str(path),
                        policy=PhasePolicy(total_steps=steps, min_warmup_steps=1,
                                           heartbeat_steps=2),
                        data_level=level, meta={"adapter": "torch"})
    rng = np.random.default_rng(0)
    for step in range(steps):
        for array in model.params.values():
            array += rng.normal(0, 5e-3, array.shape).astype(np.float32)
        recorder.step(step, loss=1.0 - 0.02 * step,
                      data=f"batch {step % 3}: the corpus line\n".encode())
    recorder.close()
    return path


def _latest_segments(reader) -> dict:
    latest = {}
    for block in reader.blocks(BlockType.DATA_SEGMENT):
        statement = json.loads(block.payload)
        latest[statement["segment_id"]] = statement
    return latest


def test_every_verified_shard_decodes_back_to_the_hash_that_names_it(tmp_path):
    reader = Reader(_record(tmp_path / "join.archx.mdl"))

    checked = 0
    for block in reader.blocks(BlockType.DATA_SHARD):
        head, _, body = block.payload.partition(b"\n")
        header = json.loads(head)
        shard = EncodedShard(shard_id=header["shard_id"], codec=header["codec"],
                             payload=body, raw_len=header["raw_len"],
                             content_crc32=header["content_crc32"],
                             reconstruction_error=header.get("reconstruction_error", 0.0))
        raw = decode_shard(shard)
        assert len(raw) == shard.raw_len
        assert zlib.crc32(raw) & 0xFFFFFFFF == shard.content_crc32, "the CRC in the file lies"
        assert content_id(raw) == header["shard_id"], "the id is not the hash of the bytes"
        checked += 1
    assert checked, "a segments recording must hold readback-verifiable shards"


def test_the_join_names_a_step_the_file_actually_holds(tmp_path):
    reader = Reader(_record(tmp_path / "join.archx.mdl"))

    stored_steps = {json.loads(block.payload)["step"]
                    for block in reader.blocks(BlockType.EVENT)}
    segments = _latest_segments(reader)
    assert segments, "the data half exists"

    for segment in segments.values():
        # `first_step` may legitimately be a skipped step: the batch arrived there, and the
        # data side records what the run saw. The promise is that the segment's *end* is
        # inside recorded history - the writer clamps it - and that its effect, if any, names
        # a stored step. An earlier version of this test asserted `first_step` too and failed
        # on a batch observed at step 1 whose effect was measured at step 2.
        assert segment["last_step"] <= max(stored_steps), (
            f"segment {segment['segment_id']} ends at step {segment['last_step']}, beyond "
            f"the last step in the file ({max(stored_steps)})")
        assert segment["first_step"] <= segment["last_step"]
        assert segment["fingerprint"], "the fingerprint is present at every level"

        # The effect's step is where the join actually lands. A batch seen on a skipped step
        # has no movement of its own, so its effect is measured at the store point that
        # followed it - and that step number must be in the file, or the join has nowhere to
        # point. `effect_step` was added for exactly this; `span_steps` could not say *which*.
        if segment["region_effects"]:
            assert segment["effect_step"] in stored_steps, (
                f"segment {segment['segment_id']} attributes movement to step "
                f"{segment['effect_step']}, which the file does not record")
            assert segment["effect_step"] >= segment["first_step"], (
                "an effect cannot be measured before the data arrived")
        else:
            assert segment["effect_step"] == -1, (
                "no movement, so no step may be claimed for it")


def test_the_join_says_what_the_data_did_to_the_weights(tmp_path):
    reader = Reader(_record(tmp_path / "join.archx.mdl"))

    segments = _latest_segments(reader)
    with_effects = [s for s in segments.values() if s["region_effects"]]
    assert with_effects, "no segment carries movement: the join answers nothing"

    for segment in with_effects:
        assert set(segment["region_effects"]) <= {"attention", "mlp"}
        for region, entry in segment["region_effects"].items():
            assert entry["delta_norm"] > 0, f"{region} moved by a non-number"
            assert entry["delta_norm"] == pytest.approx(entry["delta_norm"])  # not NaN
        # A span is shared by its batches and says so, rather than pretending to split it.
        assert segment["span_segments"] >= 1 and segment["span_steps"] >= 1


def test_a_level_that_cannot_show_the_data_says_which_questions_it_cannot_answer(tmp_path):
    """Architecture 6.8: `JOIN_INCOMPLETE` for any level that is not segments/full.

    Documented from the first design and never written, so a `none` file and a `segments`
    file looked equally complete to a reader.
    """
    for level, expected in ((DataLevel.NONE, True), (DataLevel.POINTERS, True),
                            (DataLevel.SEGMENTS, False)):
        reader = Reader(_record(tmp_path / f"{level.value}.archx.mdl", level=level))
        blocks = [json.loads(block.payload) for block in reader.blocks(BlockType.INFO)]
        incomplete = [b for b in blocks if b.get("event") == "JOIN_INCOMPLETE"]

        if not expected:
            assert not incomplete, "a complete join must not be flagged as incomplete"
            continue
        assert incomplete, f"{level.value} writes no JOIN_INCOMPLETE block"
        assert incomplete[0]["data_level"] == level.value
        assert "data_content" in incomplete[0]["cannot_answer"]
        assert incomplete[0]["detail"]


def test_data_logged_after_the_final_step_is_stored_and_flagged(tmp_path):
    """A loop that logs after `optimizer.step()` must not lose its last batch silently."""
    path = tmp_path / "unsynced.archx.mdl"
    model = ToyModel()
    recorder = Recorder(model.registry(), model.params.__getitem__, str(path),
                        policy=PhasePolicy(total_steps=6, min_warmup_steps=1,
                                           heartbeat_steps=2),
                        data_level=DataLevel.SEGMENTS, meta={"adapter": "torch"})
    recorder.step(0, loss=1.0, data=b"batch one\n" * 4)
    recorder.step(1, loss=0.9, data=b"batch two\n" * 4)
    recorder.log(loss=0.8, data=b"batch three, never stepped\n" * 4)   # after the last step
    recorder.close()

    reader = Reader(path)
    segments = _latest_segments(reader)
    assert len(segments) == 3, f"three observations, {len(segments)} in the file"

    infos = [json.loads(block.payload) for block in reader.blocks(BlockType.INFO)]
    unsynced = [b for b in infos if b.get("event") == "unsynced_log"]
    assert unsynced, "the file must say this data was never trained on"
    assert unsynced[0]["segment_id"] in segments

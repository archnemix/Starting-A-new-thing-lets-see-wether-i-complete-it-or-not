"""Unit tests for data capture: dedup, levels, sensitivity, caps, and exact round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from archx.data.capture import DataCapture, DataLevel
from archx.data.codecs import choose_data_codec, decode_shard, encode_shard


def test_the_same_shard_seen_twice_is_stored_once():
    capture = DataCapture(DataLevel.SEGMENTS)
    data = b"def f(x):\n    return x * 2\n" * 20
    first = capture.observe(data, step=0)
    second = capture.observe(data, step=1)
    assert first.segment_id == second.segment_id
    assert second.times_seen == 2, "the same object is recognised without hashing it"

    capture.seal()
    assert len(capture.shards) == 1
    assert capture.bytes_stored == len(capture.shards[first.fingerprint].payload)


def test_observe_does_no_expensive_work_so_a_training_step_never_waits():
    """Hashing a batch costs ~210 us and compressing it ~530 us: neither may run in a step."""
    capture = DataCapture(DataLevel.SEGMENTS)
    batch = b"x" * 131_072
    segment = capture.observe(batch, step=0)

    assert segment.fingerprint == "", "a new segment is not hashed on the training thread"
    assert capture.bytes_stored == 0, "a new segment is not compressed on the training thread"
    assert segment.content is batch, "the bytes are handed over, not copied"

    capture.seal()                       # the writer thread's turn
    assert segment.fingerprint and capture.bytes_stored > 0
    assert segment.content is None, "the bytes live in the shard now, not twice"


def test_an_epoch_over_the_same_data_costs_pointers_not_copies():
    """Two epochs with fresh bytes objects each time: dedup must be by content, not identity."""
    capture = DataCapture(DataLevel.SEGMENTS)
    batches = [f"batch {i}\n".encode() * 50 for i in range(10)]
    for step, batch in enumerate(batches):
        capture.observe(bytearray(batch).decode().encode(), step=step)   # new object, same bytes
    capture.seal()
    after_one_epoch = capture.bytes_stored

    for step in range(10, 20):
        capture.observe(bytearray(batches[step - 10]).decode().encode(), step=step)
    capture.seal()
    assert capture.bytes_stored == after_one_epoch, "the second epoch must add no bytes"
    assert len(capture.shards) == 10


def test_level_none_keeps_nothing_but_still_reports():
    capture = DataCapture(DataLevel.NONE)
    assert capture.observe(b"anything", step=0) is None
    assert capture.bytes_stored == 0


def test_level_pointers_keeps_no_content_but_records_characteristics():
    capture = DataCapture(DataLevel.POINTERS)
    segment = capture.observe(b"hello data", step=0, source="train.jsonl")
    capture.seal()
    assert segment is not None
    assert segment.source == "train.jsonl"
    assert capture.bytes_stored == 0
    assert segment.fingerprint, "a pointer without a hash would point at nothing"
    assert segment.shard_id in capture.pointers


def test_sensitive_shards_keep_hash_and_characteristics_only():
    capture = DataCapture(DataLevel.SEGMENTS)
    segment = capture.observe(b"patient record 12345", step=0, sensitive=True)
    capture.seal()
    assert capture.bytes_stored == 0
    assert segment.shard_id in capture.sensitive
    assert segment.shard_id == segment.fingerprint, "the hash is kept, the content is not"
    assert segment.characteristics["bytes"] == len(b"patient record 12345")
    assert segment.content is None


def test_a_byte_cap_degrades_to_pointers_instead_of_truncating():
    capture = DataCapture(DataLevel.SEGMENTS, max_bytes=200)
    capture.observe(b"a" * 500, step=0)
    capture.seal()
    assert capture.dropped_to_pointers is True
    assert capture.level is DataLevel.POINTERS
    capture.observe(b"b" * 500, step=1)
    capture.seal()
    assert capture.bytes_stored == 0, "nothing may be half-stored"
    assert capture.pointers, "a degraded shard must still be pointed at, not lost"


def test_effects_are_attributed_to_the_right_segment():
    capture = DataCapture(DataLevel.SEGMENTS)
    segment = capture.observe(b"python code", step=0)
    capture.record_effects(segment.segment_id, region="attention", delta_norm=0.5,
                           grad_norm=1.2, loss_delta=-0.01)
    capture.record_effects(segment.segment_id, region="attention", delta_norm=0.25,
                           grad_norm=0.9, loss_delta=-0.02)
    effects = capture.segments[segment.segment_id].region_effects["attention"]
    assert effects["delta_norm"] == pytest.approx(0.75)
    assert effects["grad_norm_max"] == pytest.approx(1.2)
    assert effects["loss_delta"] == pytest.approx(-0.03)


def test_text_and_token_codecs_are_chosen_by_content_and_round_trip_exactly():
    text = b"for i in range(10):\n    print(i)\n" * 100
    tokens = np.random.default_rng(0).integers(0, 127, 20_000, dtype=np.uint32).tobytes()
    assert choose_data_codec(text) == "text-zstd"
    assert choose_data_codec(tokens) == "tokens-varint"

    for data in (text, tokens):
        shard = encode_shard(data)
        assert decode_shard(shard) == data, "data codecs must be lossless"
        assert shard.reconstruction_error == 0.0


def test_token_varint_helps_only_for_genuinely_small_ids():
    """Measured, not assumed: varint wins below 128, loses above it.

    A random id in 0..499 needs two varint bytes, and two bytes of scattered varints compress
    worse than four-byte ids zstd already handles. The codec choice must reflect that.
    """
    small = np.random.default_rng(2).integers(0, 127, 40_000, dtype=np.uint32).tobytes()
    medium = np.random.default_rng(2).integers(0, 50_000, 40_000, dtype=np.uint32).tobytes()

    small_shard = encode_shard(small)
    medium_shard = encode_shard(medium)
    assert small_shard.codec == "tokens-varint"
    assert small_shard.nbytes < len(small) / 3, "one-byte ids should shrink a lot"
    assert len(medium) / medium_shard.nbytes < len(small) / small_shard.nbytes

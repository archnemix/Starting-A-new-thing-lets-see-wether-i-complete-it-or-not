"""int8 and int16 share one encoding id, and that is where a file can become unreadable.

The width of a quantized delta is carried in the block's value dtype, not in its encoding id.
A decoder table keyed by id alone decoded every int16 delta as int8 - half the expected array,
which fails on reshape. These tests pin both the round trip and the budget rule that decides
whether int16 is offered at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.values import Timeline
from archx.encoders import DenseQuant, DenseRaw
from archx.policy import PhasePolicy
from archx.recorder import Recorder
from archx.formats.registry import TensorRegistry


def _delta(seed: int = 0, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1e-3, (size, size)).astype(np.float32)


def test_int16_round_trips_through_its_own_decoder():
    delta = _delta()
    encoded = DenseQuant(bits=16).encode(delta)
    assert encoded is not None and encoded.value_dtype == "i16"
    rebuilt = DenseQuant(bits=16).decode(encoded.body, delta.shape, "i16", delta.size)
    assert rebuilt.shape == delta.shape
    assert np.linalg.norm(rebuilt - delta) / np.linalg.norm(delta) < 1e-4


def test_int16_is_cheaper_than_raw_and_more_accurate_than_int8():
    delta = _delta()
    eight = DenseQuant(bits=8).encode(delta)
    sixteen = DenseQuant(bits=16).encode(delta)
    raw = DenseRaw().encode(delta)
    assert sixteen.nbytes < raw.nbytes, "the point of quantizing is to be smaller"
    assert sixteen.rel_error < eight.rel_error, "twice the width, far less error"


def _round_trip(tmp_path, max_rel_error: float, name: str):
    """Record a run with a stated budget, read it back, and report the encodings used."""
    import struct

    rng = np.random.default_rng(0)
    params = {"w": rng.normal(0, 0.1, (32, 32)).astype(np.float32)}
    registry = TensorRegistry()
    registry.add("w", params["w"], pointer=params["w"].ctypes.data)
    path = tmp_path / name
    recorder = Recorder(registry, params.__getitem__, str(path), max_rel_error=max_rel_error,
                        policy=PhasePolicy(total_steps=6, min_warmup_steps=6))
    for step in range(6):
        params["w"] += rng.normal(0, 1e-3, params["w"].shape).astype(np.float32)
        recorder.step(step, loss=1.0)
    recorder.close()

    reader = Reader(path)
    encodings = set()
    for block in reader.blocks(BlockType.DELTA):
        body = block.payload
        count = struct.unpack("<I", body[8:12])[0]
        offset = 12
        for _ in range(count):
            _index, encoding, dtype_id = struct.unpack("<IBB", body[offset:offset + 6])
            length = struct.unpack("<I", body[offset + 6:offset + 10])[0]
            encodings.add((encoding, dtype_id))
            offset += 10 + length
    return reader, encodings, params["w"]


def test_the_default_budget_stores_deltas_that_rebuild_exactly(tmp_path):
    """At 1e-3 the recorder must not silently become lossy to save bytes."""
    reader, encodings, final = _round_trip(tmp_path, 1e-3, "default.archx.mdl")
    assert (4, 0) in encodings, f"expected raw float32 at the default budget, got {encodings}"
    rebuilt = Timeline(reader).reconstruct(5)
    assert np.array_equal(rebuilt["w"], final), "the default path rebuilt to something else"


def test_an_int16_budget_uses_int16_and_stays_readable(tmp_path):
    """The width that broke the reader: int16 shares encoding id 2 with int8.

    A budget between int16's error (3.4e-5) and int8's (8.9e-3) is the only window where the
    recorder should choose int16, and it is exactly the window a reader keyed by encoding id
    alone could not decode.
    """
    lossy_reader, lossy_encodings, _ = _round_trip(tmp_path, 3e-3, "int16.archx.mdl")
    assert (2, 5) in lossy_encodings, f"expected int16 quantization, got {lossy_encodings}"
    rebuilt = Timeline(lossy_reader).reconstruct(5)["w"]
    assert rebuilt.shape == (32, 32), "an int16 delta decoded as int8 would fail here"
    assert lossy_reader.path.stat().st_size > 0


def test_a_loose_budget_prefers_int8_and_is_still_readable(tmp_path):
    reader, encodings, _ = _round_trip(tmp_path, 1e-2, "int8.archx.mdl")
    assert (2, 4) in encodings, f"expected int8 quantization at 1e-2, got {encodings}"
    assert Timeline(reader).reconstruct(5)["w"].shape == (32, 32)

"""Unit: every shard codec is lossless, and the heuristic picks what actually wins.

The claim being protected is "exactness per data type". A codec that compresses well and loses
one byte is unusable for this product - the file's whole purpose is to answer questions about a
run that happened, so a shard that comes back different is a wrong answer, not a small one.

Two of these tests exist because the first implementation was wrong in ways that looked fine:
the token detector only recognised toy vocabularies, and a dictionary codec was written that
lost to plain zstd at every size once it had to carry its own bytes.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

from archx.data.codecs import (choose_data_codec, content_id, decode_shard, encode_shard,
                               varint_decode, varint_encode)


def cases() -> dict[str, bytes]:
    generator = np.random.default_rng(0)
    return {
        "prose": b"The optimizer stepped and the loss fell. " * 400,
        "tokens-small": generator.integers(0, 128, 2048, dtype=np.uint32).tobytes(),
        "tokens-real": generator.integers(0, 5000, 2048, dtype=np.uint32).tobytes(),
        "pixels": (generator.random((64, 64)) * 255).astype(np.uint8).tobytes(),
        "weights": generator.standard_normal(2048).astype(np.float32).tobytes(),
        "one-byte": b"\x00",
        "empty": b"",
    }


@pytest.mark.parametrize("name", sorted(cases()))
def test_every_codec_round_trips_exactly(name):
    data = cases()[name]
    shard = encode_shard(data)
    assert decode_shard(shard) == data, f"{name} did not survive the round trip"
    assert shard.reconstruction_error == 0.0, "a lossless codec must report zero error"


@pytest.mark.parametrize("name", sorted(cases()))
def test_the_content_crc_is_of_the_raw_bytes(name):
    """The CRC is what lets a reader prove identity without decompressing twice."""
    data = cases()[name]
    shard = encode_shard(data)
    assert shard.content_crc32 == zlib.crc32(data) & 0xFFFFFFFF
    assert shard.raw_len == len(data)
    assert shard.shard_id == content_id(data)


def test_the_same_bytes_always_get_the_same_id():
    data = cases()["prose"]
    assert encode_shard(data).shard_id == encode_shard(data + b"") .shard_id


def test_realistic_token_ids_take_the_varint_path():
    """Measured: zstd alone 0.560 of raw, varint+zstd 0.456 - a 1.23x saving.

    The first detector demanded every id be under 128, which no real vocabulary satisfies, so
    this path was never chosen for the case it was written for.
    """
    data = cases()["tokens-real"]
    chosen = choose_data_codec(data)
    assert chosen == "tokens-varint", f"realistic token ids fell through to {chosen}"

    stored = len(encode_shard(data).payload)
    import zstandard as zstd

    plain = len(zstd.ZstdCompressor(level=3).compress(data))
    assert stored < plain, (
        f"the varint path stored {stored} bytes against {plain} for plain zstd: "
        f"it must only be chosen where it wins")


def test_weights_and_pixels_are_not_mistaken_for_tokens():
    """Float weights and pixel bytes fill their full 32-bit width; ids do not."""
    for name in ("pixels", "weights"):
        assert choose_data_codec(cases()[name]) == "zstd", name


def test_pixels_are_reported_honestly_rather_than_faked():
    """Random pixel bytes do not compress, and the codec must not claim otherwise."""
    row = encode_shard(cases()["pixels"])
    assert len(row.payload) / row.raw_len > 0.95, (
        "incompressible content must not appear to compress")


def test_varint_round_trips_at_every_boundary():
    """126/127/128 is where a hand-written LEB128 breaks."""
    values = np.array([0, 1, 126, 127, 128, 129, 16383, 16384, 2**21 - 1, 2**21],
                      dtype=np.uint64)
    assert np.array_equal(varint_decode(varint_encode(values), values.size), values)


def test_an_explicit_codec_overrides_the_heuristic():
    data = cases()["tokens-real"]
    shard = encode_shard(data, codec="zstd")
    assert shard.codec == "zstd"
    assert decode_shard(shard) == data


def test_the_dictionary_codec_is_gone_rather_than_unused():
    """A codec that lost at every measured size was removed, not shipped dead.

    Keeping it would mean a `text-dict` branch that no heuristic ever selects, and a reader
    wondering why the docs mention a codec the encoder never produces.
    """
    import archx.data.codecs as codecs

    assert not hasattr(codecs, "train_dictionary")
    assert "text-dict" not in {choose_data_codec(data) for data in cases().values()}

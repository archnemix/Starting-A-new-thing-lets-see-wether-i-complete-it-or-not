"""Unit tests for the byte layout constants and header/footer construction."""

from __future__ import annotations

import struct

from archx.container.schema import (
    BLOCK_SIZE,
    FOOTER_SIZE,
    HEADER_SIZE,
    footer_bytes,
    header_bytes,
    zlib_crc32,
)
from archx.version import FORMAT_MAJOR, FORMAT_MINOR, MAGIC, END_MAGIC


def test_sizes_match_the_spec():
    assert (HEADER_SIZE, BLOCK_SIZE, FOOTER_SIZE) == (40, 28, 56)


def test_header_carries_magic_version_and_valid_crc():
    data = header_bytes(flags=3, created_us=1234, block_hint=7)
    magic, major, minor, flags, length, crc, created, hint, reserved = struct.unpack(
        "<8sHHII I Q I I", data
    )
    assert magic == MAGIC
    assert (major, minor) == (FORMAT_MAJOR, FORMAT_MINOR)
    assert (flags, length, created, hint, reserved) == (3, 40, 1234, 7, 0)
    assert crc == zlib_crc32(data[:20])


def test_footer_ends_with_its_magic_and_a_valid_crc():
    data = footer_bytes(index_offset=100, index_len=20, block_count=5, step_count=9, anchor_count=2)
    end_magic, ioff, ilen, blocks, steps, anchors, crc, reserved = struct.unpack(
        "<8s Q Q Q Q Q I I", data
    )
    assert end_magic == END_MAGIC
    assert (ioff, ilen, blocks, steps, anchors, reserved) == (100, 20, 5, 9, 2, 0)
    assert crc == zlib_crc32(data[:48])

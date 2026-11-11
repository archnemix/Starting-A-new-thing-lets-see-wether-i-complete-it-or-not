"""Unit tests for the reader: hostile and damaged files are rejected, not trusted."""

from __future__ import annotations

import struct

import pytest

from archx.container.reader import Reader
from archx.container.schema import BLOCK, HEADER_SIZE, BlockType, zlib_crc32
from archx.container.writer import Writer
from archx.errors import CorruptBlockError, FormatError, TruncatedFileError, UnsupportedVersionError


def _sample(tmp_path, name="run.archx.mdl"):
    path = tmp_path / name
    with Writer(path) as w:
        w.write_block(BlockType.META, b'{"a": 1}')
    return path


def test_rejects_a_foreign_file(tmp_path):
    path = tmp_path / "not-ours.bin"
    path.write_bytes(b"0" * 100)
    with pytest.raises(FormatError):
        Reader(path)


def test_rejects_a_newer_major_version(tmp_path):
    path = _sample(tmp_path)
    data = bytearray(path.read_bytes())
    data[8:10] = struct.pack("<H", 2)  # bump major
    data[20:24] = struct.pack("<I", zlib_crc32(bytes(data[:20])))
    path.write_bytes(bytes(data))
    with pytest.raises(UnsupportedVersionError):
        Reader(path)


def test_detects_a_bit_flip_in_a_block_payload(tmp_path):
    path = _sample(tmp_path)
    data = bytearray(path.read_bytes())
    data[HEADER_SIZE + 24] ^= 0xFF  # first payload byte
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptBlockError):
        Reader(path)


def test_refuses_an_absurd_claimed_length_before_allocating(tmp_path):
    path = _sample(tmp_path)
    data = bytearray(path.read_bytes())
    head = BLOCK.pack(int(BlockType.META), 0, 0, 0, 3 * 1024**3, 3 * 1024**3, 0, 0)
    data[HEADER_SIZE:HEADER_SIZE + 24] = head[:24]
    data[HEADER_SIZE + 24:HEADER_SIZE + 28] = struct.pack(
        "<I", zlib_crc32(bytes(data[HEADER_SIZE:HEADER_SIZE + 24]))
    )
    path.write_bytes(bytes(data))
    with pytest.raises(CorruptBlockError):
        Reader(path)


def test_truncated_tail_is_reported_as_torn_when_allowed(tmp_path):
    path = _sample(tmp_path)
    data = path.read_bytes()
    path.write_bytes(data[:-10])
    with pytest.raises(TruncatedFileError):
        Reader(path)
    reader = Reader(path, allow_torn=True)
    assert reader.container.torn is True


def test_unknown_block_type_is_skipped_not_rejected(tmp_path):
    path = tmp_path / "future.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, b'{"a": 2}')
        w.write_block(999, b"from a newer version")
    reader = Reader(path)
    assert reader.meta() == {"a": 2}
    assert any(b.block_type == 999 for b in reader.blocks())

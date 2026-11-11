"""Byte-level layout of the .archx.mdl container, as constants and struct formats.

This module is the single place where field offsets live. Plan/Format_Spec.md is the
normative description; if the two disagree, the spec is wrong and this file gets fixed
in the same commit.
"""

from __future__ import annotations

import struct
import zlib
from enum import IntEnum

from ..version import FORMAT_MAJOR, FORMAT_MINOR, MAGIC, END_MAGIC

HEADER = struct.Struct("<8sHHII I Q I I")
# magic, format_major, format_minor, flags, header_len, header_crc, created_us, block_hint, reserved
HEADER_SIZE = HEADER.size  # 40

#: Step carried by a block that does not belong to a training step.
STEP_NONE = 0xFFFFFFFFFFFFFFFF

BLOCK = struct.Struct("<HBB Q I I I I")
# block_type, codec, flags, step, payload_raw_len, payload_len, payload_crc, block_crc
# block_crc covers the first 24 bytes, i.e. everything before block_crc itself
BLOCK_CRC_SPAN = 24
BLOCK_SIZE = BLOCK.size  # 28

FOOTER = struct.Struct("<8s Q Q Q Q Q I I")
# end_magic, index_offset, index_len, block_count, step_count, anchor_count, footer_crc, reserved
FOOTER_SIZE = FOOTER.size  # 56

#: Refuse to allocate for a single block larger than this unless the reader is told to.
DEFAULT_MAX_BLOCK_BYTES = 2 * 1024**3  # 2 GiB


class BlockType(IntEnum):
    """Block ids. Existing ids never change meaning; new ones get new numbers."""

    META = 1
    TENSOR_REGISTRY = 2
    ANCHOR = 3
    DELTA = 4
    EVENT = 5
    DATA_SEGMENT = 17
    DATA_SHARD = 20
    DATA_POINTER = 21
    INFO = 8
    INDEX = 9
    ATLAS = 18
    EDIT_VECTOR = 19
    END = 22


class Codec(IntEnum):
    """Per-block compression ids."""

    NONE = 0
    ZSTD = 1
    ZLIB = 2
    LZ4 = 3


class Flags:
    """Header and block flag bits."""

    CLOSED = 1 << 0
    PARAMETERS_COMPLETE = 1 << 1
    RECOVERED = 1 << 2
    SHARDED_SOURCE = 1 << 3


def header_bytes(*, flags: int = 0, created_us: int = 0, block_hint: int = 0) -> bytes:
    """Build a header. CRC covers the first 20 bytes, i.e. everything before created_us."""
    packed_head = struct.pack("<8sHHII", MAGIC, FORMAT_MAJOR, FORMAT_MINOR, flags, HEADER_SIZE)
    crc = zlib_crc32(packed_head)
    return HEADER.pack(MAGIC, FORMAT_MAJOR, FORMAT_MINOR, flags, HEADER_SIZE, crc,
                       created_us, block_hint, 0)


def footer_bytes(*, index_offset: int, index_len: int, block_count: int,
                 step_count: int, anchor_count: int) -> bytes:
    """Build a footer. CRC covers the first 48 bytes."""
    body = struct.pack("<8s Q Q Q Q Q", END_MAGIC, index_offset, index_len, block_count,
                       step_count, anchor_count)
    return FOOTER.pack(END_MAGIC, index_offset, index_len, block_count, step_count,
                       anchor_count, zlib_crc32(body), 0)


def zlib_crc32(data: bytes) -> int:
    """CRC-32 (ISO-HDLC), the same polynomial zlib uses."""
    return zlib.crc32(data) & 0xFFFFFFFF

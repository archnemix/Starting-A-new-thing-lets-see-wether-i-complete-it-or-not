"""Rebuild a crashed file: keep every whole block, drop the torn tail, write a fresh footer.

A crash costs the tail, never the history. This module is the only code allowed to modify
an existing container, and it only ever truncates and re-appends.
"""

from __future__ import annotations

import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from ..errors import CorruptBlockError, FormatError
from .reader import Reader, read_header
from .schema import (
    BLOCK,
    BLOCK_CRC_SPAN,
    BLOCK_SIZE,
    STEP_NONE,
    HEADER_SIZE,
    BlockType,
    Flags,
    footer_bytes,
    header_bytes,
    zlib_crc32,
)


@dataclass
class RecoveryReport:
    """What recovery found and did, so it can be written into the file and logged."""

    path: str
    blocks_kept: int
    bytes_truncated: int
    had_footer: bool
    recovered: bool


def _good_prefix_end(data: bytes, max_block_bytes: int) -> tuple[int, int]:
    """Return (end offset of the last whole block, blocks kept)."""
    offset = HEADER_SIZE
    kept = 0
    total = len(data)
    while offset + BLOCK_SIZE <= total:
        _type, _codec, _flags, _step, raw_len, payload_len, _pcrc, block_crc = BLOCK.unpack(
            data[offset:offset + BLOCK_SIZE]
        )
        if zlib_crc32(data[offset:offset + BLOCK_CRC_SPAN]) != block_crc:
            break
        if payload_len > max_block_bytes or raw_len > max_block_bytes:
            break
        end = offset + BLOCK_SIZE + payload_len
        if end > total:
            break
        offset = end
        kept += 1
    return offset, kept


def recover_in_place(path: str | os.PathLike[str], *, max_block_bytes: int = 2 * 1024**3) -> RecoveryReport:
    """Truncate a crashed file at its last whole block and append a new footer."""
    p = Path(path)
    data = p.read_bytes()
    read_header(data)  # raises if this is not an archx file at all

    end, kept = _good_prefix_end(data, max_block_bytes)
    truncated = len(data) - end
    had_footer = False

    with open(p, "r+b") as fh:
        fh.truncate(end)
        # block count is not tracked separately: the index is rebuilt from what survived.
        index_payload = _rebuild_index(data[:end])
        fh.seek(end)
        index_offset = fh.tell()
        stored = index_payload
        prefix = _block_prefix(BlockType.INDEX, len(stored), zlib_crc32(stored))
        fh.write(prefix)
        fh.write(stored)
        index_len = len(prefix) + len(stored)
        fh.write(footer_bytes(
            index_offset=index_offset,
            index_len=index_len,
            block_count=kept + 1,
            step_count=0,
            anchor_count=0,
        ))
        fh.flush()
        os.fsync(fh.fileno())

    _set_recovered_flag(p)
    return RecoveryReport(str(p), blocks_kept=kept, bytes_truncated=truncated,
                          had_footer=had_footer, recovered=True)


def recover_copy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> RecoveryReport:
    """Recover into a new file and leave the original untouched."""
    shutil.copyfile(src, dst)
    return recover_in_place(dst)


def _block_prefix(block_type: BlockType, payload_len: int, payload_crc: int) -> bytes:
    head = BLOCK.pack(int(block_type), 0, 0, STEP_NONE, payload_len, payload_len, payload_crc, 0)
    return head[:BLOCK_CRC_SPAN] + struct.pack("<I", zlib_crc32(head[:BLOCK_CRC_SPAN]))


def _rebuild_index(data: bytes) -> bytes:
    """Walk the surviving blocks and rebuild (step, offset, length) entries.

    Step numbers come from DELTA/ANCHOR bodies; anything unreadable is skipped rather than
    guessed, because a wrong offset in the index is worse than a missing entry.
    """
    entries: list[tuple[int, int, int]] = []
    offset = HEADER_SIZE
    total = len(data)
    while offset + BLOCK_SIZE <= total:
        block_type, _codec, _flags, step, _raw_len, payload_len, _pcrc, _bcrc = BLOCK.unpack(
            data[offset:offset + BLOCK_SIZE]
        )
        end = offset + BLOCK_SIZE + payload_len
        if end > total:
            break
        if block_type != int(BlockType.INDEX) and step != STEP_NONE:
            entries.append((step, offset, BLOCK_SIZE + payload_len))
        offset = end

    parts = [struct.pack("<Q", len(entries))]
    for step, start, length in entries:
        parts.append(struct.pack("<QQIB", step, start, length, 1))
    return b"".join(parts)


def _set_recovered_flag(path: Path) -> None:
    with open(path, "r+b") as fh:
        head = fh.read(HEADER_SIZE)
        flags = (int.from_bytes(head[12:16], "little") | Flags.RECOVERED | Flags.CLOSED)
        fh.seek(0)
        fh.write(header_bytes(
            flags=flags,
            created_us=int.from_bytes(head[24:32], "little"),
            block_hint=int.from_bytes(head[32:36], "little"),
        ))

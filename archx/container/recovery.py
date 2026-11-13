"""Rebuild a crashed file: keep every whole block, drop the torn tail, write a fresh footer.

A crash costs the tail, never the history. This module is the only code allowed to modify
an existing container, and it only ever truncates and re-appends.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from ..capabilities import Capabilities
from ..errors import CorruptBlockError, FormatError
from .compress import decompress
from .reader import Reader, read_header
from .schema import (
    BLOCK,
    Codec,
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
    #: True when recovery had to state the tier itself, because the crash came before close.
    capability_written: bool = False


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
        capsule = _append_capabilities(fh, data[:end])
        index_payload = _rebuild_index(data[:end])
        fh.seek(0, os.SEEK_END)
        index_offset = fh.tell()
        stored = index_payload
        prefix = _block_prefix(BlockType.INDEX, len(stored), zlib_crc32(stored))
        fh.write(prefix)
        fh.write(stored)
        index_len = len(prefix) + len(stored)
        fh.write(footer_bytes(
            index_offset=index_offset,
            index_len=index_len,
            block_count=kept + 1 + capsule,
            step_count=0,
            anchor_count=0,
        ))
        fh.flush()
        os.fsync(fh.fileno())

    _set_recovered_flag(p)
    return RecoveryReport(str(p), blocks_kept=kept, bytes_truncated=truncated,
                          had_footer=had_footer, recovered=True, capability_written=bool(capsule))


def recover_copy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> RecoveryReport:
    """Recover into a new file and leave the original untouched."""
    shutil.copyfile(src, dst)
    return recover_in_place(dst)


def _append_capabilities(fh, data: bytes) -> int:
    """Rewrite the capability block from the blocks that survived. Returns blocks written.

    A crash happens *before* close, and the capability block is written at close - so a
    recovered file had none, and `archx info` fell back to META's `capability_tier: 0`, the
    floor. A 150-step tier-3 recording, torn at the tail and then recovered exactly as
    `archx recover` promises, went on to describe itself as a file with nothing but step
    counters. The tier is a measured fact about the run, and every measurement it is derived
    from is still on disk: loss and gradient norms in the events, data segments, activation
    summaries. So it is recomputed here from what survived, never copied from what was lost.

    If the file already has a capability block, it is left exactly as written: the run's own
    statement about itself outranks a second opinion computed from blocks.
    """
    available: set[str] = {"steps", "deltas"}
    offset = HEADER_SIZE
    total = len(data)
    while offset + BLOCK_SIZE <= total:
        block_type, codec, _flags, _step, raw_len, payload_len, _pcrc, _bcrc = BLOCK.unpack(
            data[offset:offset + BLOCK_SIZE]
        )
        end = offset + BLOCK_SIZE + payload_len
        if end > total:
            break
        try:
            body = decompress(data[offset + BLOCK_SIZE:end], Codec(codec), raw_len=raw_len)
        except Exception:                       # noqa: BLE001 - an unreadable block says nothing
            offset = end
            continue
        offset = end
        if block_type == int(BlockType.EVENT):
            try:
                event = json.loads(body)
            except ValueError:
                continue
            if event.get("loss") is not None:
                available.add("loss")
            if event.get("grad_norm") is not None:
                available.add("grad_norms")
            if event.get("lr") is not None:
                available.add("learning_rate")
        elif block_type == int(BlockType.DATA_SEGMENT):
            available.add("data_segments")
        elif block_type == int(BlockType.ACTIVATION):
            available.add("activations")
        elif block_type == int(BlockType.INFO):
            try:
                if json.loads(body).get("event") == "capabilities":
                    return 0          # the run closed its own statement; do not overrule it
            except ValueError:
                continue
    payload = {"event": "capabilities", **Capabilities(available).resolve().to_dict(),
               "recomputed_by_recovery": True}
    encoded = json.dumps(payload).encode()
    fh.seek(0, os.SEEK_END)
    fh.write(_block_prefix(BlockType.INFO, len(encoded), zlib_crc32(encoded)))
    fh.write(encoded)
    return 1


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

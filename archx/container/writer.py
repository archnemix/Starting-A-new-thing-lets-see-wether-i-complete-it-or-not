"""Append-only block writer.

Every block is written once and never moved, so a killed process leaves a readable file:
only the tail can be lost. The footer is the last thing appended and is the only proof
that the file was closed cleanly.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import BinaryIO, Iterable

from . import compress as _compress
from .schema import (
    BLOCK,
    BLOCK_CRC_SPAN,
    BLOCK_SIZE,
    STEP_NONE,
    HEADER_SIZE,
    BlockType,
    Codec,
    Flags,
    footer_bytes,
    header_bytes,
    zlib_crc32,
)


class Writer:
    """Writes one .archx.mdl file.

    Not thread safe by design: the recorder owns one writer on one thread.
    `fsync_every_bytes` / `fsync_every_seconds` trade durability against speed.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        codec: Codec = Codec.ZSTD,
        fsync_every_bytes: int = 8 * 1024 * 1024,
        fsync_every_seconds: float = 1.0,
        parameters_complete: bool = True,
    ) -> None:
        self.path = Path(path)
        self.codec = _compress.effective_codec(codec)
        self.fsync_every_bytes = fsync_every_bytes
        self.fsync_every_seconds = fsync_every_seconds
        self.block_count = 0
        self.step_count = 0
        self.anchor_count = 0
        self._closed = False
        self._dirty_bytes = 0
        self._last_fsync = time.monotonic()
        self._index: list[tuple[int, int, int]] = []  # (step, offset, length)

        flags = Flags.PARAMETERS_COMPLETE if parameters_complete else 0
        self._fh: BinaryIO = open(self.path, "wb")
        self._fh.write(header_bytes(flags=flags, created_us=int(time.time() * 1_000_000)))
        self._fsync()

    # ---------------------------------------------------------------- blocks

    def write_block(self, block_type: BlockType, payload: bytes, *, step: int | None = None) -> int:
        """Append one block and return its offset. `step` is stored in the prefix.

        The step lives in the prefix, not only in the body, so recovery can rebuild the
        index without decompressing payloads.
        """
        if self._closed:
            raise ValueError("writer is closed")
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")

        raw = bytes(payload)
        stored = _compress.compress(raw, self.codec)
        return self._write_encoded(block_type, raw, stored, self.codec, step=step)

    def write_best_block(self, candidates, *, step: int | None = None) -> int:
        """Write the smallest actual compressed candidate (or raw), recording its type/codec.

        Used for deep traces only; every candidate must represent the same information.
        Compression occurs on the writer thread, with each block independently recoverable.
        """
        if self._closed:
            raise ValueError("writer is closed")
        best = None
        for block_type, raw in candidates:
            raw = bytes(raw)
            stored = _compress.compress(raw, self.codec)
            option = (block_type, raw, stored, self.codec)
            if len(raw) < len(stored):
                option = (block_type, raw, raw, Codec.NONE)
            if best is None or len(option[2]) < len(best[2]):
                best = option
        if best is None:
            raise ValueError("no block candidates")
        return self._write_encoded(*best, step=step)

    def _write_encoded(self, block_type, raw, stored, codec, *, step):
        payload_crc = zlib_crc32(stored)
        head = BLOCK.pack(
            int(block_type),
            int(codec),
            0,
            STEP_NONE if step is None else step,
            len(raw),
            len(stored),
            payload_crc,
            0,
        )
        prefix = head[:BLOCK_CRC_SPAN] + struct.pack("<I", zlib_crc32(head[:BLOCK_CRC_SPAN]))
        assert len(prefix) == BLOCK_SIZE

        offset = self._fh.tell()
        self._fh.write(prefix)
        self._fh.write(stored)
        self.block_count += 1
        self._dirty_bytes += BLOCK_SIZE + len(stored)
        if step is not None:
            self._index.append((step, offset, BLOCK_SIZE + len(stored)))
        self._maybe_fsync()
        return offset

    def write_index(self) -> int:
        """Append the INDEX block; its offset and length are recorded in the footer."""
        parts = [struct.pack("<Q", len(self._index))]
        for step, offset, length in self._index:
            parts.append(struct.pack("<QQIB", step, offset, length, 1))
        return self.write_block(BlockType.INDEX, b"".join(parts))

    # ---------------------------------------------------------------- closing

    def close(self, *, recovered: bool = False) -> None:
        """Write the footer and close. Idempotent."""
        if self._closed:
            return
        index_offset = self.write_index()
        index_len = self._fh.tell() - index_offset
        footer = footer_bytes(
            index_offset=index_offset,
            index_len=index_len,
            block_count=self.block_count,
            step_count=self.step_count,
            anchor_count=self.anchor_count,
        )
        self._fh.write(footer)
        self._fsync()
        self._closed = True
        self._fh.close()
        # A footer is the proof of a clean close; the flag lets a reader know without
        # scanning to the end of a large file.
        self._set_flag(Flags.CLOSED)
        if recovered:
            self._set_flag(Flags.RECOVERED)

    def abort(self) -> None:
        """Close without a footer: the file stays a crashed-but-readable container."""
        if not self._closed:
            self._closed = True
            self._fh.close()

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    # ---------------------------------------------------------------- helpers

    def _maybe_fsync(self) -> None:
        now = time.monotonic()
        if (self._dirty_bytes >= self.fsync_every_bytes
                or now - self._last_fsync >= self.fsync_every_seconds):
            self._fsync()

    def _fsync(self) -> None:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._dirty_bytes = 0
        self._last_fsync = time.monotonic()

    def mark_recovered(self) -> None:
        """Set the RECOVERED flag in the header without disturbing block bytes."""
        self._set_flag(Flags.RECOVERED)

    def _set_flag(self, bit: int) -> None:
        """OR one flag bit into the header in place."""
        with open(self.path, "r+b") as fh:
            head = fh.read(HEADER_SIZE)
            flags = int.from_bytes(head[12:16], "little") | bit
            fh.seek(0)
            fh.write(header_bytes(
                flags=flags,
                created_us=int.from_bytes(head[24:32], "little"),
                block_hint=int.from_bytes(head[32:36], "little"),
            ))


def iter_blocks(data: bytes) -> Iterable[tuple[int, bytes]]:
    """Yield (block_type, stored_payload) from raw file bytes. Testing helper."""
    offset = HEADER_SIZE
    while offset + BLOCK_SIZE <= len(data):
        block_type, codec, _flags, raw_len, payload_len, payload_crc, _ = BLOCK.unpack(
            data[offset:offset + BLOCK_SIZE]
        )
        start = offset + BLOCK_SIZE
        yield block_type, data[start:start + payload_len]
        offset = start + payload_len
        _ = (codec, raw_len, payload_crc)

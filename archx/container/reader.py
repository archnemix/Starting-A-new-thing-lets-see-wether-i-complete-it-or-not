"""Reader for .archx.mdl: scan blocks, validate everything before allocating, expose an index.

Nothing here trusts the file. Every length is bounded, every CRC is checked, and an
unknown block type is skipped rather than treated as an error, per the format spec.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from ..errors import CorruptBlockError, FormatError, TruncatedFileError, UnsupportedVersionError
from ..version import FORMAT_MAJOR, MAGIC, END_MAGIC
from . import compress as _compress
from .schema import (
    BLOCK,
    BLOCK_CRC_SPAN,
    BLOCK_SIZE,
    STEP_NONE,
    DEFAULT_MAX_BLOCK_BYTES,
    FOOTER,
    FOOTER_SIZE,
    HEADER,
    HEADER_SIZE,
    BlockType,
    Codec,
    Flags,
    zlib_crc32,
)


@dataclass(frozen=True)
class Header:
    """Decoded header fields."""

    format_major: int
    format_minor: int
    flags: int
    created_us: int
    block_hint: int

    @property
    def closed(self) -> bool:
        return bool(self.flags & Flags.CLOSED)

    @property
    def parameters_complete(self) -> bool:
        return bool(self.flags & Flags.PARAMETERS_COMPLETE)

    @property
    def recovered(self) -> bool:
        return bool(self.flags & Flags.RECOVERED)


@dataclass(frozen=True)
class Block:
    """A decoded block, positioned in the file."""

    block_type: int
    codec: int
    flags: int
    step: int | None
    offset: int
    stored_len: int
    payload: bytes


@dataclass
class Container:
    """Everything a reader got out of a file."""

    header: Header
    blocks: list[Block] = field(default_factory=list)
    footer: tuple[int, int] | None = None
    torn: bool = False


def read_header(data: bytes) -> Header:
    """Parse and validate the header. Raises on anything that is not our file."""
    if len(data) < HEADER_SIZE:
        raise TruncatedFileError(f"file is {len(data)} bytes, header needs {HEADER_SIZE}")
    magic, major, minor, flags, header_len, crc, created_us, hint, _reserved = HEADER.unpack(
        data[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise FormatError(f"bad magic {magic!r}, expected {MAGIC!r}")
    if zlib_crc32(data[:20]) != crc:
        raise CorruptBlockError("header CRC mismatch")
    if header_len != HEADER_SIZE:
        raise UnsupportedVersionError(f"header_len={header_len}, this build writes {HEADER_SIZE}")
    if major != FORMAT_MAJOR:
        raise UnsupportedVersionError(
            f"format {major}.{minor} is not readable by this build (needs {FORMAT_MAJOR}.x)"
        )
    return Header(major, minor, flags, created_us, hint)


class Reader:
    """Sequential reader over a whole file, held in memory.

    v1 files are small enough to map whole; a streaming path comes with the recorder.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_block_bytes: int = DEFAULT_MAX_BLOCK_BYTES,
        allow_torn: bool = False,
    ) -> None:
        self.path = Path(path)
        self.max_block_bytes = max_block_bytes
        self._data = self.path.read_bytes()
        self.header = read_header(self._data)
        self.container = self._scan(allow_torn=allow_torn)

    # ------------------------------------------------------------------ scan

    def _scan(self, *, allow_torn: bool) -> Container:
        container = Container(header=self.header)
        offset = HEADER_SIZE
        total = len(self._data)
        while offset < total:
            # The footer check comes first: its magic shares the space a block prefix
            # would occupy, so parsing order decides whether a closed file reads cleanly.
            if self._data[offset:offset + 8] == END_MAGIC:
                if total - offset < FOOTER_SIZE:
                    if allow_torn:
                        container.torn = True
                        return container
                    raise TruncatedFileError(f"footer at {offset} is incomplete")
                container.footer = self._read_footer(offset)
                break
            if total - offset < BLOCK_SIZE:
                if allow_torn:
                    container.torn = True
                    return container
                raise TruncatedFileError(f"{total - offset} stray bytes at offset {offset}")

            fields = BLOCK.unpack(self._data[offset:offset + BLOCK_SIZE])
            block_type, codec, flags, step, raw_len, payload_len, payload_crc, block_crc = fields
            if zlib_crc32(self._data[offset:offset + BLOCK_CRC_SPAN]) != block_crc:
                if allow_torn:
                    container.torn = True
                    return container
                raise CorruptBlockError(f"block prefix CRC mismatch at offset {offset}")
            if payload_len > self.max_block_bytes or raw_len > self.max_block_bytes:
                raise CorruptBlockError(
                    f"block at {offset} claims {payload_len} bytes (cap {self.max_block_bytes})"
                )
            start = offset + BLOCK_SIZE
            end = start + payload_len
            if end > total:
                if allow_torn:
                    container.torn = True
                    return container
                raise TruncatedFileError(f"block at {offset} needs {payload_len} bytes, file ends early")
            stored = self._data[start:end]
            if zlib_crc32(stored) != payload_crc:
                raise CorruptBlockError(f"payload CRC mismatch at offset {offset}")
            try:
                payload = _compress.decompress(stored, Codec(codec), raw_len=raw_len)
            except ValueError as exc:
                raise CorruptBlockError(f"unknown codec {codec} at offset {offset}") from exc
            except Exception as exc:  # noqa: BLE001 - any decoder failure is corruption here
                raise CorruptBlockError(f"payload failed to decompress at {offset}: {exc}") from exc
            if len(payload) != raw_len:
                raise CorruptBlockError(
                    f"block at {offset} declares {raw_len} bytes, decoded {len(payload)}"
                )
            container.blocks.append(Block(block_type, codec, flags,
                                          None if step == STEP_NONE else step,
                                          offset, payload_len, payload))
            offset = end
        return container

    def _read_footer(self, offset: int) -> tuple[int, int]:
        chunk = self._data[offset:offset + FOOTER_SIZE]
        if len(chunk) != FOOTER_SIZE:
            raise TruncatedFileError("footer is incomplete")
        end_magic, index_offset, index_len, _blocks, _steps, _anchors, crc, _ = FOOTER.unpack(chunk)
        if end_magic != END_MAGIC or zlib_crc32(chunk[:48]) != crc:
            raise CorruptBlockError("footer CRC mismatch")
        return index_offset, index_len

    # ------------------------------------------------------------------ access

    def blocks(self, block_type: BlockType | None = None) -> Iterator[Block]:
        """Iterate blocks, optionally filtered by type."""
        for block in self.container.blocks:
            if block_type is None or block.block_type == int(block_type):
                yield block

    def meta(self) -> dict:
        """Decoded META block, or an empty dict when the file has none."""
        for block in self.blocks(BlockType.META):
            return json.loads(block.payload.decode("utf-8"))
        return {}

    def info(self) -> dict:
        """Small dict suitable for `archx info --json`."""
        return {
            "path": str(self.path),
            "size_bytes": len(self._data),
            "format": f"{self.header.format_major}.{self.header.format_minor}",
            "closed": self.container.footer is not None,
            "recovered": self.header.recovered,
            "parameters_complete": self.header.parameters_complete,
            "blocks": len(self.container.blocks),
            "block_types": sorted({int(b.block_type) for b in self.container.blocks}),
            "meta": self.meta(),
        }

    @staticmethod
    def index_block_payload(container: Container) -> list[tuple[int, int, int]]:
        """Decode the INDEX block into (step, offset, length) tuples."""
        for block in container.blocks:
            if block.block_type != int(BlockType.INDEX):
                continue
            count = struct.unpack("<Q", block.payload[:8])[0]
            entries = []
            pos = 8
            for _ in range(count):
                step, offset, length, _kind = struct.unpack("<QQIB", block.payload[pos:pos + 21])
                entries.append((step, offset, length))
                pos += 21
            return entries
        return []

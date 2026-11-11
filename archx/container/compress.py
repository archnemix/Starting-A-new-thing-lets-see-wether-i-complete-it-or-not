"""Compression codecs for blocks, chosen per block and recorded in the block header.

zstd comes from `zstandard` when installed, else CPython 3.14's `compression.zstd`.
Neither present is not an error: NONE is always valid.
"""

from __future__ import annotations

import zlib

from .schema import Codec

try:  # pragma: no cover - exercised by whichever branch exists on the runner
    import zstandard as _zstd
except ImportError:  # pragma: no cover
    _zstd = None

try:  # pragma: no cover
    from compression import zstd as _stdlib_zstd
except ImportError:  # pragma: no cover
    _stdlib_zstd = None


def _zstd_compress(data: bytes, level: int) -> bytes | None:
    if _zstd is not None:
        return _zstd.ZstdCompressor(level=level).compress(data)
    if _stdlib_zstd is not None:
        return _stdlib_zstd.compress(data, level)
    return None


def _zstd_decompress(data: bytes, raw_len: int | None) -> bytes:
    if _zstd is not None:
        return _zstd.ZstdDecompressor().decompress(data, max_output_size=raw_len or 0)
    if _stdlib_zstd is not None:
        return _stdlib_zstd.decompress(data)
    raise RuntimeError("zstd block found but no zstd implementation is installed")


def zstd_available() -> bool:
    """True when some zstd implementation can be used."""
    return _zstd is not None or _stdlib_zstd is not None


def compress(data: bytes, codec: Codec, *, level: int = 3) -> bytes:
    """Compress, falling back to a smaller codec when the requested one is unavailable."""
    if codec is Codec.NONE:
        return data
    if codec is Codec.ZSTD:
        out = _zstd_compress(data, level)
        return zlib.compress(data, 6) if out is None else out
    if codec is Codec.ZLIB:
        return zlib.compress(data, 6)
    raise ValueError(f"codec {codec!r} is not supported")


def decompress(data: bytes, codec: Codec, *, raw_len: int | None = None) -> bytes:
    """Reverse `compress`. `raw_len` bounds the allocation for zstd."""
    if codec is Codec.NONE:
        return data
    if codec is Codec.ZSTD:
        return _zstd_decompress(data, raw_len)
    if codec is Codec.ZLIB:
        return zlib.decompress(data)
    raise ValueError(f"codec {codec!r} is not supported")


def effective_codec(requested: Codec) -> Codec:
    """What a writer will actually use, given what is installed."""
    if requested is Codec.ZSTD and not zstd_available():
        return Codec.ZLIB
    return requested

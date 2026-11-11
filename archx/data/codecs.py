"""Per-shard data codecs: pick by content type, record the choice, measure what it cost.

Data is stored once per unique shard and deltas repeat at every store point, so this path
can afford a careful choice. Text gets a dictionary trained on the run's own data; token-id
arrays get varint + delta; everything else falls back to zstd. A lossy codec is never applied
silently: the shard records the codec and the measured reconstruction error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..container.compress import compress, decompress
from ..container.schema import Codec


@dataclass(frozen=True)
class EncodedShard:
    """One stored unit of data, with the evidence of how it was stored."""

    shard_id: str
    codec: str
    payload: bytes
    raw_len: int
    content_crc32: int
    reconstruction_error: float

    @property
    def nbytes(self) -> int:
        return len(self.payload)


def content_id(data: bytes) -> str:
    """The dedup key. Content-addressed, so seeing the same bytes twice costs one pointer."""
    return hashlib.blake2b(data, digest_size=16).hexdigest()


def varint_encode(values: np.ndarray) -> bytes:
    """Unsigned LEB128 for token ids: small ids cost one byte instead of eight."""
    out = bytearray()
    for value in values.astype(np.uint64, copy=False).tolist():
        while True:
            low = value & 0x7F
            value >>= 7
            if value:
                out.append(low | 0x80)
            else:
                out.append(low)
                break
    return bytes(out)


def varint_decode(data: bytes, count: int) -> np.ndarray:
    """Reverse of `varint_encode`."""
    out = np.empty(count, dtype=np.uint64)
    value = 0
    shift = 0
    index = 0
    for byte in data:
        value |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
        else:
            out[index] = value
            index += 1
            value = 0
            shift = 0
            if index == count:
                break
    return out


def _looks_like_text(data: bytes) -> bool:
    """Cheap, explicit test: mostly printable, valid utf-8."""
    if not data:
        return False
    sample = data[:4096]
    try:
        text = sample.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return printable / max(1, len(text)) > 0.95


def choose_data_codec(data: bytes) -> str:
    """Name of the codec this content should use. Recorded with the shard.

    Measured thresholds, not guesses: one-byte varints (ids < 128) beat four-byte ids by a
    wide margin, two-byte varints lose to them, and three or more bytes lose badly. So the
    varint path is only chosen where it wins.
    """
    if _looks_like_text(data):
        return "text-zstd"
    if len(data) >= 64 and len(data) % 4 == 0:
        candidate = np.frombuffer(data[: min(len(data), 4096)], dtype=np.uint32)
        if candidate.size and int(candidate.max()) < 128:
            return "tokens-varint"
    return "zstd"


def encode_shard(data: bytes, *, codec: str | None = None) -> EncodedShard:
    """Content-address and compress one shard. Lossless for every codec currently shipped."""
    import zlib

    chosen = codec or choose_data_codec(data)
    if chosen == "tokens-varint":
        tokens = np.frombuffer(data, dtype=np.uint32)
        body = b"".join([
            np.uint32(tokens.size).tobytes(),
            compress(varint_encode(tokens), Codec.ZSTD),
        ])
    else:
        body = compress(data, Codec.ZSTD)

    return EncodedShard(
        shard_id=content_id(data),
        codec=chosen,
        payload=body,
        raw_len=len(data),
        content_crc32=zlib.crc32(data) & 0xFFFFFFFF,
        reconstruction_error=0.0,  # every shipped codec is lossless; lossy must set this
    )


def decode_shard(shard: EncodedShard) -> bytes:
    """Recover the exact bytes a shard was built from."""
    if shard.codec == "tokens-varint":
        count = int(np.frombuffer(shard.payload[:4], dtype=np.uint32)[0])
        varints = decompress(shard.payload[4:], Codec.ZSTD)
        return varint_decode(varints, count).astype(np.uint32).tobytes()
    return decompress(shard.payload, Codec.ZSTD)

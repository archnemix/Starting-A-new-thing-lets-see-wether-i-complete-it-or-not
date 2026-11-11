"""Encoding 1: sparse (index, value) pairs, indices delta-encoded then compressed.

Wins when few elements moved a lot: MoE blocks, embeddings with small batches, layers that
are mostly frozen, and the early spikes where a handful of weights jump.
"""

from __future__ import annotations

import numpy as np

from .base import Encoded, Encoder, relative_error
from ..container.compress import compress, decompress
from ..container.schema import Codec
from ..formats.dtype import to_numpy


class Sparse(Encoder):
    """Stores only what moved, with gaps between indices rather than absolute indices."""

    encoding_id = 1
    name = "sparse"

    def __init__(self, *, threshold: float = 0.0, codec: Codec = Codec.ZSTD,
                 value_dtype: str = "f32") -> None:
        self.threshold = threshold
        self.codec = codec
        self.value_dtype = value_dtype

    def encode(self, delta: np.ndarray) -> Encoded | None:
        flat = delta.ravel()
        if flat.size == 0:
            return None
        mask = np.abs(flat) > self.threshold
        nnz = int(mask.sum())
        if nnz == 0:
            return None
        if nnz > flat.size * 0.5:
            return None  # index overhead beats density this high; let dense encoders win

        indices = np.flatnonzero(mask).astype(np.uint64)
        values = flat[mask].astype(to_numpy(self.value_dtype))
        gaps = np.diff(indices, prepend=np.uint64(0)) if nnz else indices

        gap_bytes = compress(gaps.tobytes(), self.codec)
        value_bytes = compress(values.tobytes(), self.codec)
        body = b"".join([
            np.uint64(nnz).tobytes(),
            np.uint64(len(gap_bytes)).tobytes(),
            gap_bytes,
            value_bytes,
        ])
        rebuilt = np.zeros(flat.size, dtype=np.float32)
        rebuilt[indices] = values
        rebuilt = rebuilt.reshape(delta.shape)
        err = relative_error(delta, rebuilt)
        return Encoded(self.encoding_id, body, nnz, self.value_dtype, float(err), err)

    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        nnz = int(np.frombuffer(body[:8], dtype=np.uint64)[0])
        gap_len = int(np.frombuffer(body[8:16], dtype=np.uint64)[0])
        gaps = np.frombuffer(decompress(body[16:16 + gap_len], self.codec), dtype=np.uint64)
        values = np.frombuffer(decompress(body[16 + gap_len:], self.codec),
                               dtype=to_numpy(value_dtype))
        indices = np.cumsum(gaps)
        out = np.zeros(int(np.prod(shape)), dtype=np.float32)
        out[indices[:nnz]] = values[:nnz]
        return out.reshape(shape)

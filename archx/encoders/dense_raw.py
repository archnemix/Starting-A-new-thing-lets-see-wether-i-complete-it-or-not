"""Encoding 4: the lossless fallback. Shuffle byte planes, then compress.

This encoder always exists and always wins when nothing else can be exact, which is why the
format can promise that any recorded history is reconstructable. Measurement showed the gap
it fills: at full density with unstructured movement, sparse refuses and low-rank refuses,
and a quantizer is lossy - without this, a recording would have no exact option.
"""

from __future__ import annotations

import numpy as np

from .base import Encoded, Encoder
from .plane import shuffle
from ..container.compress import compress, decompress
from ..container.schema import Codec
from ..formats.dtype import to_numpy


class DenseRaw(Encoder):
    """Byte-plane shuffled, compressed, exact."""

    encoding_id = 4
    name = "dense_raw"

    def __init__(self, *, codec: Codec = Codec.ZSTD, value_dtype: str = "f32",
                 shuffle_planes: bool = False) -> None:
        # Measured (Bench/claim_plane_shuffle.py): shuffling hurts zero-heavy deltas (0.58x)
        # and barely helps dense ones (1.07x), so the exact path compresses raw bytes and the
        # shuffle is kept for data whose planes are genuinely uniform.
        self.codec = codec
        self.value_dtype = value_dtype
        self.shuffle_planes = shuffle_planes

    def encode(self, delta: np.ndarray) -> Encoded | None:
        if delta.size == 0:
            return None
        values = delta.astype(to_numpy(self.value_dtype), copy=False)
        raw = values.tobytes(order="C")
        payload = shuffle(raw, values.dtype.itemsize) if self.shuffle_planes else raw
        body = b"".join([
            np.uint8(values.dtype.itemsize).tobytes(),
            np.uint8(1 if self.shuffle_planes else 0).tobytes(),
            compress(payload, self.codec),
        ])
        return Encoded(self.encoding_id, body, delta.size, self.value_dtype, 0.0, 0.0)

    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        from .plane import unshuffle

        itemsize = int(body[0])
        shuffled = bool(body[1])
        dtype = to_numpy(value_dtype)
        raw = decompress(body[2:], self.codec)
        if shuffled:
            raw = unshuffle(raw, itemsize)
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()

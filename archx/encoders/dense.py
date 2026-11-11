"""Encoding 2: dense delta quantized to int8/int16/fp16 with a scale and zero point.

Wins when the delta is dense and small: most full-batch fine-tuning, and mid-training where
many weights move a little. The error is measured, never assumed.
"""

from __future__ import annotations

import numpy as np

from .base import Encoded, Encoder, relative_error
from ..container.compress import compress
from ..container.schema import Codec


class DenseQuant(Encoder):
    """Symmetric per-tensor quantization of a dense delta."""

    encoding_id = 2
    name = "dense_quant"

    def __init__(self, *, bits: int = 8, codec: Codec = Codec.ZSTD) -> None:
        if bits not in (8, 16):
            raise ValueError("bits must be 8 or 16")
        self.bits = bits
        self.codec = codec
        self.value_dtype = "i8" if bits == 8 else "i16"

    def encode(self, delta: np.ndarray) -> Encoded | None:
        flat = delta.astype(np.float32).ravel()
        if flat.size == 0:
            return None
        info_max = float(np.max(np.abs(flat)))
        if info_max == 0.0:
            return None
        qmax = 127.0 if self.bits == 8 else 32767.0
        scale = info_max / qmax
        quantized = np.clip(np.rint(flat / scale), -qmax, qmax).astype(
            np.int8 if self.bits == 8 else np.int16
        )
        body = b"".join([
            np.float32(scale).tobytes(),
            compress(quantized.tobytes(), self.codec),
        ])
        # Measure against what a reader will get back, i.e. round-trip through the body.
        rebuilt = self.decode(body, delta.shape, self.value_dtype, flat.size)
        err = relative_error(delta, rebuilt)
        return Encoded(self.encoding_id, body, flat.size, self.value_dtype,
                       float(np.max(np.abs(delta.reshape(-1) - rebuilt.reshape(-1)))), err)

    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        scale = float(np.frombuffer(body[:4], dtype=np.float32)[0])
        dtype = np.int8 if self.bits == 8 else np.int16
        from ..container.compress import decompress

        quantized = np.frombuffer(decompress(body[4:], self.codec), dtype=dtype)
        return (quantized.astype(np.float32) * scale).reshape(shape)

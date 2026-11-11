"""Encoding 0: the tensor did not move. Costs nothing and must never be missed."""

from __future__ import annotations

import numpy as np

from .base import Encoded, Encoder


class Unchanged(Encoder):
    """Detects a zero delta exactly. Tied and frozen parameters hit this every step."""

    encoding_id = 0
    name = "unchanged"

    def __init__(self, *, atol: float = 0.0) -> None:
        self.atol = atol

    def encode(self, delta: np.ndarray) -> Encoded | None:
        if delta.size == 0 or np.max(np.abs(delta)) <= self.atol:
            return Encoded(self.encoding_id, b"", 0, "f32", 0.0, 0.0)
        return None

    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        return np.zeros(shape, dtype=np.float32)

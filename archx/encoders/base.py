"""The encoder contract: encode a delta, report bytes and error, decode it back.

An encoder never decides policy. It answers one question - "how would you store this delta,
and what would that cost?" - and the recorder picks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Encoded:
    """One encoded delta: which encoder, its bytes, and the error it introduced."""

    encoding: int
    body: bytes
    nnz_or_rank: int
    value_dtype: str
    max_abs_error: float
    rel_error: float

    @property
    def nbytes(self) -> int:
        return len(self.body)


class Encoder(ABC):
    """Base class for delta encoders.

    Subclasses set `encoding_id` to the id used in the file format, so the reader can pick
    the matching decoder without guessing.
    """

    encoding_id: int
    name: str

    @abstractmethod
    def encode(self, delta: np.ndarray) -> Encoded | None:
        """Return the encoding, or None when this delta is not worth it (e.g. all zeros)."""

    @abstractmethod
    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        """Rebuild an approximation of the original delta."""


def relative_error(original: np.ndarray, rebuilt: np.ndarray) -> float:
    """L2 error relative to the original magnitude; 0 when nothing changed."""
    if original.size == 0:
        return 0.0
    denom = float(np.linalg.norm(original))
    if denom == 0.0:
        return 0.0
    return float(np.linalg.norm(original.astype(np.float64) - rebuilt.astype(np.float64)) / denom)


def choose_encoder(delta: np.ndarray, candidates: list[Encoder], *,
                   max_rel_error: float) -> Encoded | None:
    """Pick the encoder with the smallest output that stays inside the error budget.

    Ties go to the earlier candidate, so callers can express a preference order.
    """
    best: Encoded | None = None
    for encoder in candidates:
        encoded = encoder.encode(delta)
        if encoded is None:
            continue
        if encoded.rel_error > max_rel_error:
            continue
        if best is None or encoded.nbytes < best.nbytes:
            best = encoded
    return best

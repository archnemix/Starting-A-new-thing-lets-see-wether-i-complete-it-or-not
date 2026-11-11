"""Encoding 3: low-rank factors of a 2-D delta (U·Vᵀ), the LoRA observation applied to history.

Wins when many weights move in a coordinated way: attention projections, full-batch
fine-tuning. Rank is chosen to fit an error target, so the encoder buys fidelity only until
it is good enough.
"""

from __future__ import annotations

import numpy as np

from .base import Encoded, Encoder, relative_error
from ..container.compress import compress
from ..container.schema import Codec


class LowRank(Encoder):
    """Truncated SVD approximation of a matrix delta."""

    encoding_id = 3
    name = "lowrank"

    def __init__(self, *, max_rank: int = 32, max_rel_error: float = 1e-3,
                 codec: Codec = Codec.ZSTD) -> None:
        self.max_rank = max_rank
        self.max_rel_error = max_rel_error
        self.codec = codec

    def encode(self, delta: np.ndarray) -> Encoded | None:
        if delta.ndim != 2 or min(delta.shape) < 2:
            return None
        matrix = delta.astype(np.float32)
        limit = min(self.max_rank, min(matrix.shape))
        if limit < 1:
            return None

        u, s, vt = np.linalg.svd(matrix, full_matrices=False)

        # The stored body is U (rows x rank) plus S*Vt (rank x cols); its size grows with
        # rank, so the smallest rank that meets the error budget is also the smallest body.
        for rank in range(1, limit + 1):
            approx = (u[:, :rank] * s[:rank]) @ vt[:rank]
            err = relative_error(matrix, approx)
            if err > self.max_rel_error:
                continue
            u_bytes = compress(u[:, :rank].astype(np.float32).tobytes(), self.codec)
            vt_bytes = compress((s[:rank, None] * vt[:rank]).astype(np.float32).tobytes(), self.codec)
            body = b"".join([
                np.uint16(rank).tobytes(),
                np.uint32(matrix.shape[0]).tobytes(),
                np.uint32(matrix.shape[1]).tobytes(),
                np.uint32(len(u_bytes)).tobytes(),
                u_bytes,
                vt_bytes,
            ])
            return Encoded(self.encoding_id, body, rank, "f32",
                           float(np.max(np.abs(matrix - approx))), err)
        return None

    def decode(self, body: bytes, shape: tuple[int, ...], value_dtype: str,
               nnz_or_rank: int) -> np.ndarray:
        from ..container.compress import decompress

        rank = int(np.frombuffer(body[0:2], dtype=np.uint16)[0])
        rows = int(np.frombuffer(body[2:6], dtype=np.uint32)[0])
        cols = int(np.frombuffer(body[6:10], dtype=np.uint32)[0])
        u_len = int(np.frombuffer(body[10:14], dtype=np.uint32)[0])
        u = np.frombuffer(decompress(body[14:14 + u_len], self.codec),
                          dtype=np.float32).reshape(rows, rank)
        vt = np.frombuffer(decompress(body[14 + u_len:], self.codec),
                           dtype=np.float32).reshape(rank, cols)
        return (u @ vt).astype(np.float32)

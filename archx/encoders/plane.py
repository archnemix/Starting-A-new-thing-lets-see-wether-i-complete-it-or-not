"""Byte-plane shuffle: split floats into their byte planes so zstd can see structure.

Raw float32 deltas look like noise to a byte compressor: every fourth byte group is an
exponent, the rest are mantissa, and they interleave. Shuffling groups equal-significance
bytes together (all exponent bytes, then all mantissa-high bytes, ...) turns a smooth
distribution into long runs, which zstd compresses well. It is reversible and exact.
"""

from __future__ import annotations

import numpy as np


def shuffle(data: bytes, itemsize: int) -> bytes:
    """Reorder bytes from value-major to plane-major. Exact and reversible."""
    if itemsize <= 1 or len(data) % itemsize:
        return data
    buf = np.frombuffer(data, dtype=np.uint8).reshape(-1, itemsize)
    return buf.T.tobytes(order="C")


def unshuffle(data: bytes, itemsize: int) -> bytes:
    """Reverse `shuffle`."""
    if itemsize <= 1 or len(data) % itemsize:
        return data
    count = len(data) // itemsize
    buf = np.frombuffer(data, dtype=np.uint8).reshape(itemsize, count)
    return buf.T.tobytes(order="C")


def xor_with_previous(plane_block: bytes, previous: bytes | None) -> bytes:
    """XOR against the previous block's planes, so unchanged exponents cancel to zeros.

    Byte-wise XOR with a prefix: identical leading bytes become 0x00 runs, which is exactly
    what a compressor wants. Missing `previous` (or a length mismatch) degrades to identity.
    """
    if not previous or len(previous) != len(plane_block):
        return plane_block
    a = np.frombuffer(plane_block, dtype=np.uint8)
    b = np.frombuffer(previous, dtype=np.uint8)
    return np.bitwise_xor(a, b).tobytes()

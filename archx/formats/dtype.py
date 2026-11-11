"""The dtype ids written in the file, mapped to numpy.

The file stores a one-byte dtype id (Plan/Format_Spec.md 5.1). numpy does not understand
those short names, so this module is the single translation point - the bug that made every
sparse encode fail with "data type 'f32' not understood".
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class DType(IntEnum):
    """Wire ids. Never renumbered."""

    F32 = 0
    F16 = 1
    BF16 = 2
    F64 = 3
    I8 = 4
    I16 = 5
    I32 = 6
    I64 = 7
    BOOL = 8


_NAMES: dict[DType, str] = {
    DType.F32: "f32",
    DType.F16: "f16",
    DType.BF16: "bf16",
    DType.F64: "f64",
    DType.I8: "i8",
    DType.I16: "i16",
    DType.I32: "i32",
    DType.I64: "i64",
    DType.BOOL: "bool",
}

_NUMPY: dict[DType, str] = {
    DType.F32: "<f4",
    DType.F16: "<f2",
    DType.BF16: "<f2",  # stored as fp16 bytes; see note below
    DType.F64: "<f8",
    DType.I8: "i1",
    DType.I16: "<i2",
    DType.I32: "<i4",
    DType.I64: "<i8",
    DType.BOOL: "?",
}


def name_of(dtype: str | DType | np.dtype) -> str:
    """Normalise anything dtype-ish to the short wire name."""
    if isinstance(dtype, DType):
        return _NAMES[dtype]
    text = str(dtype)
    for dt, name in _NAMES.items():
        if text in (name, _NUMPY[dt]):
            return name
    np_dtype = np.dtype(dtype)
    for dt, name in _NAMES.items():
        if np_dtype == np.dtype(_NUMPY[dt]) and dt is not DType.BF16:
            return name
    raise ValueError(f"unsupported dtype {dtype!r}")


def to_numpy(name: str) -> np.dtype:
    """Wire name -> numpy dtype. bf16 is handled by the caller, not here."""
    for dt, wire in _NAMES.items():
        if name == wire:
            if dt is DType.BF16:
                raise ValueError("bf16 has no numpy equivalent; convert it before decoding")
            return np.dtype(_NUMPY[dt])
    raise ValueError(f"unknown dtype name {name!r}")


def from_numpy(dtype: np.dtype) -> DType:
    """numpy dtype -> wire id."""
    np_dtype = np.dtype(dtype)
    for dt, spec in _NUMPY.items():
        if dt is DType.BF16:
            continue
        if np_dtype == np.dtype(spec):
            return dt
    raise ValueError(f"unsupported numpy dtype {dtype!r}")

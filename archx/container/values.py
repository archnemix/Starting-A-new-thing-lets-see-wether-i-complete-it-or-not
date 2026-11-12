"""Turn recorded blocks back into tensors and events.

The reader knows bytes; this module knows meaning. Deltas are decoded with the encoder the
writer recorded, then summed onto the nearest anchor - which is the only way a recording can
claim "this is what the weights were at step N".
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

from .reader import Reader
from ..container.schema import BlockType
from ..encoders import DenseQuant, DenseRaw, LowRank, Sparse, Unchanged
from ..formats.dtype import DType
from ..errors import FormatError

_DECODERS = {
    Unchanged.encoding_id: Unchanged(),
    Sparse.encoding_id: Sparse(),
    # Quantization width is a property of the *block*, carried in its value dtype, not of the
    # encoder: the same encoding id can hold int8 or int16. Keying this table by encoding id
    # alone made every int16 delta decode as int8, which halves the array and fails on reshape.
    # The reader would have rejected files this build wrote itself.
    DenseQuant.encoding_id: DenseQuant(bits=8),
    3: LowRank(),          # rank carried per delta, so a fresh instance is correct
    DenseRaw.encoding_id: DenseRaw(),
}

_DTYPE_NAMES = {int(dt): name for dt, name in zip(
    DType, ["f32", "f16", "bf16", "f64", "i8", "i16", "i32", "i64", "bool"])}


@dataclass(frozen=True)
class TensorInfo:
    """One tensor from the registry block."""

    key: str
    shape: tuple[int, ...]
    dtype: int
    frozen: bool
    component: str
    region: str
    tie_group: int


def parse_registry(payload: bytes) -> list[TensorInfo]:
    """Decode the TENSOR_REGISTRY block."""
    count = struct.unpack("<I", payload[:4])[0]
    offset = 4
    tensors: list[TensorInfo] = []
    for _ in range(count):
        name_len = struct.unpack("<H", payload[offset:offset + 2])[0]
        key = payload[offset + 2:offset + 2 + name_len].decode()
        offset += 2 + name_len
        dtype = payload[offset]
        offset += 1
        ndim = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        shape = tuple(struct.unpack("<Q", payload[offset + 8 * i:offset + 8 * (i + 1)])[0]
                      for i in range(ndim))
        offset += 8 * ndim
        frozen = bool(payload[offset])
        offset += 1
        texts = []
        for _ in range(2):
            length = struct.unpack("<H", payload[offset:offset + 2])[0]
            texts.append(payload[offset + 2:offset + 2 + length].decode())
            offset += 2 + length
        tie_group = struct.unpack("<I", payload[offset:offset + 4])[0]
        offset += 4
        tensors.append(TensorInfo(key, shape, dtype, frozen, texts[0], texts[1], tie_group))
    return tensors


def parse_anchor(payload: bytes, tensors: list[TensorInfo]) -> tuple[int, dict[str, np.ndarray]]:
    """Decode an ANCHOR block into step plus arrays (tied tensors stored once)."""
    step = struct.unpack("<Q", payload[:8])[0]
    offset = 8
    arrays: dict[str, np.ndarray] = {}
    for info in tensors:
        if info.tie_group and _primary_key(tensors, info.tie_group) != info.key:
            arrays[info.key] = arrays[_primary_key(tensors, info.tie_group)]
            continue
        numel = int(np.prod(info.shape)) if info.shape else 1
        raw = payload[offset:offset + numel * 4]
        offset += numel * 4
        arrays[info.key] = np.frombuffer(raw, dtype=np.float32).reshape(info.shape).copy()
    return step, arrays


def _primary_key(tensors: list[TensorInfo], tie_group: int) -> str:
    return next(t.key for t in tensors if t.tie_group == tie_group)


def parse_delta(payload: bytes, tensors: list[TensorInfo]) -> tuple[int, dict[str, np.ndarray]]:
    """Decode a DELTA block into step plus per-tensor approximations."""
    step = struct.unpack("<Q", payload[:8])[0]
    count = struct.unpack("<I", payload[8:12])[0]
    offset = 12
    deltas: dict[str, np.ndarray] = {}
    for _ in range(count):
        index, encoding, dtype_id = struct.unpack("<IBB", payload[offset:offset + 6])
        length = struct.unpack("<I", payload[offset + 6:offset + 10])[0]
        body = payload[offset + 10:offset + 10 + length]
        offset += 10 + length
        info = tensors[index]
        decoder = _DECODERS.get(encoding)
        if decoder is None:
            raise FormatError(f"unknown delta encoding {encoding} for {info.key}")
        dtype_name = _DTYPE_NAMES.get(dtype_id, "f32")
        if encoding == DenseQuant.encoding_id and dtype_name == "i16":
            # Same encoding id, twice the width: pick the decoder from what the block says it
            # stored rather than from the id alone.
            decoder = DenseQuant(bits=16)
        deltas[info.key] = decoder.decode(body, info.shape, dtype_name, 0)
    return step, deltas


class Timeline:
    """A recording read as state over time: anchors plus deltas, applied in order."""

    def __init__(self, reader: Reader) -> None:
        self.reader = reader
        registry_blocks = list(reader.blocks(BlockType.TENSOR_REGISTRY))
        if not registry_blocks:
            raise FormatError("file has no tensor registry")
        self.tensors = parse_registry(registry_blocks[0].payload)
        self.anchors: dict[int, dict[str, np.ndarray]] = {}
        self.deltas: list[tuple[int, dict[str, np.ndarray]]] = []
        for block in reader.blocks():
            if block.block_type == int(BlockType.ANCHOR):
                step, arrays = parse_anchor(block.payload, self.tensors)
                self.anchors[step] = arrays
            elif block.block_type == int(BlockType.DELTA):
                self.deltas.append(parse_delta(block.payload, self.tensors))
        self.deltas.sort(key=lambda item: item[0])

    @property
    def steps(self) -> list[int]:
        return sorted(set(self.anchors) | {step for step, _ in self.deltas})

    def reconstruct(self, step: int) -> dict[str, np.ndarray]:
        """Rebuild every tensor at `step` from the nearest anchor at or before it."""
        base_steps = [s for s in self.anchors if s <= step]
        if not base_steps:
            raise FormatError(f"no anchor at or before step {step}")
        base = max(base_steps)
        state = {key: array.copy() for key, array in self.anchors[base].items()}
        for delta_step, delta in self.deltas:
            if delta_step <= base or delta_step > step:
                continue
            for key, values in delta.items():
                state[key] = state[key] + values
        # Tied tensors are stored once, under the first name that registered them. Every other
        # name must come back as the same values, or a reader would see a model where two
        # aliases drifted apart - a state that never existed.
        for info in self.tensors:
            if info.tie_group:
                state[info.key] = state[_primary_key(self.tensors, info.tie_group)]
        return state

    def events(self) -> list[dict]:
        """Per-step observations: loss, gradient norm and per-region movement, by step."""
        return [json.loads(b.payload) for b in self.reader.blocks(BlockType.EVENT)]

    def annotations(self) -> list[dict]:
        """User annotations: stage boundaries, unfreezes, anything the run should explain."""
        return [json.loads(b.payload) for b in self.reader.blocks(BlockType.ANNOT)]

    def segments(self) -> list[dict]:
        """Every data segment, one entry per id, as the file's newest statement about it.

        A segment keeps growing while the run continues (its step span and how often it was
        seen), and the file is append-only, so a long segment is stated more than once. The
        last statement is the true one; earlier copies are how it got there.
        """
        latest: dict[str, dict] = {}
        for block in self.reader.blocks(BlockType.DATA_SEGMENT):
            statement = json.loads(block.payload)
            latest[statement["segment_id"]] = statement
        return sorted(latest.values(), key=lambda item: (item["first_step"], item["segment_id"]))

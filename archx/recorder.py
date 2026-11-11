"""The recorder: keep the join between state, cause and stage while training runs.

Two promises govern this file:
1. Training never waits for us. Every hook does the minimum and hands work to a writer thread.
2. Recording never changes training. Tensors are read, never written; nothing here touches
   the autograd graph, the RNG, or optimizer state.

The delta is taken from a shadow copy of the previous store point, because a post-step hook
fires after the weights have already changed and the old value no longer exists.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .container.schema import BlockType
from .container.writer import Writer
from .data.capture import DataCapture, DataLevel
from .encoders import DenseQuant, DenseRaw, LowRank, Sparse, Unchanged, choose_encoder
from .formats.registry import TensorRegistry
from .policy import Decision, PhasePolicy
from .version import __version__


@dataclass
class RecorderStats:
    """Counters a caller can print or assert on. Never affects behaviour."""

    steps_seen: int = 0
    steps_stored: int = 0
    steps_skipped: int = 0
    anchors: int = 0
    blocks_written: int = 0
    bytes_written: int = 0
    encode_seconds: float = 0.0
    copy_seconds: float = 0.0
    failed: int = 0


@dataclass
class _Pending:
    """One unit of work for the writer thread."""

    kind: str                      # "anchor" | "delta"
    step: int
    arrays: dict[str, np.ndarray]  # key -> snapshot (anchor) or previous snapshot (delta)
    current: dict[str, np.ndarray] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)


class Recorder:
    """Observes a training run and writes one .archx.mdl file.

    `read(key)` must return the current array for a registry key; the recorder calls it at
    store points only, so instrumenting a loop costs nothing on skipped steps.
    """

    def __init__(
        self,
        registry: TensorRegistry,
        read,
        path: str,
        *,
        policy: PhasePolicy | None = None,
        max_rel_error: float = 1e-3,
        queue_size: int = 8,
        meta: dict[str, Any] | None = None,
        codec: int | None = None,
        data_level: DataLevel | str = DataLevel.SEGMENTS,
        data_max_bytes: int | None = None,
    ) -> None:
        self.registry = registry
        self._read = read
        self.policy = policy or PhasePolicy()
        self.max_rel_error = max_rel_error
        self.stats = RecorderStats()
        self._queue: queue.Queue[_Pending | None] = queue.Queue(maxsize=queue_size)
        self._previous: dict[str, np.ndarray] = {}
        self._thread = threading.Thread(target=self._run, name="archx-writer", daemon=True)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._encoders = [Unchanged(), Sparse(threshold=0.0), DenseQuant(bits=8),
                          LowRank(max_rank=16, max_rel_error=max_rel_error), DenseRaw()]
        self.data = DataCapture(data_level, max_bytes=data_max_bytes)

        self._writer_kwargs = {} if codec is None else {"codec": codec}
        self._path = path
        self._meta = {
            "library": f"archx {__version__}",
            "format": "1.0",
            "capability_tier": 1,
            "attribution": "observational",
            "data_level": self.data.level.value,
            "tensors": len(registry),
            "ties": sum(1 for e in registry if e.tie_group),
            **(meta or {}),
        }
        self._thread.start()

    # ------------------------------------------------------------ observation

    def step(self, step: int, *, loss: float | None = None, grad_norm: float | None = None,
             event: bool = False, data: bytes | None = None,
             segment_id: str = "", source: str = "") -> Decision:
        """Call once per optimizer step, after the weights have been updated.

        `data` is the batch (or a fingerprint-worthy slice of it). It is deduplicated by
        content, so passing the same batch again costs a pointer, not a copy.
        """
        try:
            decision = self.policy.decide(step, loss=loss, grad_norm=grad_norm, event=event)
            self.stats.steps_seen += 1
            segment = None
            if data is not None:
                segment = self.data.observe(data, step, segment_id=segment_id, source=source)
            context = {"loss": loss, "grad_norm": grad_norm,
                       "segment_id": segment.segment_id if segment else ""}
            if decision is Decision.SKIP:
                self.stats.steps_skipped += 1
                self.policy.observe(loss=loss, grad_norm=grad_norm)
                return decision

            started = time.perf_counter()
            snapshot = self._snapshot()
            self.stats.copy_seconds += time.perf_counter() - started

            if decision is Decision.ANCHOR:
                self._enqueue(_Pending("anchor", step, snapshot, context=context))
                self.stats.anchors += 1
                # The anchor becomes the reference for the next delta; without this the next
                # delta would be measured against a stale snapshot and silently lose motion.
                self._previous = snapshot
            else:
                previous = self._previous or snapshot
                self._enqueue(_Pending("delta", step, previous, current=snapshot, context=context))
                self._previous = snapshot
            self.stats.steps_stored += 1
            self.policy.observe(loss=loss, grad_norm=grad_norm)
            return decision
        except Exception as exc:  # noqa: BLE001 - a recorder fault must not stop training
            self._degrade(exc)
            return Decision.SKIP

    def annotate(self, step: int, event: str, detail: str = "") -> None:
        """Record a user annotation (stage boundary, unfreeze, curriculum change)."""
        self._enqueue(_Pending("event", step, {}, context={"event": event, "detail": detail}))

    def close(self) -> None:
        """Flush the queue, write the footer and stop the thread."""
        self._queue.put(None)
        self._thread.join(timeout=60)
        if self._error is not None:
            raise RuntimeError(f"archx recorder failed: {self._error}") from self._error

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------ internals

    def _snapshot(self) -> dict[str, np.ndarray]:
        """Copy every tracked array. Tied tensors are read once, by their primary entry."""
        out: dict[str, np.ndarray] = {}
        for entry in self.registry.primary_entries:
            array = np.asarray(self._read(entry.key))
            out[entry.key] = array.astype(np.float32, copy=True)
        return out

    def _enqueue(self, item: _Pending) -> None:
        """Never block training: if the writer is behind, drop detail, not the run."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.stats.failed += 1

    def _degrade(self, exc: BaseException) -> None:
        self.stats.failed += 1
        if self._error is None:
            self._error = exc

    def _run(self) -> None:
        try:
            with Writer(self._path, **self._writer_kwargs) as writer:
                writer.write_block(BlockType.META, json.dumps(self._meta).encode())
                writer.write_block(BlockType.TENSOR_REGISTRY, self._registry_payload())
                while True:
                    item = self._queue.get()
                    if item is None:
                        break
                    self._handle(writer, item)
                self._write_data(writer)
        except BaseException as exc:  # noqa: BLE001 - surfaced at close()
            self._error = exc

    def _write_data(self, writer: Writer) -> None:
        """Write deduplicated shards first, then the segments that point at them."""
        for shard_id, shard in self.data.shards.items():
            payload = json.dumps({
                "shard_id": shard_id,
                "codec": shard.codec,
                "raw_len": shard.raw_len,
                "content_crc32": shard.content_crc32,
                "reconstruction_error": shard.reconstruction_error,
            }).encode()
            writer.write_block(BlockType.DATA_SHARD, payload + b"\n" + shard.payload)
        for segment in self.data.segments.values():
            writer.write_block(BlockType.DATA_SEGMENT,
                               json.dumps(segment.to_dict()).encode(), step=segment.first_step)
        if self.data.sensitive:
            writer.write_block(BlockType.DATA_POINTER, json.dumps({
                "sensitive": sorted(self.data.sensitive),
                "reason": "content withheld by request; hash and characteristics kept",
            }).encode())

    def _registry_payload(self) -> bytes:
        import struct

        parts = [struct.pack("<I", len(self.registry))]
        for entry in self.registry:
            name = entry.key.encode()
            parts.append(struct.pack("<H", len(name)) + name)
            parts.append(struct.pack("<B", _dtype_id(entry.dtype)))
            parts.append(struct.pack("<I", len(entry.shape)))
            parts.extend(struct.pack("<Q", int(d)) for d in entry.shape)
            parts.append(struct.pack("<B", 1 if entry.frozen else 0))
            for text in (entry.component, entry.region):
                encoded = text.encode()
                parts.append(struct.pack("<H", len(encoded)) + encoded)
            parts.append(struct.pack("<I", entry.tie_group))
        return b"".join(parts)

    def _handle(self, writer: Writer, item: _Pending) -> None:
        started = time.perf_counter()
        if item.kind == "event":
            writer.write_block(BlockType.EVENT,
                               json.dumps({"step": item.step, **item.context}).encode(),
                               step=item.step)
        elif item.kind == "anchor":
            writer.write_block(BlockType.ANCHOR, self._anchor_payload(item), step=item.step)
        else:
            writer.write_block(BlockType.DELTA, self._delta_payload(item), step=item.step)
        self.stats.blocks_written += 1
        self.stats.bytes_written = writer._fh.tell() if not writer._closed else self.stats.bytes_written
        self.stats.encode_seconds += time.perf_counter() - started

    def _anchor_payload(self, item: _Pending) -> bytes:
        import struct

        parts = [struct.pack("<Q", item.step)]
        for entry in self.registry.primary_entries:
            parts.append(item.arrays[entry.key].astype(np.float32).tobytes(order="C"))
        return b"".join(parts)

    def _delta_payload(self, item: _Pending) -> bytes:
        import struct

        parts = [struct.pack("<Q", item.step)]
        encoded_entries = []
        for entry in self.registry.primary_entries:
            previous = item.arrays.get(entry.key)
            current = item.current.get(entry.key)
            if previous is None or current is None:
                continue
            delta = (current - previous).astype(np.float32)
            encoded = choose_encoder(delta, self._encoders, max_rel_error=self.max_rel_error)
            if encoded is None:
                encoded = DenseRaw().encode(delta)
            encoded_entries.append((entry.index, encoded))
        parts.append(struct.pack("<I", len(encoded_entries)))
        for index, encoded in encoded_entries:
            parts.append(struct.pack("<IBB", index, encoded.encoding,
                                     _dtype_id(encoded.value_dtype)))
            parts.append(struct.pack("<I", len(encoded.body)))
            parts.append(encoded.body)
        return b"".join(parts)


_DTYPE_IDS = {"f32": 0, "f16": 1, "bf16": 2, "f64": 3, "i8": 4, "i16": 5, "i32": 6,
              "i64": 7, "bool": 8}


def _dtype_id(name: str) -> int:
    return _DTYPE_IDS[name]

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

from .capabilities import Capabilities
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
        self._last_loss: float | None = None
        self._shards_written: set[str] = set()
        self._logged: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._run, name="archx-writer", daemon=True)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._encoders = [Unchanged(), Sparse(threshold=0.0), DenseQuant(bits=8),
                          LowRank(max_rank=16, max_rel_error=max_rel_error), DenseRaw()]
        self.data = DataCapture(data_level, max_bytes=data_max_bytes)
        # What this run can promise. Filled in as facts arrive, never assumed up front.
        self.capabilities = Capabilities()
        self.capabilities.offer("steps")
        self.capabilities.offer("deltas")
        # The state at attach is what the first stored step is measured against. It is copied
        # once, here, and never read again: step 0 is always an anchor.
        self._previous = self._snapshot()

        self._writer_kwargs = {} if codec is None else {"codec": codec}
        self._path = path
        self._meta = {
            "library": f"archx {__version__}",
            "format": "1.0",
            "capability_tier": 0,   # the floor, true by construction; the achieved tier is
                                    # reported by the INFO capability block written at close
            "attribution": "observational",
            "data_level": self.data.level.value,
            "tensors": len(registry),
            "ties": sum(1 for e in registry if e.tie_group),
            **(meta or {}),
        }
        self._thread.start()

    # ------------------------------------------------------------ observation

    def log(self, *, loss: float | None = None, data: bytes | None = None,
            segment_id: str = "", source: str = "", lr: float | None = None) -> None:
        """Hand the recorder what only the training loop can see, for the step now ending.

        This is the one line an adapter cannot write for the user: the loss exists as a local
        variable and dies with the step. Calling it is always optional, and anything logged
        is used by the next `step()` and then cleared, so it cannot leak into a later step.
        """
        if loss is not None:
            self._logged["loss"] = float(loss)
        if data is not None:
            self._logged["data"] = data
        if segment_id:
            self._logged["segment_id"] = segment_id
        if source:
            self._logged["source"] = source
        if lr is not None:
            self._logged["lr"] = float(lr)

    def step(self, step: int, *, loss: float | None = None, grad_norm: float | None = None,
             event: bool = False, data: bytes | None = None,
             segment_id: str = "", source: str = "", lr: float | None = None) -> Decision:
        """Call once per optimizer step, after the weights have been updated.

        `data` is the batch (or a fingerprint-worthy slice of it). It is deduplicated by
        content, so passing the same batch again costs a pointer, not a copy. Anything not
        passed here is taken from the last `log()` call and then cleared.
        """
        logged, self._logged = self._logged, {}
        loss = loss if loss is not None else logged.get("loss")
        lr = lr if lr is not None else logged.get("lr")
        data = data if data is not None else logged.get("data")
        segment_id = segment_id or logged.get("segment_id", "")
        source = source or logged.get("source", "")
        try:
            decision = self.policy.decide(step, loss=loss, grad_norm=grad_norm, event=event)
            self.stats.steps_seen += 1
            if loss is not None:
                self.capabilities.offer("loss")
            if grad_norm is not None:
                self.capabilities.offer("grad_norms")
            if lr is not None:
                self.capabilities.offer("learning_rate")
            segment = None
            if data is not None:
                segment = self.data.observe(data, step, segment_id=segment_id, source=source)
            context = {"loss": loss, "grad_norm": grad_norm, "lr": lr,
                       "segment_id": segment.segment_id if segment else ""}
            if decision is Decision.SKIP:
                self.stats.steps_skipped += 1
                self.policy.observe(loss=loss, grad_norm=grad_norm)
                return decision

            started = time.perf_counter()
            snapshot = self._snapshot()
            self.stats.copy_seconds += time.perf_counter() - started

            previous = self._previous or snapshot
            movement = _movement_by_region(self.registry, previous, snapshot)
            context["regions"] = movement
            context["seconds"] = time.time()
            self._previous = snapshot

            if decision is Decision.ANCHOR:
                # An anchor already holds the current weights, so it is its own previous
                # snapshot: no second copy, and no stale reference either.
                self._enqueue(_Pending("anchor", step, snapshot, context=context))
                self.stats.anchors += 1
            else:
                self._enqueue(_Pending("delta", step, previous, current=snapshot, context=context))
            self.stats.steps_stored += 1
            self._last_loss = loss
            if segment is not None:
                self._attribute(segment.segment_id, movement, grad_norm, loss)
            self.policy.observe(loss=loss, grad_norm=grad_norm)
            return decision
        except Exception as exc:  # noqa: BLE001 - a recorder fault must not stop training
            self._degrade(exc)
            return Decision.SKIP

    def annotate(self, step: int, event: str, detail: str = "") -> None:
        """Record a user annotation (stage boundary, unfreeze, curriculum change)."""
        self._enqueue(_Pending("annotate", step, {},
                               context={"event": event, "detail": detail}))

    def _attribute(self, segment_id: str, movement: dict[str, float],
                   grad_norm: float | None, loss: float | None) -> None:
        """Say what the weights did while this segment was on screen (Architecture 6.9)."""
        loss_delta = None
        if loss is not None and self._last_loss is not None:
            loss_delta = loss - self._last_loss
        for region, delta_norm in movement.items():
            self.data.record_effects(segment_id, region=region, delta_norm=delta_norm,
                                     grad_norm=grad_norm, loss_delta=loss_delta)

    def close(self) -> None:
        """Flush the queue, write the footer and stop the thread.

        The capability block is written here, not at open, because a tier is what the run
        turned out to be able to see - not what was asked for at the start.
        """
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
        """The writer thread: open the file once, then drain the queue until told to stop.

        A fault here is the worst kind, because the training run continues and nobody looks
        at the recorder again until the end. So the file is always tried, even after a fault:
        if a single block was bad, the file still closes with a footer and records what
        happened, and `close()` raises so the caller cannot mistake a partial file for a
        complete one. Discovered by a crash test - a merge bug killed this thread at step 22
        and the run kept going silently for four thousand more steps.
        """
        writer: Writer | None = None
        try:
            writer = Writer(self._path, **self._writer_kwargs)
            writer.write_block(BlockType.META, json.dumps(self._meta).encode())
            writer.write_block(BlockType.TENSOR_REGISTRY, self._registry_payload())
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    self._handle(writer, item)
                except Exception as exc:  # noqa: BLE001 - one bad step, not a lost run
                    self._degrade(exc)
            self._write_data(writer)
            # Last, because writing the data is what makes tier 2 provable.
            self._write_capabilities(writer)
        except BaseException as exc:  # noqa: BLE001 - surfaced at close()
            self._error = exc
        finally:
            if writer is not None:
                try:
                    if self._error is not None:
                        writer.write_block(BlockType.INFO, json.dumps({
                            "event": "recorder_fault",
                            "error": f"{type(self._error).__name__}: {self._error}",
                            "steps_stored": self.stats.steps_stored,
                        }).encode())
                    writer.close()
                except Exception as exc:  # noqa: BLE001 - nothing left to save it with
                    self._error = self._error or exc

    def _flush_data(self, writer: Writer, upto: int | None = None) -> None:
        """Seal, then write whatever data is not in the file yet.

        This runs after every stored step, not only at close, because the crash test says
        why: a killed run that kept its deltas but lost its data would answer "the weights
        moved" and not "because of this data" - half a product. Ordering matters: a shard is
        on disk before the segment pointing at it, and both after the movement they explain,
        so no record ever names a step the file does not have.

        Sealing is the expensive half - hashing and compressing - and it happens here, on the
        writer thread, which is the whole reason `observe` is allowed to be cheap.
        """
        self.data.seal()
        for shard_id, shard in self.data.shards.items():
            if shard_id in self._shards_written:
                continue
            payload = json.dumps({
                "shard_id": shard_id,
                "codec": shard.codec,
                "raw_len": shard.raw_len,
                "content_crc32": shard.content_crc32,
                "reconstruction_error": shard.reconstruction_error,
            }).encode()
            writer.write_block(BlockType.DATA_SHARD, payload + b"\n" + shard.payload)
            self._shards_written.add(shard_id)
        if self.data.segments:
            self.capabilities.offer("data_segments")
        # A segment grows while the run continues, so it is re-stated rather than rewritten:
        # the file is append-only, and the last statement about an id is the true one. Only
        # segments whose state changed are stated again, which keeps the cost linear in the
        # number of changes instead of quadratic in the number of steps.
        for segment, statement in self.data.statements(upto=upto):
            if upto is not None and segment.first_step > upto:
                # The step this data belongs to is not in the file yet, so the segment would
                # claim a movement that is not recorded. Deferred, not dropped: the next flush
                # states it, and the final flush states every one of them unclamped.
                continue
            if statement == segment.written_payload:
                continue                      # already in the file, character for character
            writer.write_block(BlockType.DATA_SEGMENT, statement.encode(),
                               step=segment.first_step)
            segment.written_payload = statement

    def _write_data(self, writer: Writer) -> None:
        """Final flush at close, plus the blocks that record what was deliberately withheld."""
        self._flush_data(writer)
        if self.data.sensitive:
            writer.write_block(BlockType.DATA_POINTER, json.dumps({
                "sensitive": sorted(self.data.sensitive),
                "reason": "content withheld by request; hash and characteristics kept",
            }).encode())
        if self.data.pointers:
            # Level `pointers`, or a cap that was hit: the file says where to look, not what
            # was there. Never a silent downgrade - META and this block both state it.
            writer.write_block(BlockType.DATA_POINTER, json.dumps({
                "pointers": sorted(self.data.pointers),
                "reason": "content not stored: level is pointers, or the byte cap was reached",
            }).encode())

    def _write_capabilities(self, writer: Writer) -> None:
        """State the tier this run reached, and name every field it could not see."""
        payload = {"event": "capabilities", **self.capabilities.resolve().to_dict()}
        writer.write_block(BlockType.INFO, json.dumps(payload).encode())

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
        if item.kind == "annotate":
            writer.write_block(BlockType.ANNOT,
                               json.dumps({"step": item.step, **item.context}).encode(),
                               step=item.step)
        elif item.kind in ("anchor", "delta"):
            payload = (self._anchor_payload(item) if item.kind == "anchor"
                       else self._delta_payload(item))
            writer.write_block(BlockType.ANCHOR if item.kind == "anchor" else BlockType.DELTA,
                               payload, step=item.step)
            # Data goes to disk right after the movement it explains, never before. The other
            # order would let a crash leave a segment naming step N in a file whose history
            # stops at N-1, and a reader would be told about data that moved nothing. This way
            # the only thing a crash can cost is the data for the step that was being written,
            # which a reader can already tell apart from a step recorded without data.
            # `upto` is the step before this one, not this one: the delta for `item.step` is
            # written above, but the event that belongs to it is written after this call, so
            # only steps strictly below are guaranteed complete on disk.
            self._flush_data(writer, upto=item.step - 1)
            # One observation per stored step: the weight change, and the stage it happened
            # at. Loss and gradient norm are optional facts, empty when unavailable, never
            # invented - the tier in META says which of them this run could see.
            writer.write_block(BlockType.EVENT,
                               json.dumps({"step": item.step, **item.context}).encode(),
                               step=item.step)
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


def _movement_by_region(registry: TensorRegistry, previous: dict[str, np.ndarray],
                        current: dict[str, np.ndarray]) -> dict[str, float]:
    """Frobenius norm of the change, summed per region.

    Regions are the only grouping that is stable across architectures, so this is the
    number a reader gets to compare between runs and between steps. Untouched tensors are
    skipped without a copy.
    """
    totals: dict[str, float] = {}
    for entry in registry.primary_entries:
        before = previous.get(entry.key)
        after = current.get(entry.key)
        if before is None or after is None:
            continue
        diff = after - before
        delta = float(np.sqrt(np.dot(diff.ravel(), diff.ravel())))
        if delta:
            totals[entry.region] = totals.get(entry.region, 0.0) + delta
    return totals


def _dtype_id(name: str) -> int:
    return _DTYPE_IDS[name]

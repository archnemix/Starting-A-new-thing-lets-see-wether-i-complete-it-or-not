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
from collections import deque
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .capabilities import Capabilities
from .container.schema import BlockType
from .container.writer import Writer
from .data.activations import ActivationCapture
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
    waited: int = 0
    bytes_written: int = 0
    encode_seconds: float = 0.0
    copy_seconds: float = 0.0
    failed: int = 0


@dataclass
class _Pending:
    """One unit of work for the writer thread."""

    kind: str                      # "anchor" | "delta" | "annotate"
    step: int
    arrays: dict[str, np.ndarray]  # key -> snapshot (anchor) or previous snapshot (delta)
    current: dict[str, np.ndarray] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    #: Movement observed between the previous stored step and this one, kept with the item so
    #: the observation block does not have to recompute it from the arrays.
    movement: dict[str, float] = field(default_factory=dict)
    #: Set when the training thread had to encode the block itself because the writer was
    #: behind. Empty means the writer will encode it, which is the normal path.
    payload: bytes = b""
    #: Per-layer activation summaries for this step, already reduced. Small by construction:
    #: one fixed-size dict per layer, never a tensor.
    activations: list = field(default_factory=list)


#: The error budget above which int16 quantization is offered.
#:
#: Measured on a 64x64 delta of 1e-3 noise: int8 costs 8.9e-3 relative, int16 costs 3.4e-5, raw
#: float32 costs nothing. At the default budget of 1e-3 int8 is refused and the delta is stored
#: raw, so the default path rebuilds exactly - which is the promise the tests make and users
#: rely on. int16's 3.4e-5 fits inside 1e-3, so offering it there would quietly make the default
#: lossy. It is offered only when the caller chooses a larger budget, where loss is the point.
_INT16_MIN_BUDGET = 1e-3


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
        activations: "ActivationCapture | None" = None,
        full_history: bool = False,
    ) -> None:
        self.registry = registry
        self._read = read
        self.policy = policy or PhasePolicy()
        self.full_history = full_history
        self.max_rel_error = max_rel_error
        self.stats = RecorderStats()
        self._queue_size = max(1, queue_size)
        # One ordered stream, not a queue plus an overflow list. Two containers cannot keep
        # step order: the overflow holds work the writer has not reached, which is *newer*
        # than anything still queued behind it, so draining the overflow first interleaves
        # steps out of order. That is what a crash test caught.
        #
        # `_queue_size` is therefore a memory budget, not a capacity: past it, the trainer
        # encodes each block itself before appending (arrays released, bytes kept), and the
        # writer's backlog is small instead of unbounded.
        self._items: deque[_Pending] = deque()
        self._cond = threading.Condition()
        self._last_loss: float | None = None
        #: Highest step number seen, so deferred work can be stamped with a real step.
        self._last_step: int | None = None
        self._shards_written: set[str] = set()
        self._logged: dict[str, Any] = {}
        #: Segment ids observed since the last store point; attributed together when it lands.
        self._span_segments: list[str] = []
        #: The step of the previous store point, so a span knows how long it was.
        self._span_start: int | None = None
        self._thread = threading.Thread(target=self._run, name="archx-writer", daemon=True)
        self._stop = threading.Event()
        self._error: BaseException | None = None
        # Quantization is offered at two widths and the error is measured, not assumed. An
        # int8 step is max|delta|/127, which for a noisy delta of many small values is a
        # relative error near 0.2% - above the default budget, so int8 alone was never
        # choosable and every delta fell through to raw float32 at twice the size.
        #
        # int16 is a tenth of int8's error (6.1e-6 against 7.9e-4 on measured deltas), so it
        # is offered only when the budget actually permits it. Adding it unconditionally made
        # the *default* configuration lossy: reconstruction stopped being exact at anchor
        # steps, which broke a documented promise. At the default 1e-3 the recorder is still
        # lossless; an opt-in budget such as 1e-2 buys the size.
        self._encoders = [Unchanged(), Sparse(threshold=0.0), DenseQuant(bits=8),
                          LowRank(max_rank=16, max_rel_error=max_rel_error), DenseRaw(),
                          DenseRaw(shuffle_planes=True)]
        if max_rel_error > _INT16_MIN_BUDGET:
            self._encoders.insert(3, DenseQuant(bits=16))
        self.data = DataCapture(data_level, max_bytes=data_max_bytes)
        # Tier 3. The capture is owned by the adapter, which is the only layer that knows what
        # a forward pass is; the recorder only decides *when* to ask and writes the answer.
        self.activations = activations
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
            "weight_history": "every_optimizer_step" if full_history else "adaptive",
            "tensors": len(registry),
            "ties": sum(1 for e in registry if e.tie_group),
            **(meta or {}),
        }
        self._thread.start()
        if full_history:
            # In dense mode step 0 is the state BEFORE training; updates are steps 1..N.
            self._enqueue(_Pending("anchor", 0, self._previous, context={"initial": True}))
            self.stats.anchors += 1
            self.stats.steps_stored += 1

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
            self._last_step = step if self._last_step is None else max(self._last_step, step)
            decision = (Decision.STORE if self.full_history else
                        self.policy.decide(step, loss=loss, grad_norm=grad_norm, event=event))
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
                if segment is not None:
                    # The data observed since the previous store point is what this store
                    # point's movement is *caused by*. Remembering it here is what lets the
                    # effect be attributed to a batch that arrived on a skipped step: the
                    # weights moved because of every batch in the span, not only the last.
                    if segment.segment_id not in self._span_segments:
                        self._span_segments.append(segment.segment_id)
            context = {"loss": loss, "grad_norm": grad_norm, "lr": lr,
                       "segment_id": segment.segment_id if segment else ""}
            if decision is Decision.SKIP:
                self.stats.steps_skipped += 1
                self.policy.observe(loss=loss, grad_norm=grad_norm)
                return decision

            started = time.perf_counter()
            snapshot = self._snapshot()
            # Taken at the store point, not at the forward: a summary is only worth keeping
            # if the file records the step it belongs to, and this is the step.
            activation_payload = self.activations.take() if self.activations else []
            if activation_payload:
                self.capabilities.offer("activations")
            self.stats.copy_seconds += time.perf_counter() - started

            previous = self._previous or snapshot
            movement = _movement_by_region(self.registry, previous, snapshot)
            context["regions"] = movement
            context["seconds"] = time.time()
            self._previous = snapshot

            if decision is Decision.ANCHOR:
                # An anchor already holds the current weights, so it is its own previous
                # snapshot: no second copy, and no stale reference either.
                self._enqueue(_Pending("anchor", step, snapshot, context=context,
                                       movement=movement, activations=activation_payload))
                self.stats.anchors += 1
            else:
                self._enqueue(_Pending("delta", step, previous, current=snapshot,
                                       context=context, movement=movement,
                                       activations=activation_payload))
            self.stats.steps_stored += 1
            self._last_loss = loss
            if self._span_segments:
                self._attribute(list(self._span_segments), movement, grad_norm, loss, step)
                self._span_segments = []
            self._span_start = step
            self.policy.observe(loss=loss, grad_norm=grad_norm, region_movement=movement)
            return decision
        except Exception as exc:  # noqa: BLE001 - a recorder fault must not stop training
            self._degrade(exc)
            return Decision.SKIP

    def annotate(self, step: int, event: str, detail: str = "") -> None:
        """Record a user annotation (stage boundary, unfreeze, curriculum change)."""
        self._enqueue(_Pending("annotate", step, {},
                               context={"event": event, "detail": detail}))

    def _attribute(self, segment_ids: list[str], movement: dict[str, float],
                   grad_norm: float | None, loss: float | None, step: int) -> None:
        """Say what the weights did while these segments were on screen (Architecture 6.9).

        Every batch observed since the previous store point is attributed, not just the one
        that happened to arrive last. Measured before this change: with the default policy a
        20-step run stored 4-5 steps, and 17 of the 20 batches sat in the file with an empty
        effect map while the file claimed tier 2 - a data half that could not answer the only
        question it exists to answer.

        The movement is shared by the span rather than divided between its batches: the
        recorder cannot know how much of each step's movement came from which batch, and
        inventing a split would be worse than saying so. `span_steps` records how many steps
        and `span_segments` how many batches share it, so a reader can tell "this batch moved
        the weights by X" from "these four batches together moved them by X".
        """
        loss_delta = None
        if loss is not None and self._last_loss is not None:
            loss_delta = loss - self._last_loss
        span_steps = 1 if self._span_start is None else max(1, step - self._span_start)
        for segment_id in segment_ids:
            for region, delta_norm in movement.items():
                self.data.record_effects(
                    segment_id, region=region, delta_norm=delta_norm,
                    grad_norm=grad_norm, loss_delta=loss_delta,
                    span_segments=len(segment_ids), span_steps=span_steps,
                    step=step)

    def close(self) -> None:
        """Flush the queue, write the footer and stop the thread.

        The capability block is written here, not at open, because a tier is what the run
        turned out to be able to see - not what was asked for at the start.
        """
        # Detach any hook the adapter registered *before* closing: a hook left on a model that
        # outlives its recorder would keep appending summaries to a file that is already shut,
        # and the failure would surface as a confusing error inside an unrelated forward pass.
        for handle in getattr(self, "_archx_activation_handles", []):
            try:
                handle.remove()
            except Exception:  # noqa: BLE001 - a removed hook is not an error
                pass
        self._archx_activation_handles = []

        pending_trace = getattr(self, "_example_trace", None) is not None
        self._example_trace = None
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=60)
        if self._error is not None:
            raise RuntimeError(f"archx recorder failed: {self._error}") from self._error
        if pending_trace:
            raise RuntimeError("example trace had no matching optimizer step; it was not recorded")

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
        """Queue work in order; bound the large opt-in example-trace backlog.

        A trace may contain batch * parameter count floats. When the queue is full,
        its producer waits for the writer rather than retaining an unbounded sequence of
        uncompressed trace archives in RAM. No data is dropped; writer failures surface.
        Other work retains the existing caller-side encoding on backlog.
        """
        if item.kind == "example_gradients":
            with self._cond:
                waited = False
                while len(self._items) >= self._queue_size:
                    if self._error is not None or not self._thread.is_alive():
                        raise RuntimeError("cannot enqueue trace: recorder writer failed") from self._error
                    waited = True
                    self._cond.wait(timeout=0.1)
                if self._error is not None:
                    raise RuntimeError("cannot enqueue trace: recorder writer failed") from self._error
                if waited:
                    self.stats.waited += 1
                self._items.append(item)
                self._cond.notify_all()
            return
        with self._cond:
            backlogged = len(self._items) >= self._queue_size
        if backlogged and item.kind in ("anchor", "delta") and not item.payload:
            # Encoded outside the lock: it is the slow part, and the writer needs the lock to
            # make progress. A single producer appends, so doing the work here cannot reorder.
            item.payload = self._payload_for(item)
            item.arrays = {}
            item.current = {}
        with self._cond:
            if backlogged:
                self.stats.waited += 1
            self._items.append(item)
            self._cond.notify()

    def _payload_for(self, item: _Pending) -> bytes:
        """Encode an item's block payload, so the training thread can let the arrays go."""
        return self._anchor_payload(item) if item.kind == "anchor" else self._delta_payload(item)

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
                with self._cond:
                    while not self._items and not self._stop.is_set():
                        self._cond.wait(timeout=0.05)
                    if not self._items:
                        break           # stop was requested and nothing is left to write
                    item = self._items.popleft()
                    self._cond.notify_all()
                try:
                    self._handle(writer, item)
                except Exception as exc:  # noqa: BLE001 - one bad step, not a lost run
                    self._degrade(exc)
            self._write_data(writer)
            self._flush_unsynced(writer)
            self._write_join_status(writer)
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

    def _flush_unsynced(self, writer: Writer) -> None:
        """Record the data a run logged after its last optimizer step, instead of dropping it.

        A loop that calls `log(data=...)` after the final `step()` - the natural shape if the
        loss is computed per batch and printed afterwards - would otherwise have its last
        batch vanish from the file. Measured on a 20-step run: 20 observations, 19 in the
        file, and no warning anywhere. Silence is the one thing this recorder is not allowed
        to produce, so the data is stored and an INFO block says it was never trained on.
        """
        pending = self._logged
        if not pending:
            return
        data = _as_bytes(pending.get("data"))
        if data is not None and self.data.level is not DataLevel.NONE:
            segment = self.data.observe(data, self._last_step or 0)
            self.data.seal()
            self._flush_data(writer)
            current = segment.segment_id if segment is not None else ""
        else:
            current = ""
        writer.write_block(BlockType.INFO, json.dumps({
            "event": "unsynced_log",
            "detail": "log() was called with no optimizer step afterwards, so this data was "
                      "observed but never trained on. It is stored with no region effects.",
            "segment_id": current,
            "loss": pending.get("loss"),
        }).encode())
        self._logged = {}

    def _write_join_status(self, writer: Writer) -> None:
        """Say what the join cannot answer, at the levels where it cannot (Architecture 6.8).

        Documented since the first design and never written, so a `none` recording and a
        `segments` recording looked equally complete to a reader. Written as an INFO block
        rather than a comment because the reader has to be able to tell.
        """
        level = self.data.level.value
        if level in ("segments", "full"):
            return
        missing = {
            "none": ["data_content", "data_characteristics_join", "segment_spans"],
            "pointers": ["data_content", "data_characteristics_join"],
        }.get(level, [])
        if self.data.dropped_to_pointers:
            missing = sorted(set(missing) | {"data_content", "data_characteristics_join"})
        writer.write_block(BlockType.INFO, json.dumps({
            "event": "JOIN_INCOMPLETE",
            "data_level": level,
            "degraded_from_cap": self.data.dropped_to_pointers,
            "cannot_answer": missing,
            "detail": "the weights and deltas are complete; only the data side is partial",
        }).encode())

    def _write_capabilities(self, writer: Writer) -> None:
        """State the tier this run reached, and name every field it could not see."""
        if self.activations is not None and not self.capabilities.have("activations"):
            self.capabilities.deny(
                "activations",
                "a capture was attached but no forward pass happened before a store point")
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
        elif item.kind == "example_gradients":
            from .adapters.example_gradients import shuffle_trace_payload
            alternative = shuffle_trace_payload(item.payload)
            writer.write_best_block([(BlockType.EXAMPLE_GRADIENTS, item.payload),
                                     (BlockType.EXAMPLE_GRADIENTS_SHUFFLED, alternative)],
                                    step=item.step)
        elif item.kind in ("anchor", "delta"):
            payload = item.payload or self._payload_for(item)
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
            # Activations sit between the movement and the event: they describe the forward
            # pass that produced the gradient which produced the delta, so a reader that has
            # read up to here has the whole causal chain for this step.
            for summary in item.activations:
                writer.write_block(BlockType.ACTIVATION,
                                   json.dumps({"step": item.step, **summary}).encode(),
                                   step=item.step)
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


def _as_bytes(data: Any) -> bytes | None:
    """Turn whatever was logged as a batch into bytes.

    The conversion rules live in `archx.adapters.torch.as_bytes`, next to the framework they
    belong to, and this defers to them rather than keeping a second copy: a duplicate here
    already drifted once (it returned a tensor's raw buffer without a CPU/detach step, and
    could not represent a str or a list). torch is imported lazily, so a container-only
    install still works - without it, only bytes and buffer objects are understood.
    """
    if data is None or isinstance(data, (bytes, bytearray, memoryview)):
        return None if data is None else bytes(data)
    try:
        from .adapters.torch import as_bytes
    except ImportError:                        # no torch: the buffer path is all there is
        tobytes = getattr(data, "tobytes", None)
        return tobytes() if tobytes is not None else None
    return as_bytes(data)


def _movement_by_region(registry: TensorRegistry, previous: dict[str, np.ndarray],
                        current: dict[str, np.ndarray]) -> dict[str, float]:
    """How far each region moved, as a fraction of its own size.

    Regions are the only grouping that is stable across architectures, so this is the
    number a reader gets to compare between runs and between steps. Untouched tensors are
    skipped without a copy.
    """
    # Accumulate squared norms per region, then divide once. The region's movement is
    # ||delta|| / ||weights|| over the whole region, not the sum of its tensors' fractions.
    #
    # Summing the fractions was a real bug and a quiet one: a transformer region holding 14
    # tensors, each moving 1.7% in a step, reported 0.20 against a 0.02 threshold, so every
    # step of every run looked like an event. The number grew with the tensor count, which is
    # the opposite of the design goal - the threshold exists so that regions with different
    # dynamics are judged against themselves, not so that bigger models store more.
    squared: dict[str, list[float]] = {}
    for entry in registry.primary_entries:
        before = previous.get(entry.key)
        after = current.get(entry.key)
        if before is None or after is None:
            continue
        diff = (after - before).ravel()
        delta = float(np.dot(diff, diff))
        size = float(np.dot(before.ravel(), before.ravel()))
        if not delta:
            continue
        # A tensor that was all zeros has no baseline to be a fraction of, and this is not a
        # corner case: biases are usually zero-initialised. Measuring against its own new
        # magnitude instead reports 100% movement on the first step and then decays as
        # 1/sqrt(step) - above any sane threshold for thousands of steps, so the trigger
        # would quietly store every step. Its squared norm is counted on neither side; its
        # movement is still in the delta, and the heartbeat bounds the gap.
        if not size:
            continue
        bucket = squared.setdefault(entry.region, [0.0, 0.0])
        bucket[0] += delta
        bucket[1] += size
    return {region: (delta ** 0.5) / (size ** 0.5)
            for region, (delta, size) in squared.items() if size > 0}


def _dtype_id(name: str) -> int:
    return _DTYPE_IDS[name]

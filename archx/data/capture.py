"""Data capture: turn "this batch" into a shard, a segment, and a join.

Three levels, chosen by the user, recorded in the file:

- ``none``     hash and count only. The cheapest, and it cannot show the data.
- ``pointers`` hash plus where to find it. The data must still exist and must not change.
- ``segments`` (default) one compressed copy per unique shard, plus characteristics.
- ``full``     every batch exactly as fed, no deduplication.

The dedup rule is the whole point: an epoch over the same data costs pointers, not copies.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .codecs import EncodedShard, content_id, encode_shard


class DataLevel(str, Enum):
    """How much of the data to keep. Defaults to `segments` (see Architecture 6.8)."""

    NONE = "none"
    POINTERS = "pointers"
    SEGMENTS = "segments"
    FULL = "full"


@dataclass
class Segment:
    """One observed piece of training data and what it was doing to the weights."""

    segment_id: str
    shard_id: str
    source: str
    fingerprint: str
    characteristics: dict
    first_step: int
    last_step: int
    times_seen: int = 1
    region_effects: dict = field(default_factory=dict)
    component_mask: int = 0
    delta_reconstruction_error: float = 0.0
    #: How many batches and how many steps share this segment's effect measurement.
    #: A batch observed on a skipped step cannot have its own movement measured - the weights
    #: did not move on that step - so it is attributed with everything else in the span since
    #: the previous store point. These two numbers are what makes that shareable fact readable
    #: instead of a batch that looks like it did nothing.
    span_segments: int = 1
    span_steps: int = 1
    #: The store point at which this data's effect was measured, or -1 when none was.
    #: `span_steps` says how many steps the measurement covers but not *which* recorded step
    #: it belongs to, so a batch seen on a skipped step was attributed to movement the reader
    #: could not locate. With this, "which data moved which weights at which step" is answered
    #: by a step number that the file definitely holds.
    effect_step: int = -1
    #: Bytes waiting for the writer thread. Never written unless a shard is made from it.
    content: bytes | None = field(default=None, repr=False, compare=False)
    sensitive: bool = field(default=False, repr=False, compare=False)
    #: True when the caller named this segment. A name is never merged away.
    named: bool = field(default=False, repr=False, compare=False)
    #: The exact text last written to the file, so an unchanged segment is never written twice.
    written_payload: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self, *, upto: int | None = None) -> dict:
        """The join as it stands. `upto` clamps the end of the span to recorded history.

        A segment is observed on the training thread and written from the writer thread, which
        is behind it by design. A crash between the two could otherwise leave a segment saying
        it was last seen at a step the file never reached - a record of motion that is not in
        the file. Only the end of the span is clamped: where a segment *started* is a fact the
        moment it is observed, and it cannot be beyond `upto`, because `upto` is always at
        least as high as the step being observed.
        """
        last = self.last_step if upto is None else min(self.last_step, upto)
        return {
            "segment_id": self.segment_id,
            "shard_id": self.shard_id,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "characteristics": self.characteristics,
            "first_step": self.first_step,
            "last_step": max(self.first_step, last),
            "times_seen": self.times_seen,
            "region_effects": self.region_effects,
            "component_mask": self.component_mask,
            "delta_reconstruction_error": self.delta_reconstruction_error,
            "span_segments": self.span_segments,
            "span_steps": self.span_steps,
            "effect_step": self.effect_step,
        }


def characteristics(data: bytes, *, shard_id: str = "") -> dict:
    """Cheap, comparable description of the data. Same schema for every run, by design.

    `shard_id` is the content hash when the caller already has it. Hashing a 128 KiB batch
    costs ~210 us, so hashing the same bytes twice per step would be the single most
    expensive thing recording does.
    """
    info = {
        "bytes": len(data),
        "sha:blake2b16": shard_id or content_id(data),
    }
    try:
        text = data.decode("utf-8")
        info["kind"] = "text"
        info["chars"] = len(text)
        info["lines"] = text.count("\n")
        info["unique_chars"] = len(set(text))
    except UnicodeDecodeError:
        info["kind"] = "binary"
        if len(data) >= 64 and len(data) % 4 == 0:
            tokens = np.frombuffer(data, dtype=np.uint32)
            info["kind"] = "token_ids"
            info["count"] = int(tokens.size)
            info["max_id"] = int(tokens.max()) if tokens.size else 0
    return info


class DataCapture:
    """Collects data segments and deduplicated shards for the recorder to write.

    Two phases, split on purpose. `observe` runs on the training thread and does the cheapest
    thing that can answer "which segment is this?" - an identity check, then a pointer lookup
    against data already seen. `seal` runs on the writer thread and does the expensive work:
    hashing and compressing. Measured on a 128 KiB batch, hashing costs ~210 us and
    compressing ~530 us, both of which must not be felt by a training step.
    """

    def __init__(self, level: DataLevel | str = DataLevel.SEGMENTS, *,
                 max_bytes: int | None = None) -> None:
        self.level = DataLevel(level)
        self.max_bytes = max_bytes
        self.shards: dict[str, EncodedShard] = {}
        self.segments: dict[str, Segment] = {}
        self.sensitive: set[str] = set()
        self.pointers: set[str] = set()
        self._by_shard: dict[str, Segment] = {}
        self._observed = 0
        self._by_identity: dict[int, Segment] = {}
        # `observe` runs on the training thread and the flush runs on the writer thread, so
        # every read or write of these dicts is under this lock. Without it a flush can see a
        # segment that was created but not yet sealed, and write it out with no characteristics.
        self._lock = threading.Lock()
        self.bytes_stored = 0
        self.degraded_to: DataLevel | None = None
        self._started = time.monotonic()

    # ------------------------------------------------------------------ API

    def observe(self, data: bytes, step: int, *, segment_id: str = "",
                source: str = "", sensitive: bool = False) -> Segment | None:
        """Record one observation of data. Returns None when the level keeps nothing.

        Cheap by design. Same bytes under the same id is a pointer come back; a new segment is
        recorded with its content left unhashed until `seal` runs off the training thread.
        """
        if self.level is DataLevel.NONE:
            return None
        with self._lock:
            return self._observe(data, step, segment_id, source, sensitive)

    def _observe(self, data: bytes, step: int, segment_id: str,
                 source: str, sensitive: bool) -> Segment | None:
        """The body of `observe`, called with the lock held."""
        if self.level is DataLevel.FULL:
            # `full` promises every batch exactly as fed, in order, deduplicated never. An
            # epoch over 30 batches must produce 30 stored segments even when the batches
            # hold identical bytes, so no identity check may run here at all. Measured
            # before this branch existed: `full` and `segments` wrote files 10 bytes apart,
            # because `full` was documented and never implemented.
            segment_id = ""
        if segment_id:
            existing = self.segments.get(segment_id)
            if existing is not None:
                # The caller named this segment, so the name is the identity. Whether the
                # bytes changed is a question for the hash, on the writer thread.
                existing.last_step = step
                existing.times_seen += 1
                return existing
        elif self.level is not DataLevel.FULL:
            # Unnamed: the only free identity check is object identity, which is exactly true
            # when a dataloader hands over the same batch object again. Never a byte compare.
            existing = self._by_identity.get(id(data))
            if existing is not None and existing.content is data:
                existing.last_step = step
                existing.times_seen += 1
                return existing

        self._observed += 1
        key = segment_id or f"data-{self._observed}"

        segment = Segment(
            segment_id=key,
            shard_id="",                 # known only after sealing
            source=source,
            fingerprint="",              # known only after sealing
            characteristics={},
            first_step=step,
            last_step=step,
            content=data,
            sensitive=sensitive,
            named=bool(segment_id),
        )
        self.segments[key] = segment
        if data is not None:
            self._by_identity[id(data)] = segment
        return segment

    def seal(self) -> None:
        """Hash, deduplicate and compress every unsealed segment. Runs on the writer thread.

        A content cap that is hit degrades to `pointers` here, once, for the whole capture:
        the file keeps the hash and the characteristics, never a half-stored shard.
        """
        with self._lock:
            self._seal()

    def _seal(self) -> None:
        for segment in list(self.segments.values()):   # merging removes entries
            if segment.shard_id or segment.content is None:
                continue
            data = segment.content
            shard_id = content_id(data)
            segment.fingerprint = shard_id
            segment.characteristics = characteristics(data, shard_id=shard_id)

            if self.level is DataLevel.NONE:
                segment.shard_id = shard_id
            elif self.level is DataLevel.POINTERS:
                segment.shard_id = shard_id
                self.pointers.add(shard_id)
            elif segment.sensitive:
                self.sensitive.add(shard_id)
                segment.shard_id = shard_id
            elif self.max_bytes is not None and self.bytes_stored + len(data) > self.max_bytes:
                # Degrade, never truncate: a half-stored shard would look complete and be wrong.
                self.level = DataLevel.POINTERS
                self.degraded_to = DataLevel.POINTERS
                self.pointers.add(shard_id)
                segment.shard_id = shard_id
            elif self.level is not DataLevel.FULL and shard_id in self.shards:
                segment.shard_id = shard_id     # dedup: the second sighting costs nothing
            else:
                encoded = encode_shard(data)
                # Under `segments` the key is the content hash, so an epoch over the same
                # data costs pointers rather than copies - including when two *named*
                # segments share bytes, which is the join case: two ids, one stored copy.
                # Only `full` reaches the suffixed branch, because it promises every batch
                # exactly as fed, and two identical copies under one id would collapse in
                # any reader that keys shards by id.
                key = shard_id if shard_id not in self.shards else f"{shard_id}-{len(self.shards)}"
                self.shards[key] = encoded
                self.bytes_stored += encoded.nbytes
                segment.shard_id = key
            segment.content = None              # the bytes live in the shard now, not twice
            self._merge_unnamed(segment)
        self._by_shard = {s.shard_id: s for s in self.segments.values() if not s.named}
        # Forget identity references to sealed segments. The identity fast-path only proves
        # two observations are the same batch while it can still see the object; once the
        # bytes have been handed to a shard, the entry is a bare `id()` that could be reused
        # by a different object, and the segment it names may have been merged away. Keeping
        # them was a real accounting bug: observations and `times_seen` disagreed with no
        # warning. After this, a repeat observation becomes a new segment and the content
        # hash merges it on the next seal - slower, and true.
        self._by_identity = {identity: segment
                             for identity, segment in self._by_identity.items()
                             if segment.content is not None
                             and self.segments.get(segment.segment_id) is segment}

    def statements(self, upto: int | None = None) -> list[tuple["Segment", str]]:
        """The segments plus the text that should now be in the file for each.

        Comparing that text with `written_payload` is how the writer decides whether to write
        at all. A flag would be simpler and wrong twice over: the writer thread lags the
        training thread, so a flush can write a clamped, partial statement that a flag would
        mark as done, and a re-statement whose content is identical would be written anyway.
        Text comparison is exact, idempotent, and costs a JSON dump per changed segment.
        """
        with self._lock:
            return [(segment, json.dumps(segment.to_dict(upto=upto)))
                    for segment in self.segments.values()]

    def _merge_unnamed(self, segment: Segment) -> None:
        """Collapse two unnamed segments that turned out to hold the same bytes.

        While recording, identical bytes in two different objects are indistinguishable from
        two segments - telling them apart would mean hashing on the training thread. After
        sealing, the hash says they are one shard, so they are one segment too. A *named*
        segment is never merged: two ids over one shard is the join case, not a duplicate.
        """
        if segment.named or self.level is DataLevel.FULL:
            return
        previous = self._by_shard.get(segment.shard_id)
        if previous is None:
            self._by_shard[segment.shard_id] = segment
            return
        # Merging into a segment already in the file is safe: the next flush sees the changed
        # (last_step, times_seen) and re-states it, so the file never keeps an older copy.
        previous.last_step = max(previous.last_step, segment.last_step)
        previous.times_seen += segment.times_seen
        previous.first_step = min(previous.first_step, segment.first_step)
        previous.effect_step = max(previous.effect_step, segment.effect_step)
        for region, effects in segment.region_effects.items():
            held = previous.region_effects.setdefault(
                region, {"delta_norm": 0.0, "grad_norm_max": 0.0, "loss_delta": 0.0})
            held["delta_norm"] += effects["delta_norm"]
            held["grad_norm_max"] = max(held["grad_norm_max"], effects["grad_norm_max"])
            held["loss_delta"] += effects["loss_delta"]
        # Repoint every identity reference at the survivor before dropping the duplicate.
        # Without this, `_by_identity` keeps a handle on the popped object and every later
        # sighting of the same batch increments a segment that is no longer in the file -
        # measured as `times_seen` totals that did not match the number of observations.
        for identity, held in list(self._by_identity.items()):
            if held is segment:
                self._by_identity[identity] = previous
        if self._by_shard.get(segment.shard_id) is segment:
            self._by_shard[segment.shard_id] = previous
        self.segments.pop(segment.segment_id, None)

    def record_effects(self, segment_id: str, *, region: str, delta_norm: float,
                       grad_norm: float | None, loss_delta: float | None,
                       span_segments: int = 1, span_steps: int = 1,
                       step: int | None = None) -> None:
        """Attach what the weights did while this segment was being trained on.

        `span_segments` and `span_steps` say how many batches and steps share the measurement,
        so a batch that was observed on a skipped step is not mistaken for one that caused
        nothing. They are recorded rather than inferred, because the recorder is the only
        place that knows.
        """
        segment = self.segments.get(segment_id)
        if segment is None:
            return
        segment.span_segments = max(segment.span_segments if segment.region_effects else 1,
                                    span_segments)
        segment.span_steps = max(segment.span_steps if segment.region_effects else 1,
                                 span_steps)
        if step is not None:
            segment.effect_step = step
        entry = segment.region_effects.setdefault(
            region, {"delta_norm": 0.0, "grad_norm_max": 0.0, "loss_delta": 0.0}
        )
        entry["delta_norm"] += delta_norm
        if grad_norm is not None:
            entry["grad_norm_max"] = max(entry["grad_norm_max"], grad_norm)
        if loss_delta is not None:
            entry["loss_delta"] += loss_delta

    @property
    def dropped_to_pointers(self) -> bool:
        """True when the configured byte cap forced content to be dropped."""
        return self.degraded_to is DataLevel.POINTERS

    def projected_bytes(self) -> int:
        """What the file would hold if every segment seen so far were stored."""
        return self.bytes_stored

    # ------------------------------------------------------------ internals

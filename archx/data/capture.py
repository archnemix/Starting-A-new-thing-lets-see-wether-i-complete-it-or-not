"""Data capture: turn "this batch" into a shard, a segment, and a join.

Three levels, chosen by the user, recorded in the file:

- ``none``     hash and count only. The cheapest, and it cannot show the data.
- ``pointers`` hash plus where to find it. The data must still exist and must not change.
- ``segments`` (default) one compressed copy per unique shard, plus characteristics.
- ``full``     every batch exactly as fed, no deduplication.

The dedup rule is the whole point: an epoch over the same data costs pointers, not copies.
"""

from __future__ import annotations

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

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "shard_id": self.shard_id,
            "source": self.source,
            "fingerprint": self.fingerprint,
            "characteristics": self.characteristics,
            "first_step": self.first_step,
            "last_step": self.last_step,
            "times_seen": self.times_seen,
            "region_effects": self.region_effects,
            "component_mask": self.component_mask,
            "delta_reconstruction_error": self.delta_reconstruction_error,
        }


def characteristics(data: bytes) -> dict:
    """Cheap, comparable description of the data. Same schema for every run, by design."""
    info = {
        "bytes": len(data),
        "sha:blake2b16": content_id(data),
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
    """Collects data segments and deduplicated shards for the recorder to write."""

    def __init__(self, level: DataLevel | str = DataLevel.SEGMENTS, *,
                 max_bytes: int | None = None) -> None:
        self.level = DataLevel(level)
        self.max_bytes = max_bytes
        self.shards: dict[str, EncodedShard] = {}
        self.segments: dict[str, Segment] = {}
        self.sensitive: set[str] = set()
        self.bytes_stored = 0
        self.degraded_to: DataLevel | None = None
        self._started = time.monotonic()

    # ------------------------------------------------------------------ API

    def observe(self, data: bytes, step: int, *, segment_id: str = "",
                source: str = "", sensitive: bool = False) -> Segment | None:
        """Record one observation of data. Returns None when the level keeps nothing."""
        if self.level is DataLevel.NONE:
            return None

        shard_id = content_id(data)
        lookup = segment_id or shard_id
        existing = self.segments.get(lookup) if lookup else None
        if existing is not None:
            existing.last_step = step
            existing.times_seen += 1
            return existing

        stored_id = shard_id
        if self.level in (DataLevel.SEGMENTS, DataLevel.FULL):
            stored_id = self._store_shard(data, shard_id, sensitive=sensitive)

        segment = Segment(
            segment_id=lookup,
            shard_id=stored_id,
            source=source,
            fingerprint=characteristics(data)["sha:blake2b16"],
            characteristics=characteristics(data),
            first_step=step,
            last_step=step,
        )
        self.segments[lookup] = segment
        return segment

    def record_effects(self, segment_id: str, *, region: str, delta_norm: float,
                       grad_norm: float | None, loss_delta: float | None) -> None:
        """Attach what the weights did while this segment was being trained on."""
        segment = self.segments.get(segment_id)
        if segment is None:
            return
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

    def _store_shard(self, data: bytes, shard_id: str, *, sensitive: bool) -> str:
        if shard_id in self.shards or shard_id in self.sensitive:
            return shard_id  # dedup: the second sighting costs nothing
        if sensitive:
            self.sensitive.add(shard_id)
            return shard_id  # content deliberately not kept; hash and characteristics remain

        if self.max_bytes is not None and self.bytes_stored + len(data) > self.max_bytes:
            # Degrade, never truncate: a half-stored shard would look complete and be wrong.
            self.level = DataLevel.POINTERS
            self.degraded_to = DataLevel.POINTERS
            return shard_id

        encoded = encode_shard(data)
        self.shards[shard_id] = encoded
        self.bytes_stored += encoded.nbytes
        return shard_id

"""The tensor registry: what is being recorded, under what name, and how it is grouped.

Everything downstream (deltas, anchors, data joins, the reader) addresses tensors by index
into this registry, never by name. That is what makes tied weights storable once and makes
renames impossible to confuse with a different tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..formats.dtype import from_numpy, name_of


@dataclass
class TensorEntry:
    """One tracked tensor.

    `key` is the source path used to find it again ("model.layers.0.attn.q_proj.weight"),
    `index` is what goes in the file, and `tie_group` marks tensors that share storage.
    """

    key: str
    index: int
    shape: tuple[int, ...]
    dtype: str
    frozen: bool = False
    component: str = ""
    region: str = ""
    tie_group: int = 0

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))


@dataclass
class TensorRegistry:
    """Ordered, stable set of tracked tensors."""

    entries: list[TensorEntry] = field(default_factory=list)
    _by_key: dict[str, TensorEntry] = field(default_factory=dict)
    _tie_count: int = 0

    # -------------------------------------------------------------- building

    def add(
        self,
        key: str,
        array: np.ndarray,
        *,
        pointer: int | None = None,
        frozen: bool = False,
        component: str = "",
        region: str = "",
    ) -> TensorEntry:
        """Register a tensor.

        Tied weights are detected by `pointer` (the storage address). The second and later
        tensors sharing a pointer join the first one's `tie_group` and are flagged, so the
        writer never stores the same bytes twice and the reader can rebuild the alias.
        """
        existing = self._by_key.get(key)
        if existing is not None:
            return existing

        tie_group = 0
        if pointer is not None:
            for entry in self.entries:
                if getattr(entry, "_pointer", None) == pointer:
                    tie_group = entry.tie_group or self._next_tie_group()
                    entry.tie_group = tie_group
                    break

        entry = TensorEntry(
            key=key,
            index=len(self.entries),
            shape=tuple(int(d) for d in array.shape),
            dtype=name_of(array.dtype),
            frozen=frozen,
            component=component,
            region=region,
            tie_group=tie_group,
        )
        setattr(entry, "_pointer", pointer)
        self.entries.append(entry)
        self._by_key[key] = entry
        return entry

    def _next_tie_group(self) -> int:
        self._tie_count += 1
        return self._tie_count

    # -------------------------------------------------------------- queries

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def by_key(self, key: str) -> TensorEntry:
        return self._by_key[key]

    def get(self, key: str) -> TensorEntry | None:
        return self._by_key.get(key)

    @property
    def primary_entries(self) -> list[TensorEntry]:
        """One entry per distinct storage: tied tensors are stored once, here."""
        seen: set[int] = set()
        keep: list[TensorEntry] = []
        for entry in self.entries:
            if entry.tie_group and entry.tie_group in seen:
                continue
            if entry.tie_group:
                seen.add(entry.tie_group)
            keep.append(entry)
        return keep

    @property
    def components(self) -> list[str]:
        return sorted({e.component for e in self.entries if e.component})

    @property
    def regions(self) -> list[str]:
        return sorted({e.region for e in self.entries if e.region})

    def total_numel(self) -> int:
        """Parameters counted once, so a tied model does not look twice as large."""
        return sum(e.numel for e in self.primary_entries)


def registry_from_named_arrays(items, *, components: dict[str, str] | None = None) -> TensorRegistry:
    """Build a registry from (key, array) pairs, e.g. `model.named_parameters()`.

    `components` optionally maps a key prefix to a component name; the longest matching
    prefix wins, which is how a multimodal model's towers become first-class components.
    """
    registry = TensorRegistry()
    components = components or {}
    for item in items:
        if len(item) == 2:
            key, array = item
            frozen = False
        else:
            key, array, frozen = item
        pointer = array.ctypes.data if hasattr(array, "ctypes") else None
        registry.add(key, array, pointer=pointer, frozen=frozen,
                     component=_component_for(key, components))
    return registry


def _component_for(key: str, components: dict[str, str]) -> str:
    best = ""
    for prefix, name in components.items():
        if key.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return components.get(best, "") if best else ""

"""Unit tests for the tensor registry: stable ids, tie detection, components, counting."""

from __future__ import annotations

import numpy as np

from archx.formats.registry import TensorRegistry, registry_from_named_arrays


def test_indices_are_stable_and_follow_insertion_order():
    reg = TensorRegistry()
    a = reg.add("a", np.zeros((2, 3), dtype=np.float32))
    b = reg.add("b", np.zeros((4,), dtype=np.float16))
    assert (a.index, b.index) == (0, 1)
    assert reg.by_key("b") is b
    assert [e.key for e in reg] == ["a", "b"]


def test_adding_the_same_key_twice_does_not_duplicate():
    reg = TensorRegistry()
    first = reg.add("w", np.zeros((2,), dtype=np.float32))
    again = reg.add("w", np.zeros((2,), dtype=np.float32))
    assert first is again
    assert len(reg) == 1


def test_tied_weights_share_a_group_and_are_counted_once():
    shared = np.zeros((8, 8), dtype=np.float32)
    reg = TensorRegistry()
    first = reg.add("encoder.embed", shared, pointer=shared.ctypes.data)
    second = reg.add("decoder.embed", shared, pointer=shared.ctypes.data)

    assert first.tie_group != 0
    assert second.tie_group == first.tie_group
    assert len(reg.primary_entries) == 1, "the same bytes must be stored once"
    assert reg.total_numel() == 64, "a tied model must not look twice as large"


def test_distinct_storage_is_never_grouped():
    a = np.zeros((8,), dtype=np.float32)
    b = np.zeros((8,), dtype=np.float32)
    reg = TensorRegistry()
    reg.add("a", a, pointer=a.ctypes.data)
    reg.add("b", b, pointer=b.ctypes.data)
    assert len(reg.primary_entries) == 2


def test_dtypes_are_recorded_in_wire_form():
    reg = TensorRegistry()
    entry = reg.add("w", np.zeros((2, 2), dtype=np.float16))
    assert entry.dtype == "f16"


def test_components_come_from_the_longest_matching_prefix():
    items = [
        ("vision.encoder.w", np.zeros((2,), dtype=np.float32)),
        ("language.model.w", np.zeros((2,), dtype=np.float32)),
        ("language.head.w", np.zeros((2,), dtype=np.float32)),
    ]
    components = {"vision": "vision_encoder", "language": "language_model"}
    reg = registry_from_named_arrays(items, components=components)
    assert reg.by_key("vision.encoder.w").component == "vision_encoder"
    assert reg.by_key("language.head.w").component == "language_model"
    assert reg.components == ["language_model", "vision_encoder"]


def test_frozen_entries_can_be_built_from_triples():
    items = [("w", np.zeros((2,), dtype=np.float32), True)]
    reg = registry_from_named_arrays(items)
    assert reg.by_key("w").frozen is True

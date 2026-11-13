"""Unit tests for crash recovery: a torn file loses its tail, never its history."""

from __future__ import annotations

import json

from archx.container.reader import Reader
from archx.container.recovery import recover_copy, recover_in_place
from archx.container.schema import BlockType
from archx.container.writer import Writer


def _crashed_file(tmp_path, blocks=3, cut=7):
    path = tmp_path / "crashed.archx.mdl"
    w = Writer(path)
    w.write_block(BlockType.META, b'{"run": "crash-test"}')
    for step in range(blocks):
        w.write_block(BlockType.DELTA, json.dumps({"step": step}).encode(), step=step)
    w.abort()  # no footer, like a killed process
    data = path.read_bytes()
    path.write_bytes(data[:-cut])  # tear the last block
    return path


def test_recover_keeps_every_whole_block_and_writes_a_footer(tmp_path):
    path = _crashed_file(tmp_path)
    report = recover_in_place(path)
    assert report.bytes_truncated > 0
    assert report.blocks_kept >= 1

    reader = Reader(path)
    assert reader.container.footer is not None
    assert reader.header.recovered is True
    assert reader.meta() == {"run": "crash-test"}
    deltas = [json.loads(b.payload) for b in reader.blocks(BlockType.DELTA)]
    assert deltas == [{"step": 0}, {"step": 1}]


def test_recovered_file_has_a_usable_index(tmp_path):
    path = _crashed_file(tmp_path)
    recover_in_place(path)
    reader = Reader(path)
    index = Reader.index_block_payload(reader.container)
    assert [step for step, _offset, _len in index] == [0, 1]


def test_recover_copy_leaves_the_original_broken(tmp_path):
    path = _crashed_file(tmp_path)
    before = path.read_bytes()
    target = tmp_path / "recovered.archx.mdl"
    recover_copy(path, target)
    assert path.read_bytes() == before
    assert Reader(target).header.recovered is True


def test_recovery_on_a_healthy_file_is_a_no_op(tmp_path):
    path = tmp_path / "healthy.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, b"{}")
        w.write_block(BlockType.DELTA, b"abc", step=0)
    report = recover_in_place(path)
    assert report.bytes_truncated >= 0
    reader = Reader(path)
    assert reader.container.footer is not None


def test_recovery_restates_the_tier_instead_of_reporting_the_floor(tmp_path):
    """A crash happens before close, and the capability block is written at close.

    So a recovered file had none, and `archx info` fell back to META's `capability_tier: 0` -
    a run that had recorded loss, data and activations described itself afterwards as a file
    that could see nothing but a step counter. The measurements are all still on disk, so
    recovery states the tier from the blocks that survived.
    """
    path = tmp_path / "crashed-run.archx.mdl"
    w = Writer(path)
    w.write_block(BlockType.META, b'{"run": "crashed"}')
    w.write_block(BlockType.TENSOR_REGISTRY, _one_tensor_registry())
    w.write_block(BlockType.ANCHOR, _anchor_payload(), step=0)
    w.write_block(BlockType.EVENT,
                  json.dumps({"step": 0, "loss": 1.4, "grad_norm": 2.1, "regions": {}}).encode(),
                  step=0)
    w.write_block(BlockType.EVENT,
                  json.dumps({"step": 1, "loss": 1.0, "grad_norm": 2.0, "regions": {}}).encode(),
                  step=1)
    w.write_block(BlockType.DATA_SEGMENT,
                  json.dumps({"segment_id": "s1", "first_step": 0, "last_step": 1,
                              "times_seen": 2, "characteristics": {"bytes": 8}}).encode())
    w.write_block(BlockType.ACTIVATION,
                  json.dumps({"step": 1, "layer": "w", "shape": [4], "count": 4,
                              "zero_fraction": 0.0}).encode())
    w.abort()                             # killed before close: no footer, no capability block

    assert Reader(path, allow_torn=True).capabilities() == {}, (
        "this fixture must start without a capability block, or the test proves nothing")

    report = recover_in_place(path)
    assert report.capability_written

    capabilities = Reader(path).capabilities()
    assert capabilities["capability_tier"] == 3, capabilities
    assert capabilities["recomputed_by_recovery"] is True
    assert capabilities["capabilities_present"] == [
        "activations", "data_segments", "deltas", "grad_norms", "loss", "steps"]

    # And the file now answers the tier question the way `archx info` asks it.
    assert Reader(path).info()["capabilities"]["capability_tier"] == 3


def test_recovery_does_not_overrule_a_capability_block_the_run_wrote(tmp_path):
    """The run's own statement outranks a second opinion computed from its blocks.

    The tear has to be *after* the capability block, which is why the INFO block is written
    first here and the index is what gets truncated. Writing it last - where the recorder
    writes it - puts the tear before it, and there is nothing left to keep.
    """
    path = tmp_path / "closed.archx.mdl"
    w = Writer(path)
    w.write_block(BlockType.META, b"{}")
    w.write_block(BlockType.TENSOR_REGISTRY, _one_tensor_registry())
    w.write_block(BlockType.ANCHOR, _anchor_payload(), step=0)
    w.write_block(BlockType.INFO,
                  json.dumps({"event": "capabilities", "capability_tier": 3,
                              "capabilities_present": ["steps", "deltas"],
                              "capabilities_missing": {}}).encode())
    w.write_block(BlockType.INDEX, b"\x00" * 64)      # the tail that will be torn
    w.abort()

    report = recover_in_place(path)
    assert not report.capability_written
    capabilities = Reader(path).capabilities()
    assert capabilities["capability_tier"] == 3, capabilities
    assert "recomputed_by_recovery" not in capabilities

    # And recovery added no second capability block.
    blocks = [b for b in Reader(path).blocks(BlockType.INFO)
              if json.loads(b.payload).get("event") == "capabilities"]
    assert len(blocks) == 1


def _one_tensor_registry() -> bytes:
    """A registry block with one float32 tensor named `w`, shape (4,)."""
    import struct

    name = b"w"
    parts = [struct.pack("<I", 1)]
    parts.append(struct.pack("<H", len(name)) + name)
    parts.append(struct.pack("<B", 2))                       # dtype id for float32
    parts.append(struct.pack("<I", 1) + struct.pack("<Q", 4))
    parts.append(struct.pack("<B", 0))                       # not frozen
    for text in ("", ""):                                    # component, region
        parts.append(struct.pack("<H", len(text)))
    parts.append(struct.pack("<I", 0))                       # tie group
    return b"".join(parts)


def _anchor_payload() -> bytes:
    """An ANCHOR body: a step, a tensor count, then one dense f32 tensor."""
    import struct

    import numpy as np

    head = struct.pack("<Q", 0) + struct.pack("<I", 1)
    body = np.zeros((4,), dtype="<f4").tobytes()
    return head + struct.pack("<I", len(body)) + struct.pack("<B", 2) + body

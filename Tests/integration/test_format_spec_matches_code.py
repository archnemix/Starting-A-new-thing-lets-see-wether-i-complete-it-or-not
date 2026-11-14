"""Integration: the published format spec is checked against files the code actually writes.

`Plan/Format_Spec.md` describes a binary container, and it had drifted badly: it documented a
packed `DATA_SEGMENT` layout that the writer had never produced (segments are JSON), and had
no entry at all for `EVENT`, `ANNOT`, `INFO` or `META`. A specification that disagrees with the
implementation is worse than none, because it is what a second implementation would trust.

These tests parse real files against the spec's own byte layouts. They are the reason the spec
can be published as v1: every claim exercised here is checked, not asserted in prose.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

import archx                                                       # noqa: E402
from archx.container.reader import Reader                          # noqa: E402
from archx.container.schema import BlockType                       # noqa: E402
from archx.container.values import Timeline                        # noqa: E402


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """One real recording, with loss, data and a lossless delta path, used by every test."""
    torch.manual_seed(0)
    directory = tmp_path_factory.mktemp("spec")
    model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = directory / "spec.archx.mdl"
    recorder = archx.attach(model, optimizer, str(path), max_rel_error=1e-3)
    x = torch.randn(5, 4)
    recorder.annotate(0, "stage_1", "the fixture's own annotation: the join must keep it")
    for _ in range(6):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), torch.randn(5, 2))
        loss.backward()
        recorder.log(loss=float(loss.detach()), data=x, lr=0.01)
        optimizer.step()
    recorder.close()
    return path, model


def test_the_tensor_registry_parses_byte_exactly(recorded):
    """§5.1: `u32 count`, then per tensor name/dtype/ndim/dims/flags/component/region/tie."""
    path, model = recorded
    payload = next(iter(Reader(path).blocks(BlockType.TENSOR_REGISTRY))).payload

    count = struct.unpack_from("<I", payload, 0)[0]
    assert count == len(list(model.named_parameters(remove_duplicate=False)))

    offset = 4
    names = []
    for _ in range(count):
        length = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        names.append(payload[offset:offset + length].decode())
        offset += length
        offset += 1                                   # dtype
        ndim = struct.unpack_from("<I", payload, offset)[0]
        offset += 4 + ndim * 8                        # dims
        offset += 1                                   # flags
        for _ in range(2):                            # component, region
            length = struct.unpack_from("<H", payload, offset)[0]
            offset += 2 + length
        offset += 4                                   # tie group

    assert offset == len(payload), (
        f"the spec's layout consumed {offset} of {len(payload)} bytes - it does not describe "
        f"what the writer produced")
    assert names == [name for name, _ in model.named_parameters(remove_duplicate=False)]


def test_the_anchor_is_a_step_then_raw_tensor_bytes(recorded):
    """§5.3: `u64 step`, then tensor-major raw bytes in registry order."""
    path, model = recorded
    anchor = next(iter(Reader(path).blocks(BlockType.ANCHOR)))

    step = struct.unpack_from("<Q", anchor.payload, 0)[0]
    assert step == 0, "the first stored step is always an anchor, at step 0"

    expected = sum(parameter.numel() * 4 for parameter in model.parameters())
    assert len(anchor.payload) - 8 == expected, (
        "the anchor body must be exactly the raw float32 parameters")


def test_the_delta_has_a_prefix_the_spec_describes(recorded):
    """§5.2: `u64 step`, `u32 entry_count`, then per-entry encoding ids."""
    path, _model = recorded
    deltas = list(Reader(path).blocks(BlockType.DELTA))
    assert deltas, "a multi-step recording must contain deltas"

    for block in deltas:
        step = struct.unpack_from("<Q", block.payload, 0)[0]
        entries = struct.unpack_from("<I", block.payload, 8)[0]
        assert step == block.step, "the step is in the payload and in the block prefix"
        assert entries > 0
        # The first entry's encoding id must be one the spec lists.
        encoding = block.payload[12 + 4]
        assert encoding in (0, 1, 2, 3, 4), f"unknown encoding id {encoding}"


@pytest.mark.parametrize("kind", [BlockType.EVENT, BlockType.ANNOT, BlockType.INFO,
                                  BlockType.META, BlockType.DATA_SEGMENT])
def test_every_json_block_is_json_and_has_its_documented_keys(recorded, kind):
    """§5.2.3-5.2.6: these blocks are UTF-8 JSON, and the spec lists their keys."""
    path, _model = recorded
    blocks = list(Reader(path).blocks(kind))
    assert blocks, f"this recording should contain at least one {kind.name} block"

    required = {
        BlockType.EVENT: {"step", "loss", "grad_norm", "lr", "regions", "segment_id",
                          "seconds"},
        BlockType.ANNOT: {"event", "detail"},
        BlockType.INFO: {"event"},
        BlockType.META: {"library", "format", "adapter", "data_level", "tensors", "ties"},
        BlockType.DATA_SEGMENT: {"segment_id", "shard_id", "source", "fingerprint",
                                 "characteristics", "first_step", "last_step", "times_seen",
                                 "region_effects", "component_mask",
                                 "delta_reconstruction_error", "span_segments", "span_steps"},
    }[kind]
    for block in blocks:
        decoded = json.loads(block.payload)          # must be JSON, not a packed layout
        assert required <= set(decoded), (
            f"{kind.name} is missing {sorted(required - set(decoded))} that the spec lists")


def test_a_data_shard_is_a_json_header_line_then_bytes(recorded):
    """§5.2.1: the header is one JSON line, then the payload."""
    path, _model = recorded
    shards = list(Reader(path).blocks(BlockType.DATA_SHARD))
    assert shards, "data_level='segments' must produce at least one shard"

    for block in shards:
        header, _, body = block.payload.partition(b"\n")
        decoded = json.loads(header)
        assert {"shard_id", "codec", "raw_len", "content_crc32",
                "reconstruction_error"} <= set(decoded)
        assert len(body) == decoded["raw_len"] or decoded["codec"] != 0, (
            "a raw (uncompressed) shard's body must be exactly raw_len bytes")


def test_the_events_are_written_for_stored_steps_only(recorded):
    """§5.2.3 says one EVENT per stored step: the join depends on it."""
    path, _model = recorded
    timeline = Timeline(Reader(path))
    events = [json.loads(block.payload) for block in Reader(path).blocks(BlockType.EVENT)]
    # Step 0 is a pre-training anchor in the default full-history mode, not an event.
    expected = timeline.steps[1:] if Reader(path).meta().get("weight_history") == "every_optimizer_step" else timeline.steps
    assert [event["step"] for event in events] == expected


def test_a_capability_block_is_written_last(recorded):
    """§5.2.5: the tier is a result of the run, so it cannot be written at the start."""
    path, _model = recorded
    reader = Reader(path)
    info = [block for block in reader.blocks(BlockType.INFO)
            if json.loads(block.payload).get("event") == "capabilities"]
    assert len(info) == 1

    all_blocks = list(reader.blocks())
    assert all_blocks[-1].block_type != BlockType.ANCHOR
    capabilities_offset = info[0].offset
    assert capabilities_offset > max(
        block.offset for block in all_blocks if block.block_type == BlockType.DELTA), (
        "the capability block must come after the data it describes")


def test_a_reader_can_rebuild_the_spec_from_the_file_alone(recorded):
    """The whole point of publishing a spec: no access to this code, only the bytes."""
    path, _model = recorded
    reader = Reader(path)
    registry = next(iter(reader.blocks(BlockType.TENSOR_REGISTRY))).payload
    anchor = next(iter(reader.blocks(BlockType.ANCHOR))).payload

    count = struct.unpack_from("<I", registry, 0)[0]
    offset = 4
    shapes = []
    for _ in range(count):
        length = struct.unpack_from("<H", registry, offset)[0]
        offset += 2 + length + 1                      # name, dtype
        ndim = struct.unpack_from("<I", registry, offset)[0]
        offset += 4
        dims = [struct.unpack_from("<Q", registry, offset + 8 * i)[0] for i in range(ndim)]
        offset += 8 * ndim
        offset += 1                                   # flags
        for _ in range(2):
            length = struct.unpack_from("<H", registry, offset)[0]
            offset += 2 + length
        offset += 4                                   # tie group
        shapes.append(dims)

    # Parse the anchor from the bytes using only the shapes the registry gave us.
    cursor = 8
    for dims in shapes:
        count_values = int(np.prod(dims)) if dims else 1
        chunk = anchor[cursor:cursor + count_values * 4]
        assert len(chunk) == count_values * 4, "the anchor ran out of bytes"
        values = np.frombuffer(chunk, dtype="<f4")
        assert np.isfinite(values).all()
        cursor += count_values * 4
    assert cursor == len(anchor), "the anchor holds exactly the registry's tensors"

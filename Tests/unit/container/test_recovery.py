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

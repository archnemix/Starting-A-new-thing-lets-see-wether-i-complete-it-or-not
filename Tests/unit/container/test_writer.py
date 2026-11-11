"""Unit tests for the append-only writer: blocks land where declared, footer closes the file."""

from __future__ import annotations

import json

from archx.container.reader import Reader
from archx.container.schema import BLOCK, HEADER_SIZE, BlockType, Codec
from archx.container.writer import Writer


def test_blocks_roundtrip_and_footer_is_written(tmp_path):
    path = tmp_path / "run.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, json.dumps({"run": "demo"}).encode())
        offset = w.write_block(BlockType.DELTA, b"payload-1", step=1)
        assert offset > HEADER_SIZE

    reader = Reader(path)
    assert reader.header.closed is True   # closing sets the flag, footer proves it
    assert reader.container.footer is not None
    assert reader.meta() == {"run": "demo"}
    payloads = [b.payload for b in reader.blocks(BlockType.DELTA)]
    assert payloads == [b"payload-1"]
    index = Reader.index_block_payload(reader.container)
    assert [step for step, _o, _l in index] == [1]


def test_block_prefix_records_raw_and_stored_lengths(tmp_path):
    path = tmp_path / "run.archx.mdl"
    raw = b"a" * 4096
    with Writer(path, codec=Codec.ZSTD) as w:
        w.write_block(BlockType.META, raw)

    data = path.read_bytes()
    block_type, codec, _flags, _step, raw_len, stored_len, _pcrc, bcrc = BLOCK.unpack(
        data[HEADER_SIZE:HEADER_SIZE + 28]
    )
    assert block_type == int(BlockType.META)
    assert raw_len == 4096
    assert stored_len < raw_len  # zstd or zlib actually compressed it
    assert codec in (int(Codec.ZSTD), int(Codec.ZLIB))


def test_abort_leaves_a_readable_file_without_footer(tmp_path):
    path = tmp_path / "crashed.archx.mdl"
    w = Writer(path)
    w.write_block(BlockType.META, b'{"partial": true}')
    w.abort()

    reader = Reader(path, allow_torn=True)
    assert reader.container.footer is None
    assert reader.container.torn is False
    assert reader.meta() == {"partial": True}

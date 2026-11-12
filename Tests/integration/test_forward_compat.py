"""A v1.0 reader must read a v1.9 file, and say what it skipped.

The format spec's rule is: refuse a different *major* version, accept any *minor*, skip block
types you do not know. That rule is only worth anything if it is tested against a file this
build did not write, so these tests forge one: a header claiming a future minor, and a block
type (200) the build has never heard of.

Nothing here needs the future block to be understood. It needs to be *skipped*, and the reader
needs to be able to say so out loud instead of returning a shorter history and no explanation.
"""

from __future__ import annotations

import json
import struct

import pytest

from archx.container.reader import Reader
from archx.container.schema import (BLOCK, HEADER, HEADER_SIZE, Codec, Flags, MAGIC,
                                    zlib_crc32)
from archx.container.schema import BlockType
from archx.container.writer import Writer
from archx.errors import UnsupportedVersionError


def _write(base, *, minor: int, extra_block_type: int | None = None):
    """Write a small valid recording, then optionally rewrite its header minor version."""
    writer = Writer(str(base))
    writer.write_block(BlockType.META, json.dumps({"produced_by": "future"}).encode())
    if extra_block_type is not None:
        writer.write_block(extra_block_type, b"data this build has never seen")
    writer.write_block(BlockType.ANNOT, json.dumps({"step": 0, "event": "x"}).encode(), step=0)
    writer.close()

    base.write_bytes(_forge_header(base.read_bytes(), major=1, minor=minor))
    return base


def _forge_header(data: bytes, *, major: int, minor: int) -> bytes:
    """Rewrite the version fields, then recompute the header CRC over the new bytes.

    The CRC covers the version fields, so it has to be computed *after* they change - the
    first version of this test wrote the old CRC and every reader rejected the file, which is
    the reader being right and the forgery being wrong.
    """
    raw = bytearray(data)
    magic, _major, _minor, flags, header_len, _crc, created, hint, reserved = HEADER.unpack(
        bytes(raw[:HEADER_SIZE]))
    assert magic == MAGIC
    HEADER.pack_into(raw, 0, magic, major, minor, flags, header_len, 0, created, hint, reserved)
    crc = zlib_crc32(bytes(raw[:20]))
    HEADER.pack_into(raw, 0, magic, major, minor, flags, header_len, crc, created, hint, reserved)
    return bytes(raw)


def test_a_newer_minor_version_is_read_and_the_skip_is_reported(tmp_path):
    path = _write(tmp_path / "future.archx.mdl", minor=9, extra_block_type=200)

    reader = Reader(path)
    assert reader.header.format_minor == 9
    assert reader.unknown_blocks() == [200]

    # The file still reads: the known blocks are all there, in order.
    annotations = [json.loads(b.payload) for b in reader.blocks(BlockType.ANNOT)]
    assert annotations == [{"step": 0, "event": "x"}]
    assert reader.meta() == {"produced_by": "future"}

    report = reader.read_report(format_minor=reader.header.format_minor)
    assert report["usable"] is True
    assert report["unknown_block_types"] == [200]
    assert any("newer than this build" in w for w in report["warnings"])
    assert any("skipped block types [200]" in w for w in report["warnings"])


def test_a_current_file_reports_no_warnings(tmp_path):
    path = _write(tmp_path / "now.archx.mdl", minor=0)
    reader = Reader(path)
    assert reader.read_report(format_minor=0)["warnings"] == []
    assert reader.unknown_blocks() == []


def test_a_different_major_version_is_refused_with_a_clear_message(tmp_path):
    path = _write(tmp_path / "v2.archx.mdl", minor=0)
    path.write_bytes(_forge_header(path.read_bytes(), major=2, minor=0))

    with pytest.raises(UnsupportedVersionError) as caught:
        Reader(path)
    message = str(caught.value)
    assert "2.0" in message and "1.x" in message, message


def test_the_cli_info_says_the_file_is_newer_than_the_reader(tmp_path, capsys):
    from archx.cli.main import main

    path = _write(tmp_path / "future.archx.mdl", minor=9, extra_block_type=200)
    assert main(["info", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "1.9" in printed
    assert "skipped block types [200]" in printed

    assert main(["info", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["format"] == "1.9"
    assert payload["builder_format"].startswith("1.")
    assert payload["unknown_block_types"] == [200]
    assert payload["warnings"]

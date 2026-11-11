"""Integration: writer -> crash -> recovery -> reader -> CLI, as one flow.

Unit tests cover each file; this proves they still agree when wired together, including a
real killed process rather than a simulated truncation.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from archx.cli.main import main
from archx.container.reader import Reader
from archx.container.recovery import recover_in_place
from archx.container.schema import BlockType
from archx.container.writer import Writer


def test_write_read_recover_read_roundtrip(tmp_path):
    path = tmp_path / "run.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, json.dumps({"adapter": "torch", "capability_tier": 1}).encode())
        for step in range(5):
            w.write_block(BlockType.ANCHOR if step == 0 else BlockType.DELTA,
                          json.dumps({"step": step}).encode(), step=step)

    first = Reader(path)
    assert first.meta()["adapter"] == "torch"
    assert len(Reader.index_block_payload(first.container)) == 5

    # Simulate a crash: drop the footer, the index and part of the last delta.
    data = path.read_bytes()
    # 51 bytes = whole 56-byte footer plus 5 stray bytes past the last block, so the
    # truncation is unambiguous instead of landing inside a header where the CRC could
    # mask the tear. A real SIGKILL is covered by the next test.
    path.write_bytes(data[:-51])

    report = recover_in_place(path)
    assert report.recovered is True

    second = Reader(path)
    assert second.header.recovered is True
    assert second.container.footer is not None
    steps = [b.step for b in second.blocks(BlockType.DELTA)]
    # The anchor survives; the deltas that survived are a prefix of what was written,
    # and whatever the tear removed is gone rather than invented.
    assert [b.step for b in second.blocks(BlockType.ANCHOR)] == [0]
    assert steps == list(range(1, len(steps) + 1))
    assert len(steps) <= 4


def test_process_killed_mid_run_leaves_a_recoverable_file(tmp_path):
    path = tmp_path / "killed.archx.mdl"
    script = textwrap.dedent(
        f"""
        import json, os, signal
        from archx.container.writer import Writer
        from archx.container.schema import BlockType

        w = Writer({str(path)!r}, fsync_every_bytes=1)
        w.write_block(BlockType.META, b'{{"adapter": "torch"}}')
        for step in range(50):
            w.write_block(BlockType.DELTA, json.dumps({{"step": step}}).encode(), step=step)
            w._fh.flush()
        os.kill(os.getpid(), signal.SIGKILL)  # die without a footer
        """
    )
    subprocess.run([sys.executable, "-c", script], check=False, timeout=120)
    assert path.exists()

    crashed = Reader(path)
    assert crashed.container.footer is None  # no footer: this file was never closed
    assert crashed.container.torn is False   # but every whole block is readable

    recover_in_place(path)
    reader = Reader(path)
    assert reader.container.footer is not None
    assert reader.meta() == {"adapter": "torch"}
    assert len(list(reader.blocks(BlockType.DELTA))) >= 40


def test_cli_info_reports_what_the_writer_wrote(tmp_path, capsys):
    path = tmp_path / "run.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, json.dumps({"adapter": "launcher", "data_level": "full"}).encode())
    assert main(["info", str(path)]) == 0
    out = capsys.readouterr().out
    assert "launcher" in out
    assert "closed         True" in out

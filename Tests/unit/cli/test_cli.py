"""Unit tests for the CLI surface: info and recover work through the real entry point."""

from __future__ import annotations

import json

import pytest

from archx.cli.main import main
from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.writer import Writer


def _recording(tmp_path):
    path = tmp_path / "run.archx.mdl"
    with Writer(path) as w:
        w.write_block(BlockType.META, json.dumps({"adapter": "torch", "capability_tier": 1,
                                                  "data_level": "segments",
                                                  "attribution": "observational"}).encode())
        w.write_block(BlockType.ANCHOR, b"\x00" * 16, step=0)
    return path


def test_version_and_help_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert "archx" in capsys.readouterr().out

    assert main([]) == 0
    assert "info" in capsys.readouterr().out


def test_info_json_is_machine_readable(tmp_path, capsys):
    path = _recording(tmp_path)
    assert main(["info", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["closed"] is True
    assert payload["format"] == "1.0"
    assert payload["meta"]["adapter"] == "torch"


def test_info_on_a_broken_file_fails_with_a_message(tmp_path, capsys):
    bad = tmp_path / "bad.archx.mdl"
    bad.write_bytes(b"not a container")
    assert main(["info", str(bad)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_recover_through_the_cli_closes_a_crashed_file(tmp_path, capsys):
    path = tmp_path / "crashed.archx.mdl"
    w = Writer(path)
    w.write_block(BlockType.META, b'{"run": "x"}')
    w.write_block(BlockType.DELTA, b"abc", step=0)
    w.abort()
    path.write_bytes(path.read_bytes()[:-3])

    assert main(["recover", str(path)]) == 0
    assert "footer" in capsys.readouterr().out
    assert Reader(path).container.footer is not None

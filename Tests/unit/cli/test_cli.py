"""Unit tests for the CLI surface: info and recover work through the real entry point."""

from __future__ import annotations

import json

import numpy as np
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


class _Toy:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.params = {"w": rng.normal(0, 0.1, (8, 8)).astype(np.float32)}
        self.regions = {"w": "attention"}

    def registry(self):
        from archx.formats.registry import TensorRegistry

        reg = TensorRegistry()
        for key, array in self.params.items():
            reg.add(key, array, pointer=array.ctypes.data, region=self.regions[key])
        return reg

    def read(self, key):
        return self.params[key]


def _make_run(tmp_path, steps=3):
    from archx.policy import PhasePolicy
    from archx.recorder import Recorder

    model = _Toy()
    path = tmp_path / "run.archx.mdl"
    recorder = Recorder(model.registry(), model.read, str(path),
                        policy=PhasePolicy(total_steps=steps, min_warmup_steps=steps))
    rng = np.random.default_rng(2)
    live = []
    for step in range(steps):
        model.params["w"] += rng.normal(0, 1e-3, (8, 8)).astype(np.float32)
        recorder.step(step, loss=1.0 / (step + 1), data=b"batch text " * 4)
        live.append(model.params["w"].copy())
        if step == 1:
            recorder.annotate(1, "stage2_start", "unfroze")
    recorder.close()
    return path, live


def test_reconstruct_writes_weights_that_match_the_live_run(tmp_path, capsys):
    from pathlib import Path

    path, live = _make_run(tmp_path)
    out = tmp_path / "weights.npz"
    assert main(["reconstruct", str(path), "--step", "2", "--out", str(out)]) == 0

    saved = np.load(Path(out))
    assert np.allclose(saved["w"], live[2], atol=1e-5)
    assert "written" in capsys.readouterr().out


def test_reconstruct_defaults_to_the_last_step_and_refuses_unknown_steps(tmp_path, capsys):
    path, _live = _make_run(tmp_path, steps=2)
    out = tmp_path / "last.npz"
    assert main(["reconstruct", str(path), "--out", str(out)]) == 0
    assert main(["reconstruct", str(path), "--step", "99"]) == 1
    assert "not recorded" in capsys.readouterr().err


def test_events_lists_the_annotation_and_the_segment(tmp_path, capsys):
    path, _live = _make_run(tmp_path)
    assert main(["events", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(e.get("event") == "stage2_start" for e in payload["annotations"])
    assert payload["segments"]
    assert payload["observations"], "a stored step must show up as an observation"

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


def _record_two(tmp_path, *, lr=1e-3, seed=0, name="run.archx.mdl"):
    """A torch run, or None when torch is not installed."""
    torch = pytest.importorskip("torch", reason="torch is not installed on this runner")
    import archx

    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(4, 8, generator=generator)
    y = torch.randint(0, 3, (4,), generator=generator)
    path = tmp_path / name
    recorder = archx.attach(model, optimizer, path=str(path))
    for _ in range(12):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        recorder.log(loss=float(loss.item()), data=x)
        optimizer.step()
    recorder.close()
    return path


def test_diff_between_two_steps_names_the_tensors_that_moved(tmp_path, capsys):
    from archx.cli.main import main

    path = _record_two(tmp_path)
    assert main(["diff", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "comparing" in printed and "relative" in printed
    assert "weight" in printed, "the tensors are named, not summarised away"

    assert main(["diff", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["step_a"] < payload["step_b"]
    assert payload["tensors"] and payload["tensors"][0]["moved"] > 0
    assert payload["total_moved"] > 0


def test_diff_refuses_a_step_the_file_does_not_have(tmp_path, capsys):
    from archx.cli.main import main

    path = _record_two(tmp_path)
    assert main(["diff", str(path), "--step", "10000"]) == 1
    message = capsys.readouterr().err
    assert "not stored" in message
    assert "cannot be rebuilt" in message


def test_two_identical_runs_diff_as_identical(tmp_path, capsys):
    """The strongest thing `diff` can say, and it must be able to say it."""
    from archx.cli.main import main

    first = _record_two(tmp_path, name="a.archx.mdl")
    second = _record_two(tmp_path, name="b.archx.mdl")
    assert main(["diff", str(first), "--other", str(second)]) == 0
    assert "identical at every common step" in capsys.readouterr().out


def test_diff_between_two_runs_finds_the_first_disagreement(tmp_path, capsys):
    from archx.cli.main import main

    first = _record_two(tmp_path, name="slow.archx.mdl", lr=1e-3)
    second = _record_two(tmp_path, name="fast.archx.mdl", lr=5e-2)
    assert main(["diff", str(first), "--other", str(second), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["first_difference"] is not None
    # The learning rate changes the very first update, and step 0's anchor is written after
    # that update, so the disagreement is at step 0 - not "somewhere later".
    assert payload["first_difference"]["step"] == payload["differences"][0]["step"]


def test_run_says_how_to_use_it_without_running_anything(capsys):
    """`archx run --help` must not fall through to the child: there is no child."""
    from archx.cli.main import main

    assert main(["run", "--help"]) == 0
    output = capsys.readouterr().out
    assert "<command>" in output
    assert "--recording" in output


def test_run_refuses_a_stray_argument_instead_of_guessing():
    """Anything not after `--` is archx's own flag; guessing would run the wrong command."""
    from archx.cli.main import main

    assert main(["run", "python", "train.py"]) == 2


def test_run_needs_a_command():
    """`archx run --record x --` with nothing after it is an error, not a silent no-op."""
    from archx.cli.main import main

    assert main(["run", "--"]) == 2

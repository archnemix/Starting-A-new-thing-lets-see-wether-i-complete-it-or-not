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


def _record_with_gaps(tmp_path, steps=6):
    """A recording that stores some steps and skips the rest, so gaps are a fact.

    Every other fixture here stores every step (short runs stay in warmup), which is why the
    gap tests need their own: what the product promises about a skipped step is the whole
    point of the format, and a fixture with no skipped step proves nothing about it.
    """
    from archx.policy import PhasePolicy, Triggers
    from archx.recorder import Recorder

    model = _Toy()
    path = tmp_path / "gaps.archx.mdl"
    recorder = Recorder(model.registry(), model.read, str(path),
                        policy=PhasePolicy(total_steps=steps, min_warmup_steps=2,
                                           heartbeat_steps=2,
                                           triggers=Triggers(region_move_fraction=1e9,
                                                             loss_drop_fraction=1e9,
                                                             grad_norm_zscore=1e9)))
    for step in range(steps):
        model.params["w"] += np.float32(0.01) * step
        recorder.step(step, loss=1.0 / (step + 1), data=b"batch text " * 4)
    recorder.close()

    from archx.container.reader import Reader
    from archx.container.values import Timeline

    stored = Timeline(Reader(path)).steps
    assert [s for s in range(stored[0], stored[-1] + 1) if s not in stored], (
        f"the policy stored every step ({stored}); the fixture is not exercising a gap")
    return path


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


def test_reconstruct_defaults_to_the_last_step_and_refuses_steps_outside_the_file(
        tmp_path, capsys):
    path, _live = _make_run(tmp_path, steps=2)
    out = tmp_path / "last.npz"
    assert main(["reconstruct", str(path), "--out", str(out)]) == 0
    assert main(["reconstruct", str(path), "--step", "99"]) == 1
    error = capsys.readouterr().err
    assert "outside it" in error and "0-1" in error, error


def test_reconstruct_answers_a_step_between_two_store_points(tmp_path, capsys):
    """The format's headline promise, reached through the CLI rather than the library.

    `archx reconstruct --step 2` on a 20-step 1M recording exited 1 with "step 2 is not
    recorded" while `Timeline.reconstruct(2)` returned all 1,004,176 values: a command that
    refused the one thing the product exists to do. A store point is the exception, not the
    rule - the policy skips most steps in the middle of a run.
    """
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    path = _record_with_gaps(tmp_path)
    stored = Timeline(Reader(path)).steps
    target = max(s for s in range(stored[0], stored[-1] + 1) if s not in stored)
    assert main(["reconstruct", str(path), "--step", str(target),
                 "--out", str(tmp_path / "gap.npz")]) == 0
    printed = capsys.readouterr().out
    assert "stored         no" in printed
    assert f"anchor at step {max(s for s in stored if s <= target)}" in printed


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
    assert "outside the recorded range" in message
    assert "stored steps are" in message
    # It must not claim the step cannot be rebuilt: `archx reconstruct` does exactly that.
    assert "cannot be rebuilt" not in message


def test_diff_at_an_unstored_step_points_at_reconstruct_instead_of_lying(tmp_path, capsys):
    """A gap step has weights, but no recorded *movement* to compare.

    The old refusal said the step "cannot be rebuilt from the file", which is false for every
    step between two store points - and the same binary answers those through
    `archx reconstruct`. So the refusal names the real limitation and the command that helps.
    """
    from archx.cli.main import main
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    path = _record_with_gaps(tmp_path)
    stored = Timeline(Reader(path)).steps
    gap = [s for s in range(stored[0], stored[-1] + 1) if s not in stored]
    assert main(["diff", str(path), "--step", str(gap[0])]) == 1
    message = capsys.readouterr().err
    assert "not a store point" in message
    assert "archx reconstruct" in message


def test_diff_between_files_that_share_no_tensor_names_says_so(tmp_path, capsys):
    """Two adapters, one step numbering, no shared tensor name.

    `diff --other` compared only names present in both files, found nothing to disagree on,
    and printed "identical at every common step" - its strongest verdict - having compared
    zero values. A launcher recording (param[0][0]) against an attach recording
    (embed.weight) hits this every time, and the verdict was the opposite of the truth: the
    command had answered nothing at all.
    """
    from archx.cli.main import main

    named = _record_two(tmp_path, name="named.archx.mdl")      # 0.weight, 0.bias, ...
    bare, _live = _make_run(tmp_path)                          # one tensor, named "w"
    assert main(["diff", str(named), "--other", str(bare), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["common_steps"] > 0, "the test needs shared step numbers to mean anything"
    assert payload["shared_tensors"] == 0
    assert payload["compared_values"] == 0
    assert payload["first_difference"] is None

    assert main(["diff", str(named), "--other", str(bare)]) == 0
    printed = capsys.readouterr().out
    assert "cannot compare" in printed
    assert "identical at every common step" not in printed


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


def test_events_says_when_a_file_has_no_region_names(tmp_path, capsys):
    """The launcher cannot see module names, so every tensor registers with region "".

    The table printed `moved ` - an empty label, which reads as a formatting glitch instead of
    the honest answer, that this recording has no region information at all.
    """
    from archx.policy import PhasePolicy
    from archx.recorder import Recorder

    class _Bare:
        """No region at all: what the launcher's hook produces."""

        def __init__(self):
            self.params = {"param[0][0]": np.zeros((4, 4), dtype=np.float32)}

        def registry(self):
            from archx.formats.registry import TensorRegistry

            reg = TensorRegistry()
            for key, array in self.params.items():
                reg.add(key, array)
            return reg

        def read(self, key):
            return self.params[key]

    model = _Bare()
    path = tmp_path / "bare.archx.mdl"
    recorder = Recorder(model.registry(), model.read, str(path),
                        policy=PhasePolicy(total_steps=3, min_warmup_steps=3))
    for step in range(3):
        model.params["param[0][0]"] += np.float32(0.01)
        recorder.step(step, loss=1.0 / (step + 1), grad_norm=2.0, data=b"batch " * 8)
    recorder.close()

    assert main(["events", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "no region names in this file" in printed
    # An empty `moved ` column is what a formatting bug looks like; it must not be there.
    for line in printed.splitlines():
        assert not line.rstrip().endswith("moved"), line


def test_diff_says_how_many_recorded_updates_it_folded_together(tmp_path, capsys):
    """Eight intervals is not "8 stored steps between" step 0 and step 19.

    `moved` is the sum of every recorded update in the span, and the old label counted the
    intervals while calling them steps. Both numbers are true; only one of them is what the
    moved column contains, and the label named the other one.
    """
    path = _record_with_gaps(tmp_path)
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    steps = Timeline(Reader(path)).steps
    assert main(["diff", str(path), "--step", str(steps[0]), "--step-b", str(steps[-1]),
                 "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["updates_folded"] == len(steps) - 1, steps
    assert payload["steps_between"] == len(steps) - 2, steps

    assert main(["diff", str(path), "--step", str(steps[0]), "--step-b", str(steps[-1])]) == 0
    printed = capsys.readouterr().out
    assert f"({len(steps) - 1} recorded update(s) folded into the difference)" in printed
    assert "stored step(s) in the file between them" not in printed

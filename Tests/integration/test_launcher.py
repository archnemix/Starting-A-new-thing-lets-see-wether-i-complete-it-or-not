"""Integration: `archx run` records a script it was never told anything about.

The launcher injects `sitecustomize` through `PYTHONPATH` and patches torch's optimizers, so
these tests run real subprocesses. Three of them exist because the first prototype failed in
ways that looked like success:

* a `sitecustomize` that does `import torch` at start-up silently does nothing, because torch
  is not importable yet;
* patching `Optimizer.step` alone counts **zero** steps for `AdamW`, which defines no `step`
  of its own;
* a hook with a syntax error makes CPython print "Error in sitecustomize" and carry on, so
  every test on the *recording* would pass while nothing was recorded.

The last one is why `build_env` compiles the injected file before writing it, and why
`test_the_injected_hook_always_compiles` exists rather than being taken on trust.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

from archx import launcher
from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.values import Timeline

pytest.importorskip("torch", reason="torch is not installed on this runner")

SCRIPT = textwrap.dedent(
    """
    import torch
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x, y = torch.randn(4, 8), torch.randn(4, 2)
    for _ in range(6):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), y).backward()
        optimizer.step()
    """
)


def write_script(tmp_path, body=SCRIPT, name="train.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def run_script(tmp_path, body=SCRIPT, name="train.py"):
    script = write_script(tmp_path, body, name)
    recording = tmp_path / "run.archx.mdl"
    code = launcher.run([sys.executable, str(script)], recording)
    return code, recording


def test_a_script_that_was_never_touched_is_recorded(tmp_path):
    """The whole point: no import of archx, no changed loop, a recording at the end."""
    code, recording = run_script(tmp_path)
    assert code == 0
    assert recording.exists()

    timeline = Timeline(Reader(recording))
    assert timeline.steps == [0, 1, 2, 3, 4, 5], "one store point per optimizer step"


def test_adamw_is_seen_even_though_it_defines_no_step_of_its_own(tmp_path):
    """Measured failure: wrapping only `Optimizer.step` counted zero `AdamW` steps."""
    _code, recording = run_script(tmp_path)
    meta = json.loads(next(iter(Reader(recording).blocks(BlockType.META))).payload)
    assert meta["optimizer"] == "AdamW"
    assert len(Timeline(Reader(recording)).steps) == 6


def test_the_recording_rebuilds_the_weights_the_run_ended_with(tmp_path):
    """The numbers in the file must be the run's own, not a plausible-looking set."""
    body = SCRIPT + textwrap.dedent(
        """
        import json, os
        print("FINAL_WEIGHTS " + json.dumps(
            {k: v.detach().tolist() for k, v in model.named_parameters()}))
        """
    )
    script = write_script(tmp_path, body)
    recording = tmp_path / "run.archx.mdl"
    completed = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                               env=launcher.build_env(recording)[0])
    output = completed.stdout
    assert completed.returncode == 0

    timeline = Timeline(Reader(recording))
    rebuilt = timeline.reconstruct(timeline.steps[-1])
    assert len(rebuilt) == 4, "two Linear layers: a weight and a bias each"

    # The recording must hold the numbers the run ended with, not just the same shapes.
    reported = json.loads(output.split("FINAL_WEIGHTS ", 1)[1].strip())
    by_shape = sorted(rebuilt.values(), key=lambda a: (a.ndim, a.size))
    expected = sorted((torch.tensor(v).numpy() for v in reported.values()),
                      key=lambda a: (a.ndim, a.size))
    for got, want in zip(by_shape, expected):
        assert np.allclose(got, want, atol=1e-4), "the recorded weights are not the run's"

    # The launcher cannot know names, so it must say so rather than invent them.
    meta = json.loads(next(iter(Reader(recording).blocks(BlockType.META))).payload)
    assert meta["names"] == "unavailable"
    assert all(key.startswith("param[") for key in rebuilt)


def test_the_file_states_the_tier_it_reached_and_every_gap(tmp_path):
    """A launcher sees an optimizer, not a loop: tier 0, with each missing field named."""
    _code, recording = run_script(tmp_path)
    capability = next(
        json.loads(block.payload)
        for block in Reader(recording).blocks(BlockType.INFO)
        if json.loads(block.payload)["event"] == "capabilities")

    assert capability["capability_tier"] == 0
    for field in ("loss", "grad_norms", "data_segments", "activations"):
        assert field in capability["capabilities_missing"], field
    assert "activations" in capability["capabilities_missing"]


def test_a_script_with_no_optimizer_step_fails_loudly_and_leaves_nothing(tmp_path):
    """The honest failure: no file, non-zero, and instructions instead of an empty recording."""
    script = write_script(tmp_path, "print('I do not train anything')\n", "idle.py")
    recording = tmp_path / "idle.archx.mdl"

    with pytest.raises(launcher.LauncherError) as caught:
        launcher.run([sys.executable, str(script)], recording)

    message = str(caught.value)
    assert "archx.attach" in message, "the error must name a way forward"
    assert not recording.exists(), "an empty recording must never be left behind"


def test_an_existing_recording_is_never_overwritten(tmp_path):
    script = write_script(tmp_path)
    recording = tmp_path / "existing.archx.mdl"
    recording.write_bytes(b"do not lose this")

    with pytest.raises(launcher.LauncherError, match="already exists"):
        launcher.run([sys.executable, str(script)], recording)
    assert recording.read_bytes() == b"do not lose this"


def test_a_failing_command_still_reports_its_own_exit_code(tmp_path):
    """The launcher must not swallow the training script's failure."""
    script = write_script(tmp_path, SCRIPT + "\nraise SystemExit(3)\n", "fails.py")
    recording = tmp_path / "fails.archx.mdl"
    code = launcher.run([sys.executable, str(script)], recording)
    assert code == 3
    assert recording.exists(), "steps were recorded before the failure; keep them"


def test_the_injected_hook_always_compiles(tmp_path):
    """A broken sitecustomize fails at interpreter start-up and is silently ignored.

    CPython prints "Error in sitecustomize; set PYTHONVERBOSE for traceback" and then runs
    the script as if nothing happened. A marker line that was not a comment did exactly that,
    so the compile check is asserted here rather than trusted.
    """
    _env, directory = launcher.build_env(tmp_path / "x.mdl")
    try:
        source = (directory / launcher.HOOK_FILENAME).read_text(encoding="utf-8")
        compile(source, launcher.HOOK_FILENAME, "exec")
        assert source.startswith("#"), "the marker must be a comment"
    finally:
        launcher.cleanup(directory)
    assert not directory.exists(), "the temporary hook directory must not be left behind"


def test_the_hook_directory_does_not_leak_into_the_environment(tmp_path, monkeypatch):
    """The launcher's own interpreter must not patch itself, and PYTHONPATH must be prepended."""
    monkeypatch.setenv("ARCHX_LAUNCHER_ACTIVE", "1")
    env, directory = launcher.build_env(tmp_path / "y.mdl", base_env=dict(__import__("os").environ))
    try:
        assert "ARCHX_LAUNCHER_ACTIVE" not in env, "the child must not inherit the guard flag"
        assert env["PYTHONPATH"].startswith(str(directory)), (
            "the hook must come before site-packages, or torch's own modules win")
        assert env["ARCHX_LAUNCHER_PATH"].endswith("y.mdl")
    finally:
        launcher.cleanup(directory)


def test_the_launcher_reports_what_it_produced(tmp_path, capsys):
    """`main` prints the count and the honest caveat, and returns the child's status."""
    script = write_script(tmp_path, SCRIPT, "reported.py")
    recording = tmp_path / "reported.archx.mdl"

    code = launcher.main(["--", sys.executable, str(script)], recording=str(recording))

    assert code == 0
    output = capsys.readouterr().out
    assert "recorded 6 steps" in output
    assert "no loss" in output, "the caveat must be printed, not implied"

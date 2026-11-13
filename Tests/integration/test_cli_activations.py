"""The two CLI additions for tier 3: `archx demo` and `archx activations`.

Both are tested through the real entry point, because the contract a user has is the command
line and the exit code - not the Python function behind it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Through `python -m archx`, which is also the documented first thing to try."""
    return subprocess.run([sys.executable, "-m", "archx", *args],
                          capture_output=True, text=True, timeout=300, check=False)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """One real recording for the whole module: `archx demo` is the command under test."""
    directory = tmp_path_factory.mktemp("cli-demo")
    out = directory / "demo.archx.mdl"
    result = cli("demo", "--out", str(out), "--steps", "8")
    assert result.returncode == 0, result.stderr
    assert out.exists()
    return out


def test_demo_produces_a_file_and_says_what_to_type_next(tmp_path):
    out = tmp_path / "made.archx.mdl"
    result = cli("demo", "--out", str(out), "--steps", "4")

    assert result.returncode == 0, result.stderr
    assert out.stat().st_size > 0
    for command in ("archx info", "archx events", "archx activations"):
        assert command in result.stdout, f"the demo should point at `{command}`"


def test_demo_refuses_to_overwrite_an_existing_recording(tmp_path):
    out = tmp_path / "taken.archx.mdl"
    out.write_bytes(b"a recording the user already had")
    result = cli("demo", "--out", str(out), "--steps", "2")

    assert result.returncode == 1
    assert "already exists" in result.stderr
    assert out.read_bytes() == b"a recording the user already had", "it was not overwritten"


def test_activations_prints_one_row_per_layer_per_stored_step(recorded):
    result = cli("activations", str(recorded))
    assert result.returncode == 0, result.stderr
    assert "summaries" in result.stdout
    assert "zero%" in result.stdout and "sat%" in result.stdout


def test_activations_json_is_machine_readable_and_has_every_documented_field(recorded):
    result = cli("activations", str(recorded), "--json")
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert rows
    for row in rows:
        for key in ("step", "layer", "shape", "mean", "std", "min", "max",
                    "zero_fraction", "saturated_fraction", "count", "forwards"):
            assert key in row, f"{key} is missing from an activation summary"


def test_activations_can_be_filtered_to_a_layer_and_a_step(recorded):
    everything = json.loads(cli("activations", str(recorded), "--json").stdout)
    layer = everything[0]["layer"]
    step = everything[0]["step"]

    only_layer = json.loads(cli("activations", str(recorded), "--layer", layer,
                                "--json").stdout)
    assert only_layer and all(row["layer"] == layer for row in only_layer)

    only_step = json.loads(cli("activations", str(recorded), "--step", str(step),
                               "--json").stdout)
    assert only_step and all(row["step"] == step for row in only_step)


def test_a_file_without_tier_three_explains_itself_instead_of_printing_nothing(tmp_path):
    """Silence would be indistinguishable from a bug; the file knows why it has none."""
    import archx
    from archx.policy import PhasePolicy

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    path = tmp_path / "plain.archx.mdl"
    recorder = archx.attach(model, optimizer, str(path), data_level="none",
                            policy=PhasePolicy(total_steps=2, min_warmup_steps=1))
    optimizer.zero_grad()
    loss = model(torch.randn(4, 4)).sum()
    loss.backward()
    recorder.log(loss=float(loss.detach()))
    optimizer.step()
    recorder.close()

    result = cli("activations", str(path))
    assert result.returncode == 0, "no activations is not an error"
    assert "no activation summaries" in result.stdout
    assert "activations=True" in result.stdout, "it should say how to get them"

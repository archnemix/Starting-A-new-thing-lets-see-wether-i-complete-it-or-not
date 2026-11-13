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


def test_info_reports_the_achieved_tier_not_the_floor_in_meta(recorded):
    """`archx info` printed META's `capability_tier`, which is 0 by construction.

    META is written at open, before a step has happened, and carries the floor tier on purpose;
    the achieved tier is written in the capabilities block at close. The command therefore
    reported `capability_tier 0` while listing every recorded field on the next line - a file
    contradicting itself, in the first command anyone runs.
    """
    from archx.container.reader import Reader

    result = cli("info", str(recorded))
    assert result.returncode == 0, result.stderr
    achieved = Reader(str(recorded)).capabilities()["capability_tier"]
    assert achieved == 3, "this fixture records with activations=True"

    reported = [line for line in result.stdout.splitlines()
                if line.startswith("capability_tier")]
    assert reported, "info must state a tier"
    assert reported[0].split()[-1] == "3", (
        f"info said {reported[0].strip()!r} for a tier-{achieved} recording")


def test_an_unclosed_file_claims_no_tier_at_all(tmp_path):
    """A file with no capabilities block must not invent a tier.

    Measured while writing this: a recorder that is never closed leaves a 40-byte header and
    no blocks, because the writer buffers. So there is nothing to fall back to - and *that* is
    the correct answer. Printing `capability_tier 0` from a META block that does not exist, or
    from a default, would be the same class of lie this file is about.
    """
    import archx
    from archx.policy import PhasePolicy

    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    path = tmp_path / "unclosed.archx.mdl"
    recorder = archx.attach(model, optimizer, str(path), data_level="none",
                            policy=PhasePolicy(total_steps=4, min_warmup_steps=1))
    optimizer.zero_grad()
    loss = model(torch.randn(4, 4)).sum()
    loss.backward()
    recorder.log(loss=float(loss.detach()))
    optimizer.step()
    # Deliberately no close(): the capabilities block is never written.
    result = cli("info", str(path))
    assert result.returncode == 0, result.stderr
    assert "closed         no" in result.stdout
    assert "archx recover" in result.stdout, "it should say what to do about it"
    assert not [l for l in result.stdout.splitlines() if l.startswith("capability_tier")], (
        "no tier may be reported for a run that recorded none")
    recorder.close()       # let the writer thread finish so the test leaves no daemon behind


def test_a_tier_three_file_says_its_capture_is_on(recorded):
    """META said `activation_capture: false` in every tier-3 recording ever written.

    `ActivationCapture.__bool__` is "has anything been captured yet", which is false at attach
    - before a single forward pass - and META is written at attach, so the header recorded the
    wrong fact and then contradicted the body it belonged to. Nothing read the field, which is
    how it stayed wrong through every release.

    The fix is not a better guess at attach time: whether summaries arrived is unknown until
    the run ends. The field is no longer written, and `Reader.activation_capture()` answers
    from the blocks - so every file, old or new, gives one answer and it is the true one.
    """
    from archx.container.reader import Reader

    reader = Reader(recorded)
    assert "activation_capture" not in reader.meta(), reader.meta()
    assert reader.activation_capture() is True
    assert reader.capabilities()["capability_tier"] == 3

    result = cli("info", str(recorded))
    assert result.returncode == 0, result.stderr
    assert "activation_capture yes" in result.stdout
    assert "warning" not in result.stdout, (
        "a file written by this build must not warn about its own META block")


def test_info_flags_the_contradiction_in_a_file_written_before_the_fix(tmp_path):
    """Old files keep their wrong META field, and `info` names it instead of repeating it.

    The file is made the way the old recorder made it: a real tier-3 run whose META carries
    `activation_capture: false`. Nothing is patched by hand, so the bytes are exactly what a
    user upgrading from an earlier version would have on disk.
    """
    import archx

    torch.manual_seed(0)
    # Sequential, not a bare module: the activation hooks attach to named submodules, and the
    # root module is skipped by design ("the root is not a layer").
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    path = tmp_path / "old.archx.mdl"
    recorder = archx.attach(model, optimizer, str(path), activations=True,
                            meta={"activation_capture": False})
    generator = torch.Generator().manual_seed(0)
    for _ in range(4):
        x = torch.randn(8, 8, generator=generator)
        y = torch.randint(0, 4, (8,), generator=generator)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        recorder.log(loss=float(loss.detach()), data=x)
        optimizer.step()
    recorder.close()

    from archx.container.reader import Reader

    assert Reader(path).meta()["activation_capture"] is False, "the fixture must state false"
    assert Reader(path).activation_capture() is True, "the body contradicts the header"

    result = cli("info", str(path))
    assert result.returncode == 0, result.stderr
    assert "activation_capture yes" in result.stdout, result.stdout
    assert "META says activation_capture false" in result.stdout

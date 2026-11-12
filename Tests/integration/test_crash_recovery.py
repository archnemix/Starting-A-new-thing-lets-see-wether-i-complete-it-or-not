"""Integration: kill the process mid-training, then read the history back out of the wreck.

The other recovery tests tear bytes on purpose. This one does what a real run does when the
machine dies: a child process trains and records, the parent sends SIGKILL at an arbitrary
moment, and the file must still answer "what happened before that point" - no footer, no
graceful close, no chance to flush.

Windows has no SIGKILL; the child is terminated there and the same invariant must hold, which
is why the assertion is about the *prefix* of history rather than about the last step.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from archx.container.reader import Reader
from archx.container.recovery import recover_in_place
from archx.container.schema import BlockType
from archx.container.values import Timeline
from archx.errors import ArchxError

#: How much history must exist before the kill, so the test is about recovery and not about
#: how fast a particular runner starts a child process.
_MIN_STEPS_BEFORE_KILL = 3

_TRAINER = """
import sys
import numpy as np
from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy
from archx.recorder import Recorder

path = sys.argv[1]
rng = np.random.default_rng(0)
params = {"layer.weight": rng.normal(0, 0.1, (32, 32)).astype(np.float32)}
registry = TensorRegistry()
registry.add("layer.weight", params["layer.weight"], pointer=params["layer.weight"].ctypes.data)
recorder = Recorder(registry, params.__getitem__, path,
                    policy=PhasePolicy(total_steps=100000, min_warmup_steps=100000))
print("ready", flush=True)
step = 0
while True:                       # the loop a real run would be inside when the machine dies
    params["layer.weight"] += rng.normal(0, 1e-3, (32, 32)).astype(np.float32)
    recorder.step(step, loss=1.0 / (step + 1), data=b"batch %d\\n" % (step % 4))
    step += 1
"""


def _kill_mid_run(directory) -> tuple[int, str, "Path"]:
    """Start the trainer, wait until it is genuinely recording, then kill it dead."""
    directory = Path(directory)
    script = directory / "trainer.py"
    script.write_text(_TRAINER)
    recording = directory / "crash.archx.mdl"
    process = subprocess.Popen(
        [sys.executable, str(script), str(recording)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Do not guess at timing. "The file exists" and "the file is big enough" are both wrong
    # conditions: on a slower runner the file can exist with only the header and meta blocks
    # while the first delta is still queued, and the kill then leaves no history at all. The
    # condition that is actually required is history, so wait for deltas specifically.
    line = process.stdout.readline()
    assert "ready" in line, line
    deadline = time.monotonic() + 60
    seen = 0
    while time.monotonic() < deadline:
        try:
            seen = sum(1 for _ in Reader(recording, allow_torn=True).blocks(BlockType.DELTA))
        except (FileNotFoundError, ArchxError):
            seen = 0                       # still being written, or not readable yet
        if seen >= _MIN_STEPS_BEFORE_KILL:
            break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    if sys.platform == "win32":
        process.terminate()
    else:
        process.kill()                     # SIGKILL: no atexit, no flush, no footer
    _, stderr = process.communicate(timeout=30)
    assert seen >= _MIN_STEPS_BEFORE_KILL, (
        f"the child never wrote {_MIN_STEPS_BEFORE_KILL} steps in 60 s (saw {seen}); "
        f"exit {process.returncode}, stderr: {stderr[-800:]}")
    return process.returncode, stderr, recording


def test_a_killed_run_keeps_every_step_it_managed_to_write(tmp_path):
    code, stderr, path = _kill_mid_run(tmp_path)
    assert code != 0 or sys.platform == "win32", f"the child was expected to be killed: {stderr}"
    assert path.stat().st_size > 0, "the crash left no file to recover"

    # 1. The file is readable *before* recovery, with allow_torn=True: the tail is the only
    #    thing lost, and a reader can already see how far the run got.
    torn = Reader(path, allow_torn=True)
    assert torn.container.footer is None, "a killed process cannot have written a footer"
    steps_before = [b.step for b in torn.blocks(BlockType.DELTA)]
    assert steps_before, "no history survived the kill"
    assert steps_before == sorted(steps_before), "history must be a prefix, never reordered"

    # 2. Recovery turns it into a closed, self-describing file without inventing anything.
    report = recover_in_place(path)
    assert report.recovered and report.bytes_truncated >= 0
    healed = Reader(path)
    assert healed.container.footer is not None
    assert healed.header.recovered
    steps_after = [b.step for b in healed.blocks(BlockType.DELTA)]
    assert steps_after == steps_before, "recovery must keep exactly the blocks that were whole"

    # 3. The recovered history still answers the product's question: this data, these steps,
    #    this movement - rebuilt from the file alone.
    timeline = Timeline(healed)
    first = min(timeline.steps)
    state = timeline.reconstruct(first)
    assert set(state) == {"layer.weight"}
    assert state["layer.weight"].shape == (32, 32)
    segments = timeline.segments()
    assert segments, "the data written before the crash must still be there"
    assert all(segment["first_step"] <= max(timeline.steps) for segment in segments)

    # 4. Annotations and observations written before the kill are not lost either.
    observations = [json.loads(b.payload) for b in healed.blocks(BlockType.EVENT)]
    assert observations, "the per-step observations survived"


def test_a_killed_run_is_recoverable_into_a_copy_without_touching_evidence(tmp_path):
    """The crashed file is evidence. `recover --out` must leave it exactly as it was."""
    _kill_mid_run(tmp_path)
    path = tmp_path / "crash.archx.mdl"
    before = path.read_bytes()

    from archx.container.recovery import recover_copy

    copy = tmp_path / "healed.archx.mdl"
    report = recover_copy(path, copy)
    assert report.recovered
    assert path.read_bytes() == before, "recovering a copy must not modify the original"
    assert Reader(copy).container.footer is not None

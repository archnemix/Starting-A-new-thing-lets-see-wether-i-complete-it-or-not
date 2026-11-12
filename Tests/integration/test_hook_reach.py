"""Claim 7, at the level the suite can afford: what one line of help actually buys.

The full table is `Bench/claim07_hook_reach.py`. The two facts worth guarding are the ones a
user would notice: a bare optimizer hook still records gradients (they are alive inside the
hook and dead afterwards), and adding `recorder.log(loss=...)` moves the file from tier 0 to
tier 2 without restructuring the loop.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

import archx                                                       # noqa: E402
from archx.container.reader import Reader                          # noqa: E402
from archx.container.schema import BlockType                       # noqa: E402


def _train(tmp_path, *, help_the_loop: bool, steps: int = 30) -> Reader:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    path = tmp_path / ("helped.archx.mdl" if help_the_loop else "bare.archx.mdl")
    recorder = archx.attach(model, optimizer, path=str(path))

    x = torch.randn(8, 16)
    y = torch.randint(0, 4, (8,))
    for _ in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        if help_the_loop:
            recorder.log(loss=float(loss.item()), data=x)
        optimizer.step()
    recorder.close()
    return Reader(path)


def _capability(reader: Reader) -> dict:
    info = [json.loads(b.payload) for b in reader.blocks(BlockType.INFO)]
    return next(b for b in info if b.get("event") == "capabilities")


def test_a_bare_hook_still_captures_gradient_norms(tmp_path):
    """Gradients exist inside the step hook and are gone after `zero_grad()`.

    Reading them there is the reason "attach in one line" is true rather than marketing.
    """
    reader = _train(tmp_path, help_the_loop=False)
    capability = _capability(reader)
    assert capability["capability_tier"] == 0
    assert "grad_norms" in capability["capabilities_present"]
    assert "grad_norms" not in capability["capabilities_missing"]

    events = [json.loads(b.payload) for b in reader.blocks(BlockType.EVENT)]
    assert events and all(e.get("grad_norm") is not None for e in events)
    assert all(e.get("loss") is None for e in events), "the hook cannot see a loss"


def test_one_log_call_takes_the_file_to_tier_two(tmp_path):
    reader = _train(tmp_path, help_the_loop=True)
    capability = _capability(reader)
    assert capability["capability_tier"] == 2, capability
    # Tier 3 needs an activation hook the loop never installed, so the file says the adapter
    # did not provide it rather than blaming a lower tier.
    assert capability["capabilities_missing"] == {
        "activations": "not provided by this adapter"}

    events = [json.loads(b.payload) for b in reader.blocks(BlockType.EVENT)]
    assert events and all(e.get("loss") is not None for e in events)

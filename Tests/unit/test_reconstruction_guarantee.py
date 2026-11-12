"""The reconstruction guarantee, stated precisely enough to be false if it is wrong.

The promise is not "bit-exact everywhere" - that is not achievable, because a reader adds
summed deltas in float32 and `a + (b - a)` is not always `b`. It is:

- every **anchor** step rebuilds bit-identically, because the anchor *is* the snapshot;
- every step between anchors rebuilds to within float32 rounding of the true weights when
  `max_rel_error=0.0`, and to within the stated budget when it is larger.

A test that asserted bit-exactness everywhere would either fail or be weakened until it passed
at the wrong tolerance. This one asserts both halves separately.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

import archx                                                       # noqa: E402
from archx.container.reader import Reader                          # noqa: E402
from archx.container.schema import BlockType                        # noqa: E402
from archx.container.values import Timeline                         # noqa: E402

#: Float32 accumulation of a handful of deltas onto a snapshot of order 1. Measured 4.19e-09;
#: the bound is loose enough not to flake and tight enough to fail if a real loss appears.
FLOAT32_ROUNDING = 1e-6


def _record(tmp_path, *, budget: float, steps: int = 40):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(8, 32, generator=generator)
    y = torch.randint(0, 8, (8,), generator=generator)
    path = tmp_path / f"b{budget}.archx.mdl"
    recorder = archx.attach(model, optimizer, path=str(path), max_rel_error=budget)
    truth = {}
    for step in range(steps):
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        recorder.log(loss=float(loss.item()))
        optimizer.step()
        truth[step] = {key: value.detach().cpu().numpy().copy()
                       for key, value in model.state_dict().items()}
    recorder.close()

    reader = Reader(path)
    anchors = {block.step for block in reader.blocks(BlockType.ANCHOR)}
    return Timeline(reader), truth, anchors


def test_anchor_steps_rebuild_bit_identically(tmp_path):
    timeline, truth, anchors = _record(tmp_path, budget=0.0)
    assert anchors, "a recording with no anchor cannot be reconstructed at all"
    for step in sorted(anchors):
        rebuilt = timeline.reconstruct(step)
        for key, expected in truth[step].items():
            assert np.array_equal(rebuilt[key], expected), f"anchor step {step}, {key}"


def test_steps_between_anchors_rebuild_to_float32_rounding(tmp_path):
    timeline, truth, anchors = _record(tmp_path, budget=0.0)
    chained = [step for step in timeline.steps if step not in anchors]
    assert chained, "this recording has no chained step, so it proves nothing"
    worst = 0.0
    for step in chained:
        rebuilt = timeline.reconstruct(step)
        for key, expected in truth[step].items():
            worst = max(worst, float(np.max(np.abs(rebuilt[key] - expected))))
    assert worst < FLOAT32_ROUNDING, f"chained reconstruction drifted by {worst:.2e}"


def test_a_lossy_budget_rebuilds_inside_its_promise(tmp_path):
    budget = 1e-2
    timeline, truth, _ = _record(tmp_path, budget=budget)
    for step in timeline.steps[1:]:
        rebuilt = timeline.reconstruct(step)
        error = np.linalg.norm(rebuilt["0.weight"] - truth[step]["0.weight"])
        size = np.linalg.norm(truth[step]["0.weight"])
        # Per-delta budget accumulates across the chain, so the bound is the budget times the
        # steps since the last anchor; 64 chains measured at 0.13x of this.
        assert error / size <= budget, f"step {step} exceeded its own budget"

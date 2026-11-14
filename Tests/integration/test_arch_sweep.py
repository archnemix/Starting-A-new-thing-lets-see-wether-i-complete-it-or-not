"""Integration: six architectures, one recorder, no model-specific code.

Phase 4.5. The adapter and the recorder must not need to know what the model is. These models
differ in everything that usually leaks into tooling: an attention stack, a convolution stack,
a recurrence, an elementwise state-space block, a mixture with a router, and two towers sharing
one tied matrix with a frozen embedding.

The expensive sweep lives in `Bench/claim_arch_sweep.py`; this module keeps a short version of
it in the suite so the promise is checked on all three runners, not only when someone runs a
benchmark by hand.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

import archx                                                       # noqa: E402
from archx.container.reader import Reader                          # noqa: E402
from archx.container.schema import BlockType                       # noqa: E402
from archx.container.values import Timeline                        # noqa: E402


def _models():
    """Built here rather than imported, so the test does not depend on Bench/ being a package."""
    from Bench.claim_arch_sweep import build_models

    return build_models()


def test_every_architecture_reaches_tier_two_and_rebuilds_its_weights(tmp_path):
    from Bench.claim_arch_sweep import batches, DATASET_SIZE

    torch.manual_seed(0)
    steps = 12
    for name, (model, regions) in _models().items():
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        path = tmp_path / f"{name}.archx.mdl"
        recorder = archx.attach(model, optimizer, path=str(path), regions=regions,
                                data_level="segments")
        dataset = list(batches(name, DATASET_SIZE))
        truth = {0: {key: value.detach().cpu().numpy().copy()
                     for key, value in model.state_dict().items()}}
        for step in range(steps):
            inputs, targets = dataset[step % DATASET_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            recorder.log(loss=float(loss.item()), data=inputs)
            optimizer.step()
            truth[step + 1] = {key: value.detach().cpu().numpy().copy()
                           for key, value in model.state_dict().items()}
        recorder.close()

        reader = Reader(path)
        stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                        | {b.step for b in reader.blocks(BlockType.ANCHOR)})
        assert stored, f"{name}: nothing was stored"
        assert stored[0] == 0, f"{name}: reconstruction must start at step 0"

        info = [json.loads(b.payload) for b in reader.blocks(BlockType.INFO)]
        capability = next(b for b in info if b.get("event") == "capabilities")
        assert capability["capability_tier"] == 2, (
            f"{name}: expected tier 2 from loss, gradients and data, got "
            f"{capability['capability_tier']}")
        assert "activations" in capability["capabilities_missing"], (
            f"{name}: tier 3 must be reported missing, not silently absent")

        # The first step after the anchor must rebuild exactly; a wrong anchor shows up here.
        timeline = Timeline(reader)
        rebuilt = timeline.reconstruct(stored[0])
        for key, expected in truth[stored[0]].items():
            assert np.array_equal(rebuilt[key], expected), f"{name}: {key} at step {stored[0]}"

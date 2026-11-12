"""Unit: sharded models are recognised and refused, on stubbed engines.

A real FSDP or DeepSpeed run needs several GPUs, which no CI runner here has - so the
*decision* is tested against stubs that carry the same names and the same shape as the real
wrappers. What is being pinned is not "we integrate with DeepSpeed"; it is that a run whose
parameters are split across ranks cannot silently produce a file that rebuilds weights that
never existed.

Two stubs are deliberately crude (a class that is only a name, a parameter list with a `None`
in it). That is the point: the detector must work from what it can see without importing
deepspeed or torch.distributed, and must fail open rather than crash a training run.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")

from archx.parallel import Detection, Parallelism, detect                # noqa: E402


class _Wrapper:
    """The shape every parallel wrapper has: a `.module` holding the real model."""

    def __init__(self, module, **extra) -> None:
        self.module = module
        for key, value in extra.items():
            setattr(self, key, value)

    def named_parameters(self, remove_duplicate=False):
        return self.module.named_parameters(remove_duplicate=remove_duplicate)

    def parameters(self):
        return self.module.parameters()


def _named(cls_name: str, module):
    """A wrapper whose *class name* is the signal, as the real ones are."""
    return type(cls_name, (_Wrapper,), {})(module)


def test_a_plain_model_is_reported_as_plain():
    detection = detect(torch.nn.Linear(4, 2))
    assert detection.kind == Parallelism.NONE
    assert detection.is_sharded is False


@pytest.mark.parametrize("name", [
    "FullyShardedDataParallel", "FSDP", "DeepSpeedEngine", "ShardedModel", "TPModel",
    "ZeroRedundancyOptimizer",
])
def test_every_sharding_name_is_recognised(name):
    detection = detect(_named(name, torch.nn.Linear(4, 2)))
    assert detection.kind == Parallelism.SHARDED, name
    assert name in detection.describe()


@pytest.mark.parametrize("name", ["DistributedDataParallel", "DataParallel", "DDP"])
def test_replicated_wrappers_are_told_apart_from_sharded_ones(name):
    """DDP keeps a whole copy on every rank, so it is recordable - with one writer."""
    detection = detect(_named(name, torch.nn.Linear(4, 2)))
    assert detection.kind == Parallelism.DDP, name
    assert detection.needs_a_single_writer is True
    assert detection.is_sharded is False


def test_a_nested_wrapper_is_still_found():
    """`torch.compile(FSDP(model))` must not hide the sharding."""
    inner = _named("FSDP", torch.nn.Linear(4, 2))
    outer = _named("OptimizedModule", inner)
    assert detect(outer).kind == Parallelism.SHARDED


def test_parameters_missing_outside_a_forward_pass_are_recognised():
    """ZeRO-3's signature, whatever the wrapper is called."""
    model = _Wrapper(torch.nn.Linear(4, 2))

    def parameters():
        return [torch.nn.Parameter(torch.zeros(2)), None]

    model.parameters = parameters
    detection = detect(model)
    assert detection.kind == Parallelism.SHARDED
    assert "absent outside a forward pass" in detection.describe()


def test_meta_device_parameters_are_recognised():
    """A parameter on the meta device has no values on this rank, whatever the engine."""
    model = _Wrapper(torch.nn.Linear(4, 2))
    with torch.device("meta"):
        meta_parameter = torch.nn.Parameter(torch.zeros(2))
    model.parameters = lambda: [meta_parameter]
    assert detect(model).kind == Parallelism.SHARDED


def test_detection_never_raises_on_a_hostile_object():
    """A wrapper that throws on attribute access must not break the training run."""
    class Hostile:
        def __init__(self):
            object.__setattr__(self, "_boom", True)

        def __getattr__(self, name):
            raise RuntimeError("no")

    assert detect(Hostile()) == Detection()


def test_attach_refuses_a_sharded_model_before_opening_a_file(tmp_path):
    """The file must not exist at all: a half-written recording is worse than none."""
    import archx
    from archx.adapters.torch import ShardedModelError

    model = torch.nn.Linear(4, 2)
    wrapper = _named("FSDP", model)
    path = tmp_path / "refused.archx.mdl"

    with pytest.raises(ShardedModelError) as caught:
        archx.attach(wrapper, torch.optim.SGD(model.parameters(), lr=0.1), str(path))

    assert "allow_partial" in str(caught.value), "the error must name the way out"
    assert not path.exists(), "a refused recording must leave nothing behind"


def test_allow_partial_records_the_truth_about_itself(tmp_path):
    """When the user insists, the file says its parameters are incomplete - and names why."""
    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType

    model = torch.nn.Linear(4, 2)
    wrapper = _named("FullyShardedDataParallel", model)
    path = tmp_path / "partial.archx.mdl"

    recorder = archx.attach(wrapper, torch.optim.SGD(model.parameters(), lr=0.1), str(path),
                            allow_partial=True)
    recorder.close()

    meta = json.loads(next(iter(Reader(path).blocks(BlockType.META))).payload)
    assert meta["parallel"] == Parallelism.SHARDED
    assert meta["parameters_complete"] is False
    assert "FullyShardedDataParallel" in meta["parallel_evidence"]


def test_an_unsharded_run_states_that_its_parameters_are_whole(tmp_path):
    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType

    model = torch.nn.Linear(4, 2)
    path = tmp_path / "whole.archx.mdl"
    recorder = archx.attach(model, torch.optim.SGD(model.parameters(), lr=0.1), str(path))
    recorder.close()

    meta = json.loads(next(iter(Reader(path).blocks(BlockType.META))).payload)
    assert meta["parallel"] == Parallelism.NONE
    assert meta["parameters_complete"] is True
    assert "parallel_evidence" not in meta, "no evidence means no claim"


def test_a_plain_run_is_unaffected_by_the_new_check(tmp_path):
    """The guard must cost nothing and change nothing on the normal path."""
    import archx
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    torch.manual_seed(0)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    path = tmp_path / "normal.archx.mdl"
    recorder = archx.attach(model, optimizer, str(path), max_rel_error=0.0)

    x = torch.randn(8, 4)
    for step in range(4):
        optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(x), torch.randn(8, 2)).backward()
        recorder.log(loss=0.5)
        optimizer.step()
    recorder.close()

    timeline = Timeline(Reader(path))
    assert timeline.steps == [0, 1, 2, 3]
    rebuilt = timeline.reconstruct(3)
    assert np.allclose(rebuilt["weight"], model.weight.detach().numpy(), atol=1e-6)

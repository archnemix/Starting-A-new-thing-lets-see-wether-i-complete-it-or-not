"""Recognise the wrappers that make a recording incomplete, and say so before it starts.

Three situations look identical from inside an optimizer hook and are not:

| Situation | What the hook sees | Can a file hold the real weights? |
|---|---|---|
| single process, or DDP | one full copy of every parameter per rank | **yes**, on exactly one rank |
| FSDP / ZeRO-2/3 | each rank holds a *shard* | no: no rank has the full tensor |
| DeepSpeed with ZeRO-3 | parameters may be `None` or meta outside a forward pass | no |

The failure mode this module exists to prevent is the quiet one: a run under FSDP records
whatever each rank happens to hold, produces a file that opens cleanly, and rebuilds weights
that were never the model's weights. The tiers cannot catch it either - the tier says what
*fields* were seen, not whether the values were whole.

So detection runs at attach time, and the answer goes into the file:

* `PARALLEL_NONE` / `PARALLEL_DDP` - record normally (one rank only, for DDP).
* `PARALLEL_SHARDED` - the recorder refuses by default. Engine detection is best-effort, so
  the user can pass `allow_partial=True` to record anyway; the metadata then says
  `parameters_complete: false` and names the wrapper, so nobody can mistake the file for a
  full model.

Detection is by attribute and class name, not by importing deepspeed or torch.distributed,
because those are optional and expensive: a *string* in the type name cannot import anything,
and the checks are written to fail open (report NONE) rather than crash a training run.
"""

from __future__ import annotations

from dataclasses import dataclass


class Parallelism:
    """What kind of wrapper the model arrived in."""

    NONE = "none"
    DDP = "ddp"
    SHARDED = "sharded"


@dataclass(frozen=True)
class Detection:
    """The verdict, with the evidence that produced it."""

    kind: str = Parallelism.NONE
    evidence: str = ""

    @property
    def is_sharded(self) -> bool:
        return self.kind == Parallelism.SHARDED

    @property
    def needs_a_single_writer(self) -> bool:
        """True when several ranks hold the same weights and only one may write."""
        return self.kind == Parallelism.DDP

    def describe(self) -> str:
        if self.kind == Parallelism.NONE:
            return "one process, or nothing recognisable: parameters are whole"
        return self.evidence


#: Type names that mean "each rank holds a piece", across torch and the engines around it.
_SHARDING_MARKERS = (
    "FullyShardedDataParallel", "FSDP", "ZeroRedundancyOptimizer", "ShardedGradScaler",
    "DeepSpeedEngine", "DeepSpeedZeroOptimizer", "ZeRO", "FairScale", "ShardedModel",
    "sharded_model", "TPModel", "ModelParallel",
)

#: Type names that mean "every rank holds everything, one of them should write".
_REPLICATED_MARKERS = ("DistributedDataParallel", "DataParallel", "DDP")


def detect(model) -> Detection:
    """Classify the model wrapper. Never raises: an unknown wrapper is reported as NONE."""
    try:
        names = {type(model).__name__}
        # A wrapper can be nested (`torch.compile(FSDP(...))`), and only the inner one matters.
        inner = getattr(model, "module", None)
        for _ in range(4):
            if inner is None:
                break
            names.add(type(inner).__name__)
            inner = getattr(inner, "module", None)
        for attribute in ("_fsdp_wrapped_module", "sharded_module", "_orig_mod", "model"):
            wrapped = getattr(model, attribute, None)
            if wrapped is not None and wrapped is not model:
                names.add(type(wrapped).__name__)
    except Exception:                                  # noqa: BLE001 - never break a run
        return Detection()

    for name in sorted(names):
        for marker in _SHARDING_MARKERS:
            if marker in name:
                return Detection(Parallelism.SHARDED, f"wrapper {name!r} shards parameters")
    # A parameter that is `None` or on the meta device outside a forward pass is the other
    # signature of ZeRO-3, and it survives whatever the wrapper is called.
    parameter_hint = _sharded_parameter(model)
    if parameter_hint:
        return Detection(Parallelism.SHARDED, parameter_hint)

    for name in sorted(names):
        for marker in _REPLICATED_MARKERS:
            if marker in name:
                return Detection(Parallelism.DDP,
                                 f"wrapper {name!r} replicates parameters on every rank")
    return Detection()


def _sharded_parameter(model) -> str:
    """Return a reason when the parameters themselves look sharded, else an empty string."""
    try:
        parameters = list(model.parameters())
    except Exception:                                  # noqa: BLE001
        return ""
    if not parameters:
        return ""
    missing = sum(1 for parameter in parameters if parameter is None)
    if missing:
        return (f"{missing} of {len(parameters)} parameters are absent outside a forward pass, "
                f"which is what ZeRO-3/FSDP sharding looks like")
    try:
        meta = sum(1 for parameter in parameters
                   if getattr(parameter, "device", None) is not None
                   and str(parameter.device) == "meta")
    except Exception:                                  # noqa: BLE001
        return ""
    if meta:
        return (f"{meta} of {len(parameters)} parameters live on the meta device, "
                f"so their values are not available on this rank")
    return ""

"""Plain PyTorch: attach in one line, leave the loop alone, lose nothing that was recorded.

    recorder = archx.attach(model, optimizer, path="run.archx.mdl")

The hook is `optimizer.register_step_post_hook`, public since torch 2.0. It fires after the
weights have changed, which is exactly why the recorder keeps a shadow copy of the previous
store point (see archx/recorder.py). Gradients are read inside the same hook, before the
caller's `zero_grad()`, so gradient norms cost nothing extra.

The hook cannot see the loss or the batch: those are locals that die with the iteration. One
optional line hands them over, before `optimizer.step()`:

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    recorder.log(loss=loss.item(), data=x)   # optional; only these two need it
    optimizer.step()

The file then reaches tier 2 without the loop being restructured in any other way.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..data.activations import ActivationCapture
from ..formats.registry import TensorRegistry
from ..parallel import detect
from ..policy import PhasePolicy
from ..recorder import Recorder


def _label_for(key: str, labels: dict[str, str]) -> str:
    """Longest matching prefix wins, so 'vision.encoder' beats 'vision'."""
    best = ""
    for prefix in labels:
        if key.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return labels[best] if best else ""


def _build_registry(parameters: dict, *, regions: dict[str, str] | None,
                    components: dict[str, str] | None) -> TensorRegistry:
    """Registry with region tags and component names applied by longest matching prefix."""
    regions = regions or {}
    components = components or {}
    registry = TensorRegistry()
    for key, parameter in parameters.items():
        array = parameter.detach().cpu().numpy()
        registry.add(
            key,
            array,
            pointer=int(parameter.data_ptr()),
            frozen=not bool(getattr(parameter, "requires_grad", True)),
            region=_label_for(key, regions),
            component=_label_for(key, components),
        )
    return registry


def as_bytes(data: Any) -> bytes | None:
    """Turn whatever the training loop calls "a batch" into bytes, without changing it.

    Tensors are read through their CPU view; nothing is mutated and no gradient is retained.
    An unsupported type returns None, which the recorder reads as "no data recorded" - never
    as a guess about what the user meant.
    """
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    if hasattr(data, "detach"):                       # torch.Tensor
        return data.detach().cpu().numpy().tobytes()
    if isinstance(data, (list, tuple)):               # a batch of examples or tensors
        parts = [as_bytes(item) for item in data]
        if parts and all(part is not None for part in parts):
            return b"".join(parts)                    # fixed per-batch layout, byte exact
    return None


class ShardedModelError(RuntimeError):
    """Raised when a model is sharded across ranks, so one rank cannot record it whole."""


class TorchRecorder(Recorder):
    """A recorder that also understands the things a torch loop naturally has to hand.

    The core recorder only ever accepts bytes, because a container format has no opinion about
    tensors. This subclass is where "a batch" means something: it is the one place the adapter
    converts, so the conversion rules live next to the framework they belong to.
    """

    def trace_examples(self, loss_parts, examples, *, max_bytes: int = 256 * 1024 * 1024) -> dict:
        """Record gradients of additive, already-scaled per-example losses before step().

        Call after total_loss.backward(retain_graph=True), before optimizer.step(). Each
        loss_parts[i] must include its contribution to the batch reduction (for mean loss,
        divide each individual loss by batch size). The parts must sum to the gradients
        currently in .grad; accumulation, clipping, regularization or omitted parameters
        cause a clear error unless they are represented in those parts. `examples` are
        caller-supplied bytes for each input AND any target needed to identify it.

        This stores local gradients, not individual causal effects or Adam updates. It can be
        expensive: O(batch * parameters) values per step, before compression.
        """
        from .example_gradients import encode_example_gradients
        if not self.full_history:
            raise ValueError("example tracing requires full_history=True")
        if getattr(self, "_example_trace", None) is not None:
            raise RuntimeError("an example trace is already pending an optimizer step")
        payload, report = encode_example_gradients(loss_parts, examples,
                                                   self._archx_parameters, max_bytes=max_bytes)
        self._example_trace = payload
        return report

    def log(self, *, loss: float | None = None, data: Any = None, lr: float | None = None,
            segment_id: str = "", source: str = "") -> None:
        super().log(loss=loss, data=as_bytes(data), lr=lr,
                    segment_id=segment_id, source=source)


def attach(
    model,
    optimizer,
    path: str,
    *,
    policy: PhasePolicy | None = None,
    regions: dict[str, str] | None = None,
    components: dict[str, str] | None = None,
    data_level: str = "segments",
    queue_size: int = 8,
    max_rel_error: float = 1e-3,
    allow_partial: bool = False,
    meta: dict[str, Any] | None = None,
    activations: bool = False,
    activation_layers: list[str] | None = None,
    max_activation_layers: int = 256,
    full_history: bool = False,
) -> Recorder:
    """Attach a recorder and return it. The training loop stays exactly as it was.

    `model` may be a `nn.Module` or any object with `named_parameters()`. The returned
    recorder's `log()` takes the loss and the batch; both are optional.

    `max_rel_error` is the reconstruction budget per delta. The default (1e-3) is what makes
    a run small; `0.0` restricts every delta to a lossless encoder and makes every stored step
    rebuild exactly.

    Under FSDP or ZeRO-3, no rank holds a whole parameter, so a file recorded from one rank
    would rebuild weights that never existed. attach() therefore refuses, rather than writing
    a plausible-looking file; pass `allow_partial=True` to record anyway, and the file will
    say `parameters_complete: false` and name the wrapper.

    `full_history=True` records the initial weights at step 0 and a delta after every
    optimizer step. Delta encoding is lossless in this mode. This costs more.

    `activations=True` reaches tier 3: a forward hook summarises each layer's output (shape,
    mean, std, min/max, dead-unit and saturation fractions) at every store point. It is off by
    default because it costs real time inside the forward pass - measured in
    `Bench/claim15_activations.py`, reported rather than hidden - and because a recorder that
    changed a loop's speed by default would be a recorder nobody leaves switched on. Pass
    `activation_layers=[...]` to name specific modules; the default is every leaf layer that
    returns a float tensor.
    """
    parallel = detect(model)
    if parallel.is_sharded and not allow_partial:
        raise ShardedModelError(
            f"this model is sharded ({parallel.describe()}).\n"
            f"  A recording made from one rank would hold part of each parameter, so every\n"
            f"  step would rebuild weights that never existed. Nothing was opened.\n"
            f"  Either record on an unsharded run, or pass allow_partial=True and the file\n"
            f"  will state that its parameters are incomplete.")
    parameters = dict(_named_parameters(model))
    if full_history:
        import torch
        if any(p.dtype != torch.float32 for p in parameters.values()):
            raise ValueError("full_history currently requires float32 parameters")
        # Dense mode is a fidelity contract, not the adaptive lossy budget.
        max_rel_error = 0.0
    registry = _build_registry(parameters, regions=regions, components=components)

    def read_parameter(key: str) -> np.ndarray:
        return parameters[key].detach().cpu().numpy()

    # This is a seed label, not a saved RNG state or a replay guarantee. User-supplied
    # run configuration (dataset, model, schedule and RNG states) belongs in meta.
    import torch
    import base64
    groups = []
    for group in optimizer.param_groups:
        groups.append({k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v)
                       for k, v in group.items() if k != "params"})
    recorder = TorchRecorder(registry, read_parameter, path, policy=policy,
                        queue_size=queue_size,
                        max_rel_error=max_rel_error,
                        data_level=data_level,
                        full_history=full_history,
                        meta={"adapter": "torch", "torch": _torch_version(),
                              "torch_initial_seed": int(torch.initial_seed()),
                              "torch_cpu_rng_state_base64": base64.b64encode(
                                  torch.get_rng_state().cpu().numpy().tobytes()).decode("ascii"),
                              "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
                              "optimizer_groups": groups,
                              "parallel": parallel.kind,
                              "parameters_complete": not parallel.is_sharded,
                              **({"parallel_evidence": parallel.describe()}
                                 if parallel.is_sharded else {}),
                              **(meta or {})})
    recorder._archx_parameters = parameters        # type: ignore[attr-defined]
    handle = optimizer.register_step_post_hook(_make_step_hook(recorder, parameters))
    recorder._archx_handle = handle                # type: ignore[attr-defined]
    if activations:
        _attach_activation_hooks(model, recorder, layers=activation_layers,
                                 max_layers=max_activation_layers)
    return recorder


def _attach_activation_hooks(model, recorder: Recorder, *, layers: list[str] | None,
                             max_layers: int) -> None:
    """Summarise every forward pass, for tier 3.

    Hooks are registered here, in `attach`, and removed by `detach`; a hook left behind on a
    model that outlives its recorder would keep writing into a closed file. The handle list is
    kept on the recorder so closing it is enough.
    """
    capture = ActivationCapture(max_layers=max_layers)
    wanted = set(layers) if layers else None
    handles = []

    for name, module in model.named_modules():
        if not name:
            continue                                  # the root is not a layer
        if wanted is not None and name not in wanted:
            continue
        handles.append(module.register_forward_hook(_make_activation_hook(name, capture)))

    recorder.activations = capture
    recorder._archx_activation_handles = handles      # type: ignore[attr-defined]
    if not handles:
        recorder.capabilities.deny(
            "activations",
            "activations=True but the model exposes no named modules to hook")
        return
    recorder._meta_activation_layers = sorted(       # type: ignore[attr-defined]
        {name for name, module in model.named_modules() if name})


def _make_activation_hook(name: str, capture: ActivationCapture):
    """A forward hook that records a summary and returns nothing.

    Returning a value from a forward hook *replaces* the layer's output, which would silently
    change the model's behaviour - the one thing a recorder must never do. This returns None,
    explicitly.
    """

    def hook(_module, _inputs, output) -> None:
        capture.observe(name, output)
        return None

    return hook


def _named_parameters(model):
    """Every parameter, with its dotted name - including both names of a tied tensor.

    `remove_duplicate=False` is deliberate: torch hides the second name of a shared parameter
    by default, and that name is exactly what tells the registry the two are one storage unit.
    """
    try:
        return model.named_parameters(remove_duplicate=False)
    except TypeError:            # older torch, or an object that only takes no arguments
        return model.named_parameters()


def _grad_norm(parameters: dict) -> float | None:
    """Global L2 norm of the gradients, read in one pass over the CPU views.

    Measured on a 6-tensor model: `pow(2).sum().item()` costs ~40 us per call, the numpy
    reduction ~19 us, and this runs on every step whether or not the step is stored. The norm
    is summed in float64 because a float32 accumulation loses the small tensors entirely.
    """
    total = 0.0
    seen = False
    for parameter in parameters.values():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        flat = grad.detach().cpu().numpy().ravel()
        total += float(np.dot(flat, flat))
        seen = True
    return total ** 0.5 if seen and total else None


def _torch_version() -> str:
    """The framework version goes into the file: a recording must say what produced it."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement of this adapter
        return "unknown"
    return torch.__version__


def _make_step_hook(recorder: Recorder, parameters: dict):
    """Build the post-step hook: read the gradients, then let the recorder decide.

    Gradient norms are computed here because this is the last moment they exist: the caller's
    loop calls zero_grad() immediately afterwards. Nothing is kept alive afterwards.
    """
    state = {"step": 0 if recorder.full_history else -1}

    def hook(_optimizer, _args, _kwargs) -> None:
        state["step"] += 1
        decision = recorder.step(state["step"], grad_norm=_grad_norm(parameters))
        pending = getattr(recorder, "_example_trace", None)
        if pending is not None:
            from ..recorder import _Pending
            if decision.value == "skip":
                raise RuntimeError("example trace requires a stored optimizer step")
            recorder._enqueue(_Pending("example_gradients", state["step"], {}, payload=pending))
            recorder._example_trace = None

    return hook

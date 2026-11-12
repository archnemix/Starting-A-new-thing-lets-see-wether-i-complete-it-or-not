"""Hugging Face `Trainer`: one callback, no fork of the training loop.

    from archx.adapters.hf import ArchxCallback
    trainer = Trainer(..., callbacks=[ArchxCallback("run.archx.mdl")])
    trainer.train()

Everything below was measured against a real `Trainer` on torch 2.15 / transformers 5.22,
not read off the documentation. The three facts that set the design:

1. **Where the weights change.** `on_pre_optimizer_step` fires *before*
   `optimizer.step()`, `on_optimizer_step` fires *after* it but before `model.zero_grad()`.
   So the weights are new at `on_optimizer_step`, and the gradients are still alive there.
   `on_step_end` is too late for the gradients - the loop has already zeroed them - which is
   why it is not the primary hook. Both step hooks fire only on real update steps, so with
   `gradient_accumulation_steps > 1` the recorder gets one call per optimizer step, not one
   per micro-batch.

2. **`state.global_step` is not yet incremented** when the step hooks fire (it is bumped
   just before `on_step_end`). It is therefore exactly the 0-based index of the update that
   just happened, which is what the recorder wants.

3. **A step-post-hook cannot be registered on the optimizer the callback is given.** The
   `optimizer` in the callback kwargs is accelerate's `AcceleratedOptimizer`, which is *not*
   a `torch.optim.Optimizer`; `register_step_post_hook` is inherited from the `Optimizer`
   base class and reaches for `_optimizer_step_post_hooks`, which the wrapper does not have.
   Registering there raises `AttributeError` inside `on_train_begin` and kills the run. The
   callback events above are the supported path, and they are enough on their own - so this
   adapter registers no optimizer hooks at all.

What it cannot see, and says so in the file rather than guessing:

- **The loss.** It is a local variable inside `Trainer.training_step`, and neither `on_log`
  nor `on_step_end` can carry it honestly. `on_log` fires *after* the step has been recorded
  and reports the mean loss over the whole logging window, so using it would attach an
  average of the last N steps to the wrong step number - and with `logging_steps=50` it
  would leave 49 steps of every 50 with no loss while the file still claimed tier 1. It is
  therefore not used at all. Two honest routes remain: `loss_hook(trainer)` for every step
  with one line at setup, or `callback.log(loss=...)` from code that already has the loss.
  With neither, no loss is written and the file says tier 1 was unreachable and why.
- **The gradient norm is the clipped one** when `max_grad_norm > 0` (the default is 1.0),
  because `Trainer` clips before it calls any callback. The metadata records
  `grad_norm_clipped` so a reader is never left to assume.

- **The batch.** The dataloader is not passed to any step hook. `data_level` therefore
  defaults to `none` here (unlike the plain-torch adapter, where the loop calls `log(data=)`).
  Asking for `data_level="segments"` without a way to see the data is refused at
  construction, not ignored.

TRL's `SFTTrainer`/`DPOTrainer` and Axolotl's trainer are subclasses of the same stack, so
they take this callback unchanged. DPO-style trainers run two backward passes per update;
keying off the optimizer-step events (not backward counts) is what keeps that correct.
"""

from __future__ import annotations

from typing import Any

from ..formats.registry import TensorRegistry
from ..policy import PhasePolicy
from ..recorder import Recorder


def _label_for(key: str, labels: dict[str, str]) -> str:
    """Longest matching prefix wins, so 'vision.encoder' beats 'vision'."""
    best = ""
    for prefix in labels:
        if key.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return labels[best] if best else ""


def _unwrap(model):
    """The real model, not the DDP/FSDP wrapper.

    `DistributedDataParallel` prefixes every parameter with `module.`, which would put a
    name in the file that the user's model does not have. accelerate's `unwrap_model`
    understands every wrapper the Trainer can produce; without accelerate the model is
    already bare.
    """
    try:
        from accelerate import Accelerator
        from accelerate.utils import extract_model_from_parallel
    except ImportError:                    # pragma: no cover - accelerate comes with Trainer
        return model
    del Accelerator
    return extract_model_from_parallel(model)


def _named_parameters(model):
    """Every parameter with its dotted name, including both names of a tied tensor."""
    try:
        return model.named_parameters(remove_duplicate=False)
    except TypeError:                      # an object with the older signature
        return model.named_parameters()


def _build_registry(model, *, regions, components) -> TensorRegistry:
    regions = regions or {}
    components = components or {}
    registry = TensorRegistry()
    for key, parameter in _named_parameters(model):
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


def _grad_norm(parameters: dict) -> float | None:
    """Global L2 norm of the gradients, summed in float64 so small tensors still count."""
    total = 0.0
    seen = False
    for parameter in parameters.values():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        flat = grad.detach().to("cpu").float().numpy().ravel()
        total += float((flat * flat).sum(dtype="float64"))
        seen = True
    return total ** 0.5 if seen and total else None


class HfRecorder(Recorder):
    """A recorder that also accepts a loss from user code (`callback.log(loss=...)`)."""


_TRAINER_BASE: type | None = None


def _with_trainer_base(cls: type) -> type:
    """Bind the callback to the real `TrainerCallback`, once, at first construction.

    `transformers` is imported here rather than at module import so that the container,
    the CLI and the plain-torch adapter keep working on a machine that has no transformers.
    The rebuilt class keeps the name, docstring and module of the original, so tracebacks
    and `isinstance` checks against the imported name behave as if it had always been a
    plain subclass.
    """
    global _TRAINER_BASE
    if _TRAINER_BASE is not None:
        return _TRAINER_BASE
    try:
        from transformers import TrainerCallback
    except ImportError as exc:                     # pragma: no cover - depends on install
        raise ImportError(
            "ArchxCallback needs `transformers`; install it with "
            "`pip install transformers` (the plain-torch adapter, archx.attach, does not)"
        ) from exc
    _TRAINER_BASE = type(cls.__name__, (cls, TrainerCallback), {
        "__module__": cls.__module__,
        "__doc__": cls.__doc__,
        "__qualname__": cls.__qualname__,
    })
    globals()[cls.__name__] = _TRAINER_BASE
    return _TRAINER_BASE


class ArchxCallback:
    """A `TrainerCallback` that records the run. Importable without transformers at import time.

    Subclassing `TrainerCallback` directly would make `import archx.adapters.hf` fail on a
    machine without transformers - and the rest of the library (container, CLI, plain torch)
    must keep working there. So the class is built against the real base class on first
    instantiation, which is the only moment transformers is genuinely required.
    """

    def __new__(cls, *args, **kwargs):
        return object.__new__(_with_trainer_base(cls))

    def __init__(
        self,
        path: str,
        *,
        policy: PhasePolicy | None = None,
        regions: dict[str, str] | None = None,
        components: dict[str, str] | None = None,
        data_level: str = "none",
        queue_size: int = 8,
        max_rel_error: float = 1e-3,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if data_level not in ("none", "pointers"):
            raise ValueError(
                f"data_level={data_level!r} cannot be honoured by this adapter: the Trainer "
                "does not hand the batch to any callback, so no segment could be recorded. "
                "Use data_level='pointers' to keep the data this callback *is* given (none by "
                "default), or record a plain torch loop with archx.attach, which does see it."
            )
        self.path = path
        self.policy = policy
        self.regions = regions or {}
        self.components = components or {}
        self.data_level = data_level
        self.queue_size = queue_size
        self.max_rel_error = max_rel_error
        self.meta = meta or {}
        self.recorder: HfRecorder | None = None
        self._pending_step = 0
        self._pending_loss: float | None = None
        self._pending_grad_norm: float | None = None
        self._pending_lr: float | None = None
        self._skipped_steps = 0
        self._closed = False

    # ------------------------------------------------------------ public surface

    def log(self, *, loss: float | None = None) -> None:
        """Hand over the loss for the update that is about to happen.

        The only thing the callback genuinely cannot reach. Call it from the loop or from a
        trainer subclass method that sees the loss; whatever is logged is attached to the
        next optimizer step and then cleared, so it cannot leak into a later step.
        """
        if loss is not None:
            self._pending_loss = float(loss)

    # --------------------------------------------------- transformers callbacks

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_main_process(args):
            # Every rank has identical weights under DDP; N ranks appending to one file would
            # interleave N streams. Silent, deliberate, and recorded in the file's metadata.
            return control
        model = kwargs.get("model")
        if model is None:                              # pragma: no cover - older signature
            model = kwargs.get("trainer").model
        model = _unwrap(model)
        parameters = dict(_named_parameters(model))

        def read_parameter(key: str):
            return parameters[key].detach().to("cpu").numpy()

        meta = {
            "adapter": "hf",
            "transformers": _version("transformers"),
            "torch": _version("torch"),
            "recording_rank": 0,
            # A callback cannot see gradients before the Trainer clips them, so which norm
            # this file holds is stated rather than left for the reader to assume.
            "grad_norm_clipped": bool(getattr(args, "max_grad_norm", 0) and
                                      args.max_grad_norm > 0),
            **({"model_class": type(model).__name__} if model is not None else {}),
            **self.meta,
        }
        self.recorder = HfRecorder(
            _build_registry(model, regions=self.regions, components=self.components),
            read_parameter,
            self.path,
            policy=self.policy,
            queue_size=self.queue_size,
            max_rel_error=self.max_rel_error,
            data_level=self.data_level,
            meta=meta,
        )
        self._parameters = parameters
        self._model = model
        # Nothing has told us yet whether the loss will arrive, so the file states the route
        # out rather than a shrug. `step()` clears this the moment a real loss is recorded.
        self.recorder.capabilities.deny(
            "loss",
            "the Trainer keeps the loss inside training_step; call "
            "archx.adapters.hf.loss_hook(trainer) before train(), or callback.log(loss=...)")
        self.recorder.capabilities.deny(
            "data_segments",
            "the Trainer does not hand the batch to any callback; data_level stays 'none'")
        # A run resumed from a checkpoint starts at a non-zero global_step; the file says so
        # instead of pretending this recording began at step 0.
        if state is not None and state.global_step:
            self.recorder.annotate(state.global_step, "resumed",
                                   f"trainer resumed at global_step {state.global_step}")
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Gradients and learning rate, read while they still exist and before they change.

        Two measured facts shape what is read here:

        * `Trainer` calls `_clip_grad_norm` *before* this callback, so `max_grad_norm > 0`
          (the default of 1.0) means the number available here is the **clipped** norm. No
          callback can see the unclipped one, and the public alternative
          (`register_post_accumulate_grad_hook`) fires per parameter, so on a model with a
          frozen or unused tensor it can silently never fire. The clipped norm is therefore
          recorded, as a fact, and `grad_norm_clipped` in the metadata says which it is.
        * The learning rate in `args.learning_rate` is the *initial* one; the scheduler only
          ever writes to the optimizer. The rate the update actually used is the one sitting
          in the optimizer's parameter groups right now, because the scheduler is stepped
          after the update.
        """
        if self.recorder is None:
            return control
        self._pending_step = int(state.global_step) if state is not None else 0
        self._pending_grad_norm = _grad_norm(self._parameters)
        self._pending_lr = _current_lr(kwargs.get("optimizer"))
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        """The weights have just changed: this is the recorder's own step boundary."""
        if self.recorder is None:
            return control
        # A skipped step (gradient overflow on an fp16 run) did not change the weights, so
        # recording it as a store point would put a delta of zero in the file and claim the
        # run took a step it did not take. Counted, not hidden.
        if _step_was_skipped(args, kwargs):
            self._skipped_steps += 1
            return control
        loss, self._pending_loss = self._pending_loss, None
        grad_norm = getattr(self, "_pending_grad_norm", None)
        self._pending_grad_norm = None
        learning_rate = self._pending_lr
        try:
            self.recorder.step(self._pending_step, loss=loss, grad_norm=grad_norm,
                               lr=learning_rate)
        except Exception as exc:                       # noqa: BLE001
            # Never let the recorder stop the run - but never swallow it either: `close()`
            # re-raises, so a broken recording cannot pass for a good one.
            self.recorder._degrade(exc)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.close()
        return control

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the recorder once. Safe to call from `on_train_end` or from user code."""
        if self._closed or self.recorder is None:
            self._closed = True
            return
        self._closed = True
        if self._skipped_steps:
            self.recorder.annotate(self._pending_step, "skipped_steps",
                                   f"{self._skipped_steps} optimizer steps were skipped "
                                   f"and are not store points")
        self.recorder.close()

    @property
    def stats(self):
        return self.recorder.stats if self.recorder is not None else None

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _is_main_process(args) -> bool:
        """True on the one rank that should write. Single-process runs are always main."""
        try:
            from accelerate import PartialState
        except ImportError:                    # pragma: no cover
            return True
        return PartialState().is_main_process


def _current_lr(optimizer) -> float | None:
    """The learning rate this update is about to use, read from the optimizer itself."""
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        return None
    return float(groups[0]["lr"])


def _step_was_skipped(args, kwargs) -> bool:
    """Did this update get skipped (an fp16 overflow)? Read from accelerate, if present.

    accelerate's `AcceleratedOptimizer` flags it as `_is_overflow` and never forwards an
    `accelerator` attribute, so both spellings are checked. A skipped step did not change
    the weights: recording it would put a zero delta in the file and claim a step that the
    run did not take.
    """
    optimizer = kwargs.get("optimizer")
    if getattr(optimizer, "_is_overflow", False):
        return True
    accelerator = getattr(optimizer, "accelerator", None)
    if accelerator is not None:
        return bool(getattr(accelerator, "optimizer_step_was_skipped", False))
    return False


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception:                                  # noqa: BLE001 - a version is a fact
        return "unknown"


def loss_hook(trainer) -> None:
    """Make the loss visible for every update, with one line of code at setup time.

    Call it after the trainer is constructed and before `train()`:

        callback = ArchxCallback("run.archx.mdl")
        trainer = Trainer(..., callbacks=[callback])
        loss_hook(trainer)

    The Trainer computes the loss inside `training_step`, where nothing outside can see it,
    and `on_log` is throttled by `logging_steps`. The documented extension point -
    "subclass and override `compute_loss` for custom behavior" - is used here at the instance
    level: the original method is called unchanged, so the run is bit-identical, and only the
    number it returns is observed.

    The rejected alternative was `compute_loss_func`. That seam *pops the labels out of the
    inputs* before the model runs, so the model never computes a loss and the wrapper would
    have to rebuild one from the logits - which means guessing whether this model wants a
    shifted cross-entropy, a plain one, an MSE, or something else entirely. Measured on a
    small regression model it produced a confidently wrong loss. Observing the real value is
    the only version that cannot be wrong.
    """
    original = trainer.compute_loss
    if getattr(original, "_archx_wrapped", False):      # idempotent: never wrap twice
        return

    def compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        result = original(model, inputs, return_outputs, num_items_in_batch)
        # `return_outputs=True` is used by prediction and by custom training steps; the loss
        # is then the first element. Both shapes are real, so both are handled rather than
        # assuming the common one.
        loss = result[0] if return_outputs else result
        if loss is not None and hasattr(loss, "detach"):
            for callback in _archx_callbacks(trainer):
                callback.log(loss=float(loss.detach()))
        return result

    compute_loss._archx_wrapped = True
    trainer.compute_loss = compute_loss


def _archx_callbacks(trainer):
    """Every Archnemix callback on this trainer, in the order it was registered."""
    handler = getattr(trainer, "callback_handler", None)
    for callback in getattr(handler, "callbacks", []) or []:
        if isinstance(callback, ArchxCallback):
            yield callback

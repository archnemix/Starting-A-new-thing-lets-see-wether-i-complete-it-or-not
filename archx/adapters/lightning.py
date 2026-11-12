"""PyTorch Lightning: one callback, and the loss comes for free.

    from archx.adapters.lightning import ArchxCallback
    trainer = pl.Trainer(..., callbacks=[ArchxCallback("run.archx.mdl")])
    trainer.fit(model)

Measured against lightning 2.6.6 / torch 2.15.0, the hook order on an update step is:

    on_after_backward            gradients alive, weights old
    on_before_optimizer_step     gradients alive, weights old, optimizer has this lr
    <optimizer.step(), zero_grad>
    on_train_batch_end           weights new, gradients gone, outputs carry the loss
    trainer.global_step          incremented after that

Two consequences, both better than the Hugging Face adapter's:

* **The loss is available without any user code.** Lightning hands `outputs` to
  `on_train_batch_end`, and `outputs["loss"]` is the training loss (a dict since 2.0; a bare
  tensor in older versions). So this adapter reaches tier 1 with one line and no `loss_hook`.
* **The update boundary is the hooks, not the step counter.** With
  `accumulate_grad_batches > 1`, `on_train_batch_end` fires once per micro-batch while
  `trainer.global_step` does not move - measured: two batch-ends at gs=1, then one
  `on_before_optimizer_step`. And `global_step` is incremented *after* the batch-end hook, so
  the step number in the file is `trainer.global_step` at `on_before_optimizer_step`, not at
  batch end. Recording at batch end would have written two store points for one update, both
  claiming the same step number.

The gradient norm recorded here is the *unclipped* one: unlike the Hugging Face `Trainer`,
Lightning clips inside `on_before_optimizer_step` (its own gradient-clipping component runs
there), so a callback registered before it sees the true norm. That is stated in the metadata
as `grad_norm_clipped: false` rather than left to be assumed.

Manual optimization (`self.manual_backward` + `self.optimizers().step()` in `training_step`)
is handled: the same hooks fire, and the loss is taken from `on_train_batch_end` as usual.
"""

from __future__ import annotations

from typing import Any

from ..formats.registry import TensorRegistry
from ..policy import PhasePolicy
from ..recorder import Recorder

_BASE: type | None = None


def _with_callback_base(cls: type) -> type:
    """Bind to the real `pl.Callback` at first use, so `import archx.adapters.lightning`
    does not require lightning to be installed (the container and CLI must not need it)."""
    global _BASE
    if _BASE is not None:
        return _BASE
    try:
        from lightning.pytorch import Callback
    except ImportError as exc:                       # pragma: no cover - depends on install
        raise ImportError(
            "ArchxCallback needs `lightning`; install it with `pip install lightning` "
            "(archx.attach and the container do not)"
        ) from exc
    _BASE = type(cls.__name__, (cls, Callback), {
        "__module__": cls.__module__,
        "__doc__": cls.__doc__,
        "__qualname__": cls.__qualname__,
    })
    globals()[cls.__name__] = _BASE
    return _BASE


def _label_for(key: str, labels: dict[str, str]) -> str:
    best = ""
    for prefix in labels:
        if key.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return labels[best] if best else ""


def _named_parameters(model) -> dict:
    try:
        return dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return dict(model.named_parameters())


def _grad_norm(parameters: dict) -> float | None:
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


def _loss_from(outputs: Any) -> float | None:
    """Pull the training loss out of whatever Lightning passed to `on_train_batch_end`.

    Lightning 2.x passes a dict, older versions a bare tensor, and a manual-optimization
    step passes whatever `training_step` returned. Anything unrecognised returns None -
    a loss is never guessed.
    """
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        value = outputs.get("loss")
        return _scalar(value)
    if isinstance(outputs, (list, tuple)):
        values = [_scalar(item.get("loss") if isinstance(item, dict) else item)
                  for item in outputs]
        values = [value for value in values if value is not None]
        return values[-1] if values else None
    return _scalar(outputs)


def _scalar(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.detach() if hasattr(value, "detach") else value)
    except (TypeError, ValueError):
        return None


class LightningRecorder(Recorder):
    """A recorder that also accepts a loss from user code (`callback.log(loss=...)`)."""


class ArchxCallback:
    """Records a `pl.Trainer.fit()` run without changing the model or the loop."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(_with_callback_base(cls))

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
        self.path = path
        self.policy = policy
        self.regions = regions or {}
        self.components = components or {}
        self.data_level = data_level
        self.queue_size = queue_size
        self.max_rel_error = max_rel_error
        self.meta = meta or {}
        self.recorder: LightningRecorder | None = None
        self._pending_loss: float | None = None
        self._pending_lr: float | None = None
        self._pending_grad_norm: float | None = None
        self._window_losses: list[float] = []
        self._last_global_step = 0
        self._closed = False

    # ------------------------------------------------------------ public surface

    def log(self, *, loss: float | None = None) -> None:
        """Override the recorded loss for the next update (rarely needed: it is automatic)."""
        if loss is not None:
            self._pending_loss = float(loss)

    # ---------------------------------------------------- lightning callbacks

    def on_fit_start(self, trainer, pl_module) -> None:
        """Build the recorder here: the model and its parameters exist."""
        if not self._is_main_process(trainer):
            return
        parameters = _named_parameters(pl_module)
        registry = TensorRegistry()
        for key, parameter in parameters.items():
            registry.add(
                key,
                parameter.detach().cpu().numpy(),
                pointer=int(parameter.data_ptr()),
                frozen=not bool(getattr(parameter, "requires_grad", True)),
                region=_label_for(key, self.regions),
                component=_label_for(key, self.components),
            )

        def read(key: str):
            return parameters[key].detach().cpu().numpy()

        self._parameters = parameters
        self._window_losses: list[float] = []
        self._last_global_step = int(getattr(trainer, "global_step", 0))
        self._pending_grad_norm: float | None = None
        self.recorder = LightningRecorder(
            registry, read, self.path,
            policy=self.policy, queue_size=self.queue_size,
            max_rel_error=self.max_rel_error, data_level=self.data_level,
            meta={
                "adapter": "lightning",
                "lightning": _version("lightning"),
                "torch": _version("torch"),
                "model_class": type(pl_module).__name__,
                "grad_norm_clipped": False,
                **self.meta,
            })
        self.recorder.capabilities.deny(
            "data_segments",
            "this adapter is not given the batch by any callback it uses; data_level is 'none'")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Where the weights are already new - and where a micro-batch is told from an update.

        Measured, and the opposite of what the hook names suggest: at
        `on_before_optimizer_step` the weights are still **old**, and by
        `on_train_batch_end` they are new. So the store point is taken here, and the gradients
        and learning rate cached just before the update are attached to it.

        `trainer.global_step` has advanced by exactly one when this micro-batch completed an
        update, and not at all when it is an accumulation micro-batch. That is the only
        reliable signal: recording unconditionally at batch end wrote two store points for one
        update, both labelled with the same step number.
        """
        if self.recorder is None:
            return
        loss = _loss_from(outputs)
        if loss is not None:
            self._window_losses.append(loss)

        current = int(getattr(trainer, "global_step", 0))
        if current == self._last_global_step:
            return                                   # an accumulation micro-batch
        self._last_global_step = current

        # The gradient is the mean over the accumulated micro-batches, so the recorded loss
        # is their mean too - the same quantity the update was driven by, rather than
        # whichever micro-batch happened to be last.
        loss = (sum(self._window_losses) / len(self._window_losses)
                if self._window_losses else self._pending_loss)
        self._window_losses = []
        self._pending_loss = None
        grad_norm, self._pending_grad_norm = self._pending_grad_norm, None

        try:
            self.recorder.step(current - 1, loss=loss, grad_norm=grad_norm,
                               lr=self._pending_lr)
        except Exception as exc:                       # noqa: BLE001 - never stop training
            self.recorder._degrade(exc)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        """Cache what only exists before the update: the gradient norm and this lr.

        Nothing is stored here: the weights have not changed yet.
        """
        if self.recorder is None:
            return
        self._pending_lr = float(optimizer.param_groups[0]["lr"])
        self._pending_grad_norm = _grad_norm(self._parameters)
        self._pending_loss = None

    def on_fit_end(self, trainer, pl_module) -> None:
        self.close()

    # ------------------------------------------------------------ lifecycle

    def close(self) -> None:
        if self._closed or self.recorder is None:
            self._closed = True
            return
        self._closed = True
        self.recorder.close()

    @property
    def stats(self):
        return self.recorder.stats if self.recorder is not None else None

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _is_main_process(trainer) -> bool:
        """Only one rank writes. Lightning knows this locally and globally."""
        try:
            from lightning.pytorch.utilities.rank_zero import rank_zero_only  # noqa: F401
        except ImportError:                            # pragma: no cover
            return True
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "is_global_zero"):
            return bool(strategy.is_global_zero)
        return True


def _version(module: str) -> str:
    try:
        return __import__(module).__version__
    except Exception:                                  # noqa: BLE001 - a version is a fact
        return "unknown"

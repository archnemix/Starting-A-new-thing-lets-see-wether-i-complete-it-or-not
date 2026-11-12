"""The code injected by `archx run`: patched into a training process it does not own.

This file is copied verbatim into a temporary directory as `sitecustomize.py`, which the
interpreter imports during start-up. It therefore runs in a process that never asked for it,
against a torch it has not imported yet, and it must be defensive about everything.

Two facts from measurement decide the shape of this file:

1. **`torch` is not importable at `sitecustomize` time.** A first prototype did
   `import torch; patch(...)` inside `try/except` and silently did nothing. The patch is
   installed through a `sys.meta_path` finder that fires when `torch.optim` finishes loading.
2. **Patching `Optimizer.step` alone captures nothing for `AdamW`.** `AdamW` does not define
   `step`; it resolves to `Adam.step`. Measured: zero calls counted for the most common
   optimizer in the world, with no error anywhere. Every optimizer class in `torch.optim` is
   wrapped at its own `step`, and only classes that define one are touched, so nothing is
   double-counted.

Everything is wrapped in `except Exception: pass` on purpose: this code runs inside somebody
else's training job, and a recorder that crashes a run is worse than one that records nothing.
The launcher checks afterwards whether anything was actually recorded, and removes the file
and exits non-zero when it was not - so silence here is never mistaken for success.
"""

from __future__ import annotations

import atexit
import os
import sys

#: Written by the launcher; a copy of this file is placed on `PYTHONPATH` as sitecustomize.
RECORDING_PATH = os.environ.get("ARCHX_LAUNCHER_PATH", "run.archx.mdl")
#: Set in the environment we hand to the child, and removed before spawning it, so the
#: launcher's own interpreter does not patch itself.
FLAG = "ARCHX_LAUNCHER_ACTIVE"

STATE_MODULE = "archx_launcher_state"


def _state():
    """One shared namespace for everything the patch touches, visible across processes."""
    module = sys.modules.get(STATE_MODULE)
    if module is None:
        module = type(sys)(STATE_MODULE)
        sys.modules[STATE_MODULE] = module
        module.steps_seen = 0
        module.patched = []
        module.recorder = None
        module.parameters = {}
        module.pending = False
        module.registry = None
    return module


def _install():
    """Wrap `step` on every optimizer class that defines one. Idempotent."""
    import torch.optim as optim

    state = _state()
    if state.recorder is None:
        try:
            from archx.formats.registry import TensorRegistry  # noqa: F401
            from archx.recorder import Recorder                 # noqa: F401
        except Exception:                       # archx not importable in this interpreter
            return
        state.pending = True

    for name in dir(optim):
        cls = getattr(optim, name, None)
        if not isinstance(cls, type) or not issubclass(cls, optim.Optimizer):
            continue
        if name in state.patched:
            continue
        original = cls.__dict__.get("step")
        if original is None:                    # inherited: the base class is wrapped instead
            continue
        try:
            cls.step = _wrap(original)
            state.patched.append(name)
        except (AttributeError, TypeError):
            continue


def _wrap(original):
    def step(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            _observe(self)
        except Exception:                       # never stop somebody's training
            pass
        return result

    step.__name__ = getattr(original, "__name__", "step")
    step.__doc__ = "archx: wrapped to record the step; the original ran unchanged."
    step._archx_wrapped = True
    return step


def _observe(optimizer):
    state = _state()
    state.steps_seen += 1
    if state.recorder is None and state.pending:
        _open(optimizer)
        if state.recorder is None:
            return
    if state.recorder is not None and state.parameters:
        state.recorder.step(state.steps_seen - 1)


def _open(optimizer):
    """Build a registry from the optimizer's parameters - the only arguments available.

    Their names are not knowable here: a launcher sees an optimizer, not a model. The file
    says so (`names: unavailable`) rather than inventing plausible names.
    """
    state = _state()
    state.pending = False
    try:
        from archx.formats.registry import TensorRegistry
        from archx.recorder import Recorder
    except Exception:
        return
    registry = TensorRegistry()
    parameters = {}
    for group_index, group in enumerate(optimizer.param_groups):
        for position, parameter in enumerate(group.get("params", ())):
            if not hasattr(parameter, "detach"):
                continue
            key = f"param[{group_index}][{position}]"
            parameters[key] = parameter
            registry.add(key, parameter.detach().cpu().numpy(),
                         pointer=int(parameter.data_ptr()),
                         frozen=not bool(getattr(parameter, "requires_grad", True)))
    if not parameters:
        return
    state.parameters = parameters
    state.registry = registry

    def read(key):
        return state.parameters[key].detach().cpu().numpy()

    try:
        state.recorder = Recorder(
            registry, read, RECORDING_PATH,
            meta={"adapter": "launcher", "names": "unavailable",
                  "optimizer": type(optimizer).__name__})
    except Exception:
        state.recorder = None


class _Finder:
    """Install the patch the moment `torch.optim` finishes importing.

    Delegating to the other finders keeps the normal machinery in charge of locating the
    module; only `exec_module` is wrapped, so the module is loaded exactly as it would be.
    """

    tested = ("torch.optim",)

    def find_spec(self, name, path=None, target=None):
        if name not in self.tested:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = getattr(spec, "loader", None)
        if loader is None or not hasattr(loader, "exec_module"):
            return None
        original = loader.exec_module

        def exec_module(module):
            original(module)
            try:
                _install()
            except Exception:
                pass

        loader.exec_module = exec_module
        return spec


def activate():
    """Called once, at import. Safe to call more than once."""
    if os.environ.get(FLAG) == "1":
        return                                  # the launcher's own process, or a re-import
    os.environ[FLAG] = "1"
    try:
        import torch.optim  # noqa: F401 - a late sitecustomize may run after torch loads
    except ImportError:
        sys.meta_path.insert(0, _Finder())
    except Exception:
        sys.meta_path.insert(0, _Finder())
    else:
        try:
            _install()
        except Exception:
            pass


def report() -> dict:
    """Close the recording and return what actually happened. Used by the launcher."""
    state = _state()
    result = {"steps": state.steps_seen, "patched": list(state.patched),
              "recorder": state.recorder is not None, "path": RECORDING_PATH}
    if state.recorder is not None:
        try:
            state.recorder.close()
        except Exception as exc:                # the launcher reports it, not this module
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _atexit():
    try:
        report()
    except Exception:
        pass


if __name__ == "__main__":                      # pragma: no cover - diagnostics only
    print(report())

activate()
atexit.register(_atexit)

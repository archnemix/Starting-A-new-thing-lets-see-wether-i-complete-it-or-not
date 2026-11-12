"""Archnemix Model Life Line: record how a model changes, why, and when.

The public surface is small on purpose:

    import archx
    recorder = archx.attach(model, optimizer, path="run.archx.mdl")

Everything else (encoders, container internals, adapters) is imported from its own module.
"""

from __future__ import annotations

from .version import FORMAT_MAJOR, FORMAT_MINOR, __version__

FORMAT_VERSION = (FORMAT_MAJOR, FORMAT_MINOR)

__all__ = ["__version__", "FORMAT_VERSION", "attach"]


def attach(model, optimizer, path: str, **kwargs):
    """Attach a recorder to an existing PyTorch training loop.

    Imported lazily so `import archx` works without torch installed: the container layer and
    the CLI do not need it.
    """
    from .adapters.torch import attach as _attach

    return _attach(model, optimizer, path, **kwargs)

"""Command line entry points. One module per command family."""

from __future__ import annotations

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Lazy import so `import archx` never pays for the CLI."""
    from .main import main as _main

    return _main(argv)

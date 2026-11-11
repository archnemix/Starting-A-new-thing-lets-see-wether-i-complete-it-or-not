"""`archx` entry point: argument parsing and dispatch only.

Every command lives in its own module; this file must stay readable at a glance.
"""

from __future__ import annotations

import argparse
import sys

from ..version import FORMAT_MAJOR, FORMAT_MINOR, __version__


def build_parser() -> argparse.ArgumentParser:
    """The full CLI surface. Commands are added as they are implemented."""
    parser = argparse.ArgumentParser(
        prog="archx",
        description="Archnemix Model Life Line: record how a model changes, why, and when.",
    )
    parser.add_argument("--version", action="version",
                        version=f"archx {__version__} (format {FORMAT_MAJOR}.{FORMAT_MINOR})")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    info = sub.add_parser("info", help="describe a .archx.mdl file")
    info.add_argument("file", help="path to the recording")
    info.add_argument("--json", action="store_true", help="machine-readable output")

    recover = sub.add_parser("recover", help="rebuild the index of a crashed recording")
    recover.add_argument("file", help="path to the recording")
    recover.add_argument("--out", help="recover into a copy instead of in place")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "info":
        from .info import run_info

        return run_info(args.file, as_json=args.json)

    if args.command == "recover":
        from .recover import run_recover

        return run_recover(args.file, out=args.out)

    parser.error(f"unknown command {args.command!r}")
    return 2

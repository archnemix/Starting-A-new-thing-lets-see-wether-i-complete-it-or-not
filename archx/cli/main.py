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

    reconstruct = sub.add_parser("reconstruct", help="rebuild the weights at a step")
    reconstruct.add_argument("file", help="path to the recording")
    reconstruct.add_argument("--step", type=int, help="step to rebuild (default: the last)")
    reconstruct.add_argument("--out", help="output .npz path")

    events = sub.add_parser("events", help="list events, annotations and data segments")
    events.add_argument("file", help="path to the recording")
    events.add_argument("--json", action="store_true", help="machine-readable output")

    diff = sub.add_parser("diff", help="compare two steps, or two recordings")
    diff.add_argument("file", help="path to the recording")
    diff.add_argument("--other", help="a second recording to compare against")
    diff.add_argument("--step", type=int, help="first step (default: the second-to-last)")
    diff.add_argument("--step-b", type=int, help="second step (default: the last)")
    diff.add_argument("--json", action="store_true", help="machine-readable output")

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

    if args.command == "reconstruct":
        from .reconstruct import run_reconstruct

        return run_reconstruct(args.file, step=args.step, out=args.out)

    if args.command == "events":
        from .events import run_events

        return run_events(args.file, as_json=args.json)

    if args.command == "diff":
        from .diff import run_diff

        return run_diff(args.file, other=args.other, step=args.step, step_b=args.step_b,
                        as_json=args.json)

    parser.error(f"unknown command {args.command!r}")
    return 2

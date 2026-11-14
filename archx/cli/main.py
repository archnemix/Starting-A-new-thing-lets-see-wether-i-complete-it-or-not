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

    demo = sub.add_parser("demo", help="record a small run you can inspect immediately")
    demo.add_argument("--out", default="demo.archx.mdl", help="where to write (default: %(default)s)")
    demo.add_argument("--steps", type=int, default=20, help="how many steps (default: %(default)s)")

    acts = sub.add_parser("activations", help="layer-by-layer activation summaries")
    acts.add_argument("file", help="path to the recording")
    acts.add_argument("--layer", help="only this module name")
    acts.add_argument("--step", type=int, help="only this step")
    acts.add_argument("--json", action="store_true", help="machine-readable output")

    diff = sub.add_parser("diff", help="compare two steps, or two recordings")
    diff.add_argument("file", help="path to the recording")
    diff.add_argument("--other", help="a second recording to compare against")
    diff.add_argument("--step", type=int, help="first step (default: the second-to-last)")
    diff.add_argument("--step-b", type=int, help="second step (default: the last)")
    diff.add_argument("--json", action="store_true", help="machine-readable output")

    history = sub.add_parser("history", help="show batch-linked history of a tensor or weight")
    history.add_argument("file", help="path to the recording")
    history.add_argument("--tensor", required=True, help="recorded tensor name")
    select = history.add_mutually_exclusive_group()
    select.add_argument("--index", help="comma-separated indices, e.g. 0,3")
    select.add_argument("--row", type=int, help="tensor row (not automatically a neuron)")
    history.add_argument("--data", action="store_true", help="include recorded batch contents")
    history.add_argument("--cochanges", type=int, default=0, metavar="N", help="show up to N co-changing weights per update (max 10000)")
    history.add_argument("--json", action="store_true", help="machine-readable output")

    # `archx run` takes everything after `--` verbatim, which is why it must not be
    # parsed by argparse beyond the recording path: argparse would eat the training
    # script's own flags, including `--help`.
    launch = sub.add_parser(
        "run", help="record a training script without editing it",
        description="Record any command that steps a torch optimizer, without touching its "
                    "code. Everything after `--` is the command.")
    launch.add_argument("--recording", default="run.archx.mdl",
                        help="where to write the recording (default: run.archx.mdl)")
    # Deliberately no positional for the command: see `main`, which splits the argv itself.
    # `argparse.REMAINDER` looks like the right tool and is not - inside a subparser it
    # swallows the subcommand name, so `archx run -- python x.py` parsed `run` as the
    # command. The training script's own `--help` would also have been consumed by archx.

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and return a process exit code."""
    argv = list(argv if argv is not None else sys.argv[1:])

    # `archx run` is split off before argparse runs, so the command after `--` reaches the
    # child untouched - including flags archx itself defines, and `--help`.
    if argv and argv[0] == "run":
        return _run_launcher(argv[1:])

    parser = build_parser()
    args = parser.parse_args(argv)

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

    if args.command == "demo":
        from .demo import run_demo

        return run_demo(args.out, steps=args.steps)

    if args.command == "activations":
        from .activations import run_activations

        return run_activations(args.file, as_json=args.json, layer=args.layer,
                               step=args.step)

    if args.command == "events":
        from .events import run_events

        return run_events(args.file, as_json=args.json)

    if args.command == "history":
        from .history import run_history
        return run_history(args.file, tensor=args.tensor, index=args.index, row=args.row,
                           include_data=args.data, cochanges=args.cochanges, as_json=args.json)

    if args.command == "diff":
        from .diff import run_diff

        return run_diff(args.file, other=args.other, step=args.step, step_b=args.step_b,
                        as_json=args.json)

    parser.error(f"unknown command {args.command!r}")
    return 2


def _run_launcher(argv: list[str]) -> int:
    """Split `archx run [--recording PATH] -- command...` and hand it to the launcher."""
    from ..launcher import main as run_launcher

    recording = "run.archx.mdl"
    rest: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--":
            rest = argv[index + 1:]
            break
        if item == "--recording":
            if index + 1 >= len(argv):
                print("archx: --recording needs a path", file=sys.stderr)
                return 2
            recording = argv[index + 1]
            index += 2
            continue
        if item.startswith("--recording="):
            recording = item.split("=", 1)[1]
            index += 1
            continue
        if item in ("-h", "--help"):
            print("usage: archx run [--recording PATH] -- <command> [args...]\n\n"
                  "Records any command that steps a torch optimizer, without editing it.\n"
                  "Everything after `--` is run unchanged.")
            return 0
        print(f"archx: unexpected argument {item!r}; put the command after `--`",
              file=sys.stderr)
        return 2
    return run_launcher(rest, recording=recording)


if __name__ == "__main__":      # `python -m archx.cli.main`, used by the CLI tests
    raise SystemExit(main())

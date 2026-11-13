"""`archx reconstruct`: rebuild the weights at a step and write them out.

Writes a plain .npz - no pickle - so the result is inspectable with numpy alone and cannot
execute code when loaded. This is also the path that proves the final anchor is a usable
model snapshot.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..container.reader import Reader
from ..container.values import Timeline
from ..errors import ArchxError


def run_reconstruct(path: str, *, step: int | None = None, out: str | None = None) -> int:
    """Reconstruct one step (default: the last) into `out`. Returns a shell exit code.

    A step the file does not store is still answerable, from the anchor at or before it plus
    every delta up to it - that is the format's whole promise. Only a step outside the
    recorded range is refused.
    """
    # `allow_torn=True` for the same reason `archx info` uses it: a crashed recording is
    # exactly when someone asks for the weights, and every store point before the torn tail is
    # intact. Reading the file strictly made this command refuse a file that `info` had just
    # read and reported nine recoverable steps in - the two commands disagreed about what the
    # same bytes were, and the stricter one was the one with the answer.
    try:
        reader = Reader(path, allow_torn=True)
        timeline = Timeline(reader)
    except ArchxError as exc:
        print(f"archx: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    torn = not reader.info()["closed"]

    steps = timeline.steps
    if not steps:
        print(f"archx: {path} contains no recorded steps", file=sys.stderr)
        return 1
    target = steps[-1] if step is None else step
    # A step between two stored steps is answerable from the file: the nearest anchor at or
    # before it, plus every delta up to it. This command used to refuse those with "step N is
    # not recorded", which is the library's own guarantee contradicted by its own CLI - the
    # product's headline claim, and `archx reconstruct --step 2` on a real run exited 1. What
    # cannot be answered is a step before the first anchor or after the last stored step, and
    # those still exit 1 with the range it does have.
    if target < steps[0] or target > steps[-1]:
        print(f"archx: the file covers steps {steps[0]}-{steps[-1]}; step {target} is outside it",
              file=sys.stderr)
        return 1

    try:
        state = timeline.reconstruct(target)
    except ArchxError as exc:
        print(f"archx: cannot reconstruct step {target}: {exc}", file=sys.stderr)
        return 1

    destination = Path(out or f"step-{target}.npz")
    np.savez(destination, **state)
    total = sum(int(a.size) for a in state.values())
    print(f"step           {target}")
    if target in steps:
        print("stored         yes")
    else:
        # Say where the answer came from, so a reconstruction of a skipped step is never
        # mistaken for one taken at a store point.
        base = max(s for s in steps if s <= target)
        print(f"stored         no  (latest prior stored state is step {base}; not exact for step {target})")
    print(f"tensors        {len(state)}")
    print(f"values         {total}")
    if torn:
        print("warning        the file was not closed by its writer; this is rebuilt from the "
              "last whole block (run `archx recover` to close it)")
    print(f"written        {destination}")
    return 0

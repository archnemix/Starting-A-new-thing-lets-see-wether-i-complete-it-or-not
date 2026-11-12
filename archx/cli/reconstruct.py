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
    """Reconstruct one step (default: the last) into `out`. Returns a shell exit code."""
    try:
        timeline = Timeline(Reader(path))
    except ArchxError as exc:
        print(f"archx: cannot read {path}: {exc}", file=sys.stderr)
        return 1

    steps = timeline.steps
    if not steps:
        print(f"archx: {path} contains no recorded steps", file=sys.stderr)
        return 1
    target = steps[-1] if step is None else step
    if target not in steps:
        print(f"archx: step {target} is not recorded; steps are {steps[:5]}..."
              if len(steps) > 5 else f"archx: step {target} is not recorded; steps are {steps}",
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
    print(f"tensors        {len(state)}")
    print(f"values         {total}")
    print(f"written        {destination}")
    return 0

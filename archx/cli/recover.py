"""`archx recover`: turn a crashed recording into a closed, indexed one."""

from __future__ import annotations

import sys

from ..container.recovery import recover_copy, recover_in_place
from ..errors import ArchxError


def run_recover(path: str, *, out: str | None = None) -> int:
    """Recover in place, or into `out` when given. Returns a shell exit code."""
    try:
        report = recover_copy(path, out) if out else recover_in_place(path)
    except ArchxError as exc:
        print(f"archx: cannot recover {path}: {exc}", file=sys.stderr)
        return 1

    print(f"recovered      {report.path}")
    print(f"blocks kept    {report.blocks_kept}")
    print(f"bytes dropped  {report.bytes_truncated}")
    print("footer         written")
    # Without this line the run's own capability statement is simply gone from a recovered
    # file, and `archx info` reports the floor tier for a recording that is entirely intact
    # except for its tail. Say what was rebuilt, and show the tier it settled on.
    if report.capability_written:
        from ..container.reader import Reader

        tier = Reader(report.path).capabilities().get("capability_tier")
        print(f"capabilities   rebuilt from the blocks that survived; tier {tier}")
    else:
        print("capabilities   kept as the run wrote them")
    return 0

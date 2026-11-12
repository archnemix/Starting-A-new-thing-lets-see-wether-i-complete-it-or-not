"""`archx events`: the timeline - per-step observations, user annotations, data segments.

Three streams, printed separately because they answer three different questions: what the
run measured at a step, what the user declared happened, and which data produced which
movement. Merging them into one list would hide that distinction.
"""

from __future__ import annotations

import json
import sys

from ..container.reader import Reader
from ..container.values import Timeline
from ..errors import ArchxError


def run_events(path: str, *, as_json: bool = False) -> int:
    """Print the event stream, annotations and segment spans. Returns a shell exit code."""
    try:
        timeline = Timeline(Reader(path))
    except ArchxError as exc:
        print(f"archx: cannot read {path}: {exc}", file=sys.stderr)
        return 1

    observations = timeline.events()
    annotations = timeline.annotations()
    segments = timeline.segments()
    if as_json:
        print(json.dumps({"steps": timeline.steps, "observations": observations,
                          "annotations": annotations, "segments": segments},
                         indent=2, sort_keys=True))
        return 0

    print(f"steps          {len(timeline.steps)}")
    print(f"observations   {len(observations)}")
    for event in observations[:50]:
        loss = event.get("loss")
        grad = event.get("grad_norm")
        regions = event.get("regions") or {}
        moved = max(regions.items(), key=lambda item: item[1])[0] if regions else ""
        print(f"  step {event.get('step'):>6}  loss {_num(loss):>10}  "
              f"grad {_num(grad):>10}  moved {moved}")
    print(f"annotations    {len(annotations)}")
    for note in annotations:
        print(f"  step {note.get('step'):>6}  {note.get('event', ''):<16} {note.get('detail', '')}")
    print(f"segments       {len(segments)}")
    for segment in segments[:50]:
        chars = segment.get("characteristics", {})
        print(f"  {segment['segment_id'][:16]:<16} steps {segment['first_step']}-"
              f"{segment['last_step']}  seen {segment['times_seen']}x  "
              f"{chars.get('kind', '?')} {chars.get('bytes', 0)}B")
    return 0


def _num(value) -> str:
    """Format an optional float without inventing a zero."""
    return "-" if value is None else f"{value:.4g}"

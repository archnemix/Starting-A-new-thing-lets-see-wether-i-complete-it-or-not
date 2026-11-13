"""`archx diff`: what changed between two steps, or between two recordings.

Two questions wear the same name and deserve different answers:

- two steps of one run - which tensors moved, how far, and which regions account for it. This
  is the "where did the model change" question, and it is answerable from one file.
- two files - whether the same steps exist, how the histories diverge, and where the first
  difference is. This is the "did these two runs do the same thing" question.

Distance is reported relative to each tensor's own size, because an absolute norm says nothing
on its own: 0.5 is a rounding error for a layer of norm 40 and a catastrophe for one of norm
0.1, and the point is to compare tensors of different shapes.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from ..container.reader import Reader
from ..container.values import Timeline
from ..errors import ArchxError


def run_diff(path: str, *, other: str | None = None, step: int | None = None,
             step_b: int | None = None, as_json: bool = False) -> int:
    """Compare two steps of one file, or one file against another. Returns a shell exit code."""
    try:
        if other:
            return _diff_files(path, other, as_json=as_json)
        return _diff_steps(path, step, step_b, as_json=as_json)
    except ArchxError as exc:
        print(f"archx: {exc}", file=sys.stderr)
        return 1


def _diff_steps(path: str, step_a: int | None, step_b: int | None, *, as_json: bool) -> int:
    timeline = Timeline(Reader(path))
    steps = timeline.steps
    if not steps:
        print(f"archx: {path} stores no steps", file=sys.stderr)
        return 1
    # The defaults answer the question people actually ask - "what changed since the last
    # thing I looked at" - so the last two stored steps are the pair.
    step_a = steps[-2] if step_a is None and len(steps) > 1 else (step_a if step_a is not None else steps[0])
    step_b = steps[-1] if step_b is None else step_b
    for name in (step_a, step_b):
        if name not in steps:
            # Not "cannot be rebuilt": the file *can* answer a step between two store points
            # (that is `archx reconstruct`, and it is the format's headline promise). This
            # command compares store points, so the honest refusal names the wrong thing and
            # points at the command that does answer it - the earlier wording contradicted
            # the library it belongs to.
            if steps[0] < name < steps[-1]:
                print(f"archx: step {name} is not a store point, so there is no recorded "
                      f"movement to compare at it; stored steps are {_span(steps)}. "
                      f"For the weights at step {name}, use `archx reconstruct`.", file=sys.stderr)
            else:
                print(f"archx: step {name} is outside the recorded range {steps[0]}-{steps[-1]}; "
                      f"stored steps are {_span(steps)}", file=sys.stderr)
            return 1

    before = timeline.reconstruct(step_a)
    after = timeline.reconstruct(step_b)
    regions = _regions_by_step(timeline)
    rows = []
    for key in before:
        if key not in after:
            continue
        b = np.asarray(before[key], dtype=np.float64).ravel()
        a = np.asarray(after[key], dtype=np.float64).ravel()
        if a.size != b.size:
            continue
        moved = float(np.linalg.norm(a - b))
        size = float(np.linalg.norm(b))
        regions_of = regions.get(key, "")
        rows.append({
            "tensor": key,
            "region": regions_of,
            "shape": list(np.asarray(after[key]).shape),
            "moved": moved,
            "relative": moved / size if size else None,
        })
    rows.sort(key=lambda row: row["moved"], reverse=True)

    total_moved = sum(row["moved"] for row in rows)
    payload = {
        "path": path,
        "step_a": step_a,
        "step_b": step_b,
        # The number of recorded updates the difference is the sum of, so that `moved` can be
        # read per update. It used to be called "steps_between" and printed as "8 stored
        # steps in the file between them" for step 0 -> 19 with 9 store points in the file:
        # eight is the count of intervals, not of steps in between (seven), and the label
        # said the wrong one of the two true numbers.
        "updates_folded": len([s for s in steps if step_a < s <= step_b]),
        "steps_between": len([s for s in steps if step_a < s < step_b]),
        "tensors": rows,
        "total_moved": total_moved,
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"comparing  step {step_a} -> step {step_b}  "
          f"({payload['updates_folded']} recorded update(s) folded into the difference)")
    print(f"{'tensor':<44}{'shape':>14}{'moved':>12}{'relative':>11}  region")
    print("-" * 92)
    for row in rows[:25]:
        relative = "-" if row["relative"] is None else f"{row['relative']:.3e}"
        print(f"{row['tensor'][:43]:<44}{str(tuple(row['shape'])):>14}"
              f"{row['moved']:>12.4g}{relative:>11}  {row['region']}")
    if len(rows) > 25:
        print(f"... {len(rows) - 25} more tensors")
    unchanged = sum(1 for row in rows if row["moved"] == 0)
    if unchanged:
        print(f"\n{unchanged} tensor(s) did not move at all")
    return 0


def _diff_files(path: str, other: str, *, as_json: bool) -> int:
    first, second = Timeline(Reader(path)), Timeline(Reader(other))
    common = sorted(set(first.steps) & set(second.steps))
    only_first = sorted(set(first.steps) - set(second.steps))
    only_second = sorted(set(second.steps) - set(first.steps))

    # The first step where the two files disagree is the useful answer: after that, every later
    # step differs because it inherited the difference, and listing them all says nothing new.
    first_difference = None
    rows = []
    compared = 0
    for step in common:
        left, right = first.reconstruct(step), second.reconstruct(step)
        worst = None
        for key in left:
            if key not in right:
                continue
            a = np.asarray(left[key], dtype=np.float64).ravel()
            b = np.asarray(right[key], dtype=np.float64).ravel()
            if a.size != b.size:
                continue
            compared += 1
            size = float(np.linalg.norm(a))
            relative = float(np.linalg.norm(b - a)) / size if size else 0.0
            if relative and (worst is None or relative > worst["relative"]):
                worst = {"tensor": key, "relative": relative}
        if worst:
            rows.append({"step": step, **worst})
            if first_difference is None:
                first_difference = {"step": step, **worst}

    # Two files recorded by different adapters name their tensors differently (`embed.weight`
    # against `param[0][0]`), so they can share step numbers and compare nothing. The emptiness
    # check above read as "they agree at every shared step" - the strongest verdict this
    # command can give - on zero comparisons. It is not a difference and it is not agreement;
    # it is a file pair this command cannot answer, and it has to say so.
    shared_keys = {info.key for info in first.tensors} & {info.key for info in second.tensors}
    payload = {
        "a": path,
        "b": other,
        "steps_a": len(first.steps),
        "steps_b": len(second.steps),
        "common_steps": len(common),
        "compared_values": compared,
        "shared_tensors": len(shared_keys),
        "only_in_a": only_first,
        "only_in_b": only_second,
        "first_difference": first_difference,
        "differences": rows,
    }
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"a              {path}  ({len(first.steps)} stored steps)")
    print(f"b              {other}  ({len(second.steps)} stored steps)")
    print(f"common steps   {len(common)}")
    if only_first:
        print(f"only in a      {_span(only_first)}")
    if only_second:
        print(f"only in b      {_span(only_second)}")
    if compared == 0:
        print(f"shared tensors 0 of {len(first.tensors)} and {len(second.tensors)}: the two files "
              f"name their weights differently")
        print("verdict        cannot compare - no tensor name is present in both files")
        return 0
    if first_difference is None:
        print(f"verdict        identical at every common step "
              f"({compared:,} values compared)")
        return 0
    print(f"verdict        first difference at step {first_difference['step']}: "
          f"{first_difference['tensor']} by {first_difference['relative']:.3e} "
          f"relative")
    print(f"steps differing {len(rows)} of {len(common)}")
    return 0


def _regions_by_step(timeline: Timeline) -> dict[str, str]:
    """Tensor -> region, from the registry every value block is parsed against."""
    return {info.key: info.region for info in timeline.tensors}


def _span(steps: list[int]) -> str:
    """`1-9, 40, 99` - the steps present, without spelling out a long run."""
    if not steps:
        return "none"
    parts, start, previous = [], steps[0], steps[0]
    for step in steps[1:]:
        if step == previous + 1:
            previous = step
            continue
        parts.append(f"{start}" if start == previous else f"{start}-{previous}")
        start = previous = step
    parts.append(f"{start}" if start == previous else f"{start}-{previous}")
    return ", ".join(parts)

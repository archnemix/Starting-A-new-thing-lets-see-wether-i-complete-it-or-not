"""`archx activations`: what the network saw, layer by layer, at each stored step.

The third stream of the recording, and the one most likely to be read by a person looking for
a broken model: a layer whose outputs collapse to zero, or saturate, is visible here and
nowhere else - the weights look fine and the loss curve looks fine.
"""

from __future__ import annotations

import json
import sys

from ..container.reader import Reader
from ..container.schema import BlockType
from ..errors import ArchxError


def run_activations(path: str, *, as_json: bool = False, layer: str | None = None,
                    step: int | None = None) -> int:
    """Print activation summaries. Returns a shell exit code."""
    try:
        reader = Reader(path)
    except ArchxError as exc:
        print(f"archx: cannot read {path}: {exc}", file=sys.stderr)
        return 1

    rows = []
    for block in reader.blocks(BlockType.ACTIVATION):
        payload = json.loads(block.payload)
        if layer is not None and payload.get("layer") != layer:
            continue
        if step is not None and payload.get("step") != step:
            continue
        rows.append(payload)

    if not rows:
        # Not an error: a file below tier 3 legitimately has none, and saying so is the point.
        tiers = reader.capabilities()
        tier = tiers.get("capability_tier")
        reason = (tiers.get("capabilities_missing") or {}).get("activations", "")
        print("no activation summaries in this file.")
        if tier is not None:
            print(f"  the run reached tier {tier}.")
        if reason:
            print(f"  activations missing: {reason}")
        print("  record with `archx.attach(..., activations=True)` to reach tier 3.")
        return 0

    if as_json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    layers = sorted({row["layer"] for row in rows})
    print(f"layers         {len(layers)}  {', '.join(layers[:8])}"
          f"{' ...' if len(layers) > 8 else ''}")
    print(f"summaries      {len(rows)}")
    print(f"{'step':>6} {'layer':>10} {'shape':>12} {'mean':>9} {'std':>9} "
          f"{'min':>9} {'max':>9} {'zero%':>6} {'sat%':>6}")
    print("-" * 88)
    for row in rows[:60]:
        shape = "x".join(str(d) for d in row.get("shape", []))
        print(f"{row.get('step', -1):>6} {row.get('layer', '')[:10]:>10} {shape:>12} "
              f"{row.get('mean', 0):>9.4g} {row.get('std', 0):>9.4g} "
              f"{row.get('min', 0):>9.4g} {row.get('max', 0):>9.4g} "
              f"{row.get('zero_fraction', 0) * 100:>6.2f} {row.get('saturated_fraction', 0) * 100:>6.2f}")
    if len(rows) > 60:
        print(f"... {len(rows) - 60} more")

    # The questions this stream exists to answer, stated rather than left to the reader's eye.
    dead = [(r["step"], r["layer"], r["zero_fraction"]) for r in rows
            if r.get("zero_fraction", 0) >= 0.99 and r.get("count", 0)]
    hot = [(r["step"], r["layer"], r["saturated_fraction"]) for r in rows
           if r.get("saturated_fraction", 0) >= 0.5]
    nonfinite = [(r["step"], r["layer"], r.get("nonfinite")) for r in rows if r.get("nonfinite")]
    if dead:
        print(f"\ndead layers (>=99% exactly zero): {len(dead)} "
              f"e.g. step {dead[0][0]} layer {dead[0][1]}")
    if hot:
        print(f"saturated layers (>=50% |x| >= 4): {len(hot)} "
              f"e.g. step {hot[0][0]} layer {hot[0][1]}")
    if nonfinite:
        print(f"non-finite values: {len(nonfinite)} summaries, "
              f"first at step {nonfinite[0][0]} layer {nonfinite[0][1]}")
    return 0

"""Claim 7: what an optimizer hook can reach, and what it costs to reach the rest.

The claim on the box is "attach in one line". That is true, and it is also a boundary: the hook
fires after `optimizer.step()`, so it can read the weights and the gradients - both still alive
at that moment - and it cannot read the loss or the batch, which are locals that died with the
iteration.

So there are three levels of help, and this benchmark measures each one rather than arguing
about it:

- **hook only**: `archx.attach(...)`, nothing else. No loss, no batch, no learning rate.
- **hook + log**: one line before `optimizer.step()`, `recorder.log(loss=..., data=x)`.
- **hook + log + annotate**: the same, plus a stage annotation at the point the loop changes
  what it is doing.

For each level: the capability tier the file reports, which fields are in it, what fraction of
the steps carry each field, and the per-step cost on the training thread.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CI without torch reports the skip
        raise SkipBench(f"torch is not installed: {exc}") from exc
    return torch


class SkipBench(RuntimeError):
    """The benchmark cannot run here. Reported as a skip, never as a pass."""


def make_model(seed: int = 0):
    torch = _torch()
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.ReLU(),
                               torch.nn.Linear(128, 10))


def data(steps: int, seed: int = 1):
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(8 * steps, 64, generator=generator)
    y = torch.randint(0, 10, (8 * steps,), generator=generator)
    for step in range(steps):
        yield x[step * 8:(step + 1) * 8], y[step * 8:(step + 1) * 8]


def run(level: str, steps: int, path: Path) -> dict:
    """Train under one level of help and report what the file ended up with."""
    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType

    torch = _torch()
    model = make_model(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    recorder = archx.attach(model, optimizer, path=str(path), data_level="segments")
    batches = data(steps)
    charged = 0.0          # time our code is responsible for, on the training thread
    for step, (inputs, targets) in enumerate(batches):
        started = time.perf_counter()
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        charged += time.perf_counter() - started

        if level in ("log", "annotate"):
            mark = time.perf_counter()
            recorder.log(loss=float(loss.item()), data=inputs)
            charged += time.perf_counter() - mark
            if level == "annotate" and step == steps // 2:
                recorder.annotate(step, "lr_decay", "halfway")
        optimizer.step()
    recorder.close()

    reader = Reader(path)
    events = [json.loads(b.payload) for b in reader.blocks(BlockType.EVENT)]
    info = [json.loads(b.payload) for b in reader.blocks(BlockType.INFO)]
    capability = next(b for b in info if b.get("event") == "capabilities")
    annots = [json.loads(b.payload) for b in reader.blocks(BlockType.ANNOT)]

    def share(field: str) -> float:
        if not events:
            return 0.0
        return sum(1 for e in events if e.get(field) is not None) / len(events)

    return {
        "level": level,
        "tier": capability["capability_tier"],
        "present": capability.get("capabilities_present"),
        "missing": sorted(capability["capabilities_missing"]),
        "events": len(events),
        "steps": steps,
        "has_loss": share("loss"),
        "has_grad_norm": share("grad_norm"),
        "has_data": 1.0 if any(True for _ in reader.blocks(BlockType.DATA_SEGMENT)) else 0.0,
        "annotations": len(annots),
        "bytes": path.stat().st_size,
        "charged_ms": charged / max(1, steps) * 1000.0,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        _torch()
    except SkipBench as exc:
        print(f"skipped: {exc}")
        return 2

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for level in ("hook", "log", "annotate"):
            rows.append(run(level, args.steps, Path(tmp) / f"{level}.archx.mdl"))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = (f"{'help given':<10}{'tier':>5}{'events':>8}{'loss':>7}{'grad':>7}{'data':>6}"
              f"{'notes':>7}{'bytes':>10}{'charged ms/step':>17}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['level']:<10}{row['tier']:>5}{row['events']:>8}{row['has_loss']:>7.0%}"
              f"{row['has_grad_norm']:>7.0%}{row['has_data']:>6.0%}{row['annotations']:>7}"
              f"{row['bytes']:>10}{row['charged_ms']:>17.3f}")
    print("\nmissing fields, per level")
    for row in rows:
        print(f"  {row['level']:<10} {', '.join(row['missing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

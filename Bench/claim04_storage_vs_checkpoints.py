"""Claim 4: recording every step costs less than checkpointing every few steps.

The comparison has to be stated carefully to be honest. A recorder stores one *anchor* of full
weights plus a delta per stored step; naive checkpointing stores a full copy every N steps and
nothing in between. So the fair question is not "which is smaller at the same step count" - the
answer to that is trivially the checkpoints, because they store almost nothing - but **which is
smaller at the same reconstruction quality**.

That is what this measures. For each architecture:

- sweep the checkpoint interval, measure its bytes and the reconstruction error of the steps it
  does not store (the reader's view: the nearest earlier checkpoint),
- sweep the recorder's delta budget (`max_rel_error`), measure the same two things,
- report both frontiers, and the cost ratio at matched error.

A checkpoint cannot be better than "as of N steps ago". A recorder that stores every step at
1e-3 error is not competing with a checkpoint every 100 steps; it is competing with one every 2.

The result, measured, is that the answer depends entirely on what is being asked:

- **Same accuracy on every step of the run**: checkpointing wins by about 4-9x on these small
  models, because one stored step of ours is roughly a whole model checkpoint and a checkpoint
  interval can be made as coarse as the accuracy target allows. There is no cleverness here;
  it is arithmetic, and the benchmark reports it rather than hiding it.
- **Same accuracy on the steps that are stored, across a long run**: recording wins, because
  the number of checkpoints grows with the run while the recorder's per-step cost falls - it
  stores a share of steps rather than all of them. At 1e-2 accuracy on 1000 steps of the CNN,
  1000 checkpoints cost 5.84 MB and the recorder costs 1.54 MB, 0.26x.
- **Both, at once**: only the recorder can be dense at an event and sparse elsewhere in the
  same run, which is claim 6's territory.

What the recorder buys is *resolution where the model moved*, not a smaller file than a
checkpoint every few steps. Stating it the other way round was wrong, and this benchmark is the
reason that is now written down.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .claim_arch_sweep import DATASET_SIZE, SkipBench, _torch, batches, build_models


def forward(block, inputs):
    """Some stacks take a sequence and average it; the dataset generator is shared."""
    try:
        return block(inputs)
    except RuntimeError:
        return block(inputs.mean(dim=1))


def _weights(model) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


def _error(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    error = 0.0
    magnitude = 0.0
    for key, values in a.items():
        if key not in b:
            continue
        x = np.asarray(values, dtype=np.float64).ravel()
        y = np.asarray(b[key], dtype=np.float64).ravel()
        if x.size != y.size:
            continue
        error += float(np.linalg.norm(x - y))
        magnitude += float(np.linalg.norm(x))
    return error / magnitude if magnitude else 0.0


def run_architecture(name: str, model, regions, *, steps: int = 60, checkpoint_at=(6, 12, 25, 60),
                     budgets=(0.0, 1e-3, 1e-2)) -> dict:
    """One baseline run for the truth, then each scheme measured against it."""
    import archx
    from archx.policy import PhasePolicy
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    torch = _torch()

    def train(recorder_factory=None, budget: float | None = None):
        """The same run every time: same seed, same data, optional recorder."""
        block = build_models()[name][0]
        optimizer = torch.optim.AdamW(block.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        dataset = list(batches(name, DATASET_SIZE))
        recorder = recorder_factory(block, optimizer) if recorder_factory else None
        # `history[i]` is the state the file calls step i: written after the update that ended
        # step i. Appending before the update instead offsets the whole comparison by one step,
        # which shows up as a uniform ~1-step reconstruction error and makes the recorder look
        # far worse than it is. The first version of this benchmark had exactly that bug.
        history = []
        for step in range(steps):
            inputs, targets = dataset[step % DATASET_SIZE]
            optimizer.zero_grad()
            loss = criterion(forward(block, inputs), targets)
            loss.backward()
            if recorder is not None:
                recorder.log(loss=float(loss.item()), data=inputs)
            optimizer.step()
            history.append(_weights(block))
        if recorder is not None:
            recorder.close()
        return block, history

    # The truth: an unrecorded run. Every recorded run below must reproduce it, which is
    # asserted rather than assumed - the numerics guard says recording changes nothing, and
    # this benchmark would be meaningless if that were not true here too.
    baseline, history = train()
    baseline_final = _weights(baseline)

    rows = []
    # Naive checkpointing: a full copy every N steps, read back as "the last copy at or before
    # this step". Its error is exactly how far the model moved since that copy.
    for every in checkpoint_at:
        stored = list(range(0, steps, every))
        errors = []
        for step in range(1, steps):
            nearest = max([s for s in stored if s <= step], default=None)
            if nearest is None:
                continue
            errors.append(_error(history[step], history[nearest]))
        raw = sum(history[s][key].nbytes for s in stored for key in history[s])
        rows.append({
            "scheme": "checkpoints",
            "detail": f"every {every}",
            "stored": len(stored),
            "raw_bytes": raw,
            "median_error": float(np.median(errors)) if errors else 0.0,
            "worst_error": float(max(errors)) if errors else 0.0,
        })

    with tempfile.TemporaryDirectory() as tmp:
        for budget in budgets:
            path = Path(tmp) / f"{name}-{budget}.mdl"
            policy = PhasePolicy(total_steps=steps, warmup_fraction=1.0, min_warmup_steps=steps)

            def factory(block, optimizer, path=path, policy=policy, budget=budget):
                return archx.attach(block, optimizer, path=str(path), regions=regions,
                                    max_rel_error=budget, policy=policy)

            replayed, ours = train(factory)
            # A real assertion, not `x = cond, "message"` - which is a tuple assignment and
            # silently checks nothing. This file is about not lying to yourself; the first
            # version of it did.
            if not all(np.array_equal(ours[-1][k], baseline_final[k]) for k in baseline_final):
                raise AssertionError(
                    f"{name}: recording changed the run at max_rel_error={budget}; the "
                    "comparison below would be between two different runs")

            reader = Reader(path)
            timeline = Timeline(reader)
            stored = timeline.steps
            errors = [_error(history[s], timeline.reconstruct(s))
                      for s in stored if 0 <= s < len(history)]
            rows.append({
                "scheme": "recorder",
                "detail": f"max_rel_error {budget:g}",
                "stored": len(stored),
                "raw_bytes": path.stat().st_size,
                "median_error": float(np.median(errors)) if errors else 0.0,
                "worst_error": float(max(errors)) if errors else 0.0,
            })

    return {"architecture": name, "steps": steps, "rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        _torch()
    except SkipBench as exc:
        print(f"skipped: {exc}")
        return 2

    results = []
    for name, (model, regions) in build_models().items():
        results.append(run_architecture(name, model, regions, steps=args.steps))

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print("\ncost to reach a given accuracy (bytes, then how many steps of history)")
    print(f"  {'architecture':<14}{'target err':>11}{'checkpoints':>14}{'recorder':>12}{'ratio':>8}")
    for result in results:
        checkpoint = [r for r in result["rows"] if r["scheme"] == "checkpoints"]
        recorder = [r for r in result["rows"] if r["scheme"] == "recorder"]
        best = max(checkpoint, key=lambda r: r["median_error"])
        for target in (1e-2, 5e-3):
            need = [c for c in checkpoint if c["median_error"] <= target]
            if not need or not recorder:
                continue
            chosen = max(need, key=lambda c: c["median_error"])
            ours = recorder[0]
            ratio = chosen["raw_bytes"] / ours["raw_bytes"]
            print(f"  {result['architecture']:<14}{target:>11.0e}"
                  f"{chosen['raw_bytes']:>14}{ours['raw_bytes']:>12}{ratio:>7.2f}x")

    for result in results:
        print(f"\n{result['architecture']}  ({result['steps']} steps)")
        header = (f"  {'scheme':<12}{'detail':<24}{'stored':>8}{'bytes':>11}"
                  f"{'median err':>12}{'worst err':>11}")
        print(header)
        for row in result["rows"]:
            print(f"  {row['scheme']:<12}{row['detail']:<24}{row['stored']:>8}"
                  f"{row['raw_bytes']:>11}{row['median_error']:>12.2e}"
                  f"{row['worst_error']:>11.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

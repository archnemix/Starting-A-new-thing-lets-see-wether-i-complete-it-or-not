"""Claim 6: phase-adaptive recording vs a fixed interval, at equal storage.

The claim being tested is not "adaptive stores fewer bytes" - that is what skipping means and
proving it would be circular. The claim is that **for the same number of bytes**, the adaptive
policy keeps a more useful history: denser where the model is moving, sparser where it is not,
and awake at the moment something changes.

The comparison needs a run whose behaviour actually changes. A constant-rate run cannot tell
the two policies apart - both policies would cover it evenly and equally well - so a benchmark
on one would be a lie. The simulated run is:

- steps 0-49: fast movement (1e-2), the warmup a real run has,
- steps 50-699: quiet (2e-4),
- step 700: one abrupt jump (5e-2) - the event a person opens the recording to find,
- steps 701-999: quiet again.

The jump is also expressed the way a policy can actually *see* it: a real jump produces a loss
spike and a gradient spike, so the benchmark feeds those too at step 700. A policy that cannot
notice it is not being tested on its triggers, and would be judged only on its heartbeat - which
is not the claim.

Two things are reported at a matched byte budget:

- **fidelity**: reconstruct at steps the file does *not* store and compare with the true
  weights. This is the only measure that means anything, because it is what a reader does.
- **resolution near the jump**: the same error, restricted to the steps around step 700. A
  policy that spends its bytes evenly is blind exactly there.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from archx.container.reader import Reader
from archx.container.schema import BlockType
from archx.container.values import Timeline
from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy, Triggers
from archx.recorder import Recorder

#: Where the weights move fast, then stop, then jump once.
WARMUP_STEPS = 50
JUMP_STEP = 700


def movement_schedule(steps: int) -> np.ndarray:
    """How far the weights move at each step."""
    rates = np.full(steps, 2e-4, dtype=np.float64)
    rates[:WARMUP_STEPS] = 1e-2
    rates[JUMP_STEP] = 5e-2
    return rates


class Model:
    """A small model whose movement per step follows a known schedule."""

    def __init__(self, size: int = 64, count: int = 4, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.params = {f"block{i}.weight": rng.normal(0, 0.05, (size, size)).astype(np.float32)
                       for i in range(count)}
        self._rng = np.random.default_rng(seed + 1)
        self.registry = TensorRegistry()
        for index, (key, array) in enumerate(self.params.items()):
            self.registry.add(key, array, pointer=array.ctypes.data, region=f"block{index}")

    def step(self, scale: float) -> None:
        for array in self.params.values():
            array += self._rng.normal(0, scale, array.shape).astype(np.float32)


def run(policy: PhasePolicy, steps: int, *, seed: int = 0, label: str = "",
        sample: int = 400) -> dict:
    """Train under one policy, then read the file back the way a user would."""
    model = Model(seed=seed)
    rates = movement_schedule(steps)
    truth: dict[int, dict[str, np.ndarray]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{label}.archx.mdl"
        recorder = Recorder(model.registry, model.params.__getitem__, str(path), policy=policy)
        for step in range(steps):
            model.step(rates[step])
            # The loss and the gradient are what the policy is allowed to look at. A real jump
            # shows up in both; a benchmark that hides it would be testing the heartbeat only.
            spike = step == JUMP_STEP
            # Flat between events on purpose. A decaying loss drops by more than 5% per step
            # early on, which would trip the loss trigger constantly and turn the "fixed"
            # baseline into an event-driven recorder - the comparison would then be between
            # two trigger sets, which is not the claim.
            loss = 0.8 + (0.5 if spike else 0.0)
            grad_norm = 1.0 + (25.0 if spike else 0.0)
            recorder.step(step, loss=loss, grad_norm=grad_norm)
            truth[step] = {key: value.copy() for key, value in model.params.items()}
        recorder.close()

        reader = Reader(path)
        stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                        | {b.step for b in reader.blocks(BlockType.ANCHOR)})
        size = path.stat().st_size
        timeline = Timeline(reader)

        # Fidelity: for every step the file does not store, rebuild the weights from the file
        # and compare with what they really were. Relative error over the whole model.
        stored_set = set(stored)
        unstored = [step for step in range(steps) if step not in stored_set]
        # Long runs make a full sweep expensive, so fidelity is measured on a fixed random
        # sample of unstored steps plus every step around the jump. Same seed for both
        # policies, so the two rows are compared on the same steps.
        if sample and len(unstored) > sample:
            rng = np.random.default_rng(12345)
            near = [s for s in unstored if abs(s - JUMP_STEP) <= 25]
            rest = [s for s in unstored if abs(s - JUMP_STEP) > 25]
            keep = min(len(rest), max(1, sample - len(near)))
            picked = rng.choice(rest, size=keep, replace=False)
            unstored = sorted(set(near) | {int(x) for x in picked})
        errors = {step: _relative_error(timeline, step, truth[step]) for step in unstored}

    near_jump = [errors[step] for step in unstored if abs(step - JUMP_STEP) <= 25]
    return {
        "policy": label,
        "bytes": size,
        "stored": len(stored),
        "skipped": steps - len(stored),
        "median_error": float(np.median(list(errors.values()))) if errors else 0.0,
        "worst_error": float(max(errors.values())) if errors else 0.0,
        "jump_error": float(np.median(near_jump)) if near_jump else 0.0,
        "anchor_at_jump": JUMP_STEP in stored,
        "coverage": _coverage(stored, steps),
        "stored_steps": stored,
    }


#: Twelve events at irregular gaps, so neither policy can satisfy them by cadence.
MULTI_EVENTS = [137, 271, 449, 613, 700, 811, 1024, 1157, 1390, 1502, 1733, 1891]


def run_multi(policy: PhasePolicy, steps: int, *, label: str = "", seed: int = 0) -> dict:
    """Train a run with several irregular jumps and ask whether each one got stored.

    One event can be stored by luck; a fixed interval lands within two steps of an event
    about 12% of the time. Twelve irregular events make luck uninteresting. Fidelity is
    measured on the five steps after each event, where a missed event hurts most.
    """
    model = Model(seed=seed)
    events = [e for e in MULTI_EVENTS if e < steps]
    spikes = set(events)
    truth: dict[int, dict[str, np.ndarray]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{label}.multi.mdl"
        recorder = Recorder(model.registry, model.params.__getitem__, str(path), policy=policy)
        for step in range(steps):
            model.step(5e-2 if step in spikes else 2e-4)
            recorder.step(step, loss=0.8 + (0.5 if step in spikes else 0.0),
                          grad_norm=1.0 + (25.0 if step in spikes else 0.0))
            truth[step] = {key: value.copy() for key, value in model.params.items()}
        recorder.close()

        reader = Reader(path)
        stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                        | {b.step for b in reader.blocks(BlockType.ANCHOR)})
        stored_set = set(stored)
        timeline = Timeline(reader)
        covered = [e for e in events if any(0 <= s - e <= 2 for s in stored_set)]
        after = [max(_relative_error(timeline, s, truth[s]) for s in range(e + 1, min(e + 6, steps)))
                 for e in events if e + 5 < steps]
        return {
            "policy": label,
            "bytes": path.stat().st_size,
            "stored": len(stored),
            "events": len(events),
            "events_stored": len(covered),
            "error_after_event": float(np.median(after)) if after else 0.0,
        }


def _relative_error(timeline: Timeline, step: int, truth: dict[str, np.ndarray]) -> float:
    """How wrong the file is at a step it does not store, relative to the weights' size."""
    rebuild = timeline.reconstruct(step)
    error = 0.0
    magnitude = 0.0
    for key, true_values in truth.items():
        rebuilt = rebuild[key].astype(np.float64)
        true_values = true_values.astype(np.float64)
        error += float(np.linalg.norm(rebuilt - true_values))
        magnitude += float(np.linalg.norm(true_values))
    return error / magnitude if magnitude else 0.0


def _coverage(stored: list[int], steps: int) -> dict:
    """Where the stored steps fall, in thirds: the shape of the recording."""
    third = steps // 3
    return {
        "first_third": sum(1 for s in stored if s < third),
        "middle_third": sum(1 for s in stored if third <= s < 2 * third),
        "last_third": sum(1 for s in stored if s >= 2 * third),
    }


def fixed_interval(every: int, total_steps: int) -> PhasePolicy:
    """A plain fixed-interval recorder: store every `every` steps, nothing else considered.

    Expressed through the same policy surface, so this compares two policies rather than two
    code paths. Phase acceleration is switched off, the heartbeat *is* the interval, and there
    is no anchor cadence - otherwise every stored step would be a full snapshot and the
    measurement would be about anchor density instead of about which steps were chosen.
    """
    # Triggers are switched off rather than left at their defaults. A fixed-interval recorder
    # by definition does not react to loss or gradients; leaving the defaults on produced a
    # baseline that stored 57 of its steps inside the first third of the run and was not
    # fixed at all.
    return PhasePolicy(total_steps=total_steps, warmup_fraction=0.0, cooldown_fraction=0.0,
                       min_warmup_steps=1, heartbeat_steps=every, anchor_every=None,
                       triggers=Triggers(grad_norm_zscore=float("inf"),
                                         loss_drop_fraction=float("inf"),
                                         region_move_fraction=float("inf")))


def _bisect_multi(budget: int, steps: int) -> dict:
    """The fixed interval whose byte count is closest to `budget` on the multi-event run."""
    low, high, best = 1, steps, None
    for _ in range(12):
        if low > high:
            break
        mid = (low + high) // 2
        candidate = run_multi(fixed_interval(mid, steps), steps, label=f"fixed-{mid}")
        if best is None or abs(candidate["bytes"] - budget) < abs(best["bytes"] - budget):
            best = candidate
        if candidate["bytes"] < budget:
            high = mid - 1
        else:
            low = mid + 1
    return best


def best_interval(total_steps: int, best: dict) -> int:
    """The interval the bisection ended on, recovered from its label (`fixed-<n>`)."""
    try:
        return int(best["policy"].split("-", 1)[1])
    except (KeyError, IndexError, ValueError):
        return max(1, total_steps // max(1, best.get("stored", 1)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    adaptive = run(PhasePolicy(total_steps=args.steps), args.steps, label="adaptive")
    budget = adaptive["bytes"]

    # "At equal storage" has to mean the same budget, so bisect on the fixed interval until
    # its byte count is as close to the adaptive one as an integer interval can get.
    low, high, best = 1, args.steps, None
    for _ in range(12):
        if low > high:
            break
        mid = (low + high) // 2
        candidate = run(fixed_interval(mid, args.steps), args.steps, label=f"fixed-{mid}")
        if best is None or abs(candidate["bytes"] - budget) < abs(best["bytes"] - budget):
            best = candidate
        if candidate["bytes"] < budget:
            high = mid - 1                # a wider interval stores less: close the gap
        else:
            low = mid + 1

    results = [adaptive, best]
    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{args.steps} steps, jump at {JUMP_STEP}, matched on bytes written\n")
    header = (f"{'policy':<14}{'bytes':>10}{'stored':>8}{'median err':>12}"
              f"{'err near jump':>14}{'jump stored':>13}")
    print(header)
    print("-" * len(header))
    for row in results:
        print(f"{row['policy']:<14}{row['bytes']:>10}{row['stored']:>8}"
              f"{row['median_error']:>12.2e}{row['jump_error']:>14.2e}"
              f"{str(row['anchor_at_jump']):>13}")

    print("\ncoverage by third of the run")
    for row in results:
        c = row["coverage"]
        print(f"  {row['policy']:<14} first {c['first_third']:>4}  middle {c['middle_third']:>4}"
              f"  last {c['last_third']:>4}")

    difference = abs(adaptive["bytes"] - best["bytes"]) / max(adaptive["bytes"], 1)
    print(f"\nbyte difference {difference * 100:.1f}%")
    if difference >= 0.15:
        print("warning: budgets differ by more than 15%; do not read these rows as equal")

    # The single jump above is one sample, and a fixed interval can catch one event by luck.
    # The question the policy is actually built to answer is whether it catches events it
    # could not have guessed - so ask it repeatedly, at irregular gaps.
    multi_steps = max(args.steps, MULTI_EVENTS[-1] + 109)
    adaptive_multi = run_multi(PhasePolicy(total_steps=multi_steps), multi_steps,
                               label="adaptive")
    # Matched on this run's own budget: the adaptive policy stores a different number of
    # bytes here than above, so reusing the single-jump interval would compare two budgets
    # and quietly give the fixed row an advantage or a handicap it did not earn.
    fixed_multi = _bisect_multi(adaptive_multi["bytes"], multi_steps)
    print(f"\nrepeated events, {len([e for e in MULTI_EVENTS if e < multi_steps])} jumps at "
          f"irregular steps, {multi_steps} steps")
    multi = [adaptive_multi, fixed_multi]
    header = (f"\n{'policy':<14}{'bytes':>10}{'stored':>8}{'events stored':>15}"
              f"{'err after event':>17}")
    print(header)
    print("-" * (len(header) - 1))
    for row in multi:
        print(f"{row['policy']:<14}{row['bytes']:>10}{row['stored']:>8}"
              f"{str(row['events_stored']) + '/' + str(row['events']):>15}"
              f"{row['error_after_event']:>17.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

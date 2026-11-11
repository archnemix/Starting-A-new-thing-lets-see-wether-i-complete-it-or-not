"""Claim 5: what recording costs per step, measured instead of guessed.

Compares a bare loop against the same loop with a recorder attached, on the default policy
and with every step stored. Reports copied bytes and time per step, because copying tensors
is the only unavoidable cost and pretending otherwise would be dishonest.
"""

from __future__ import annotations

import argparse
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

from archx.formats.registry import TensorRegistry
from archx.policy import PhasePolicy
from archx.recorder import Recorder


def make_params(size: int, count: int, rng) -> dict[str, np.ndarray]:
    return {f"layer{i}.w": rng.normal(0, 0.05, (size, size)).astype(np.float32)
            for i in range(count)}


def run(steps: int, params: dict[str, np.ndarray], recorder: Recorder | None, rng):
    times = []
    for step in range(steps):
        started = time.perf_counter()
        for array in params.values():
            array += rng.normal(0, 1e-4, array.shape).astype(np.float32)
        if recorder is not None:
            recorder.step(step, loss=1.0 / (step + 1))
        times.append(time.perf_counter() - started)
    return times


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv[1:])

    rng = np.random.default_rng(args.seed)
    params = make_params(args.size, args.layers, rng)
    tensors = sum(a.size for a in params.values())

    base = run(args.steps, params, None, np.random.default_rng(args.seed + 1))

    rows = []
    for label, policy in (
        ("adaptive (default)", PhasePolicy(total_steps=args.steps)),
        ("every step", PhasePolicy(total_steps=args.steps, warmup_fraction=1.0,
                                   min_warmup_steps=args.steps)),
    ):
        working = {k: v.copy() for k, v in params.items()}
        registry = TensorRegistry()
        for key, array in working.items():
            registry.add(key, array, pointer=array.ctypes.data)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(registry, working.__getitem__, str(Path(tmp) / "o.archx.mdl"),
                                policy=policy, queue_size=64)
            measured = run(args.steps, working, recorder, np.random.default_rng(args.seed + 1))
            recorder.close()
            bytes_written = Path(tmp, "o.archx.mdl").stat().st_size
        rows.append((label, statistics.median(measured), bytes_written, recorder.stats))

    baseline = statistics.median(base)
    print(f"tensors={tensors} ({tensors * 4 / 1024:.0f} KiB), steps={args.steps}, "
          f"layers={args.layers}, size={args.size}")
    print("| policy | median step | overhead | recorded bytes | stored/skipped |")
    print("|---|---|---|---|---|")
    print(f"| none | {baseline * 1e3:.2f} ms | - | 0 | - |")
    for label, median, written, stats in rows:
        overhead = (median - baseline) / baseline * 100
        print(f"| {label} | {median * 1e3:.2f} ms | {overhead:+.1f}% | {written / 1024:.1f} KiB | "
              f"{stats.steps_stored}/{stats.steps_skipped} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))

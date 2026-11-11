"""Claim 2: when low-rank beats sparse, and when it does not, per delta density.

Sweeps density from 0.1% to 100% twice: once asking "what is smallest at any error" and once
"what is smallest when the delta must be reproduced (error under 1e-6)". The two answers are
different, and only the second one is usable for an exact-history recording. That gap is the
finding this script exists to expose.
"""

from __future__ import annotations

import argparse

import numpy as np

from archx.encoders import DenseQuant, DenseRaw, LowRank, Sparse


def best_at(rng, density: float, structured: bool, size: int = 128):
    delta = np.zeros((size, size), dtype=np.float32)
    count = max(1, int(delta.size * density))
    if structured:
        a = rng.normal(size=(size, 4)).astype(np.float32)
        b = rng.normal(size=(4, size)).astype(np.float32)
        base = (a @ b) * 1e-3
    else:
        base = rng.normal(0, 1e-3, (size, size)).astype(np.float32)
    idx = rng.choice(delta.size, count, replace=False)
    delta.reshape(-1)[idx] = base.reshape(-1)[idx]

    candidates = [Sparse(), DenseQuant(bits=8), LowRank(max_rank=16, max_rel_error=1e-2),
                  DenseRaw()]
    scored = []
    for encoder in candidates:
        encoded = encoder.encode(delta)
        if encoded is None:
            continue
        rebuilt = encoder.decode(encoded.body, delta.shape, encoded.value_dtype,
                                 encoded.nnz_or_rank)
        rel = float(np.linalg.norm(delta - rebuilt) / np.linalg.norm(delta))
        scored.append((encoder.name, encoded.nbytes, rel))
    scored.sort(key=lambda row: row[1])
    smallest = scored[0] if scored else ("none", None, None)
    exactish = next((row for row in scored if row[2] is not None and row[2] <= 1e-6),
                    ("none", None, None))
    return smallest, exactish


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv[1:])
    rng = np.random.default_rng(args.seed)

    print("| density | structure | smallest (any error) | bytes | err | smallest exact | bytes |")
    print("|---|---|---|---|---|---|---|")
    for structured in (False, True):
        for density in (0.001, 0.01, 0.05, 0.2, 0.5, 1.0):
            (small, small_bytes, small_err), (exact, exact_bytes, _e) = best_at(
                rng, density, structured)
            kind = "rank-4" if structured else "noise"
            b1 = "-" if small_bytes is None else str(small_bytes)
            e1 = "-" if small_err is None else f"{small_err:.1e}"
            b2 = "-" if exact_bytes is None else str(exact_bytes)
            print(f"| {density:.1%} | {kind} | {small} | {b1} | {e1} | {exact} | {b2} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))

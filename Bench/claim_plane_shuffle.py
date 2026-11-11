"""Does byte-plane shuffling actually help, and by how much?

The design claims shuffling is what makes float deltas compressible. That claim is cheap to
test and easy to get wrong, so it is measured on three delta shapes instead of assumed.
"""

from __future__ import annotations

import argparse

import numpy as np

from archx.container.compress import compress
from archx.container.schema import Codec
from archx.encoders.plane import shuffle


def cases(rng):
    yield "sparse 1%", (lambda d: d)(_sparse(rng))
    yield "dense small", rng.normal(0, 1e-5, (256, 256)).astype(np.float32)
    a = rng.normal(size=(256, 4)).astype(np.float32)
    b = rng.normal(size=(4, 256)).astype(np.float32)
    yield "rank-4", (a @ b) * 1e-3


def _sparse(rng):
    d = np.zeros(256 * 256, dtype=np.float32)
    moved = d.size // 100
    d[rng.choice(d.size, moved, replace=False)] = rng.normal(0, 1e-3, moved)
    return d.reshape(256, 256)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv[1:])
    rng = np.random.default_rng(args.seed)

    print("| delta | raw bytes | zstd(raw) | zstd(shuffled) | shuffle gain |")
    print("|---|---|---|---|---|")
    for label, delta in cases(rng):
        raw = delta.astype(np.float32).tobytes(order="C")
        plain = len(compress(raw, Codec.ZSTD))
        shuffled = len(compress(shuffle(raw, 4), Codec.ZSTD))
        gain = plain / shuffled if shuffled else 0.0
        print(f"| {label} | {len(raw)} | {plain} | {shuffled} | {gain:.2f}x |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))

"""Claim 3: chained-delta error as a function of anchor spacing.

A recording rebuilds a stored step by starting at the nearest anchor at or before it and adding
deltas. Each delta is allowed to be off by `max_rel_error`, and those errors accumulate along the
chain - so the format needs an answer to: **how far apart may anchors be before the accumulated
error stops being acceptable?**

Both directions of the trade are measured, not argued:

- more deltas in a chain means more accumulated error, and
- an anchor is exact, so fewer anchors also means the steps far from one are further from
  anything exact.

The measured quantity is the error at the far end of a chain of `k` quantized deltas, against the
true weights. An anchor spacing is recommendable when the worst chain it allows stays within a
stated multiple of the per-delta budget.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from archx.encoders import (DenseQuant, DenseRaw, LowRank, Sparse, Unchanged,
                            choose_encoder)


def _encoders(budget: float) -> list:
    """The same candidate set the recorder builds for this budget."""
    candidates = [Unchanged(), Sparse(threshold=0.0), DenseQuant(bits=8),
                  LowRank(max_rank=16, max_rel_error=budget), DenseRaw()]
    if budget > 1e-3:
        candidates.insert(2, DenseQuant(bits=16))
    return candidates


def _decode(encoded, shape: np.shape) -> np.ndarray:
    """A decoder chosen the way the reader chooses it: by encoding id *and* value width."""
    encoding = int(encoded.encoding)
    if encoding == DenseQuant.encoding_id:
        decoder = DenseQuant(bits=16 if encoded.value_dtype == "i16" else 8)
    else:
        decoder = {0: Unchanged(), 1: Sparse(), 3: LowRank(), 4: DenseRaw()}[encoding]
    return decoder.decode(encoded.body, shape, encoded.value_dtype, encoded.nnz_or_rank)


_NAMES = {0: "unchanged", 1: "sparse", 2: "dense_quant", 3: "lowrank", 4: "dense_raw"}


def _name_of(encoded) -> str:
    """Encoder name plus its width, since encoding id 2 covers both int8 and int16."""
    name = _NAMES.get(int(encoded.encoding), f"id{int(encoded.encoding)}")
    return f"{name}:{encoded.value_dtype}" if name == "dense_quant" else name


def chained_error(budget: float, chain: int, *, seed: int = 0, size: int = 128) -> dict:
    """Relative error after `chain` quantized deltas are added onto one exact anchor."""
    rng = np.random.default_rng(seed)
    anchor = rng.normal(0, 0.05, (size, size)).astype(np.float32)
    candidates = _encoders(budget)

    rebuilt = anchor.copy()
    true = anchor.copy()
    used = set()
    for _ in range(chain):
        step = rng.normal(0, 1e-3, (size, size)).astype(np.float32)
        true = true + step
        encoded = choose_encoder(step, candidates, max_rel_error=budget)
        if encoded is None:                       # pragma: no cover - raw float32 always passes
            raise RuntimeError("no encoder accepted the delta even at raw float32")
        used.add(_name_of(encoded))
        # A reader adds the *reconstructed* delta, which is what accumulates the error.
        rebuilt = rebuilt + _decode(encoded, step.shape)

    return {
        "chain": chain,
        "relative_error": float(np.linalg.norm(rebuilt - true) / np.linalg.norm(true)),
        "encodings": sorted(used),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    rows = [{"budget": args.budget, **chained_error(args.budget, chain)}
            for chain in (1, 2, 4, 8, 16, 32, 64)]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"delta budget {args.budget:g}, one exact anchor, quantized deltas added in order\n")
    header = f"{'deltas in chain':>16}{'relative error':>17}{'x budget':>11}  encodings"
    print(header)
    print("-" * len(header))
    for row in rows:
        multiple = row["relative_error"] / args.budget if args.budget else float("inf")
        print(f"{row['chain']:>16}{row['relative_error']:>17.2e}{multiple:>10.2f}x"
              f"  {','.join(row['encodings'])}")

    # "Acceptable" has to be stated, or the recommendation is a feeling. Ten times the per-delta
    # budget is the bar used in the docs: it keeps a chained step within an order of magnitude of
    # any single delta's promised accuracy.
    inside = [row["chain"] for row in rows if row["relative_error"] <= 10 * args.budget]
    if inside:
        print(f"\nchains up to {max(inside)} stay within ten times the per-delta budget")
    else:
        print("\nno chain length stays within ten times the per-delta budget at this setting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

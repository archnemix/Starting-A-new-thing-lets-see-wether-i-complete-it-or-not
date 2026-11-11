"""Claim 1: reconstruction error versus compression ratio, per delta encoder.

Produces the table the storage design depends on. Until this runs, every ratio in the docs
is a hypothesis; after it runs, the numbers here are the ones quoted.
"""

from __future__ import annotations

import argparse
import numpy as np

from archx.encoders import DenseQuant, DenseRaw, LowRank, Sparse, Unchanged


def shapes(kind: str, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    """Deltas shaped like what training actually produces, not like ideal test data."""
    if kind == "sparse":           # MoE block / frozen-ish layer: 1% of entries moved
        d = np.zeros(256 * 256, dtype=np.float32)
        moved = d.size // 100
        d[rng.choice(d.size, moved, replace=False)] = rng.normal(0, 1e-3, moved)
        return "1% of entries moved", d.reshape(256, 256)
    if kind == "dense_small":      # typical mid-training step: everything moved a little
        return "dense, all moved", rng.normal(0, 1e-5, (256, 256)).astype(np.float32)
    if kind == "lowrank":          # coordinated movement, the structure LoRA exploits
        a = rng.normal(size=(256, 4)).astype(np.float32)
        b = rng.normal(size=(4, 256)).astype(np.float32)
        return "rank-4 structure", (a @ b) * 1e-3
    raise ValueError(kind)


def measure(encoders, delta):
    raw = delta.nbytes
    rows = []
    zero = Unchanged().encode(delta)
    if zero is not None:
        rows.append(("unchanged", 0, 0.0, "exact"))
    for encoder in encoders:
        encoded = encoder.encode(delta)
        if encoded is None:
            rows.append((encoder.name, None, None, "refused"))
            continue
        rebuilt = encoder.decode(encoded.body, delta.shape, encoded.value_dtype,
                                 encoded.nnz_or_rank)
        rel = float(np.linalg.norm(delta - rebuilt) / np.linalg.norm(delta))
        exact = "exact" if rel == 0.0 else f"{rel:.2e}"
        rows.append((encoder.name, encoded.nbytes, raw / encoded.nbytes, exact))
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv[1:])

    rng = np.random.default_rng(args.seed)
    encoders = [Sparse(threshold=0.0), DenseQuant(bits=8),
                LowRank(max_rank=16, max_rel_error=1e-3), DenseRaw()]
    results = {}
    print("| delta shape | encoder | bytes | ratio vs raw | rel. error |")
    print("|---|---|---|---|---|")
    for kind in ("sparse", "dense_small", "lowrank"):
        label, delta = shapes(kind, rng)
        rows = measure(encoders, delta)
        results[kind] = {"label": label, "raw_bytes": int(delta.nbytes), "rows": []}
        for name, nbytes, ratio, err in rows:
            shown_ratio = "-" if ratio is None else f"{ratio:.1f}x"
            shown_bytes = "-" if nbytes is None else str(nbytes)
            print(f"| {label} | {name} | {shown_bytes} | {shown_ratio} | {err} |")
            results[kind]["rows"].append({"encoder": name, "bytes": nbytes, "error": err})

    if args.json:
        import json

        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))

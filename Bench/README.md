# Bench

> Purpose: the measurement scripts behind the claims. Each file measures one claim and
> prints numbers; none of them asserts, so a number can be read and argued with.

Run one with `.venv/bin/python Bench/<file>.py`. Results are recorded in
`Docs/Research.json` under the matching id.

| File | Claim | What it prints |
|---|---|---|
| `claim01_error_vs_ratio.py` | 1 | encoder size and reconstruction error at 1% / dense movement |
| `claim02_sparse_vs_lowrank.py` | 2 | smallest encoder per sparsity level, exact vs error allowed |
| `claim_plane_shuffle.py` | - | whether byte-plane shuffling helps or hurts |
| `claim05_overhead.py` | 5 | median step time with and without a recorder |

The adapter-level version of claim 5 lives in `Tests/integration/test_torch_adapter.py`,
because a claim about recording a real optimizer is only meaningful on a real optimizer.

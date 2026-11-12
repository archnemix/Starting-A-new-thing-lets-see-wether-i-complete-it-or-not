# Bench

> Purpose: the measurement scripts behind the claims. Each file measures one claim and
> prints numbers; none of them asserts, so a number can be read and argued with.

Run one with `.venv/bin/python Bench/<file>.py`. Results are recorded in
`Docs/Research.json` under the matching id.

| File | Claim | What it prints |
|---|---|---|
| `claim01_error_vs_ratio.py` | 1 | encoder size and reconstruction error at 1% / dense movement |
| `claim02_sparse_vs_lowrank.py` | 2 | smallest encoder per sparsity level, exact vs error allowed |
| `claim04_storage_vs_checkpoints.py` | 4 | both schemes at matched accuracy, and which one that favours |
| `claim04_long_run.py` | 4 | a 1000-step run: recording vs a checkpoint at every step |
| `claim05_overhead.py` | 5 | median step time with and without a recorder |
| `claim06_adaptive_vs_fixed.py` | 6 | adaptive vs a fixed interval at the same byte budget |
| `claim07_hook_reach.py` | 7 | tier and resolution with a bare hook, with `log()`, with annotations |
| `claim_arch_sweep.py` | - | six architectures through one adapter: tier, stored steps, error |
| `claim_plane_shuffle.py` | - | whether byte-plane shuffling helps or hurts |

The rule for all of them: a result that goes against the product is printed, not filtered. Two of
these ended that way - claim 4 favours checkpointing on every-step accuracy, and claim 6 favours
the fixed interval on a stationary run.

The adapter-level version of claim 5 lives in `Tests/integration/test_torch_adapter.py`,
because a claim about recording a real optimizer is only meaningful on a real optimizer.

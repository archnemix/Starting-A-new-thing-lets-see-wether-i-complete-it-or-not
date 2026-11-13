# Docs: Changelog

> Purpose: what each version shipped, and what changed between versions - added,
> removed, replaced. Concise, but complete per release.

Version numbering: `MAJOR.MINOR.PATCH`. While the format and API are unstable the
MAJOR stays `0`. The container format has its own `format_major.minor` recorded in
every file.

---

## Unreleased (design phase)

No code yet. This stage produced documentation only:

- Added the full doc set (`Docs/`, `Plan/`), repository skeleton, folder READMEs.
- Added research log with verified hook/API facts (`Docs/Research.json`).
- Added CI spine: trigger-file-driven workflow that commits results into the repo.
  Verified end to end: GitHub run `43052619656`, three runners (ubuntu, windows,
  macos) green, each committing `Actions/out/latest-env/<os>.json` back to the branch.
- Decided: Python core instead of a Rust core (no Rust toolchain reachable; the
  brief's requirement was "must work in Python").
- Decided: the product is a **recording triple** (state + cause + stage), not a storage
  format (Architecture 6.0).
- Decided: training data is **stored by default** (`data_level=segments`), deduplicated and
  content-addressed, opt-out recorded; the brief's "fingerprints only" is superseded.
- Format: `DATA_SEGMENT`, `DATA_SHARD`, `DATA_POINTER` block types; `data_level`,
  `attribution`, `base_model`, `ancestor` in META.
- Added claims 11-14 (dedup, codecs, data overhead, join round-trip) to the measurement list.
- Naming decided: PyPI distribution `archx.mdl` (normalises to `archx-mdl`), import and CLI
  stay `archx`, npm `archx` reserved as a signpost package. No cap on stored data by default.

## v0.1.0 - not released yet (Phase 0 and Phase 1 complete on the branch)

Shipped on the branch:

- Packaging: distribution `archx.mdl`, import `archx`, CLI `archx`, MIT, Python >= 3.10.
- Container format 1.0: 40-byte header, 28-byte block prefix (step inside), JSON/binary
  payloads, zstd or zlib per block, INDEX block, 56-byte footer, CRC everywhere.
- Append-only `Writer` with batched fsync and CLOSED flag.
- Validating `Reader`: refuses foreign files, newer majors, absurd lengths, bad CRCs; skips
  unknown block types; reports torn tails.
- `archx recover`: keeps whole blocks, truncates the tail, rebuilds the index from prefixes
  (no decompression), writes a footer, sets RECOVERED.
- `archx info` and `archx recover` commands, human and `--json` output.
- 31 tests (unit + integration + property), green on Linux, Windows and macOS.
- CI writes machine-readable run records into `Actions/out/test-runs/`.

## v0.2.0 - not released yet (tier 3, and a usable first run)

The last two roadmap rows, and the first version a stranger can pick up without the plan.

### Tier 3: what the network saw (roadmap 8.3)

`archx.attach(..., activations=True)` now reaches **tier 3**. A forward hook reduces each leaf
layer's output to a fixed-size summary, written as `ACTIVATION` blocks (type 23) at every
stored step:

```console
$ archx activations demo.archx.mdl
layers         5  0, 1, 2, 3, 4
summaries      55
  step  layer   shape   mean     std     min     max  zero%   sat%
     0      0  64x32  -0.018   0.614  -2.13    2.10   0.00   0.00
     0      1  64x64   0.234   0.349   0.00    1.63  49.80   0.00
```

- **It is not a tensor cache.** Shape, mean, std, min/max, dead-unit fraction, saturation
  fraction, forward count. Bounded by layer count, never by batch size.
- **Measured cost: +1.6 ms on a 1.4 ms forward pass** for a 9-layer 256-wide stack, off by
  default. The cost **plateaus** above the sampling cap: batch 64/256/1024 costs
  +0.84/+1.51/+2.03 ms, so 4x the data is 1.3x the price.
- **It finds what the loss cannot.** A model with half its ReLU units killed is flagged in 4
  summaries at a 49.8% dead-unit fraction while the loss moves by 0.0014.
- Above 16,384 elements per layer the reduction is **stride-sampled across the whole tensor**,
  never truncated from the start, and the block records how many elements the moments cover.

Getting the number right took three attempts, all recorded: the bench first timed
`recorder.step()` alongside the forward (writer thread on another core) and reported +208% then
+503% for identical code; the first per-element reduction allocated a float64 copy per layer;
and the "sick model" probe reported tier 0 until it called `optimizer.step()`, because tiers are
strictly sequential and only a real optimizer step produces gradients.

### A first run that does not need a training loop (roadmap 0.5 follow-up)

- **`archx demo`** records a small run in one command and prints the five commands to read it.
- **`archx activations`** is the new read command; a file below tier 3 explains why instead of
  printing nothing.
- **`python -m archx`** works. It did not: there was no `__main__`, so the first thing many
  people try did nothing at all, silently.
- **`README.md` rewritten.** It still said "design only, no code yet", which had been false for
  nine phases.
- **`examples/quickstart.py`**: the tour, ending with the rebuilt weights compared against the
  live tensors (max abs error 1.9e-09 on the last step).

### `archx info` reported the wrong tier

Found by reading the output of the command in the new guide: `archx info` printed
`capability_tier 0` for a recording that **listed every field it had recorded on the next
line**. META carries the floor tier on purpose - it is written at open, before a step has
happened - and the achieved tier is written in the capabilities block at close. The command
printed META's.

It now prints the achieved tier, and an unclosed file (which has no capabilities block, and in
fact no blocks at all, because the writer buffers) reports no tier rather than a default. Two
tests pin both halves.

This is the kind of defect only a person running the product finds: every test passed, because
every test read `reader.capabilities()` directly instead of reading what a user is shown.

### Roadmap 7.4 decided

Ship the launcher as a **fallback**, not a headline. The decision is recorded as made on the
evidence, with the user delegating the call, rather than as an approval that never happened.
`archx run` stays: it works on an untouched script and never lies about what it could not see,
which is exactly what a fallback should be.

Roadmap: **53 / 53 rows done.** Tests: **263 passed** locally, and CI run
`43429355746` is green on Linux, Windows and macOS at the same commit `14fc5bc` - 263 passed,
0 failed, 0 skipped on each, with an **identical set of 263 test names** verified
programmatically. First attempt, no re-runs.

## The CI record for the Phase 2 work: run 43259502790

Three runners, python 3.11, torch + transformers + lightning installed on all three:

| runner | result | skipped |
|---|---|---|
| ubuntu-latest | 245 passed, 0 failed, 0 errors | 0 |
| windows-latest | 245 passed, 0 failed, 0 errors | 0 |
| macos-latest | 245 passed, 0 failed, 0 errors | 0 |

The three junit files hold an **identical set of 245 test names**, verified programmatically,
at the same commit `81bc8ff`. The npm signpost suite is green on Linux/Windows/macOS at node
18 and 20 (`npm-signpost` run `43251525556`).

Getting there took four CI iterations, and every one of them found a real defect rather than
a flake to re-run:

| run | what it caught |
|---|---|
| 43251129935 | the Windows *record-commit* step failed after its 241 tests passed - two workflows racing for one branch |
| 43251525614 | `subprocess.run(["npm", ...])` -> `WinError 2` on Windows (`npm` is a `.cmd` shim) |
| 43253521609 | macOS: asserted the literal `# fail 0`, but node 22 prints `ℹ fail 0` |
| 43256908848 | a 20% overhead *ratio* on a 1% effect - passed one run, failed the next |

## v1.0.1 - not released yet (Phase 2 complete: codecs and data capture measured)

Two roadmap rows finished, five silent defects found, four of them in code that was
documented as working.

### Claims 13-14: capture cost and the join round-trip

Measured, `Bench/claim13_capture_levels.py`, numbers in `Docs/capture_levels.json`. 60 steps,
batch 256x128, 18,128-parameter model, 5 repeats, charged to `log()` + `step()`:

| level | file bytes | step() median | p95 | seal (writer thread) | join answered |
|---|---|---|---|---|---|
| `none` | 1,455,641 | 0.018 ms | 14.4 ms | 0.17 s | n/a - no data |
| `pointers` | 1,478,206 | 0.033 ms | 14.9 ms | 0.20 s | no - content withheld |
| `segments` | 8,756,493 | 0.032 ms | 14.2 ms | 0.25 s | **60/60** |
| `full` | 8,756,491 | 0.033 ms | 12.1 ms | 0.43 s | **60/60** |

- The marginal cost of capture is **~15 us per step**. Hashing and compression are on the
  writer thread, not the training thread.
- **The tail belongs to the weights, not the data.** 12-17% of steps exceed 1 ms at every
  level, including `none` with nothing handed over, because the trainer encodes a held weight
  block when the writer falls behind. Charging that to capture would be wrong.
- The bench's first version timed `optimizer.step()` and reported a *negative* overhead for
  `pointers`. It was measuring AdamW; the optimizer is now outside the timed region.

### Fixed: five defects that were silently wrong

1. **`full` was never implemented.** Documented since the first design ("every batch exactly
   as fed, no dedup"), it wrote the same deduplicated file as `segments` - 4 bytes apart on
   60 identical batches. It now stores one shard per observation (60 shards where `segments`
   writes 1 for 60 sightings). Test: `test_full_stores_every_batch_even_when_the_bytes_repeat`.
2. **A merged segment stayed reachable through the identity map.** `_merge_unnamed` popped the
   duplicate from `segments`, but `_by_identity` kept it and sealing had already set
   `content = None`, so the stale entry was permanent. Later sightings of the same batch
   incremented a segment no longer in the file: a 20-step run reported `times_seen` summing to
   19. The map is now pruned at seal; observations and `times_seen` agree exactly.
3. **The join had nowhere to point.** `span_steps` said how long a measurement's span was,
   never which recorded step it belonged to. 36 of 60 segments in the first measurement
   carried movement that landed on no stored step. New additive v1 field `effect_step` (the
   store point at which the movement was measured, `-1` when there is none): 24/60 -> 60/60.
4. **Data logged after the final optimizer step vanished.** 20 observations, 19 in the file,
   no warning. It is now stored and an `unsynced_log` block says it was never trained on.
5. **`none` recorded no fingerprints.** Both its own module summary ("hash and count only")
   and Architecture 6.8 ("the fingerprint is always present at every level") promised the
   hash; the code returned early and wrote nothing, so a `none` run could say how many batches
   it saw but not which. It now writes one fingerprinted row per unique shard, with no content
   and no shard block.

### Added, both documented since the first design and never emitted

- **`JOIN_INCOMPLETE`** (`INFO`) for `none`, `pointers`, and any run whose byte cap degraded
  to `pointers`, listing what the reader cannot answer.
- Spec: the `segments` vs `full` distinction is now stated in `Plan/Format_Spec.md` §4 with
  the shard-id rule (`<hash>-<n>` under `full`) a second implementation must follow.

### Claims 11-12: shard codecs

`Bench/claim11_codecs.py`, `Docs/codec_results.json`. Every case exact, every CRC matching:

| case | raw | stored | ratio | codec |
|---|---|---|---|---|
| prose | 30,749 | 153 | 0.0050 | text-zstd |
| code | 51,200 | 116 | 0.0023 | text-zstd |
| tokens-128 | 16,384 | 3,620 | 0.2209 | tokens-varint |
| tokens-5000 | 16,384 | 7,397 | 0.4515 | tokens-varint |
| pixels | 16,384 | 16,394 | 1.0006 | zstd |
| weights | 16,384 | 15,167 | 0.9257 | zstd |

- **The token path was dead code**: its detector required every id below 128, so a real
  vocabulary never took it (0.560 -> 0.451 with shape-based detection).
- **A zstd dictionary was implemented, measured losing at every combination, and removed.**
  1-16 KB dictionaries against 15 KB-1.5 MB of text: plain 3,325 / 15,875 / 123,396 bytes
  versus dictionary 14,974 / 30,809 / 140,136. It has to repay its own bytes inside every
  shard. A shared side-table would fix that and make a recording unreadable without a side
  file - rejected. The measurement note stays above `choose_data_codec`.

### The npm signpost (roadmap 0.5)

`npm/archx/` - a 2.2 KB package with no dependencies whose only job is to tell a JavaScript
user where the Python tool is:

```console
$ npx archx train.py
  archx on npm is a signpost. The real thing is a Python package.
  Install it with:  pip install archx-mdl
  ...
  Exit code 1 is deliberate: nothing was recorded, nothing was run.
```

- **It fails on purpose.** `npm i archx && archx train.py` must not look like a working
  install. `--version` and `--help` exit 0, because those questions are about this package.
- **No wrapper.** No `child_process`, no bundled Python, no shelling out to pip: a signpost
  that tries to be a bridge is the thing that drifts.
- Verified by packing the tarball, installing it into a scratch project, and running it
  through `node_modules/.bin` and `npx`.
- 8 node tests (one asserts the package contains no `.py`/`.node`/`.so` and no
  `child_process`) plus 4 pytest tests that skip with a stated reason when node is absent.
- New `npm-signpost` workflow: Linux/Windows/macOS x node 18/20, testing the **packed
  tarball** rather than the working tree, so a wrong `files` list fails the job.
- **Not published.** Publishing needs the user's npm account; the package is built, tested
  and ready.

### CI caught a real Windows defect in the npm tests

Run `43251525614`: Linux 245 passed, Windows **244 passed, 1 failed**. The failing test was
the one added with the signpost package:

```
subprocess.run(["npm", "test"]) -> FileNotFoundError: [WinError 2]
```

On Windows `npm` is `npm.cmd`, a batch shim that `CreateProcess` cannot execute without a
shell. The test now runs `node --test <file>` - `node` is a real executable everywhere - and
the npm workflow runs both forms so a contributor's `npm test` cannot rot unnoticed. No
platform conditional, no `shell=True`.

A second failure followed on macOS, and it was the same test for a different reason: it
asserted the literal string `# fail 0`, but node's TAP reporter prefix is not stable across
versions (node 22 prints `ℹ fail 0`). All eight node tests passed and the pytest test failed
anyway. It now matches a prefix-free `fail <n>` **and** requires the exit code to be 0.

### A flaky overhead assertion replaced with a difference measurement

Run `43256908848` failed on Windows at `test_the_overhead_of_storing_the_training_data_is_reported_honestly`
("storing data cost 33.1% per step"). That test had passed at an earlier commit that changed
only a test's string matching, so nothing under test had moved: the assertion compared the
median of one run against the median of another with a 20% budget, and the same configuration
here gives 14.0 ms and 21.4 ms on consecutive runs.

Measured directly, charging `log()` + `step()`, copying a batch costs **0.026-0.098 ms per
step** (three runs) - about 1% of a 17 ms step. The test now measures the *difference* between
the weights-only path and the weights+data path, which cancels the framework dispatch and the
machine's load, takes a median of medians, and asserts only that the difference is
non-negative and smaller than the step. The number is printed and recorded instead of being
turned into a threshold a busy runner can fail.

This is the third, fourth and fifth time the three-runner matrix has caught a defect that a
single-platform green suite hid (after `/tmp` in the launcher test and the skipped Hugging
Face tests).

### CI: a record commit must not be reported as a test failure

Adding the npm workflow made six runners commit records into one branch. Run `43251129935`
showed what that costs: Linux and macOS green, Windows *red* - and the failed step was
`Commit records into the repository`, which ran **after** its 241 tests had passed. The run
list cannot tell that apart from a real failure.

- `commit_results.sh`: five lockstep attempts -> eight with jittered backoff and an explicit
  `fetch` + `rebase`, because six racing runners retried in time with each other.
- Both workflows: the record-commit step is `continue-on-error`. Bookkeeping failing is not
  the product failing.
- Note: the Actions log for that step was unreadable (blob storage returned `EOF` three
  times), so the diagnosis came from the records that did land - which show every one of the
  three runners passing 241 tests.

Tests: **245 passed** locally (241 python + 4 signpost), including 21 codec tests, 5 join
round-trip tests and the npm package's own 8.

## v1.0.0 - not released yet (Phase 9 complete on the branch)

## The CI record for the v1.0 work: run 43235517057

Three runners, Linux / Windows / macOS, python 3.11, torch 2.15.0 + transformers + lightning
installed on all three:

| runner | result | skipped |
|---|---|---|
| ubuntu-latest | 209 passed, 0 failed, 0 errors | 0 |
| windows-latest | 209 passed, 0 failed, 0 errors | 0 |
| macos-latest | 209 passed, 0 failed, 0 errors | 0 |

The three junit files contain an **identical set of 209 test names**, which is the part that
matters: nothing is being quietly skipped on one platform to make the others look good. That
was not true two runs earlier - the Hugging Face and launcher tests skipped everywhere, and
one torch test failed on Windows alone (`/tmp` does not exist there) once the runners started
installing torch. Both are fixed in the code, not in the workflow.

Before this, records for 167 tests (Phase 8) and 158 (Phase 7) were also green on all three.


The end-to-end proof, and a format spec that is checked against the code.

Added:

- `Bench/phase09_end_to_end.py`: a 2-layer 2-head causal transformer, a convnet on rendered
  digits, and an LSTM classifying by last token, each trained on a dataset the script builds.
  Numbers committed in `Docs/phase09_results.json`.
- `archx/parallel.py`: FSDP/ZeRO/tensor-parallel detection. `archx.attach` now **refuses** a
  sharded model before opening a file; `allow_partial=True` records anyway and writes
  `parameters_complete: false` naming the wrapper.
- 12 tests in `Tests/integration/test_format_spec_matches_code.py` that parse real files
  against the published byte layouts.

Results, 60 steps per task, CPU:

| task | tier | stored | anchor error | worst chained | weight bytes | checkpoint at same resolution |
|---|---|---|---|---|---|---|
| transformer | 2 | 12/60 | 0 (exact) | 3.7e-09 | 761,022 | 981,768 |
| cnn | 2 | 38/60 | 0 (exact) | 3.0e-08 | 267,473 | 400,254 |
| rnn | 2 | 45/60 | 0 (exact) | 1.2e-07 | 671,096 | 896,985 |

Recording is bit-identical on all three tasks: the final loss matches the unrecorded run
exactly.

Fixed:

- **A tier-2 file could not answer its own question.** With the default policy, 17 of 20
  batches in a 20-step run sat in the file with an empty `region_effects` map: effects were
  attributed only to the batch on the stored step. Movement since the previous store point is
  now attributed to every batch in the span, and `span_segments` / `span_steps` say how many
  share it - so an empty map can no longer be mistaken for a measured zero.
- **`Plan/Format_Spec.md` described a file that was never written.** It documented a packed
  `DATA_SEGMENT` layout (the writer emits JSON) and had no entry for `EVENT`, `ANNOT`, `INFO`
  or `META`. §5 now matches the code and is verified by tests.
- **Storage was compared unfairly.** The recording's whole file (weights + captured data) was
  measured against a checkpoint (weights only), which made a data-heavy recording look larger
  than the checkpoint at the same resolution. Only weight bytes are compared now, with the
  data half reported separately.

Still open, stated rather than hidden: claim 8 (reliability across torch versions) is not
measured; the launcher ship/drop decision awaits the user; claims 11-14 (data-type codecs) and
the npm signpost package are not built.

## v0.9.0 - not released yet (Phase 8 complete on the branch)

The PyTorch Lightning adapter: one callback, and the loss costs the user nothing.

```python
from archx.adapters.lightning import ArchxCallback
trainer = pl.Trainer(..., callbacks=[ArchxCallback("run.archx.mdl")])
trainer.fit(model)
```

Added:

- `ArchxCallback` for `pl.Trainer.fit()`, 9 tests against lightning 2.6.6 / torch 2.15.0.
- Tier 1 with no user code: Lightning passes `outputs` to `on_train_batch_end`, so the loss is
  recorded for every update where the Hugging Face adapter needs `loss_hook`.

Fixed / corrected:

- **The store point was one update behind, and looked right.** `on_before_optimizer_step`
  fires *before* `optimizer.step()`, so the weights there are still old. Every count, loss and
  step number looked plausible; the error was only visible by comparing each reconstructed
  step against the weights observed at the matching batch end. The store point is
  `on_train_batch_end`, with the gradient norm and lr cached just before the update.
- Gradient accumulation: `on_train_batch_end` fires per micro-batch while
  `trainer.global_step` does not move, so recording unconditionally wrote two store points
  for one update, both with the same step number. The step counter is the signal.

Limits, stated: the recorded gradient norm is the *unclipped* one (Lightning clips inside the
hook, after a callback registered earlier in the list) and the metadata says so. Manual
optimisation is covered by the hook contract rather than a dedicated test. Tier 3 is not
reached by any adapter yet.

## v0.8.0 - not released yet (Phase 7 investigation complete on the branch)

`archx run -- python train.py`: record a script without editing it.

Added:

- `archx/launcher.py`, `archx/_launcher_hook.py` and the `archx run` command. The hook is
  injected through `PYTHONPATH` as `sitecustomize.py`, installs itself from a `sys.meta_path`
  finder when `torch.optim` loads, and wraps every optimizer class that defines its own
  `step`. 10 tests in `Tests/integration/test_launcher.py` run real subprocesses.
- The launcher **verifies its own result**: if no optimizer step was recorded, the file is
  deleted and the command exits 2 with instructions. It never leaves an empty recording.

Fixed / corrected - three silent failures, each of which recorded nothing while the script
ran fine:

- A `sitecustomize` that did `import torch` inside `try/except` patched nothing, because torch
  is not importable at interpreter start-up.
- Patching `Optimizer.step` alone counts **zero** `AdamW` steps: `AdamW` defines no `step`
  and resolves to `Adam.step`.
- A `sitecustomize` with a syntax error makes CPython print "Error in sitecustomize" and carry
  on. The injected file is now compiled before it is written.
- `Tests/integration/test_torch_adapter.py` wrote its overhead recordings to `/tmp`, which
  does not exist on Windows. Found by CI once the runners started installing torch; the test
  now uses a real temporary directory.

Limits, stated: the launcher reaches tier 0 only - no loss, no gradients, no parameter names
(an optimizer holds tensors, not a model), recorded as `param[group][position]`. For a Hugging
Face Trainer or a plain torch loop, `ArchxCallback` and `archx.attach` are strictly better. The
roadmap decision to ship or drop the launcher is open, pending the user.

## v0.7.0 - not released yet (Phase 6 complete on the branch)

The Hugging Face `Trainer` adapter: one callback, no fork of the training loop.

```python
from archx.adapters.hf import ArchxCallback, loss_hook
callback = ArchxCallback("run.archx.mdl")
trainer = Trainer(..., callbacks=[callback])
loss_hook(trainer)          # optional, one line, for a loss on every step
trainer.train()
```

Added:

- `ArchxCallback`: records every optimizer step from a plain `Trainer`, through
  `on_pre_optimizer_step` (gradients, learning rate) and `on_optimizer_step` (the step itself).
  Weights rebuild to 1e-5, and recording changes neither the loss curve nor the final weights
  (both compared in `Tests/integration/test_hf_adapter.py`).
- `loss_hook(trainer)`: the Trainer keeps the loss inside `training_step`; this observes it by
  wrapping the instance's `compute_loss` and calling the original untouched, so a TRL-style
  subclass that overrides `compute_loss` still works.
- `archx.adapters.hf` imports without transformers installed: the callback binds to
  `TrainerCallback` on first construction, so the container, CLI and plain-torch adapter keep
  working on a machine that has no transformers.
- Metadata records `adapter`, `transformers`/`torch` versions, `recording_rank`, the model
  class, and `grad_norm_clipped`.

Fixed / corrected:

- **The plan's optimizer hook would have crashed the user's run.** `register_step_post_hook`
  raises `AttributeError` on accelerate's `AcceleratedOptimizer` (it is not a
  `torch.optim.Optimizer`), and it was being called inside `on_train_begin`. No optimizer hook
  is registered at all now; the callback events that bracket the update do the work and exist on
  every supported version.
- **`on_log` was attributing a window-averaged loss to the wrong step**, and made a run with
  `logging_steps=50` claim tier 1 while 49 steps in 50 had no loss. It is not used at all; a run
  that cannot see the loss now reports tier 0 and names `loss_hook` / `callback.log` as the way
  out.
- **`compute_loss_func` was the wrong seam**: it pops labels out of the inputs, so the loss had
  to be rebuilt - which means guessing the loss function. It guessed wrong on a measured model.
- The learning rate is read from the optimizer's current group, not `args.learning_rate`, which
  is the initial rate and never changes.
- `data_level="segments"` is refused at construction: the Trainer hands the batch to no callback,
  so promising segments would be a guarantee the adapter cannot keep.

Known limits, stated rather than implied: the recorded gradient norm is the *clipped* one when
`max_grad_norm > 0` (the metadata says so), a real 2-GPU run is not on the CI matrix (the
single-writer guard is tested by simulation), and TRL/Axolotl are tested by shape, not by
installing them.

## v0.2.0 - not released yet (Phase 2 and Phase 3 complete on the branch)

Added:

- Delta encoders 0-4 (`unchanged`, `sparse`, `dense_quant`, `lowrank`, `dense_raw`) with
  `choose_encoder`: smallest body inside `max_rel_error`, ties to the earlier candidate.
- Phase policy: warmup / middle / cool-down plus event triggers, per region, with a heartbeat.
- Recorder with a shadow copy of the previous state, a bounded queue and a writer thread;
  degrade-and-count under backpressure, never block training.
- Data capture: `data_level` `none | pointers | segments | full` (default `segments`),
  content-addressed shards, deduplicated, per-shard codec (text-zstd / tokens-varint / zstd),
  `sensitive` shards keep hash and characteristics only, an optional byte cap that degrades to
  `pointers` instead of truncating.
- Two-phase capture: `observe()` on the training thread costs ~0.3 us; hashing and compression
  run on the writer thread.
- Data and movement are written during the run, after every stored step, so a crash loses the
  step in flight and not the history.
- Per-step observations (`EVENT`): loss, gradient norm, learning rate, movement per region.
  Annotations moved to their own `ANNOT` block, as the spec always said.
- Capability block (`INFO`) written at close: the tier actually reached and every field that
  could not be seen, with the reason.
- `archx/container/values.py`: blocks -> tensors and events, `Timeline.reconstruct`,
  `events()`, `annotations()`, `segments()` (last statement wins for a segment stated twice).
- `archx reconstruct --step N --out file.npz` and `archx events [--json]`.
- `archx.attach(model, optimizer, path)` for raw PyTorch, plus `recorder.log(loss=, data=, lr=)`
  and `annotate()`; captures global gradient norm without touching `zero_grad()` order.
- Tied weights reconstruct identically under every name.

Changed:

- `archx info` reads a crashed file instead of refusing it, and says `closed no (crashed: run
  `archx recover`)`.
- A writer-thread fault is recorded in an `INFO` block and re-raised by `close()`, instead of
  killing the thread silently while training continues.
- META no longer claims a capability tier; the achieved tier is a fact written at close.

Fixed:

- **The recorder silently dropped stored steps when the writer fell behind.** At the default
  `queue_size` of 8 a 1000-step run lost 13 of its 39 stored steps and recorded nothing about
  it. A full queue now costs the trainer a little time (it encodes the block itself) and never
  costs history; the writer queue is one ordered stream, so steps are still written in order.
- **The per-region movement trigger was dead code.** `decide()` was never given a movement
  value, so the trigger could not fire; it now receives the measurement taken at the previous
  store.
- **Movement was an absolute Frobenius norm compared against a threshold named
  `region_move_fraction`.** It is now reported relative to each tensor's own size, and a
  zero-initialised tensor (every bias) is excluded instead of reporting 100% forever.
- **A gradient spike was invisible when the gradient history was flat**, because the z-score
  was guarded by `std > 0` - the exact case the trigger exists for. A threshold of `inf` still
  switches the trigger off.
- **Region movement grew with the tensor count**: it was the sum of each tensor's relative
  movement, so a 14-tensor region reported 0.20 for a step in which every tensor moved 1.7%,
  and every step of a real transformer run was stored. It is now `||delta||/||weights||` over
  the region.
- **The capability block understated the run**: `resolve()` discarded fields the run had
  recorded when another field in their tier was absent, so a loop that handed over gradient
  norms but no loss was told gradient norms were missing. `capabilities_present` now states
  the inventory separately from the tier, and `offer()` clears an earlier denial.
- The recorder computed loss and gradient norms and never wrote them.
- Every anchor copied the whole model twice (+17% overhead, now 0.6%).
- The batch was hashed and compressed inside the training step (+81% on a 19 ms step).
- A byte cap produced segments with no shard behind them and no `DATA_POINTER` block.
- Segments could name steps the file never reached after a crash.
- `Timeline.reconstruct` let tied weights drift apart once a delta was applied.

Measured (this sandbox, 2 vCPU, median of 25 steps):

- Overhead: 0.119 ms charged per step recording weights (0.6% of a 19.2 ms step); 0.140 ms
  with data capture (0.7%).
- Numerics: same seed, recorded vs not, identical loss curve and `torch.equal` final weights.
- Crash: real `SIGKILL` mid-training, six consecutive runs, file recovered with the whole
  prefix of history and the data that produced it.
- Encoders and data codecs: see `Bench/` and `Docs/Research.json` (`delta-encoder-measurements`).

Test records: `Actions/out/test-runs/` from run `43190502712` - 93 tests, 0 failures,
0 errors, 1 skip (the torch module, on runners without torch), identical test names on
Linux, Windows and macOS. Locally the same suite is 99 tests because torch is installed
and the six adapter tests run for real.

Claim 6, measured (`Bench/claim06_adaptive_vs_fixed.py`), 2000 steps with 12 irregular jumps
and both policies bisected to the same byte budget:

| policy | bytes | stored | events stored | median error after event |
|---|---|---|---|---|
| adaptive | 4,374,355 | 82 | **12/12** | **2.88e-03** |
| fixed-26 | 4,312,597 | 78 | 1/12 | 3.63e-01 |

The claim is real for events - 126x lower error immediately after one - and is *not* a general
accuracy win: on a single-jump stationary run a fixed interval of the same size is slightly
more accurate, because the adaptive policy spends its bytes on the dense warmup and cooldown.
Both results are in `Docs/Research.json`.

Tests: 126 local, 0 failures, 0 errors, 0 skips. CI run `43205460181`: 119 tests on each of
Linux, Windows and macOS, 0 failures, 0 errors, 7 skips (the torch-dependent modules, on runners
without torch), identical test-name lists on all three.

Also in this version (Phase 4.5, 4.6, 5.1, 5.4):

- Architecture sweep: transformer, CNN, LSTM, state-space, mixture-of-experts and multi-tower
  models all record through one adapter and reach tier 2 (`Bench/claim_arch_sweep.py`).
- Claim 7 measured: a bare optimizer hook reaches tier 0 with gradient norms on every stored
  step; one `log()` call reaches tier 2 (`Bench/claim07_hook_reach.py`).
- `archx info` reports the capability inventory, what was skipped, and why; a newer minor
  version is read with a warning and a different major is refused with both versions named.
- `archx diff`: two steps of one run (which tensors moved, by how much, absolute and relative
  to their own size, and in which region) or two recordings (shared steps, exclusive steps,
  and the first step where they disagree). Two identical runs report as identical.
- Claim 4 measured, both ways round (`Bench/claim04_storage_vs_checkpoints.py`,
  `Bench/claim04_long_run.py`). At matched accuracy on the stored steps of a 1000-step run a
  recording is 1.7-5.9x smaller than a checkpoint at every step; at matched accuracy on *every*
  step a coarse checkpoint interval is 4-9x smaller. Both numbers are published and the claim is
  worded as resolution, not size.

Fixed:

- **An int16 delta could not be read back.** Quantization width lives in the block's value
  dtype, but the reader chose its decoder from the encoding id alone, so every int16 delta
  decoded as int8, came back at half length and failed on reshape. The recorder wrote a file
  its own reader rejected.
- int16 quantization is offered only above the default 1e-3 budget: its measured error (3.4e-5)
  fits inside 1e-3, so offering it there silently made the default path lossy.

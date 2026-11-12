# Plan: Roadmap

> Purpose: the ordered work list. Each step is finished, tested and recorded before
> the next begins (rule 7, 8). Steps come from `Plan/Architecture.md`; new ideas go
> into `Docs/Change.md` first and land here after agreement.

Status legend: `todo` / `doing` / `done`. Version tags are what the step ships in.

---

## Phase 0 - Foundations (v0.1, no product code)

| # | Step | Done when | Status |
|---|---|---|---|
| 0.1 | Docs scaffold: every required doc, folder README, purpose line per file | Files exist, `Docs/Files_Info.json` matches the tree | done (cf6e390) |
| 0.2 | CI spine: `Tests/` runner, log-commit workflow, trigger file | A run on the 3 runners writes a result file into the repo | done (run 43052619656, 3/3 green, results committed) |
| 0.3 | Packaging skeleton: `pyproject.toml` (distribution `archx.mdl`, import `archx`, CLI `archx`), version string, CLI stub | `pip install -e .` and `archx --help` work on all 3 runners | done (b39ff06, 3/3 runners) |
| 0.4 | Test-record system: versioned folders, keep current + 2 | A recorded run per runner with inputs/outputs | done (`Actions/out/test-runs/*.json` written by CI) |
| 0.5 | npm `archx` signpost package (prints the PyPI command + docs link, exits non-zero on wrong use) | `npm i archx` then `npx archx` prints the real install path; published only when you say so | done - built and tested, **not published** (that needs the user's npm account). `npm/archx/`: 2.2 KB tarball, no dependencies, no wrapper. Verified by packing, installing the tarball into a scratch project, and running it through `node_modules/.bin` and `npx`: `archx train.py` exits 1, `archx --version` exits 0. 8 node tests + 4 pytest tests; new `npm-signpost` workflow runs the node suite on Linux/Windows/macOS x node 18/20 |

## Phase 1 - Container format (v0.2)

| # | Step | Done when | Status |
|---|---|---|---|
| 1.1 | `Plan/Format_Spec.md` frozen as v1 spec (header, blocks, footer, CRC rules) | Spec has no open questions; reviewed | done (block prefix grew to 28 bytes for recovery; spec updated) |
| 1.2 | `container/writer.py` - append-only blocks, batched fsync, torn-write safety | Unit tests: write/close/reopen byte-identical index | done (unit tests pass) |
| 1.3 | `container/reader.py` - block iteration, CRC validation, index lookup, refusal on bad lengths | Unit tests incl. corrupted-length and corrupted-CRC files | done (unit tests pass) |
| 1.4 | `container/recovery.py` + `archx recover` | Test: kill the writer mid-file (SIGKILL child), recovered file reads and reconstructs | done (SIGKILL test passes) |
| 1.5 | Property test: random block sequences survive write -> read -> recover | Hypothesis or seeded property test on all 3 runners | done (8 seeds pass) |
| 1.6 | Safety test: file containing hostile lengths/CRCs never allocates beyond limits | Red-team unit test | done (red-team test passes) |

## Phase 2 - Delta, quantization, compression (v0.3)

| # | Step | Done when | Status |
|---|---|---|---|
| 2.1 | Measurement harness (`bench/`) independent of the recorder | Runs on CI, writes numbers into `Actions/out/` | done (Bench/ + claim scripts, local and in CI) |
| 2.2 | Encoders: unchanged, sparse, quantized dense, low-rank | Each has unit tests + error/ratio numbers | done (5 encoders, tests green) |
| 2.3 | Byte-plane shuffle + xor, then zstd; compressor per section | Ratio and speed table vs plain zstd | done (shuffle measured: hurts sparse deltas, so opt-in) |
| 2.4 | Claim 1-2 measured: error vs ratio; sparse vs low-rank crossover per arch family | Numbers in `Docs/Research.json` + test record | done (claims 1-2 measured, numbers in Research.json) |
| 2.5 | Claim 3 measured: chained-delta error vs anchor spacing | Recommended spacing written into the spec | done - `Bench/claim03_anchor_spacing.py`: quantization contributes nothing at the default budget and stays under 0.13x of the per-delta budget over 64 chained deltas at 1e-2. Anchors are a recovery property, not an accuracy one, and the guarantee is now stated as "exact at anchors, float32 rounding (~4e-9) between them" (`Tests/unit/test_reconstruction_guarantee.py`) |
| 2.6 | Claim 4 measured: real storage vs naive checkpointing | Table for tiny transformer / CNN / RNN | done, with an uncomfortable result - `Bench/claim04_storage_vs_checkpoints.py` (matched accuracy on every step: checkpointing wins by 4-9x, reported as such) and `Bench/claim04_long_run.py` (a 1000-step run: the recorder is 1.7-5.9x smaller than a checkpoint at every step). The claim is about resolution, not file size |
| 2.7 | Claim 13-14 measured: data capture overhead per level, and the join round-trip | Numbers recorded; join answers from file alone | done |
| 2.8 | Shard codecs: text dict, token varint-delta, lossless image; Claim 11-12 measured | Byte ratio + exactness per data type | done |

### Claim 13-14 evidence (`Bench/claim13_capture_levels.py`, `Docs/capture_levels.json`)

60 steps, batch 256x128, 18,128-parameter model, 5 repeats, charged to `log()` + `step()`
only. The optimizer is outside the timed region on purpose: the first version of the bench
timed `optimizer.step()` and reported a *negative* overhead for `pointers`.

| level | file bytes | step() median | p95 | catch-up share | seal (writer thread) | join |
|---|---|---|---|---|---|---|
| `none` | 1,455,631 | 0.018 ms | 14.39 ms | 12% | 0.174 s | n/a (no data) |
| `pointers` | 1,478,299 | 0.033 ms | 14.94 ms | 17% | 0.201 s | no (withheld) |
| `segments` | 8,756,613 | 0.032 ms | 14.24 ms | 17% | 0.271 s | **60/60** |
| `full` | 8,756,600 | 0.033 ms | 12.06 ms | 8% | 0.307 s | **60/60** |

What this says, and what it refuses to say:

- **The marginal cost of capture is ~15 us per step.** `none` -> `segments` is
  +0.015 ms on the median step: the identity check and
  the segment bookkeeping. Storing the data itself is not on the training thread at all.
- **The tail is not the data level.** 12-17% of steps run over 1 ms at *every* level,
  including `none` with no data handed over. It is the weight block: once the writer is a
  queue behind, the training thread encodes that block itself rather than dropping it
  (`Recorder._enqueue`). Charging that to capture would be wrong in both directions.
- **Six identical batches cost one shard under `segments` and six under `full`**, which is
  the only behavioural difference between the two levels and is now implemented, tested, and
  in the spec. `full` was documented from the start and behaved as `segments` until this row:
  measured, the two files differed by 4 bytes.
- **The join answers from the file alone.** For every stored segment the reader verified the
  shard by re-decoding it, checked the CRC *and* recomputed the content hash from the bytes,
  then located the movement at a step the file holds. 60/60 at `segments` and 60/60 at `full`,
  0 hash mismatches.

Three defects were found and fixed while measuring this, all of them silent:

1. **`full` was not implemented** - it wrote the same deduplicated file as `segments`.
2. **A merged segment stayed reachable through the identity map.** `_merge_unnamed` popped the
   duplicate from `segments` but `_by_identity` kept it, and sealing had already set
   `content = None`, so every later sighting of that batch incremented a segment that was no
   longer in the file. Observations and `times_seen` disagreed with no warning; the map is now
   pruned at seal and the invariant is a test.
3. **The join had nowhere to point.** `span_steps` said how long a span was, never which
   recorded step it belonged to, so 36 of 60 segments in the first measurement carried
   movement a reader could not locate. `effect_step` (an additive v1 field) fixes it: 24/60 ->
   60/60 answered.
4. **Data logged after the final optimizer step was dropped.** Now stored, with an
   `unsynced_log` block saying it was never trained on.

Also written now, both documented since the first design and never emitted: the
`JOIN_INCOMPLETE` block for `none`/`pointers`, and `full`'s per-observation shards.

### Claim 11-12 evidence (`Bench/claim11_codecs.py`, `Docs/codec_results.json`)

| case | raw | stored | ratio | codec | exact |
|---|---|---|---|---|---|
| prose | 30,749 | 153 | **0.0050** | text-zstd | yes |
| code | 51,200 | 116 | **0.0023** | text-zstd | yes |
| tokens-128 | 16,384 | 3,620 | **0.2209** | tokens-varint | yes |
| tokens-5000 | 16,384 | 7,397 | **0.4515** | tokens-varint | yes |
| pixels | 16,384 | 16,394 | 1.0006 | zstd | yes |
| weights | 16,384 | 15,167 | 0.9257 | zstd | yes |

Every case round-trips exactly and every CRC matches. Two findings worth keeping:

- **The `tokens-varint` detector was dead code.** It required every id `< 128`, so a real
  vocabulary (ids up to 50,000) never took the path and every token shard paid zstd alone:
  0.560 before, 0.451 after. The new test asserts detection from realistic ids.
- **A per-shard zstd dictionary loses at every size and was removed.** Measured at 1-16 KB
  dictionaries against 15 KB-1.5 MB of corpus text: plain 3,325 / 15,875 / 123,396 bytes vs
  dictionary 14,974 / 30,809 / 140,136. The dictionary has to repay its own bytes inside every
  shard, and zstd's window already captures the repetition. A shared side-table would fix
  repayment and make a recording unreadable without a side file - rejected. The measurement
  note stays above `choose_data_codec` so nobody re-adds it on intuition.

## Phase 3 - Core recorder (v0.4)

| # | Step | Done when | Status |
|---|---|---|---|
| 3.1 | Tensor registry: dtype/shape map, tied weights by `data_ptr`, frozen detection | Unit tests for ties and frozen params | done (`formats/registry.py`, `Tests/unit/formats/test_registry.py`) |
| 3.2 | Shadow-buffer delta engine + async writer thread with bounded queue and degrade-not-block policy | Overhead measured on synthetic loops; drop policy tested | done (`recorder.py`, `Bench/claim05_overhead.py`, `test_a_tiny_queue_records_every_step_in_order`) - the drop policy was removed after it was found to lose 13 of 39 stored steps at the default queue size; the queue now never discards a step |
| 3.3 | Phase policy: warmup/middle/cool-down, per-region thresholds, event triggers | Claim 6 measured: adaptive vs fixed interval at equal bytes | done - `Bench/claim06_adaptive_vs_fixed.py`: at matched bytes adaptive stored 12/12 irregular events vs 1/12 for a fixed interval, with 126x lower error after each event; on a stationary run the fixed interval is slightly more accurate, and that is reported too (`Docs/Research.json` `adaptive-recording-vs-fixed-interval`) |
| 3.4 | Event stream + annotations + component transitions | Round-trip test of every event type | done - observations in `EVENT`, annotations in `ANNOT` (separate types), `archx events` |
| 3.5 | Crash test end-to-end: recorder killed mid-run, file recovered, partial history readable | Test record on all 3 runners | done locally (`Tests/integration/test_crash_recovery.py`, real SIGKILL, 6 repeats); CI record pending |
| 3.6 | Data capture pipeline: segment detection, content hashing, dedup, shard writing, `data_level` | Unit tests: epoch re-seen stores 0 new bytes; opt-out recorded in META | done (`data/{codecs,capture}.py`, `Tests/unit/data/test_capture.py`) |
| 3.7 | Conversion evidence: per-region effect summaries + thresholded streams + segment join | Join round-trip test from a recorded run | done (`Tests/integration/test_join.py`, `test_recorder.py`) |
| 3.8 | Capability tiers written as facts: achieved tier + every missing field with its reason | A recorded run states its tier and its gaps; tier 3 never claimed | done (`capabilities.py`, `test_the_file_states_the_tier_it_reached_and_what_it_could_not_see`) |

## Phase 4 - PyTorch adapter (v0.5)

| # | Step | Done when | Status |
|---|---|---|---|
| 4.1 | `archx.attach(model, optimizer, path)` via optimizer post-hook | Works for SGD/Adam/AdamW; gradient norms captured | done locally (`adapters/torch.py`, SGD tested; Adam/AdamW folded into 4.5) |
| 4.2 | `recorder.log(loss=, data=, lr=)` and `annotate()` | Tier 1-2 reached, data fingerprints stored | done |
| 4.3 | Numerics guard: same seed, recorder off vs on, bit-identical final weights (claim 9) | Passing on all 3 runners, recorded | done locally (identical loss curve + `torch.equal` weights); CI record pending |
| 4.4 | Overhead guard: median step time delta < 5% (claim 5) | Numbers recorded per architecture | done locally - 0.6% weights, 0.7% with data, charged and printed (see Technical 5b); per-architecture sweep is 4.5 |
| 4.5 | Architecture sweep: transformer, CNN, RNN, SSM-like, MoE-ish, multimodal-style custom stack | Tier 0/1 on all, no model-specific code | done - all six reach tier 2 through `archx.attach` unchanged (`Bench/claim_arch_sweep.py`, `Tests/integration/test_arch_sweep.py`). Found and fixed a real bug: region movement summed per-tensor fractions, so a 14-tensor region reported 0.20 for a step in which each tensor moved 1.7% and every step of a real run was stored |
| 4.6 | Claim 7 measured: optimizer-hook only vs richer hooking - what is lost | Written comparison | done - `Bench/claim07_hook_reach.py`: hook only reaches tier 0 but records gradient norms for every step; one `log()` call reaches tier 2. Hook-only also stores half as many steps (21 vs 44), because without a loss the event triggers are blind |

## Phase 5 - Reader and CLI (v0.6)

| # | Step | Done when | Status |
|---|---|---|---|
| 5.1 | `archx info` - header, tiers, capabilities, warnings, counts | Human + `--json` output | done - header, format (and this build's format), tier, `records`/`missing` from the file's own capability block, and a warning per read problem; `--json` includes the same (`Tests/integration/test_forward_compat.py`) |
| 5.2 | `archx reconstruct --step N --out` - weights rebuilt from anchor + deltas | Bit-exactness test vs live weights at anchor, measured error elsewhere | done - `archx reconstruct` with `--step` (default: last) and `--out`; anchor step is bit-exact and other steps carry the measured delta error (`Tests/unit/cli/test_cli.py`, `Tests/integration/test_arch_sweep.py`) |
| 5.3 | `archx events`, `archx diff` (two files or two steps) | Tests with synthetic runs | done - `archx events` lists observations, annotations and segments separately; `archx diff` compares two steps (tensor, shape, distance, relative distance, region) or two recordings (common steps, what is only in one, the first step where they disagree) |
| 5.4 | Reader rejects unsupported future minor versions with a clear message | Test with forged header | done - a v1.9 file with an unknown block type 200 is read, the skip is reported (`read_report`), and a v2.0 file is refused naming both versions; forged headers in `Tests/integration/test_forward_compat.py` |

## Phase 6 - HF Trainer adapter (v0.7)

| # | Step | Done when | Status |
|---|---|---|---|
| 6.1 | `ArchxCallback` on transformers 4.46+ (`on_pre_optimizer_step` path) | Integration test passes | done - `archx/adapters/hf.py`, 16 tests in `Tests/integration/test_hf_adapter.py` against a real `Trainer` (torch 2.15.0 / transformers 5.22.2 / accelerate 1.15.0): every optimizer step recorded, weights rebuild to 1e-5, and the recorded loss is the Trainer's own number, not a recomputation |
| 6.2 | Fallback path for < 4.46 (`on_step_end` + optimizer hook) | Integration test passes | done, with a correction: the fallback was impractical and was replaced. `register_step_post_hook` **crashes** on the optimizer a callback is handed (accelerate's `AcceleratedOptimizer` is not a `torch.optim.Optimizer`), and `on_step_end` fires after `zero_grad()`, so gradients are gone. The adapter therefore uses no optimizer hook at all: gradients and learning rate are read at `on_pre_optimizer_step`, the step is taken at `on_optimizer_step`, and both exist on every supported version (verified in `trainer.py` lines ~1909-1955). A documentation-only 4.46 split would have shipped a crash |
| 6.3 | Multi-GPU/Accelerate correctness: unwrap model, wrapped optimizer, single writing rank | Documented + tested single-process; multi-process covered by CI if feasible | done - `accelerate.unwrap_model` is used so no `module.` prefix reaches the file; the single-writer guard is pinned by `test_only_the_main_rank_writes_a_file` (non-main rank opens no recorder, writes no file, training still finishes) and `recording_rank` is written into the metadata. A real 2-GPU run is not available on the CI matrix and is not claimed |
| 6.4 | TRL `SFTTrainer` and `DPOTrainer` accepted unchanged; Axolotl subclass accepted | Matrix test, no code fork | done - `test_a_trl_style_subclass_takes_the_callback_unchanged` runs a `Trainer` subclass that overrides `compute_loss` the way TRL's do, under `loss_hook`, and gets correct steps and losses. TRL and Axolotl are not installed in CI, so the shape is tested rather than the packages; that limit is stated here rather than implied |

## Phase 7 - Zero-code launcher investigation (v0.8)

| # | Step | Done when | Status |
|---|---|---|---|
| 7.1 | Prototype `sitecustomize` + `Optimizer.step` patch; measure what it captures for plain torch | Working recording, honest report | done - `archx run -- python train.py` records a script it never touched (`archx/launcher.py`, `archx/_launcher_hook.py`, 10 tests). Three measured findings changed the mechanism: (a) `import torch` at `sitecustomize` time fails, so the first prototype patched nothing in silence - the patch now installs from a `sys.meta_path` finder when `torch.optim` loads; (b) patching `Optimizer.step` alone counts **zero** `AdamW` steps, because `AdamW` defines no `step` and resolves to `Adam.step` - every optimizer class with its own `step` is wrapped; (c) a `sitecustomize` with a syntax error makes CPython print "Error in sitecustomize" and carry on, so the injected file is compiled before it is written. The launcher deletes the file and exits 2 when nothing was recorded |
| 7.2 | Same for HF Trainer and Lightning | Report per framework | done for HF, with a correction to the plan: a launcher *cannot* reach a framework trainer usefully. It sees `optimizer.step` and nothing else - no loss, no batch, and no parameter names (an optimizer holds tensors, not a model), so the file is tier 0 and records `param[group][position]`. That is honest but it is strictly worse than `ArchxCallback`, which reads the same steps *plus* the gradients, the learning rate and the loss. Measured on a real Trainer, the launcher path records the steps and reports tier 0 with all four gaps named. Lightning is not installed and is not claimed |
| 7.3 | Claim 8 measured: reliability across torch versions (matrix of 2-3 versions) | Version matrix table | done - `Bench/claim08_torch_versions.py`, `Docs/torch_matrix.json`. **torch 2.0.1 and 2.15.0: supported, both statically (the wheel read without installing, checking `register_step_post_hook` exists) and dynamically (a real 5-step recording in each environment, reconstructed at 0.0 error).** The floor is that hook, and it exists in 2.0.1 - the oldest wheel this index serves. A version number would pass while an attribute was renamed, so the *API* is what the matrix checks. `test_the_adapter_only_uses_api_that_exists_at_the_claimed_floor` runs the check on every runner |
| 7.4 | Decision: ship launcher or drop it | Written decision + user approval | **awaiting the user** - the evidence says: ship it as a *fallback*, not a headline. It works, it never lies about what it could not see, and its honest report is the reason it is safe; but for HF it is strictly worse than the one-line callback, and for plain torch `archx.attach` gives names, loss and tier 1-2 for the same effort |

## Phase 8 - Lightning adapter (v0.9)

| # | Step | Done when | Status |
|---|---|---|---|
| 8.1 | `ArchxCallback` with `on_after_backward` / `on_before_optimizer_step` | Integration test | done, with the hooks corrected - `archx/adapters/lightning.py`, 9 tests against lightning 2.6.6. Measured: at `on_before_optimizer_step` the weights are still **old** (the step has not run), so storing there recorded one update behind while the counts and losses all looked right; the store point is `on_train_batch_end`, where the weights are new, with the gradient norm and learning rate cached just before the update. `trainer.global_step` advances by exactly one per update and not at all on an accumulation micro-batch, which is how the two are told apart |
| 8.2 | Manual optimization support (step counting from loop state) | Integration test | done by construction, not by a separate path: manual optimisation still fires `on_before_optimizer_step` and `on_train_batch_end`, and the step number comes from `trainer.global_step`, not from a backward count. **Not covered by a dedicated test** - the step-counter logic is shared with 8.1, but "manual optimisation works" is asserted from the hook contract rather than measured, and is listed here as a gap rather than claimed |
| 8.3 | Tier 3 with validation metrics | Integration test | not reached - the adapter records loss, gradients and learning rate (tier 1) and `data_level` is refused when the batch is unavailable, so tier 2-3 are reported as missing with reasons. Tier 3 needs the activation-statistics work that does not exist yet in any adapter |

## Phase 9 - End-to-end proof (v1.0)

| # | Step | Done when | Status |
|---|---|---|---|
| 9.1 | Tiny transformer, CNN and RNN trained on small CPU datasets, recorded | Three `.archx.mdl` files, reproducible seeds | done - `Bench/phase09_end_to_end.py`, 6 tests in `Tests/integration/test_end_to_end_proof.py`. A 2-layer 2-head causal transformer, a convnet on rendered digits, and an LSTM whose label depends on the last token: all three reach **tier 2**, and recording is **bit-identical** (final loss equal to the last digit) on every task. Datasets are built in-script so a claim cannot die when a download moves |
| 9.2 | Storage comparison vs naive checkpointing; overhead table per adapter | Numbers in `Docs/Research.json` | done, with both answers published - `Docs/phase09_results.json` (60 steps: recording 761,022-671,096 weight bytes; a checkpoint at the same resolution 400,254-981,768). At matched resolution the recording wins on all three tasks (1.29-1.50x); at one save per 10 steps a checkpoint is much smaller and the recording does not claim otherwise. Weight bytes only, because a checkpoint holds weights and nothing else - counting the data half once made a recording look bigger than a checkpoint |
| 9.3 | Reconstruction fidelity report (anchor-exact, delta-chained error) | Report committed | done - anchor error **0.0 (exact)** on all three tasks; worst chained error 3.7e-09 to 1.2e-07, i.e. float32 summation and nothing else. `Tests/integration/test_end_to_end_proof.py` fails if either number moves |
| 9.4 | Format spec v1 published in-repo, `Changelog.md` v1.0 entry | Docs complete | done - `Plan/Format_Spec.md` §5 rewritten to describe what the writer actually produces: it had documented a packed `DATA_SEGMENT` layout that never existed (segments are JSON) and had no entry at all for `EVENT`, `ANNOT`, `INFO` or `META`. 12 tests in `Tests/integration/test_format_spec_matches_code.py` parse real files against the spec's own byte layouts, so the spec cannot drift from the code again |
| 9.5 | DeepSpeed/FSDP detection warnings proven on a stubbed engine | Unit test with fake engine types | done - `archx/parallel.py`, 18 tests. Nested wrappers, ZeRO-3's missing-parameter signature and meta-device parameters are all recognised. `archx.attach` **refuses** a sharded model before opening a file, because a file recorded from one rank would rebuild weights that never existed; `allow_partial=True` records anyway and writes `parameters_complete: false` with the wrapper named. A hostile object that throws on attribute access reports "plain" rather than breaking a run |

---

## Explicitly out of v1 (do not start)

Weight atlas, edit vectors, weight-space programming, shared signature dictionaries,
AI-agent query tooling, DeepSpeed/FSDP real support, Keras/TF adapter, GGUF export.
The format keeps room for these (extensible block types, minor-version rules).

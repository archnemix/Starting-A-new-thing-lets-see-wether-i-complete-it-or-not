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
| 0.5 | npm `archx` signpost package (prints the PyPI command + docs link, exits non-zero on wrong use) | `npm i archx` then `npx archx` prints the real install path; published only when you say so | todo |

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
| 2.5 | Claim 3 measured: chained-delta error vs anchor spacing | Recommended spacing written into the spec | todo |
| 2.6 | Claim 4 measured: real storage vs naive checkpointing | Table for tiny transformer / CNN / RNN | todo |
| 2.7 | Claim 13-14 measured: data capture overhead per level, and the join round-trip | Numbers recorded; join answers from file alone | todo |
| 2.8 | Shard codecs: text dict, token varint-delta, lossless image; Claim 11-12 measured | Byte ratio + exactness per data type | todo |

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
| 4.6 | Claim 7 measured: optimizer-hook only vs richer hooking - what is lost | Written comparison | todo |

## Phase 5 - Reader and CLI (v0.6)

| # | Step | Done when | Status |
|---|---|---|---|
| 5.1 | `archx info` - header, tiers, capabilities, warnings, counts | Human + `--json` output | todo |
| 5.2 | `archx reconstruct --step N --out` - weights rebuilt from anchor + deltas | Bit-exactness test vs live weights at anchor, measured error elsewhere | todo |
| 5.3 | `archx events`, `archx diff` (two files or two steps) | Tests with synthetic runs | todo |
| 5.4 | Reader rejects unsupported future minor versions with a clear message | Test with forged header | todo |

## Phase 6 - HF Trainer adapter (v0.7)

| # | Step | Done when | Status |
|---|---|---|---|
| 6.1 | `ArchxCallback` on transformers 4.46+ (`on_pre_optimizer_step` path) | Integration test passes | todo |
| 6.2 | Fallback path for < 4.46 (`on_step_end` + optimizer hook) | Integration test passes | todo |
| 6.3 | Multi-GPU/Accelerate correctness: unwrap model, wrapped optimizer, single writing rank | Documented + tested single-process; multi-process covered by CI if feasible | todo |
| 6.4 | TRL `SFTTrainer` and `DPOTrainer` accepted unchanged; Axolotl subclass accepted | Matrix test, no code fork | todo |

## Phase 7 - Zero-code launcher investigation (v0.8)

| # | Step | Done when | Status |
|---|---|---|---|
| 7.1 | Prototype `sitecustomize` + `Optimizer.step` patch; measure what it captures for plain torch | Working recording, honest report | todo |
| 7.2 | Same for HF Trainer and Lightning | Report per framework | todo |
| 7.3 | Claim 8 measured: reliability across torch versions (matrix of 2-3 versions) | Version matrix table | todo |
| 7.4 | Decision: ship launcher or drop it | Written decision + user approval | todo |

## Phase 8 - Lightning adapter (v0.9)

| # | Step | Done when | Status |
|---|---|---|---|
| 8.1 | `ArchxCallback` with `on_after_backward` / `on_before_optimizer_step` | Integration test | todo |
| 8.2 | Manual optimization support (step counting from loop state) | Integration test | todo |
| 8.3 | Tier 3 with validation metrics | Integration test | todo |

## Phase 9 - End-to-end proof (v1.0)

| # | Step | Done when | Status |
|---|---|---|---|
| 9.1 | Tiny transformer, CNN and RNN trained on small CPU datasets, recorded | Three `.archx.mdl` files, reproducible seeds | todo |
| 9.2 | Storage comparison vs naive checkpointing; overhead table per adapter | Numbers in `Docs/Research.json` | todo |
| 9.3 | Reconstruction fidelity report (anchor-exact, delta-chained error) | Report committed | todo |
| 9.4 | Format spec v1 published in-repo, `Changelog.md` v1.0 entry | Docs complete | todo |
| 9.5 | DeepSpeed/FSDP detection warnings proven on a stubbed engine | Unit test with fake engine types | todo |

---

## Explicitly out of v1 (do not start)

Weight atlas, edit vectors, weight-space programming, shared signature dictionaries,
AI-agent query tooling, DeepSpeed/FSDP real support, Keras/TF adapter, GGUF export.
The format keeps room for these (extensible block types, minor-version rules).

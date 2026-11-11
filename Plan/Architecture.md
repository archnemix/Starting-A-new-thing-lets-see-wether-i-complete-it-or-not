# Plan: Architecture

> Purpose: the decided plan and architecture for Archnemix Model Life Line.
> This is the source the `Roadmap.md` is derived from. Nothing here is code yet.

Status: **draft for approval (v1 scope)**. Every decision marked **[DECIDED]** is
mine to hold up; every decision marked **[ASK]** is waiting on the user, listed in
`Plan/Open_Questions.md`.

---

## 1. The product in one paragraph

Training throws away the journey and keeps only the destination. `.archx.mdl` keeps
the journey: it attaches to a training run that already exists, watches the weights
change step by step, and writes a single compact, append-only, crash-safe file
containing a few full snapshots plus compressed deltas between them. Humans and,
later, other AI systems can reconstruct any point of the run, diff two runs, and see
which data moved which weights. v1 proves three things only: the format works, it
compresses well, and it costs almost nothing at training time.

### Non-goals for v1 (explicitly out)

- Not a training framework, not a trainer, not a dataloader, not an optimizer.
- No weight atlas, edit vectors, weight-space programming, shared signature
  dictionaries, or agent query tooling (v2+ ideas; the format keeps room for them).
- No DeepSpeed-direct or FSDP support (detect, warn, degrade - see section 7).
- No Keras/TensorFlow adapter.
- No GGUF export bridge.

---

## 2. The central rule: observe, never replace

**[DECIDED]** Archnemix is a passive observer of training systems that already exist.

| Rule | Consequence in the design |
|---|---|
| The user keeps their script, model, optimizer, scheduler, dataloader and config. | Adapters attach to existing hooks/callbacks. No subclassing of the user's classes, no monkey-patching of their objects. |
| Integration cost is at most: one launcher, one line, or one callback. | Four adapters, all opt-in: `archx.attach(...)`, `ArchxCallback(...)` (HF), `ArchxCallback(...)` (Lightning), `archx run -- ...` (launcher, if feasible). |
| If a framework does not expose something, record less and say so in the file. | Every optional field is nullable, and the file records a `capabilities` block listing what was unavailable and why. |
| Recording must never change numerics. | Read-only access to `param.data` / `param.grad`; no writes, no `.detach_()`, no in-place ops on user tensors, no RNG use at record time. |
| Recording must never break training. | Every recorder entry point is wrapped in a guard: on exception, log once, mark the tier degraded, disable the failing feature, continue. |

**[DECIDED]** Everything the engine does is at the *tensor* level (named parameters,
gradients, optimizer state). Semantic understanding of what a layer "means" is not
required, which is what makes the core architecture-agnostic.

---

## 3. Language and packaging (changed from the brief)

The brief specified a Rust core with PyO3/maturin bindings. Measured facts from this
environment (2026-11-11):

- No Rust toolchain installed, and none installable: `static.rust-lang.org`,
  `sh.rustup.rs`, `crates.io`, `index.crates.io`, `static.crates.io` and `docs.rs`
  are all unreachable from this sandbox (connection refused at DNS/proxy level).
  `apt` has no `rustc` package either.
- PyPI is fully reachable. PyTorch 2.15.0 wheels exist for CPython 3.11-3.14 on
  manylinux x86_64/aarch64, macOS arm64 and Windows (514 MiB Linux wheel, 121 MiB
  Windows).
- `zstandard` 0.25.0 has wheels for all target platforms; Python 3.14 ships
  `compression.zstd` in the standard library.
- The user's correction: the library must be usable from Python; the language the
  core is written in does not matter.

**[DECIDED]** v1 core is **pure Python with a narrow, typed boundary around hot
loops**, plus an optional native accelerator later.

Layout (proposed, confirmable):

```
Python package:      archx/                 (import archx)
CLI entry point:     archx / python -m archx
Native accelerator:  (deferred) archx/_native built with maturin; pure-Python
                     fallback must always work and is the reference implementation.
Format spec:         Docs/Format_Spec.md   (public from day one, versioned)
```

Why this is not a downgrade: the expensive parts are (a) copying tensors off an
accelerator and (b) byte transforms plus zstd. (b) is already C inside zstd/numpy;
(a) is memcpy either way. A Rust core would speed up the byte-shuffle and delta
kernels, which is a measurable, optional win - not a correctness requirement. It is
therefore designed as an *accelerator behind the same interface* (see section 6.6),
which also means v1 can be measured before that bet is made.

**[ASK]** Whether to keep a Rust/maturin accelerator on the v1 table at all, and
whether the distribution is named `archx` (the PyPI name `archx` is already taken by
an unrelated project) or ships as `archx-mdl` while still importing as `archx`.

---

## 4. Recording model: what a "step record" is

The recorder observes a *step* (one optimizer update) and stores:

| Piece | Source | Available in |
|---|---|---|
| step index, wall-clock time | hook | all adapters |
| per-parameter delta (new weights - previous weights) | copy taken **before** the weights change | all adapters |
| per-parameter gradient norm | `param.grad` right after backward, before zero_grad | most adapters |
| global grad norm, loss, learning rate | hook payload or `recorder.log()` | Tier 1+ |
| data segment fingerprint | `recorder.log(data_id=...)` or adapter plumbing | Tier 2 |
| active/frozen mask, region tags, component id | `requires_grad`, config | all adapters |
| eval metrics, activation/attention statistics | adapter-specific | Tier 3 |

**[DECIDED] The delta must be computed from a copy taken at the previous step, not
from the weights at the moment of the hook.** This is the single most important
implementation detail in the whole engine, and it follows from how PyTorch behaves:
by the time an optimizer *post-step* hook runs, `p.data` already holds the new value
and the old value is gone. So the recorder keeps a shadow buffer per tracked tensor:

```
step N:   hook fires  ->  enqueue(copy of p.data)      [cheap if p is on CPU]
step N:   worker      ->  delta = copy_N - copy_{N-1}  [off the training thread]
          hook ends   ->  training continues immediately
```

Copies are made at hook time (that is unavoidable, and it is the cost we measure);
everything else happens on the worker. `torch.utils.swap_tensors` is deliberately
**not** used: it would steal the tensor from the user's model, which breaks the
numerics rule.

---

## 5. Adapters (one per setup in the brief)

### 5.1 Setup 1: raw PyTorch loop

```python
import archx
recorder = archx.attach(model, optimizer, path="run.archx.mdl")
for batch in dataloader:          # unchanged loop
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

- **User does:** one line at setup; optionally one `recorder.log(loss=..., data_id=...)`
  line inside the loop.
- **Engine does:** registers `optimizer.register_step_post_hook(...)` (public API,
  present since PyTorch 2.0 - verified in `torch/optim/optimizer.py` at tag v2.0.0,
  and absent at v1.13.1, which only had the internal `profile_hook_step`) and
  `module.register_full_backward_hook` **only if** gradient norms are requested.
- **Measured constraint:** the hook path only exists inside `Optimizer.profile_hook_step`,
  so the *first* `step()` call pays a one-time class patching cost; the hook itself
  then runs inside `torch.autograd.profiler.record_function`.
- **Tier:** 0 without `log()`, 1-2 with it.

### 5.2 Setup 2: Hugging Face `Trainer` (and TRL, Axolotl)

```python
from archx.adapters.hf import ArchxCallback
trainer = Trainer(..., callbacks=[ArchxCallback("run.archx.mdl")])
trainer.train()
```

- **Verified in current `transformers` (`trainer_callback.py`, `trainer.py`):** the
  callback receives `args, state, control, **kwargs`; the `Trainer` itself is passed
  through `**kwargs` when available, which is how we reach the model, optimizer and
  accelerator. Hook names available: `on_train_begin/end`, `on_epoch_begin/end`,
  `on_step_begin/end`, `on_pre_optimizer_step`, `on_optimizer_step`,
  `on_substep_end`, `on_log`, `on_evaluate`, `on_save`, `on_predict`.
- **`on_pre_optimizer_step` and `on_optimizer_step` were added in transformers 4.46**
  (absent in 4.44). Below 4.46 the adapter falls back to `on_step_end`, which fires
  immediately after `optimizer.step()` and *before* `zero_grad()` - verified in the
  inner training loop, so gradients are still readable there.
- **Gradient availability (verified in the inner loop):** the order is
  `optimizer.step()` -> `model.zero_grad()` -> `global_step += 1` -> `on_step_end`.
  So by `on_step_end` the gradients are already zeroed and the *previous* weights are
  already gone. Gradient norms therefore come from `on_pre_optimizer_step` /
  `on_optimizer_step` (transformers >= 4.46) and, as the version-independent default,
  from an optimizer `register_step_post_hook` attached to `trainer.optimizer` at
  `on_train_begin` - that hook fires on every version and is what sections 4 and 6.6
  describe.
- **Accelerate / multi-GPU (honest answer to the brief's question):** the callback is
  constructed once and shared; `Trainer` is wrapped by `Accelerator`, which owns
  `model` (possibly `DistributedDataParallel`), `optimizer` (wrapped: `AcceleratedOptimizer`),
  and may shard parameters via `FullyShardedDataParallel` on each rank. Consequence:
  parameters must be read from `accelerator.unwrap_model(model)` (DDP wrapper adds a
  `module.` prefix that would corrupt names) while the *optimizer* must be the wrapped
  instance, because that is the one whose `step()` is called. With DDP every rank
  holds identical weights, so **recording must be enabled on exactly one rank
  (`is_main_process`)**, otherwise N ranks append N interleaved streams to one file.
  With sharded state (FSDP/ZeRO) no single rank holds full parameters -> detection
  and degradation, as in section 7.
- **TRL / Axolotl:** `SFTTrainer(_BaseTrainer)` and `DPOTrainer(_BaseTrainer)` are
  verified subclasses of the Trainer stack, so a `TrainerCallback` is accepted
  unchanged; Axolotl's `AxolotlTrainer` also subclasses the Trainer hierarchy. Plan:
  no separate adapter; add an integration test per stack instead of a code path. TRL's
  DPO-style trainers perform *two* backward passes per step (policy + reference), so
  the adapter must key off `optimizer.step()` rather than backward counts.
- **Tier:** 1-2 with no user change, 3 when `evaluation_strategy` is enabled.

### 5.3 Setup 3: PyTorch Lightning

```python
from archx.adapters.lightning import ArchxCallback
trainer = pl.Trainer(..., callbacks=[ArchxCallback("run.archx.mdl")])
```

- **Verified hook surface (`lightning.pytorch.callbacks.Callback`):** `on_fit_start/end`,
  `on_train_batch_start/end`, `on_before_backward`, `on_after_backward`,
  `on_before_optimizer_step`, `on_before_zero_grad`, epoch and validation hooks,
  `on_save_checkpoint`, `on_exception`.
- **Gradient capture:** `on_after_backward` (grads populated, not yet clipped) or
  `on_before_optimizer_step` (after clipping - this is what the optimizer will
  actually see, so it is the default). Manual optimization (`self.manual_backward()` +
  `self.optimizer.step()`) still goes through the same strategy hooks, but the
  adapter must not assume one `optimizer.step()` per batch, so it counts steps from
  `trainer.fit_loop.epoch_loop` state rather than batch indices.
- **Automatic optimization detail:** Lightning calls the optimizer with a closure
  (`Closure`/`_optimizer_step`), so `optimizer.step(closure)` re-enters the closure;
  the adapter's optimizer post-hook must be cheap and re-entrant, and must not
  capture the closure's intermediate state.
- **Tier:** 1-2 with no user change, 3 with validation.

### 5.4 Setup 4: DeepSpeed (direct) and Setup 5: FSDP

**[DECIDED] Not supported in v1; detected and reported.** Detection is cheap and
specific: `deepspeed.runtime.engine.DeepSpeedEngine` (or `model.engine`), and
`torch.distributed.fsdp.FullyShardedDataParallel` / `FSDPModule` in the module tree,
plus ZeRO stage from the DeepSpeed config. On detection the recorder:

1. writes a `WARN_UNSUPPORTED_SHARDING` event naming the wrapper, world size and
   ZeRO stage / FSDP sharding strategy;
2. records whatever is genuinely complete on rank 0 (usually: step count, loss, lr,
   timings, and *full-parameter* snapshots only at points where parameters are
   gathered, e.g. `full_optim_state_dict` availability);
3. sets capability tier to 0 and marks `parameters_complete=false` in the file header.

It never pretends to have deltas it does not have. Design room for v2: the delta
encoder already works on *shards*, so per-rank streams with a merge step are additive,
not a rewrite.

### 5.5 Setup 6: Keras / TensorFlow

Out of scope for v1. The container is framework-agnostic (named tensors, dtypes,
bytes), so a future adapter is a new producer, not a format change.

### 5.6 Setup 7: custom trainers

`archx.attach(model, optimizer)` is the universal fallback and works for any
`torch.nn.Module`, including architectures invented after this document was written.

### 5.7 Setup 8: zero-code launcher, honest verdict

`archx run -- python train.py`. Options evaluated on evidence:

| Approach | Works for | Fragility | Verdict |
|---|---|---|---|
| `PYTHONPATH` + `sitecustomize.py` injection | any process that imports torch | Depends on site-packages init order; breaks if the script freezes `sys.path`, or on embedded interpreters; silent no-op is a real risk | **Recommended base mechanism** |
| `PYTHONSTARTUP` | - | Only interactive shells | Not usable |
| Import-hook injection of a wrapper module | plain torch | Order-dependent, fights with other import hooks, hardest to debug | Avoid |
| Monkey-patching `Optimizer.step` at import time | plain torch + HF Trainer + Lightning | Survives most versions because `step` is a public method; can be observed/undone by user code; breaks for custom optimizers that bypass `step` | **Used for optimizer capture** |
| Patching `Trainer.train` / `pl.Trainer.fit` | HF / Lightning | Version-sensitive signatures, but only used to *find* the objects, then normal callbacks are registered | Acceptable |

**[DECIDED]** Feasible but *conditional*: supported for plain PyTorch loops and the HF
Trainer, best-effort for Lightning, and the launcher prints exactly what it attached to
and marks `adapter=launcher` in the file. If nothing is found, it exits non-zero with
instructions rather than writing a plausible-looking empty recording. Whether it
becomes a v1 feature is **[ASK]**; the investigation is a v1 roadmap step regardless.

---

## 6. Storage design

### 6.1 What is stored

- **Anchors:** full parameter snapshots - at start, at phase boundaries, at major
  events (freeze/unfreeze, data domain change), at end. Final state always full.
- **Deltas:** per tensor per step (subject to the phase-adaptive policy), encoded.
- **Events:** the always-on timeline plus discrete events (grad-norm anomaly, loss
  drop, eval result, annotation, component transition).
- **Data fingerprints:** hash + token/byte count + characteristics per segment. Never
  the data itself.
- **Registry + index + footer.**

### 6.2 Phase-adaptive detail (defaults [DECIDED], tunable, all tested)

| Phase | Boundary (default) | Policy |
|---|---|---|
| warmup | first 1% of steps (min 20) | every step, anchor at start |
| middle | until 99% | event-triggered only: grad-norm spike (z-score > 4), loss drop, region weight move above threshold, new data domain, component change; plus a heartbeat every K steps (K=100 default) |
| cool-down | last 1% | every step |
| end | last step | full anchor |

Thresholds are **per region** (per parameter, or per user region tag), never global:
CNNs and RNNs have layer rates that differ by orders of magnitude, so a global
threshold silently drops the layers that matter. Anchoring policy for chained deltas
is a measured question (section 10).

### 6.3 Encodings, chosen per tensor per step by measurement

| Encoding | When it wins | Notes |
|---|---|---|
| sparse (index + value pairs) | few changed elements: embeddings with small batches, MoE blocks, frozen-ish layers | index cost dominates when density > ~2-5%; benchmark decides |
| low-rank (U·V factors) | coordinated, structured movement: attention projections, most full-batch fine-tuning | rank chosen to hit an error target |
| quantized dense delta (int8 / fp16 / bf16 + scale) | dense small deltas | error measured, not assumed |
| "unchanged" marker | ties, frozen parameters, zero delta | must never double-count tied tensors |

Encoder choice is recorded in the file per tensor block, so the reader never guesses.

### 6.4 Compression pipeline

Order: `delta -> (optional quantize) -> byte-plane shuffle -> xor/delta against
previous plane -> zstd (level tuned per section)`.

- Byte-plane shuffling (split fp32/fp16 into exponent/mantissa planes) is what makes
  float deltas compressible; plain text compressors on raw floats do almost nothing.
- **zstd** via `zstandard` (wheels everywhere; `compression.zstd` in CPython 3.14).
  `lz4` is not a default: worse ratio at similar speed for our payload.
- Small sections may use `lz4`/`zlib` when decompression latency matters more than
  bytes; the compressor is a per-section field, not a global assumption.
- Every number in this section is a hypothesis to benchmark in roadmap step 2, not a
  claim.

### 6.5 Container format (draft v1, spec text lives in `Docs/Format_Spec.md`)

```
[ header ][ block ][ block ] ... [ footer ]
```

Header (fixed size, at offset 0):

| field | type | value |
|---|---|---|
| magic | 8 bytes | `ARCHXMDL` |
| format major / minor | u16 / u16 | 1 / 0 |
| flags | u32 | bit 0: closed cleanly, bit 1: parameters complete, bit 2: recovered |
| header_len | u32 | |
| header_crc | u32 | crc32 over header |

Each **block** is self-describing (this is what makes crash recovery possible):

| field | type |
|---|---|
| block_type | u16 (META, TENSORS, ANCHOR, DELTA, EVENT, DATA, ANNOT, INFO) |
| codec | u8 (none, zstd, zlib, lz4) |
| flags | u8 |
| reserved | u16 |
| payload_len | u64 |
| payload_crc | u32 |
| reserved | u32 |
| payload | bytes |

Blocks are appended sequentially and fsynced in batches; the footer is written last:

| field | type |
|---|---|
| block_index_len, block_index_offset | u64, u64 |
| step_count, anchor_count | u64, u32 |
| elapsed_seconds | f64 |
| footer_crc, flags | u32, u32 |
| end magic | 8 bytes `ARCHXEND` |

**Crash recovery:** a file without the footer is still valid data. `archx recover`
scans from the header, reads each block header, validates the CRC, stops at the first
torn block (truncating it), and rebuilds the index. Because deltas are self-contained
per tensor per step, a lost tail costs only the tail.

**Safety:** no executable content is possible by design - no pickle, no code
dereferencing, only fixed-width numbers, bytes and UTF-8 JSON metadata. The reader
validates every length and CRC before allocating.

### 6.6 Async writer

- One writer **thread** (not a process) with a bounded queue; hook-side work is:
  copy tensors -> enqueue -> return. Backpressure policy: if the queue is full, drop
  detail (degrade to heartbeat) rather than block training. That is the rule: *the
  training loop never waits on us*.
- Copies use `tensor.detach().to("cpu", copy=True)`, non-blocking plus a pinned
  staging buffer when CUDA is present; CPU tensors are a plain `clone()`.
- Optional future: encode in a second process to escape the GIL. Python 3.14's
  free-threaded build is a real option (torch 2.15 ships `cp314t` wheels) - **[ASK]**.

### 6.7 Capability tiers (recorded in every file)

| Tier | Contents |
|---|---|
| 0 | step counter, weight deltas, timing |
| 1 | + loss, learning rate, gradient norms |
| 2 | + data segment fingerprints and user tags |
| 3 | + activation statistics, per-head attention statistics, eval/benchmark results |

Tier is a recorded fact, not a request: the file says which tier was actually reached
and which fields were unavailable.

---

## 7. Architecture coverage (section 3 of the brief)

No model-specific code paths in the core. What the engine may know:

- **Region tags** (optional, from config/env): `attention`, `vision_encoder`, `head`...
  stored per region; falls back to per-layer when absent.
- **Frozen detection:** `requires_grad == False` -> anchor at start and end only, plus
  a transition event if it changes during the run.
- **Tied weights:** detected by `data_ptr()` identity across named parameters; stored
  once, aliased in the registry, never double-counted in deltas.
- **Components:** top-level `named_children()` (or user config) become named
  components with their own statistics, phase detection and anchors, so a multimodal
  run with one frozen tower is reconstructable component by component.
- **MoE:** per-expert tensors are just tensors; sparse encoding carries the cost.
  FSDP+MoE sharding is explicitly deferred (see 5.4).

Per-architecture notes (what we must verify in Research before code):

| Arch | What matters to the recorder | Risk |
|---|---|---|
| Transformers | huge slowly-changing embeddings, per-head stats for Tier 3 | sparse index cost on embeddings |
| CNNs | wildly different per-layer rates | needs per-region thresholds (6.2) |
| RNN/LSTM | vanishing/exploding gradients | grad-norm anomaly events are primary signal |
| SSMs (Mamba/Hyena) | structured state matrices, possible custom parameter layout | unusual shapes, `dt_proj`-style parameters; verify in research |
| MoE | sparse activation across blocks | sparse encoder efficiency |
| Multimodal | multiple components, staged freezing | detection of staged training |
| Custom | anything `nn.Module` | none by design (Tier 0/1 guaranteed) |

---

## 8. Cost, safety, and how "never hurts training" is proven

- Budget: **< 5%** of step time, measured per adapter and per architecture on all
  three runners.
- Measurement harness exists before the adapter ships: same seed, same data order,
  recorder on vs off, compare (a) median step time, (b) final weights bit-exact,
  (c) loss curve bit-exact.
- Guard rails: every public recorder method is exception-safe; a `strict=False`
  default means recorder errors never propagate into training.
- No writes to user tensors: enforced by code review plus a test that hashes every
  parameter before and after recorder calls.

---

## 9. Python package layout (proposal)

```
archx/
  __init__.py          public surface: attach, log, annotate, open
  recorder.py          orchestration: policy, queueing, lifecycle
  policy.py            phase detection + what to store per step
  encoders/
    sparse.py          index/value encoding
    lowrank.py         truncated SVD deltas
    quant.py           int8/fp16/bf16 delta quantization
    plane.py           byte-plane shuffle + xor
  container/
    writer.py          append-only block writer, fsync policy
    reader.py          block reader, index, integrity checks
    recovery.py        rebuild index from a crashed file
    schema.py          block type definitions, version constants
  formats/
    tensor.py          dtype/shape registry, tied-weight handling
  adapters/
    torch.py           plain loop (optimizer + module hooks)
    hf.py              transformers Trainer callback
    lightning.py       Lightning callback
    launcher.py        zero-code sitecustomize/monkey-patch entrance
  cli/
    main.py, info.py, reconstruct.py, diff.py, recover.py
  _native/             optional future native accelerator (pure-Python fallback)
```

One file per responsibility, each with a purpose line, per the rules.

---

## 10. Claims to measure (not assume) - becomes roadmap step content

1. Reconstruction error vs compression ratio, per delta encoder.
2. Low-rank vs sparse crossover, per architecture family.
3. Accumulated error when chaining deltas; correct anchor spacing.
4. Real storage vs naive checkpointing on the benchmark models.
5. Training slowdown per adapter, per architecture.
6. Phase-adaptive vs fixed-interval at equal storage.
7. What the optimizer hook alone loses vs full trainer integration.
8. Zero-code launcher reliability across PyTorch versions.
9. Same seed, recorder on/off -> identical weights (hard requirement, must pass).
10. A crashed file is readable after `archx recover`.

Each is a record in `Tests/Records/<version>/` and an entry in `Docs/Research.json`.

---

## 11. Known unknowns (honest list)

- Real deltas of real training runs are unmeasured; every compression number here is
  a hypothesis until roadmap step 2 runs.
- Whether a Python writer thread can stay under 5% overhead with GIL contention is
  unmeasured; the free-threaded build is the hedge.
- HF + Accelerate + DDP correctness is understood on paper (unwrap model, use wrapped
  optimizer, single writing rank) but not yet tested.
- The launcher's reliability across PyTorch versions is asserted from mechanism
  analysis, not from a test matrix - that is roadmap step 7.
- No GPU exists in this sandbox; all accelerator claims must be validated on CI
  runners, and runner hardware is CPU-only on GitHub-hosted runners unless the
  benchmark is CPU-only by design (which the v1 benchmarks will be).

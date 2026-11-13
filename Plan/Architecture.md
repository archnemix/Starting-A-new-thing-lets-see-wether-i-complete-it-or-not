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

### Where the idea came from

The product is the third idea in a longer conversation, and the shape of that
conversation still explains the design. In short: it started from model file formats
(safetensors vs GGUF), moved through "computers inside models", and landed on a simple
observation - *a checkpoint keeps the destination and throws away the journey*. That
journey is the product. The full extraction, including what the original chat asked for
that this brief left out, is in `Plan/Origin_Chat.md`.

Two properties inherited from that origin and kept as requirements:

- **The final anchor doubles as a model snapshot.** A closed file with history removed is
  still a usable weights file. That is why the last anchor is always full, never
  delta-only.
- **Parts are independently usable.** Every block is self-describing and independently
  compressed, so history, metadata and final weights can be read separately.

### The priority rule (decided by the user, 2026-11-11)

**The origin chat outranks the generated brief.** Where they disagree, the chat wins and
the deviation is recorded.

The real starting point was **2026-11-11 19:51** (user turn: "What if we build rather than a
format a way to store model during training..."). Everything before it - model formats,
computers inside models, 2D attention, the Pygame idea - is the road that led there. The
generated brief was written 20 minutes later and inherited the earlier "fingerprints only"
assumption from that road. That is the origin of the disagreement, and why the chat wins.

The user's newest statement of intent is:

> "the real chatting about it started later ... this is the main thing"

and the main thing is a system that records *when this data was introduced, how the state
changed, at which stage* - with the data itself, because "that would show the conversion
that this data had this effect on the weights at this stage". So data content is **in**,
on by default, and the recording is optimised around keeping the data-to-weight join cheap
(6.0 and 6.8), not around avoiding it.

### Non-goals for v1 (explicitly out)

The atlas, edit vectors and query tooling are out of *v1 scope*, but they are the reason
v1 exists: every v1 file must contain enough information for an atlas to be computed from
it later without retraining. That constraint is now a requirement (6.10), not a nice-to-have.

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
| Report evidence, never claim causation. | Segment summaries record correlation under the observed schedule and META says `attribution: "observational"`. |
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

### Naming [DECIDED by the user, 2026-11-11]

`archx` is the brand identity and stays the name people *use*. Because the bare name is
taken on PyPI, the **extension is the disambiguator**:

| Surface | Name | Note |
|---|---|---|
| file extension | `.archx.mdl` | unchanged |
| import | `import archx` | unchanged, always |
| CLI | `archx` | unchanged |
| PyPI distribution | `archx.mdl` | normalises to `archx-mdl` (PEP 503), so both `pip install archx.mdl` and `pip install archx-mdl` resolve to the same project |
| npm package | `archx` | **free** (measured 404); used as a seam that tells people where the real tool is |

Measured 2026-11-11: PyPI `archx` = 200 (someone else's "framework to explore the design
space of computer systems", v0.1.1), PyPI `archx-mdl` / `archx.mdl` / `archnmix` variants =
404. npm `archx` = 404, so it is available. npm *usernames* cannot be checked without
auth (401), which only matters if the user's npm account name has to match.

**The npm seam (why an npm package at all):** someone who hears about the tool types
`npm i archx`. That package installs a tiny CLI that prints the PyPI install command and
the docs link, and exits with a clear message. It is a signpost, never a reimplementation:
no Python code, no native bindings, nothing that can drift. It exists so the brand is not
squatted and so the wrong install is impossible to reach silently. Roadmap step 0.5.

**[ASK]** Whether to keep a Rust/maturin accelerator on the v1 table at all. The core is
Python either way; the accelerator is an optional later step (Open_Questions Q4).

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
| activation statistics | built in, `activations=True` | Tier 3 |
| eval metrics, attention statistics | adapter-specific | Tier 3, **planned** (not recorded) |

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
  already gone. Gradient norms therefore come from `on_pre_optimizer_step`, where the
  gradients are still alive, and the update is recorded at `on_optimizer_step`.

  **Corrected in Phase 6, against a real Trainer:** the "version-independent default" of
  attaching an optimizer `register_step_post_hook` at `on_train_begin` does not work. The
  optimizer a callback is handed is accelerate's `AcceleratedOptimizer`, which is not a
  `torch.optim.Optimizer`; the inherited method reaches for `_optimizer_step_post_hooks`
  and raises `AttributeError`, inside `on_train_begin`, which kills the training run. It is
  `trainer.optimizer.optimizer` that is the real optimizer. Since the two callback events
  above bracket the update on every supported version, no optimizer hook is registered at
  all. Also measured: `_clip_grad_norm` runs *before* `on_pre_optimizer_step`, so the norm
  a callback can see is the clipped one whenever `max_grad_norm > 0` (the default), which
  the file records as `grad_norm_clipped`.
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
  **The loss is not obtainable from a callback.** `on_log` fires after the step record and
  reports a window mean (measured: losses on steps 1-3 and none on step 0, and a run with
  `logging_steps=50` silently claimed tier 1 while 49 steps in 50 had no loss). The
  `compute_loss_func` seam pops the labels out of the inputs, so it requires rebuilding the
  loss - a guess that measured wrong on a regression model. `loss_hook(trainer)` wraps the
  instance's `compute_loss`, calls it untouched and reads the real value; `callback.log()`
  is the manual route. Without either, the file reports tier 0 and names both.

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

### 6.0 The recording triple: state, cause, stage

The brief describes "a recording engine". The chat describes something more specific, and
what must be recorded is not the weights over time. It is the **pairing of three things at
every moment**:

| Element | What it is | Where it lives |
|---|---|---|
| **State** | what the weights were and how they changed | anchors + deltas |
| **Cause** | which data produced that change, with the data itself retrievable | data segments + per-step join |
| **Stage** | where in training it happened: step, phase, epoch, freeze/unfreeze, eval | timeline + events |

Any two of the three is a weaker, already-existing product:

- state + stage, no cause -> checkpointing with metadata (what the world has today)
- cause + state, no stage -> an offline dataset ablation
- stage + cause, no state -> a training log

The join of all three answers the question the product exists to answer: *which data moved
which weights, when, and by how much*. Every design decision below is judged by one test:
**does it keep that join exact?** Byte savings that break the join are rejected, and that
is why compression work targets float deltas (6.5, 6.6) while data content is stored
losslessly per unique shard (6.8).

### 6.1 What is stored

- **Anchors:** full parameter snapshots - at start, at phase boundaries, at major
  events (freeze/unfreeze, data domain change), at end. Final state always full.
- **Deltas:** per tensor per step (subject to the phase-adaptive policy), encoded.
- **Events:** the always-on timeline plus discrete events (grad-norm anomaly, loss
  drop, eval result, annotation, component transition).
- **Data segments (the cause):** every distinct unit of training data the recorder sees,
  stored **once**, content-addressed, compressed per shard, each with a fingerprint,
  characteristics, source reference and first/last-seen steps. Cost scales with the amount
  of *unique* data, not with steps or epochs - re-seeing data is a pointer, never a copy.
- **The join:** every delta block and every event carries the segment id(s) active during
  those steps and the component mask. "This data moved these weights at this stage" is a
  direct read of the file, not something reconstructed later.
- **Registry + index + footer.**

### 6.2 Phase-adaptive detail (defaults [DECIDED], tunable, all tested)

| Phase | Boundary (default) | Policy |
|---|---|---|
| warmup | first 1% of steps (min 20) | every step, anchor at start |
| middle | until 99% | event-triggered only: grad-norm spike (z-score > 4), loss drop, region weight move above threshold, new data domain, component change; plus a heartbeat every K steps (K=100 default) |
| cool-down | last 1% | every step |
| end | last step | full anchor |

A stored delta is the exact accumulated movement since the last stored state (the sum of
every skipped step), not a single step's change. Copying every step and accumulating it
would cost more and store nothing more, so copies happen at store points only.

Thresholds are **per region** (per parameter, or per user region tag), never global:
CNNs and RNNs have layer rates that differ by orders of magnitude, so a global
threshold silently drops the layers that matter. Anchoring policy for chained deltas
is a measured question (section 10).

### 6.3 Encodings, chosen per tensor per step by measurement

Measured on 256x256 float32 deltas, seed 0 (`Bench/claim01_error_vs_ratio.py`):

| Delta shape | sparse | dense_quant (int8) | lowrank | dense_raw (exact) |
|---|---|---|---|---|
| 1% of entries moved | **77x, exact** | 125x, 8.9e-3 | refused | 34x, exact |
| dense, all moved | refused | **4.7x, 1.1e-2** | refused | 1.2x, exact |
| rank-4 structure | refused | 5.0x, 1.5e-2 | **34x, 8.4e-8** | 1.1x, exact |

| Encoding | When it wins | Notes |
|---|---|---|
| sparse (index + value pairs) | few changed elements: embeddings with small batches, MoE blocks, frozen-ish layers | loses to int8 at **any density when error is allowed**, and is the *only* exact choice below full density. Threshold must come from a noise floor; a zero threshold keeps essentially everything (measured at 0.8% of the max delta) |
| low-rank (U·V factors) | coordinated, structured movement: attention projections, most full-batch fine-tuning | rank chosen to hit an error target |
| quantized dense delta (int8 / fp16 / bf16 + scale) | dense small deltas | error measured, not assumed |
| "unchanged" marker | ties, frozen parameters, zero delta | must never double-count tied tensors |

Encoder choice is recorded in the file per tensor block, so the reader never guesses.

### 6.4 Compression pipeline

Order: `delta -> (optional quantize) -> byte-plane shuffle -> xor/delta against
previous plane -> zstd (level tuned per section)`.

- **Measured, and it corrected the design:** byte-plane shuffling does *not* make float
  deltas compressible. On realistic deltas it costs bytes (sparse 1%: 4,533 -> 7,774, i.e.
  0.58x) because it scatters the long zero runs zstd already exploits, and it gains only
  ~1.07x on dense deltas. So the lossless path compresses raw bytes and the shuffle is
  opt-in (`Bench/claim_plane_shuffle.py`). Plain zstd on raw floats is weak, as expected:
  1.1-1.2x. The bytes come from *sparsification and low rank*, not from shuffling.
- **zstd** via `zstandard` (wheels everywhere; `compression.zstd` in CPython 3.14).
  `lz4` is not a default: worse ratio at similar speed for our payload.
- Small sections may use `lz4`/`zlib` when decompression latency matters more than
  bytes; the compressor is a per-section field, not a global assumption.
- **A lossless fallback is mandatory** (`dense_raw`, encoding 4). Measurement showed the gap:
  at full density with unstructured movement, sparse refuses (indices cost more than values)
  and low rank refuses (noise is not low rank), so without it a recording would have had no
  exact option. Every tensor must have an exact encoding available, always.
- **Deltas before data.** Compressing the data is far easier than compressing float deltas
  and saves less of the total (deltas repeat at every store point; data is stored once).
  So the split is: data gets a per-shard codec chosen automatically, deltas get the
  measured treatment above. That is where the bytes are.
- Distance-based encoding for correlated tensors (`w_2 - 2w_1 + w_0`, the chat's "distance
  based weights encoding") is a per-tensor encoding to test in Phase 2, not a default.
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

**Reserved for later, defined now so v1 files stay composable:**

- `META.base_model` (fingerprint + registry hash of a parent file) and `META.ancestor`
  (content hash of the file this run continued from). With a valid base reference the
  first anchor becomes optional and a file can be a pure delta against a parent - the
  "fine-tune B = model A + changes" case the origin chat called the most valuable
  feature. Not implemented in v1; the fields ship so v1 files compose later.
- `DATA_BLOB` / `DATA_SHARD` / `DATA_SEGMENT` / `DATA_POINTER` block types: the data
  itself, on by default (6.8). The brief said fingerprints only; the chat outranks it.

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

<!-- 6.7 is intentionally unused: the numbering was reflowed when 6.8-6.10 were inserted;
     renumbering would break cross-references from Docs/ and Plan/Origin_Chat.md. -->
### 6.8 Data capture: on by default, four levels, opt-out only

The data a run saw is part of the record, not an accessory. Default level: **`segments`**.

| Level | Stores | Typical size (illustrative, to be measured) | Breaks the join? |
|---|---|---|---|
| `none` | only a hash + count per batch | tiny | yes - the data cannot be recovered |
| `pointers` | where to get each shard (path/URL/hub id + content hash) + characteristics | KB | only if the source later moves or changes |
| `segments` **(default)** | one compressed copy per unique shard + characteristics + join | data size x (1/ratio) x dedup factor | no |
| `full` | every batch exactly as fed, no dedup, no normalisation | 100%+ | no, and exact sample order |

Rules for `segments`:

- **Deduplicate by content hash.** An epoch over the same data adds 2,000 pointers, not
  2,000 copies.
- **Per-shard codec chosen automatically**, recorded in the shard header: text-like data
  -> zstd dictionary trained on the run's own data; token-id arrays -> varint + delta +
  zstd; images -> lossless (webp/png) or zstd. Lossy codecs are never applied silently;
  if used, the shard records the codec and the measured reconstruction error.
- **The fingerprint is always present** at every level, so a `none`/`pointers` run is still
  analysable, it simply cannot show the data.
- **Opting out is allowed but must be explicit**: `data="none"` or an env var. The file
  records `data_level` and, when the level is not `segments`/`full`, a
  `JOIN_INCOMPLETE` info block stating what the reader will not be able to answer.
- **No size cap by default** (user decision). The recorder prints a projected data size
  when it starts, warns once per doubling, and reports the real total at close. A cap
  (`data_max_bytes`) exists only if the user sets one; when it is hit the level degrades to
  `pointers` and the file records it, because silently truncating data would break the join
  while still looking complete.
- **Legal/PII guard**: shards can be marked `sensitive=true`, which stores the hash and
  characteristics only for that shard. This is the one place where redaction beats
  fidelity, and it is per shard, not per run.

### 6.9 Conversion evidence: what "this data had this effect at this stage" costs

The user's requirement is stronger than "store the data next to the deltas". Evidence means
the join can be read back and defended. What is stored, per data segment:

| Field | Why |
|---|---|
| `segment_id`, content hash, source ref, time span | identity and provenance |
| characteristics: language, domain, complexity, token/byte count, dedup ratio | what kind of data it is, for cross-run comparison later |
| first/last step seen, times seen, batches per epoch | the "stage" half of the join |
| **per-region weight movement during the segment's steps** | the effect, already summarised so it does not need recomputation |
| **per-region gradient norm during those steps** | the pressure the data applied |
| **local loss trend over those steps** | whether the movement helped |
| component mask (which sub-model was active) | multimodal/staged runs |
| reconstruction error of the deltas in those steps | how much of the effect is missing due to compression |

The last six rows are computed by the recorder from data it already holds, so the cost is
CPU on the writer thread, not extra storage. They are what makes a file answer
"python data -> moved attention in layers 4-7 -> loss fell 0.023 -> reconstruction error
1e-5" in one query instead of an offline re-analysis.

**What is *not* stored per step.** The per-region streams above are thresholded: a region's
entry is written only when it crosses its own threshold, otherwise the steps fold into the
next written entry (which covers a range, exactly like a flushed delta). If a region never
crosses, its summary is still written at the end of the segment. This is what keeps the
join affordable at millions of steps: no per-step rows, no unbounded metadata growth.

For extreme run lengths, the recorder also writes a **coarse dense tail**: a rolling
summary of N steps at a time, so the timeline stays complete at low resolution even where
nothing triggered. `Docs/Rules.md` applies - this is measured in Phase 2, not assumed.

**Honesty requirement:** that is *evidence of correlation under a controlled schedule*, not
proof of causation. The format states this in META (`attribution: "observational"`) so no
downstream reader overclaims. Designed-schedule experiments (same seed, one variable
changed) are the way to strengthen it, and the format supports them by tagging runs that
share a schedule id.

### 6.10 Atlas-ready from day one

The chat's end goal is a queryable map: which data shapes which weights, which capabilities
live where, and directions in weight space to edit them. None of that is v1 *scope*, but v1
must not make it require retraining. Three requirements follow:

1. **Region and component statistics are written continuously** (6.9), not computed later.
2. **Segment characteristics are comparable across runs** (same schema, same field names),
   so signatures can be matched between models in v2.
3. **Per-layer data links are recorded in run one** (chat, 20:11): layer ids in the
   file are stable and named exactly as PyTorch names them (`model.layers.4.self_attn.q_proj`),
   so an atlas built from one run can be reused on another model when names coincide. A
   tool that groups by layer must be able to do so without a mapping table.
4. **The join is machine-readable without the recorder**: a reader can extract
   `(segment, step range, region, movement)` tuples straight from the file, which is exactly
   the input an atlas builder needs.

### 6.11 Capability tiers (recorded in every file)

| Tier | Contents | State |
|---|---|---|
| 0 | step counter, weight deltas, timing | recorded |
| 1 | + loss, learning rate, gradient norms | recorded |
| 2 | + data segment fingerprints and user tags | recorded |
| 3 | + activation statistics | recorded |
| 3 | + per-head attention statistics | **planned, not recorded** |
| 3 | + eval/benchmark results | **planned, not recorded** |

Tier is a recorded fact, not a request: the file says which tier was actually reached
and which fields were unavailable.

The last two rows are the reason this table gained a "State" column. Until v1.0.2 the capability
block in every file said tier 3 meant "activation statistics, eval results", and no adapter had
ever recorded an eval result: every tier-3 recording promised, in its own header, something the
build could not produce. A planned field belongs in the plan, not in the file, so the file now
says only what it holds. When there is a block for attention statistics or eval results, it is
added to `TIER_FIELDS` and to this row's state at the same time -
`test_every_tier_meaning_describes_exactly_the_fields_that_tier_promises` fails if the prose runs
ahead of the code again.

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
11. Data-shard dedup: an N-epoch run stores N pointers, not N copies - measured bytes.
12. Per-shard codec choice: byte ratio and reconstruction exactness per data type
    (text, token ids, images), including whether text-like data is lossless.
13. Recording overhead with data capture on vs off, per level (none/pointers/segments/full) -
    the data path is I/O, so this is measured separately from the delta path.
14. The join round-trip: given a segment id, the file returns its steps, regions moved,
    gradient norms and loss trend without re-reading weights.

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

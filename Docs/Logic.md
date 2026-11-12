# Docs: Logic

> Purpose: the flow of control. Which call leads to which file, in order, so a break
> can be traced from the user's line to the exact file. Updated whenever a call chain
> changes.

Implemented so far: the container layer, the CLI, the recorder, data capture and the raw
PyTorch adapter (Phases 0-3). The HF Trainer and Lightning chains are still the contract the
code will be built against. `[live]` marks a chain that exists in code today; `[planned]`
marks one that does not.

---

## 1. `archx.attach(model, optimizer, path=...)` - plain PyTorch  [live]

```
user script
  └─ archx.attach(model, optimizer, path)          archx/__init__.py   (lazy import)
       └─ attach()                                 archx/adapters/torch.py
            ├─ named_parameters(remove_duplicate=False)
            ├─ TensorRegistry.add(..., data_ptr, frozen, region, component)
            │                                       archx/formats/registry.py
            ├─ TorchRecorder(...)                   archx/recorder.py
            │    ├─ writer thread opens the file, writes META + TENSOR_REGISTRY
            │    └─ shadow copy of every primary tensor   (the step-0 baseline)
            └─ optimizer.register_step_post_hook(hook)   archx/adapters/torch.py
```

Per training step:

```
optimizer.step()
  └─ step_post_hook                                archx/adapters/torch.py
       ├─ global grad norm (one numpy dot per grad, ~19 us)
       └─ Recorder.step(step, grad_norm=...)       archx/recorder.py
            ├─ pick up anything from log()  (loss, data, lr) for this step only
            ├─ DataCapture.observe()               archx/data/capture.py   (~0.3 us)
            ├─ PhasePolicy.decide()                archx/policy.py
            ├─ SKIP  -> count, feed the policy, done (nothing is copied)
            ├─ STORE -> copy every primary tensor (shadow copy)
            │            movement per region       archx/recorder.py
            │            attribute effects to the segment   archx/data/capture.py
            └─ enqueue anchor | delta              (bounded queue; full = drop + count)

writer thread
  ├─ ANCHOR | DELTA block written first            archx/container/writer.py
  ├─ EVENT block: step, loss, grad_norm, lr, regions
  ├─ DataCapture.seal()                            archx/data/capture.py
  │    ├─ content id (blake2b-16)
  │    ├─ dedup against stored shards  -> pointer only
  │    └─ codec by content -> compress            archx/data/codecs.py
  ├─ DATA_SHARD (once per unique content)          then DATA_SEGMENT (per statement)
  └─ encode deltas                                 archx/encoders/*  (choose_encoder)

recorder.close()
  ├─ final clamp-free data flush + DATA_POINTER blocks (sensitive, pointers)
  ├─ INFO capabilities: the tier reached, every missing field with its reason
  ├─ INDEX + footer, CLOSED flag                   archx/container/writer.py
  └─ re-raise any writer fault                     archx/recorder.py
```

Reading it back - `archx reconstruct`, `archx events`, or the library:

```
Reader(path)                                       archx/container/reader.py
  └─ Timeline(reader)                              archx/container/values.py
       ├─ parse_registry -> TensorInfo[]
       ├─ parse_anchor / parse_delta -> arrays
       ├─ reconstruct(step): nearest anchor + deltas  -> tensors (ties re-aliased)
       ├─ events() / annotations() / segments()      -> the join, as records
       └─ steps                                       -> what the file claims to have
```

## 2. `ArchxCallback(...)` - Hugging Face Trainer

```
Trainer.train()
  ├─ on_train_begin        -> if not main process: do nothing at all (no recorder, no file)
  │                          else unwrap model (accelerate.unwrap_model), build registry,
  │                          open the recorder; annotate "resumed" if global_step > 0
  ├─ on_pre_optimizer_step -> gradient norm (CLIPPED when max_grad_norm > 0) and the learning
  │                          rate from optimizer.param_groups[0]["lr"], plus state.global_step
  ├─ on_optimizer_step     -> the weights have just changed: recorder.step(...). A skipped
  │                          (fp16 overflow) update is counted, not recorded
  ├─ on_evaluate           -> (not used yet; tier 3 activations are not reached)
  └─ on_train_end          -> close: flush the queue, write the capability block and footer
```

The loss is *not* taken from `on_log`: that fires after the step has been recorded and reports a
window mean. `loss_hook(trainer)` observes `compute_loss` instead, or the caller passes
`callback.log(loss=...)`. With neither, the file reports tier 0 and names the way out.

## 2b. `archx run -- python train.py` - zero-code launcher

```
sitecustomize (injected via PYTHONPATH)
  ├─ start-up  -> torch not importable yet: register a sys.meta_path finder for torch.optim
  ├─ torch.optim loads -> wrap every optimizer class that defines its own step
  │                       (AdamW does NOT: it resolves to Adam.step)
  ├─ first step -> build the registry from optimizer.param_groups (no names available:
  │                keys are param[group][position]), open the recorder
  ├─ every step -> recorder.step(...)
  └─ atexit     -> close: footer + capability block
launcher (parent)
  └─ after exit -> count the steps in the file; 0 means delete it and exit 2 with instructions
```

Reaches tier 0: steps and deltas only. No loss, gradients, data or parameter names.

## 3. `ArchxCallback(...)` - Lightning

```
on_fit_start            -> unwrap to pl_module, build the registry, open the recorder
on_train_batch_end      -> THE store point: weights are new here, and
                           trainer.global_step advanced by one only on a real update
on_before_optimizer_step-> cache the gradient norm and the lr (weights are still OLD here)
on_fit_end              -> close
```

Measured on lightning 2.6.6: `on_before_optimizer_step` fires *before* the step, so storing
there records one update behind. The loss needs no user code - `outputs["loss"]` arrives with
the batch end. Tier 1 with one line.

```
Trainer.fit()
  ├─ on_fit_start -> resolve pl_module + trainer.optimizers, register optimizer post-hook
  ├─ on_before_optimizer_step -> gradient norms (post-clipping, what the optimizer sees)
  ├─ on_train_batch_end       -> step count from loop state, not batch index
  ├─ on_validation_epoch_end  -> metrics (Tier 3)
  └─ on_fit_end               -> final anchor, footer, close
```

## 4. `archx run -- python train.py` - launcher (if it ships)

```
archx/cli/main.py
  ├─ put a sitecustomize shim on PYTHONPATH        archx/adapters/launcher.py
  ├─ exec the user's command in a child process
child process imports torch
  └─ shim wraps Optimizer.step and Trainer.train / pl.Trainer.fit
       └─ on first match, auto-attach (chain 1 / 2 / 3) and log what it found
nothing matched -> exit non-zero with instructions (never an empty file)
```

## 5. Reading a file `[live]`

```
archx info <file>                         archx/cli/main.py -> archx/cli/info.py
  └─ Reader(path)                         archx/container/reader.py
       ├─ read_header()  magic/version/CRC    archx/container/reader.py
       ├─ _scan(): per block
       │     ├─ pair_crc over bytes 0..23     archx/container/schema.py
       │     ├─ length cap (anti-DoS)         archx/container/schema.py
       │     ├─ decompress (zstd/zlib/none)   archx/container/compress.py
       │     └─ payload CRC + declared length
       └─ footer found -> closed file, index usable
            └─ Reader.index_block_payload()   archx/container/reader.py
```

## 5a. Writing a file `[live]`

```
Writer(path)                               archx/container/writer.py
  ├─ header_bytes()  -> 40-byte header     archx/container/schema.py
  ├─ write_block(type, payload, step=N)
  │     ├─ compress.compress()             archx/container/compress.py
  │     ├─ 28-byte prefix (step inside)    archx/container/schema.py
  │     └─ batched fsync (bytes or time)
  └─ close()
        ├─ write_index()     INDEX block
        ├─ footer_bytes()    footer last
        └─ CLOSED flag set in the header
```

## 5b. Recovering a crashed file `[live]`

```
archx recover <file> [--out copy]          archx/cli/recover.py
  └─ recover_in_place()                    archx/container/recovery.py
       ├─ read_header()                    archx/container/reader.py
       ├─ _good_prefix_end(): walk prefixes until one fails
       ├─ truncate at the last whole block
       ├─ _rebuild_index(): steps read from prefixes, no decompression
       ├─ append INDEX + footer
       └─ set RECOVERED | CLOSED flags
```

## 6. Failure paths `[live]` (must never propagate into training)

```
any exception inside Recorder.step()
  └─ caught in archx/recorder.py: stats.failed += 1, remember the first error,
     return SKIP -> training continues untouched

any exception while writing one block
  └─ caught per step in the writer thread: counted, remembered, the queue keeps draining;
     the file still closes with a footer and an INFO block naming the fault

close()
  └─ raises RuntimeError if anything was swallowed, so a partial recording can never be
     mistaken for a complete one

any reader exception
  └─ typed errors in archx/errors.py: FormatError / CorruptBlockError /
     TruncatedFileError / UnsupportedVersionError -> CLI prints and exits 1
```

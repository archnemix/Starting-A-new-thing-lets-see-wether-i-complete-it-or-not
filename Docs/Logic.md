# Docs: Logic

> Purpose: the flow of control. Which call leads to which file, in order, so a break
> can be traced from the user's line to the exact file. Updated whenever a call chain
> changes.

Nothing is implemented yet; these chains are the contract the code will be built
against. Files named here appear in `Plan/Architecture.md` section 9.

---

## 1. `archx.attach(model, optimizer, path=...)` - plain PyTorch

```
user script
  └─ archx.attach()                       archx/__init__.py
       └─ Recorder(...)                   archx/recorder.py
            ├─ build tensor registry      archx/formats/tensor.py
            │    (names, dtypes, shapes, tied weights by data_ptr, frozen flags)
            ├─ open container             archx/container/writer.py
            │    └─ header block + META block + first ANCHOR
            ├─ start writer thread        archx/recorder.py  (bounded queue)
            ├─ optimizer.register_step_post_hook(_on_step)   archx/adapters/torch.py
            └─ per-parameter grad hooks (torch >= 2.1)       archx/adapters/torch.py
```

Per training step:

```
optimizer.step()
  └─ step_post_hook                        archx/adapters/torch.py
       ├─ read param.grad -> norms          archx/adapters/torch.py
       ├─ copy param.data -> queue          archx/adapters/torch.py
       └─ policy.decide(step, stats)        archx/policy.py
            └─ enqueue(delta_request | skip | anchor_request)
writer thread (never blocks training)
  ├─ delta = current - shadow              archx/recorder.py
  ├─ encode (unchanged|sparse|quant|lowrank) archx/encoders/*
  ├─ byte-plane shuffle + xor              archx/encoders/plane.py
  ├─ zstd compress                         archx/container/writer.py
  └─ append block + batched fsync          archx/container/writer.py
```

## 2. `ArchxCallback(...)` - Hugging Face Trainer

```
Trainer.train()
  ├─ on_train_begin   -> resolve model via accelerator.unwrap_model, optimizer via trainer.optimizer
  │                      then register the same optimizer post-hook as chain 1
  ├─ on_pre_optimizer_step (transformers >= 4.46) -> gradient norms
  ├─ on_optimizer_step -> timestamp / skip flag from accelerator.optimizer_step_was_skipped
  ├─ on_log           -> loss, lr
  ├─ on_evaluate      -> eval metrics (Tier 3)
  └─ on_train_end     -> final anchor, footer, close
```

## 3. `ArchxCallback(...)` - Lightning

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

## 5. Reading a file

```
archx info / reconstruct / diff / recover      archx/cli/*
  └─ container/reader.py  (header -> block scan -> CRC check -> index)
       ├─ index present  -> random access by step
       └─ index missing  -> container/recovery.py rebuilds it
  └─ delta chain apply: anchor + sum(deltas)     archx/container/reader.py
```

## 6. Failure paths (must never propagate into training)

```
any recorder exception
  └─ guard in archx/recorder.py: log once, mark feature unavailable,
     write CAPABILITY_DEGRADED event, continue training
```

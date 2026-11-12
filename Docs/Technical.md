# Docs: Technical

> Purpose: technical reference for the whole project - stack, layout, what each file
> is for, its inputs and outputs, and what the important terms mean.

---

## 1. Stack (v1, decided)

| Layer | Choice | Why |
|---|---|---|
| Language | Python >= 3.10 | must run inside PyTorch training; no Rust toolchain reachable from this sandbox |
| Arrays | numpy | tensor staging, byte-plane work |
| Compression | zstandard (or stdlib `compression.zstd` on 3.14+) | best ratio/speed for our payload, wheels everywhere |
| Optional numeric | numpy only (SVD via `numpy.linalg.svd`) | avoid a scipy dependency for v1 |
| Data capture | content-addressed shards, per-shard codec, zstd dictionaries trained per run | one copy per unique unit of data; dedup is what keeps epochs free |
| Framework glue | torch >= 2.0/2.1, transformers >= 4.46 (soft), lightning 2.x | public hooks verified by reading source at tags |
| Tests | pytest, run on GitHub Actions (ubuntu / windows / macos) | cross-runner requirement; 31 tests green on all three |
| Packaging | pyproject.toml + hatchling | simple, no build backend trickery |
| CLI | `archx` console script + `python -m archx` | two access paths, one code path |

Not in v1: Rust/PyO3 accelerator (see Architecture 3), CUDA-specific code paths
(detected and degraded, untestable here), DeepSpeed/FSDP, TensorFlow.

## 2. Repository layout

```
README.md              what the project is, where to start reading
Docs/                  rules, reasoning, research, reference, secrets
Plan/                  architecture, roadmap, format spec, open questions
Tests/                 test code per layer + recorded results per version
Actions/               CI trigger files, inputs, committed outputs
internels/             non-public helper material, skills
examples/              tiny runnable examples (empty until code exists)
```

Each folder has a README that says what belongs there and what does not.

## 3. File-by-file reference

The live, machine-readable version of this table is `Docs/Files_Info.json`. It is the
first thing to consult before reading code; this section explains the *why*, that file
lists the *what*.

### Docs/

| File | What it is | Read it when |
|---|---|---|
| `Rules.md` | binding working rules | before every change |
| `Thinking.md` | reasoning log per round | before changing a decision |
| `Technical.md` | this file: stack, layout, terms | onboarding to the code |
| `Logic.md` | call chains (user call -> file -> file) | tracing a break |
| `Changelog.md` | features per version | before a release |
| `Change.md` | user-requested additions, waiting to enter the roadmap | when the user says "add X" |
| `Workaround.md` | sandbox/CI limitations and their workarounds | when something "cannot be done" |
| `Secrets.md` | exact secret names and what they are for | before a workflow needs a secret |
| `Files_Info.json` | per-file purpose, inputs, outputs, handles | instead of re-reading code |
| `Research.json` | every lookup, with URLs | before using any library or API |

### Plan/

| File | What it is |
|---|---|
| `Architecture.md` | the decided design: product, adapter strategy, storage, tiers, non-goals |
| `Roadmap.md` | ordered steps, each finished before the next |
| `Format_Spec.md` | normative container format, versioned (`format_major.minor`) |
| `Open_Questions.md` | decisions waiting on the user, with recommendations |
| `Origin_Chat.md` | what the original conversation asked for, and how it maps to v1 |

### archx/ (container layer and CLI are live)

| File | Role |
|---|---|
| `archx/version.py` | version and format constants, the only place they are defined |
| `archx/errors.py` | typed exceptions (format, corruption, truncation, version, dependency) |
| `archx/container/schema.py` | the only file with byte offsets: structs, ids, flags, CRC helpers |
| `archx/container/compress.py` | per-block codec, zstd -> zlib -> none fallback |
| `archx/container/writer.py` | append-only writing, batched fsync, index, footer, closed flag |
| `archx/container/reader.py` | validation and decoding, hostile-file refusal |
| `archx/container/recovery.py` | prefix scan, truncate, rebuild index, re-close |
| `archx/cli/main.py` | argparse and dispatch |
| `archx/cli/info.py`, `recover.py`, `reconstruct.py`, `events.py` | one command each |

| File | Role | State |
|---|---|---|
| `archx/formats/dtype.py` | numpy/torch dtype -> the 9 wire dtype ids | live |
| `archx/formats/registry.py` | tensor registry: stable indices, tied weights by `data_ptr`, regions, components | live |
| `archx/encoders/base.py` | the encoder contract and `choose_encoder` (smallest body inside the error budget) | live |
| `archx/encoders/{unchanged,sparse,dense,dense_raw,lowrank}.py` | encodings 0-4, one file each | live |
| `archx/policy.py` | phase (warmup/middle/cool-down) plus event triggers -> store / skip / anchor | live |
| `archx/recorder.py` | the hook surface, the shadow copy, the bounded queue, the writer thread | live |
| `archx/data/codecs.py` | content id, per-shard codec choice (text / tokens / zstd), lossless round-trip | live |
| `archx/data/capture.py` | data levels, two-phase capture (observe cheap, seal off-thread), dedup, effects | live |
| `archx/capabilities.py` | what the run could actually see: the tier and every missing field | live |
| `archx/container/values.py` | blocks -> meaning: registry, anchor, delta, `Timeline` | live |
| `archx/adapters/torch.py` | `archx.attach(model, optimizer, path)` via `register_step_post_hook` | live |

Rule: one file per responsibility, purpose line at the top of each.

## 4. Terms (so the docs stay short)

| Term | Meaning here |
|---|---|
| anchor | a full snapshot of all recorded parameters at a step |
| delta | per-tensor difference between consecutive recorded states |
| shadow copy | the recorder's own copy of the weights from the previous step, taken at hook time |
| step | one optimizer update (not one batch, not one backward) |
| region | a user-named group of layers (`attention`, `head`, ...) used for statistics |
| component | a named sub-model in a multi-part model (vision tower, language model, ...) |
| tier | capability tier 0-3, recorded in the file as a fact |
| block | the self-describing unit of storage; the file is a sequence of blocks |
| phase | warmup / middle / cool-down, controlling recording density |
| observation (`EVENT`) | what the run measured at one stored step: loss, gradient norm, learning rate, movement per region |
| annotation (`ANNOT`) | what the user declared happened: a stage boundary, an unfreeze, a curriculum change |
| sealed | a segment or shard whose hash and compressed form are final; done on the writer thread |
| capability block | header section listing what could not be observed and why |

## 5. Environment facts that constrain the design

- Sandbox: 2 vCPU, ~3.6 GiB RAM, no swap, ~20 GB disk, no GPU.
- Reachable: github.com, api.github.com, codeload.github.com, pypi.org,
  files.pythonhosted.org, registry.npmjs.org.
- Blocked: all Rust hosts, raw.githubusercontent.com, docs.rs, docs.github.com,
  arbitrary sites. Source reading therefore happens through the GitHub contents API.
- Token: contents read/write and actions metadata read; no secrets, variables, logs
  or artifacts. Consequence: CI writes results into the repo.

## 5a. The data path (v1, on by default)

```
training loop
  └─ batch (as the user's dataloader produced it)
       ├─ hash content -> shard id                       (writer thread)
       ├─ already stored? -> pointer only                (dedup, the epoch win)
       └─ new -> codec by type -> compress -> DATA_SHARD
              + DATA_SEGMENT (characteristics, step span,
                 per-region effects, gradient norms, loss trend)
```

Levels: `none | pointers | segments (default) | full`. A `sensitive` shard keeps hash and
characteristics and drops content, which is the only allowed fidelity exception, and it is
recorded per shard in a `DATA_POINTER` block.

**The capture is split in two, and the split is a measured decision.** `observe()` runs on the
training thread and does only what costs nothing: an identity check against the previous batch,
and a dictionary insert. `seal()` runs on the writer thread and does the expensive half.
Measured on this sandbox (2 vCPU):

| Operation | Cost | Where it runs |
|---|---|---|
| `observe()` of a new 128 KiB batch | ~0.3 us | training thread |
| content hash of 128 KiB | ~210 us | writer thread |
| content hash of a 2 MiB batch | ~3.4 ms | writer thread |
| compress+encode 128 KiB | ~530 us | writer thread |
| `detach().cpu().numpy().tobytes()` of a 2 MiB tensor | ~150 us | training thread |

The last row is the only unavoidable cost on the training thread, because the loader may reuse
the buffer as soon as the step returns. That is why `data_level` has the escape hatch it has.

**Data reaches the disk during the run, not at close.** Before, a killed process kept its
deltas and lost every batch - the "cause" half of the product vanished exactly when someone
needed to investigate. Now, after each stored step, the writer seals and appends the new shards
and segments. Two orderings are enforced:

- a shard is written before the segment that points at it;
- movement is written before the data that explains it, and no segment may name a step the
  file has not reached yet.

The second rule is why a segment can be stated more than once with a wider span: the file is
append-only, so the segment grows by re-statement, and `Timeline.segments()` resolves the
statements to the newest one per id.

## 5b. What a recording costs (measured, not claimed)

The overhead budget in the brief is <5%. It holds for the weight half and not for the data
half, and the difference is reported rather than hidden.

Method: `Tests/integration/test_torch_adapter.py` wraps the recorder's own `step` and `log`
and sums their wall time, so the number is the work added, not the difference between two
noisy runs of a whole loop (on this sandbox the same loop varies by +/-8% between runs, which
is bigger than the effect being measured). Step size 512-wide MLP, batch 1024, 25 steps,
median.

| Recorded | Charged per step | Share of a 19.2 ms step |
|---|---|---|
| weights only (`data_level=none`) | 0.119 ms | 0.6% |
| weights + data (`segments`, default) | 0.140 ms | 0.7% |

Two older measurements, kept because they show where the money went: the same run cost
**+17%** when the snapshot was taken twice per anchor, and **+81%** when the batch was hashed
and compressed on the training thread. Both are now writer-thread work.

On a tiny model the same code is a much larger share - a 0.3 ms step is dominated by framework
dispatch, so a fixed 5% budget is not meaningful there. The honest statement, and the one the
docs use: **recording weights costs under 1% of a real training step; recording the data adds
one buffer copy and the documented levels exist to stop paying it.**

## 6. CI runners (measured 2026-11-11, run 43052619656)

| Runner | CPU | Arch | Python |
|---|---|---|---|
| ubuntu-latest | 4 vCPU | x86_64 (`Linux-7.0.0-1017-azure`) | 3.11.18 |
| windows-latest | 4 vCPU | X64 (`Windows-10.0.26100`) | 3.11.x |
| macos-latest | (arm64) | `macOS-26.7.2-arm64` | 3.11.9 |

Each runner is ~2x the sandbox CPU. Results are readable only as commits into
`Actions/out/`, never through the Actions UI - verified for this repository.

## 7. Data safety rules baked into the format

- No pickle, no code loading, no executable section: only fixed-width numbers, bytes
  and UTF-8 JSON.
- Every length and CRC validated before allocation.
- Append-only writes; the footer is the only thing written "backwards" (as a final
  append), so a truncated file is always readable up to the last whole block.
- Format version in the header from day one; unknown block types are skipped, unknown
  major version is refused.
- The writer thread never dies silently. A fault is counted, remembered, written into the
  file as an `INFO` block naming the error, and re-raised by `close()` so a partial recording
  cannot be mistaken for a complete one. This was found by the crash test: a merge bug killed
  the thread at step 22 and the run continued for four thousand more steps with nothing
  recorded.
- Every file states the tier it reached (`INFO` `capabilities`) and names each field it could
  not see, with the reason. A tier is a measured fact about one run, never a request.

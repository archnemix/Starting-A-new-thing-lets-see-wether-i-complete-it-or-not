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
| Framework glue | torch >= 2.0/2.1, transformers >= 4.46 (soft), lightning 2.x | public hooks verified by reading source at tags |
| Tests | pytest, run on GitHub Actions (ubuntu / windows / macos) | cross-runner requirement |
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

### archx/ (planned, not created yet)

See Architecture section 9 for the package tree. Rule: one file per responsibility,
purpose line at the top of each.

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
| capability block | header section listing what could not be observed and why |

## 5. Environment facts that constrain the design

- Sandbox: 2 vCPU, ~3.6 GiB RAM, no swap, ~20 GB disk, no GPU.
- Reachable: github.com, api.github.com, codeload.github.com, pypi.org,
  files.pythonhosted.org, registry.npmjs.org.
- Blocked: all Rust hosts, raw.githubusercontent.com, docs.rs, docs.github.com,
  arbitrary sites. Source reading therefore happens through the GitHub contents API.
- Token: contents read/write and actions metadata read; no secrets, variables, logs
  or artifacts. Consequence: CI writes results into the repo.

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

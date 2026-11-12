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

Planned for the same version: claim 6 (adaptive vs fixed interval at equal bytes).

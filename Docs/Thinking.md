# Docs: Thinking

> Purpose: the reasoning log. Why something was built or changed, what problem came
> up, how it was split, how it was solved. One entry per round of work, newest last.

---

## Round 1 - 2026-11-11 - Repository skeleton and the language decision

**Input:** the setup brief (rules + doc set + structure) plus a correction: the
library must be usable from Python; the implementation language is the assistant's
call.

**Problems found, in order of severity**

1. *Rust is not buildable here, at all.* The brief's section 7 asks for a Rust core
   with PyO3/maturin. I checked before designing around it: no rustc/cargo/rustup,
   no `apt` candidate, and `static.rust-lang.org`, `sh.rustup.rs`, `crates.io`,
   `index.crates.io`, `static.crates.io`, `docs.rs` all refuse connections. Only
   GitHub, PyPI and npm answer. That means any Rust file written here could never be
   compiled, tested, or even format-checked - it would be unverifiable code, which
   breaks the "fully built and tested" rule.
2. *The token cannot see Actions logs.* Verified by API: `actions/runs` and
   `actions/workflows` return 200, but `actions/secrets`, `actions/variables` and
   `actions/permissions` return 403. So the CI design has to write its results into
   the repository as commits. That was already the user's instruction; now it is
   evidence-backed instead of assumed.
3. *The delta cannot be taken at the hook.* Optimizer post-step hooks fire after the
   weights have changed, so the old value is gone. The design needs a shadow copy per
   tensor, which is also the only unavoidable per-step cost. This reframes "overhead"
   as "copy bandwidth", which is measurable and can be designed around.

**Decisions taken**

- Core in Python with a narrow hot-loop boundary; optional native accelerator later.
  Rationale: the only two expensive operations are a memcpy and zstd, and both are
  already C. This is written up in Architecture section 3 so the user can overturn it
  with one sentence.
- Shadow-buffer delta engine, async writer thread, "degrade, never block" backpressure.
- Per-region thresholds, not global, because CNNs/RNNs have per-layer rates that
  differ by orders of magnitude.

**Verification done instead of assuming** (all through the GitHub API, since raw
content and docs sites are blocked)

- `Optimizer.register_step_post_hook`: present at v2.0.0, absent at v1.13.1 -> minimum
  torch 2.0 for `attach()`.
- `Tensor.register_post_accumulate_grad_hook`: present at v2.1.0, absent at v2.0.0,
  defined in `torch/_tensor.py` (not on `nn.Module`).
- HF Trainer inner loop order: `step()` -> `zero_grad()` -> `global_step += 1` ->
  `on_step_end`, so gradients are gone at `on_step_end`; `on_pre_optimizer_step`
  exists only from transformers 4.46.
- TRL `SFTTrainer`/`DPOTrainer` and Axolotl's trainer are real subclasses of the same
  Trainer stack -> one callback, no forks.
- Lightning automatic optimization uses a closure and can call `step()` itself ->
  step counting must come from loop state, not batch index.

**Still open / not claimed**

- Real compression numbers: nothing measured yet, and Architecture marks every ratio
  as a hypothesis.
- Whether a Python writer thread stays under the overhead budget with the GIL -
  unmeasured. Free-threaded CPython 3.14 (`cp314t` torch wheels exist) is the hedge.
- Launcher reliability across torch versions - mechanism-level reasoning only.
- Where exactly the docs live (Q10) and JSON "purpose comments" (Q11) - asked, not
  guessed. Current placement is a proposal, cheap to move.

**Process note:** docs were written before any code, per the brief's "do not write
code yet" and the rules' "nothing new until the current thing is built and tested".

## Round 2 - 2026-11-11 - The origin chat arrives

**Input:** the full export of the conversation that produced the idea (2387 lines), with
the instruction that it must be kept but never pushed to GitHub, not even in a deeply
nested folder.

**What was done**

1. Stored verbatim at `internels/chat/2026-11-10-to-11-11_origin-chat.md` with an added
   header comment (body untouched), ignored by `.gitignore`, and made a rule (25) so the
   exclusion is policy, not a stray pattern. Verified with `git check-ignore`.
2. Read all 11 user turns and wrote `Plan/Origin_Chat.md`, because the raw text is local
   only: if the workspace is ever recreated, the *conclusions* must survive.
3. Compared the chat against the approved brief instead of trusting the brief to be
   complete. Three real gaps came out, listed below.

**Gaps found (and what I did about them)**

- *Training data.* The brief says "never the data itself". The chat says the data should
  be stored so a human or AI can see what caused which change. These are not compatible,
  so it was left as a question at the time (Q17). **Resolved in round 3: the data is
  stored, default on** - the brief's sentence was a leftover from the earlier format
  discussion and the user's 20:11 turn overrode it.
- *Base-model composition.* The chat calls "store only what changed from the base model"
  the most valuable format feature - and it is **not** the same thing as in-run history,
  which is what v1 builds. Reserved `META.base_model` / `META.ancestor` now, composition
  in v2 (Q18), because two keys today prevent a format major bump later.
- *The file must stay a model file.* The origin chat started from safetensors/GGUF, and
  the user expects a closed recording to still be usable as weights. That became a stated
  property: the final anchor is always full, and blocks are independently readable.

**One equivalence that saved complexity**

The chat's "accumulate small deltas and flush when the threshold is crossed" looked like
a different scheme from the phase-adaptive policy. It is not: a delta flushed at a trigger
is exactly `state(trigger) - state(last stored)`. Accumulating every step would cost
more copies and store the same bytes. So the design copies at store points only, and the
wording in Architecture 6.2 now says the stored delta is the accumulated movement. One
mechanism, not two.

**Numbers deliberately not adopted**

The chat's 244x / 20-100x / 60-250x ratios are arithmetic on a hypothetical 4096x4096
matrix with no measured error. They are marked as hypotheses in `Plan/Origin_Chat.md`
section 6 and stay out of the design until Phase 2 measures them.

**Still open:** Q17, Q18, Q19 (new), plus Q2-Q16 from round 1.

## Round 3 - 2026-11-11 - The main thing is the join, and the data is stored

**Input:** decision that real training data is stored by default (opt-out allowed, cost
optimised), instruction that the origin chat outranks the generated brief, and the exact
turn where the idea started (11-11 19:51) plus everything after it.

**What re-reading the chat in detail changed**

1. **The product is not "a recording engine".** The main turn says the point is to know
   *when this data was introduced and how the state changed at that stage*. Recorded as the
   recording triple in Architecture 6.0: state + cause + stage. Any two of the three is a
   product that already exists (checkpointing, ablation, training log), which is a useful
   test for every feature: does it keep the join, or does it only look like it does?
2. **The brief's "never the data itself" is a leftover from the earlier format discussion**
   (the chat's own Strategy 5 from 20:05 said fingerprints were enough). The user's 20:11
   message overrode it - the data is what proves the effect. Note the ordering: the brief
   was generated 20 minutes after the main turn and inherited the older assumption. That is
   exactly why rule 12a now puts the chat above generated prompts.
3. **Storage has to be sub-linear in steps and epochs** (20:00: trillions of tokens,
   millions of steps). Two independent mechanisms: dedup for data (an epoch writes pointers,
   not copies), and phase-adaptive deltas for weights. Written as a hard measurement
   (claims 11-14), not an assumption.
4. **"Different levels of detail" was already the user's idea before the brief** (20:04),
   including the numerical-vs-text compression distinction. So the phase policy is not an
   invention to justify; it is the spec.

**Design that came out of it**

- `data_level` with `segments` default; `sensitive` per-shard redaction; explicit recorded
  opt-out with a `JOIN_INCOMPLETE` block. Silence is not allowed to look like compliance.
- Compression effort moved to float deltas: data is stored once, deltas repeat at every
  store point, so that is where bytes actually are.
- Per-region effect summaries written during the run (not recomputed later), thresholded so
  they stay affordable at millions of steps, with a coarse dense tail for long runs.
- `attribution: "observational"` in META. The chat's own reply warned that attribution is
  probabilistic; the format must not let a reader overclaim causation from correlation.

**Deliberately not adopted**

The chat's 244x/50-200x/100-200GB figures stay hypotheses. The shared-signature and
edit-vector ideas stay post-v1, but 6.10 now requires that v1 files contain enough to build
an atlas later without retraining - which is a v1 requirement created by a v2 feature
accepting the constraint in advance.

**Open:** Q20 (a byte cap for stored data), plus Q2-Q16.

## Round 4 - 2026-11-11 - Naming and the data cap

**Decisions:** no cap on stored data by default; PyPI distribution `archx.mdl` while the
import and CLI stay `archx`; npm `archx` used as a signpost.

**Why the extension does the disambiguating:** the bare name is taken on PyPI by an
unrelated project (v0.1.1). The user's rule is that the brand never changes and the
extension represents the tool, so the distribution name carries the extension - the same
`.archx.mdl` idea applied to packaging. PEP 503 normalises `archx.mdl` to `archx-mdl`, so
either spelling resolves; that is checked, not assumed.

**Why an npm package at all:** not to build anything in JavaScript. `npm i archx` is what a
person tries first, and an empty name is a name someone else can take later. A signpost
package that prints the real install command is ~20 lines and cannot drift from the Python
code because it contains none.

**No-cap decision, applied honestly:** "no cap" without reporting would mean a surprise
terabyte. So: projected size at start, a warning per doubling, real total at close. The cap
still exists as an opt-in and degrades to `pointers` rather than truncating - truncation
would leave a file that looks complete but has a broken join.

**Not decided here:** whether the npm package gets published now or after v1 (roadmap step
0.5 says built when ready, published on your word), and the npm account name (uncheckable
without auth).

## Round 5 - 2026-11-11 - Phase 0.3 and Phase 1: the container exists

**Built:** packaging (`archx.mdl` distribution, `archx` import, `archx` CLI, MIT, py>=3.10),
container schema/writer/reader/recovery, `archx info`, `archx recover`, 31 tests, and a CI
matrix that runs them on Linux, Windows and macOS and commits the records.

**Problems met, and how each was solved**

1. *The reader parsed the footer as a block.* Symptom: every closed file failed with "block
   prefix CRC mismatch at offset 181". Cause: I checked "fewer than a block prefix left"
   before checking for the END magic, and a footer is bigger than a prefix - so once the
   index block was read, the footer looked like a corrupt block. Fix: check for END magic
   before any size check, and treat an incomplete footer as a torn tail, not as corruption.
   This is a real bug that only appears with a multi-block file, which is why the first
   roundtrip test failed.
2. *Recovery could not rebuild the index.* The step only existed inside the compressed
   payload, so `archx recover` would have had to decompress every block to know where steps
   were - slow, and it made the index of a recovered file wrong (the first attempt produced
   step `25063509241869608`). Fix: moved the step into the block prefix (8 bytes). The
   prefix grew from 24 to 28 bytes and the spec was updated in the same commit. This is the
   one place where implementation changed the format, and it changed it for a good reason:
   recovery is a headline promise, so it must be a prefix scan, not a decompression pass.
3. *The 28-byte struct was miscounted twice.* `struct.Struct.size` is padding-free, so
   `"<HBBQIIII"` is 28, not 32, and a CRC constant copied from the old layout made the
   prefix 28 bytes while the code asserted 32. Fix: derive `BLOCK_SIZE` from the struct,
   assert it once in the schema test, and let the constant be checked rather than typed.
4. *`argparse` exits instead of returning.* `--version` raises `SystemExit`, so the test had
   to expect it; the CLI contract is "return an int for commands, exit for argparse flags",
   now written down in the test.
5. *A tear can land inside an uncompressed block's prefix.* Random truncation at a
   payload boundary can leave a short payload that still passes its CRC (2^-32). Not
   realistically reachable, but not impossible; recorded as a documented limitation in the
   spec (6a) with three options, and left to step 1.6 rather than hidden.
6. *The test record had an empty summary.* `pytest -q` prints only dots when everything
   passes, so parsing for "N passed" found nothing. Fix: parse the dot line into counts and
   only fall back to parsing words. The record now says `31 passed, 0 failed`.

**Verified, not assumed**

- CI run 43060156597: 3/3 runners green on the first real matrix.
- CI run 43060440992: 3/3 green again after the closed-flag change, and each runner
  committed its own `Actions/out/test-runs/<os>.json`.
- Local: 31 tests, 0 failures, 0 skips, including a real `SIGKILL` of a writer subprocess
  and an 8-seed tear-point property test.

**Deliberate choices**

- `Reader` loads the whole file. v1 files are small; a streaming path belongs with the
  recorder, not before it.
- Unknown block types are skipped; unknown codecs are errors. Skipping unknown data is safe,
  silently mis-decoding it is not.
- The property test uses fixed seeds instead of hypothesis: a failure is then reproducible
  from the test id alone, which matters more than input variety at this stage.

## Round 6 - 2026-11-12 - Phase 3 finishes: the torch adapter, the crash test, and what it caught

**Built:** `archx.attach()` for raw PyTorch; `max_rel_error` exposed on attach; per-step
observations (`EVENT`) carrying loss, gradient norm, learning rate and movement per region;
annotations moved to their own `ANNOT` block; a capability block written at close; `archx
reconstruct` and `archx events`; data written during the run instead of at close; 99 tests,
green locally and stable over three consecutive full runs.

**The measurement that started this round.** torch from PyPI would not import here: it is the
CUDA wheel and `libcublasLt.so` was missing. I installed the nvidia wheels it asks for until
`import torch` worked: 8 GB of the 20 GB disk, about two and a half minutes of downloading, and
a CPU-only machine that never uses any of it. That is written into `Docs/Workaround.md` so
nobody repeats the search.
On step 1 of the plan I had planned to skip the adapter test in CI; it stays skippable there
(`pytest.importorskip`) because a runner without torch is a real configuration, but it now
runs for real locally, and the numbers below are from a real `nn.Module`, not a stub.

**Bugs found, in the order they appeared - each one is a reason the previous design was wrong**

1. *The recorder computed loss and gradient norms and then threw them away.* `step()` built a
   context dict for the writer that nothing ever wrote. The file therefore claimed tier 1 in
   META while containing none of tier 1. Fix: every stored step writes an `EVENT` block with
   those values, and META no longer claims a tier at all - the achieved tier is written at
   close, when it is a fact.
2. *Annotations and observations shared one block type.* `annotate()` wrote `{"event": ...}`
   into `EVENT`, which the spec reserves for what the run measured. A reader could not tell
   "the user unfroze the model at step 900" from "loss was 0.31 at step 900", and the join lost
   the distinction between cause and commentary. Fix: `ANNOT` (id 7, already allocated in the
   spec) is the user's word; `EVENT` is the run's measurement. The CLI prints them apart.
3. *Data capture did the hashing and compression inside the training step.* It cost +81% on a
   19 ms step, and on the 128 KiB batch used by the crash test the hash alone is 210 us. Fix:
   `observe()` now only checks identity and stores a reference; `seal()` hashes and compresses
   on the writer thread. This also fixed a subtler thing: the batch is hashed once, not twice
   (once for the shard id, once for the characteristics).
4. *Every anchor copied the whole model twice.* The anchor's payload and the shadow buffer were
   the same arrays, duplicated. Fix: an anchor *is* its own previous state, so the copy is
   reused. Measured: +17% overhead became +0.6%.
5. *A data cap silently produced segments with no shard.* At level `pointers`, or after the cap
   was hit, the segment said `shard_id` with nothing behind it and no `DATA_POINTER` block.
   Fix: pointers are collected and written, so "where to look" is a fact in the file.
6. *The crash test found that data never reached the disk during the run.* A killed process
   kept its deltas and lost every batch: "the weights moved" with no "because of this data",
   which is half of the product and precisely what someone investigating a crashed run needs.
   Fix: the writer seals and appends after every stored step.
7. *That fix exposed a race that killed the writer thread.* `seal()` merged segments into the
   same dict the flush was iterating, so the thread died with "dictionary changed size during
   iteration" at step 22 - and the training run continued for four thousand more steps, silently
   recording nothing. Two fixes, both kept: a lock around the shared dicts, and a writer thread
   that can no longer die silently. It now catches per-step faults, writes an `INFO` block
   naming the error, still closes the file, and re-raises from `close()`.
8. *Segments claimed steps the file did not have.* The segment grew on the training thread while
   the writer lagged, so a crash could leave a segment naming step N in a file whose history
   stopped at N-2. Fix: a flush states a segment only once its first step is on disk, and clamps
   the end of its span to the last step whose *event* is guaranteed written - which is the
   previous step, not the current one, because the event for the current step is appended after
   the call. Getting that off-by-one wrong the first time produced the "segment at step 0 while
   only step -1 exists" bug.
9. *The dirty flag was the wrong mechanism, twice.* First it was cleared before the clamped
   write (so a stale span was never corrected); then it was kept set (so identical content was
   rewritten each flush, giving nine statements for four segments). Fix: compare the text of
   the statement with the text last written, and write only when it differs. Exact, idempotent,
   and it removes a whole class of "when do I clear this" reasoning.
10. *Tied weights were reconstructed as if they had drifted apart.* Deltas are stored under the
    primary name only, so applying them and forgetting the alias produced a model where
    `encoder.weight` and `decoder.weight` held different values - a state that never existed.
    Fix: `Timeline.reconstruct` copies the primary value onto every name in the tie group.
11. *`archx info` refused the file it was most needed for.* A torn tail raised, so the command
    you run on a crashed run failed on a crashed run. Fix: `info` reads with `allow_torn=True`
    and says `closed no (crashed: run archx recover)`.
12. *The overhead test was measuring the machine, not the code.* Timing two whole runs and
    subtracting gave +550%, +320%, -0.6% for the same code on consecutive runs - the sandbox
    varies by more than the effect. Fix: charge the recorder's own `step` and `log` with a timer,
    and assert the charge against the measured step time. The wall-clock numbers are still
    printed but no longer asserted on.

**What the numbers actually say, and what I am not claiming**

Recording the weights costs 0.119 ms on a 19.2 ms step: **0.6%**, inside the 5% budget.
Recording the data as well costs 0.140 ms: **0.7%**, because only a buffer copy is left on the
training thread. Both are measured by charging our code, not inferred from a noisy difference.

I am *not* claiming 5% on a toy model, and the test says so in a comment: at 0.3 ms per step,
framework dispatch dominates and any fixed percentage is meaningless. The <5% claim belongs to
a step big enough to have a meaningful cost, which is where it is measured.

**Verified, not assumed**

- 99 tests locally, 0 failures, 0 errors, 0 skips; three consecutive full runs identical.
- CI run `43190502712`: 3/3 runners green, 93 tests each, 1 skip each (no torch on the
  runner), identical test-name lists across Linux, Windows and macOS.
- The crash test is a real `SIGKILL` of a child process, repeated six times without a flake.
  Its first CI run failed on macOS for a reason worth keeping in the design: the test waited
  for "the file exists and is big enough", and on a slower runner that was true with only the
  header and meta blocks on disk, so the kill landed before any history existed. The condition
  is now the thing the test is about - deltas present, read back through the reader - and the
  test fails loudly if a runner cannot produce three steps in a minute instead of failing a
  different assertion later.
- The numerics guard runs the same loop twice with the same seed - once recorded, once not -
  and compares the loss curve element by element and the final weights with `torch.equal`.
- The lossless path (`max_rel_error=0.0`) rebuilds to 1e-10, and that residual is float32
  accumulation when five deltas are summed onto an anchor, not an encoding loss; with an
  anchor at the same step it is exact.

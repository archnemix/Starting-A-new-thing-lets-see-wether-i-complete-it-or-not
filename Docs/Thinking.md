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
- The lossless path (`max_rel_error=0.0`) is exact at anchor steps and carries a float32
  rounding residual elsewhere: measured 4.19e-09 absolute on weights of order 1, from `a +
  (b - a) != b` in float32, not from encoding. The earlier note here said 1e-10; the measured
  figure is about 4e-09, and the guarantee is now stated as "exact at anchors, float32
  rounding between them" rather than "exact" (`Tests/unit/test_reconstruction_guarantee.py`).

---

## Claim 6: the adaptive policy, measured honestly

**The number I expected and the number I got are not the same, so both are written down.**

At a matched byte budget (1.4% apart) on a 2000-step run with 12 irregular jumps, the adaptive
policy stored **12 of 12** events. A fixed interval of the same size stored **1 of 12**. Median
reconstruction error over the five steps after each event: **2.88e-03 vs 3.63e-01**, a factor of
126. That is the claim, and it holds.

It does not hold as "adaptive is more accurate". On a single-jump stationary run the fixed-12
interval is *better* on both median (5.27e-03 vs 1.46e-02) and post-jump error (5.57e-03 vs
9.91e-03). A fixed grid is a good estimator of a stationary run, and the adaptive policy spends
mostly on the dense warmup and cooldown phases. The product claim is about event resolution,
so that is what is stated; the accuracy comparison is reported next to it rather than dropped.

**The policy had a floor that made "matched bytes" harder than it looked.** 1% warmup plus 1%
cooldown are stored densely by construction, so the adaptive policy cannot be tuned below
roughly that spend. Bisecting its heartbeat to a 700 kB budget ran into its bound and kept
returning the same 4.1 MB run - a measurement that looked like a result until the bound was
checked. The fix was to bisect *both* families and to compare at budgets the policy can
actually reach.

**Three benchmark bugs, each of which would have produced a flattering answer.**
1. `anchor_every=every` on the fixed baseline made every stored step a full snapshot, so the
   comparison was about anchor cadence.
2. The baseline kept the default triggers, and a decaying toy loss trips the 5% loss trigger on
   almost every early step: it stored 57 of its steps in the first third of the run. That is an
   event-driven recorder wearing a fixed interval's name.
3. A weight jump with no loss or gradient change is invisible to a policy that only sees loss,
   gradients and movement. Both policies missed it and the harness measured their heartbeats.
   An event the policy cannot observe is not a test of the policy.

**Two real product bugs came out of this**, and both were worse than the benchmark being wrong:

- The recorder **dropped stored steps** when the writer fell behind: `stats.failed += 1` and
  the item was discarded. At the default `queue_size=8` a 1000-step run kept 39 steps and lost
  13, silently. Losing history while the run continues is the one thing a recorder must never
  do. Two repairs were tried and thrown away first - coalescing movements into the queued item
  (the writer starves, the file begins at step 12), and re-appending the oldest item to the back
  (same starvation). A third, a separate overflow list drained first, reordered the history and
  the crash test caught it: the overflow holds *newer* work than what is still queued. The
  answer is one ordered stream that never discards, where a backlog costs the trainer an encode
  and the writer's memory stays bounded because encoded bytes replace the arrays.
- The **movement trigger could never fire**: `decide()` was never passed a movement value. And
  when it was wired up, the value was an absolute Frobenius norm compared against a threshold
  named `region_move_fraction` - a 256x256 layer of weights near 0.1 drifts by an absolute ~2.7
  per step, so every step read as an event. It is now a fraction of each tensor's own size, and
  a zero-initialised tensor is excluded: `||delta|| / ||W||` with `W == 0` is 100% on the first
  step, decaying like `1/sqrt(n)`, which would have kept the trigger firing for thousands of
  steps on nothing but biases.

Also fixed in passing: a gradient spike was invisible when the gradient history was flat,
because the z-score was guarded by `std > 0`. A quiet stretch has zero variance - exactly the
situation the trigger exists for.

**Verified:** 106 tests, 0 failures, 0 errors, 0 skips locally, including new tests for the
lost-step path, the anchor being step 0, the flat-history spike, the disabled-trigger sentinel,
and the movement fraction (which is what found the zero-bias problem).

---

## Phase 4.5: six architectures, and the trigger that fired on everything

The sweep's purpose is narrow: prove nothing in the adapter or the recorder knows what the
model is. Six stacks - transformer encoder, CNN, LSTM, a gated state-space block, a 4-expert
mixture with a router, and two towers sharing one tied matrix with a frozen embedding - all go
through `archx.attach` unchanged and all reach tier 2, reporting activations as the only
missing field. Ties and frozen parameters needed no special handling, which is the point of
having detected them in the registry rather than in the adapter.

**The sweep found a bug the unit tests could not.** On the transformer, every step was stored.
The usual explanation - "gradients are large, so the triggers fire" - was wrong: the gradient
z-score never reached 4. The movement trigger fired on all 130 middle-phase steps, at 0.12 to
0.41 against a 0.02 threshold.

The cause was in how a region's movement was computed. Each tensor contributed its own
`||delta|| / ||weights||`, and those fractions were **summed** per region. A transformer region
holds 14 tensors; each of them moved 1.5-2.2% in a step, which individually is unremarkable,
and the sum came to 0.20. The number grew with the tensor count, so a bigger model crossed the
threshold sooner and the "per-region threshold" quietly became a per-tensor-count threshold.

The fix is to accumulate squared norms per region and divide once, so the region reports
`||delta|| / ||weights||` over all of its tensors - the same 1.7% the tensors actually moved.
A test pins the property directly: one tensor, three tensors and ten tensors, each moving by
the same fraction, must report the same number.

After the fix the six stacks store 132-144 of 300 steps, with per-skipped-step reconstruction
error between 5.6e-04 and 3.2e-03. Two smaller things were fixed on the way: the models were
built before the seed was set, so the table changed between runs (82 stored steps, then 54,
then 66 for the same model), and the multimodal stack needed 300 steps rather than 150 for its
loss to fall - it does learn, at 400 steps the loss reaches 0.01, and lengthening the run is
honest where lowering the bar would not have been.

---

## Claim 7: what one line buys

The claim is "attach in one line", and the useful thing to measure is where that line stops.
Inside the optimizer's step post-hook the gradients are still alive - the caller's `zero_grad()`
runs afterwards - so a bare `archx.attach` records a gradient norm on **every** stored step.
The loss and the batch are locals that died with the iteration, and no hook can recover them.

So the file from a bare hook is tier 0, and it says so: loss, data and activations all listed
missing with a reason. One `recorder.log(loss=..., data=x)` before `optimizer.step()` makes it
tier 2. Measured on 200 steps of a 3-layer MLP: 0.415 ms/step charged to us with the bare hook,
0.700 ms/step with the logging call - the difference is the loss conversion and the batch copy,
which is exactly what the caller asked us to do.

**The finding I did not expect: help is not only about tiers.** The bare hook stored 21 of the
200 steps; with a log call it stored 44. Without a loss the policy's loss trigger can never
fire, so it falls back on phases and the heartbeat and records half as much history. A user who
attaches and never logs gets a file that is honest and thin, and the capability block is how
they find that out.

**Two contradictions in the honesty block itself, found while testing this.** The module exists
to stop the file from overstating what it recorded, and it was *understating*: `resolve()`
discarded every field of an unreached tier, so a run that recorded gradient norms but no loss
reported `grad_norms` as missing - the file told a reader to ignore data it had. Fields the run
actually recorded are now left alone, and a new `capabilities_present` list states the
inventory separately from the tier, which is a threshold rather than a description. Separately,
`deny()` then `offer()` left a field both present and listed as missing; offering now clears the
denial.

---

## Forward compatibility: the rule was written, then tested

The format spec's rule is one line - refuse a different major, accept any minor, skip unknown
block types - and the reader was only half-implemented against it. Unknown block types *were*
skipped, because `block_type` is a plain integer and every accessor filters by type. Nothing
said so, and a reader that quietly drops a block looks exactly like a reader that understood
everything, which is the same class of problem as the capability block claiming fields it did
not have.

`Reader.read_report()` now names the skips, the newer minor, a torn tail and an unclosed file;
`archx info` prints them as warnings and `--json` carries them. The tests forge the file with
`struct` - a header claiming 1.9, a block type 200 this build has never heard of - because a
test of forward compatibility that uses this build's own writer is not testing forward
compatibility.

The first forgery was wrong in an instructive way: it changed the version fields and left the
old header CRC, so every reader rejected it with "header CRC mismatch". The reader was right.
The CRC covers the fields being changed, so it has to be computed after them.

`archx info` also reads the capability facts from the file now instead of recomputing them. A
tier belongs to the run that produced it; working it out again at read time would be a second
opinion presented as a record.

---

## Claim 4: the claim was aimed at the wrong comparison

"Smaller than naive checkpointing" is the kind of claim that is easy to state and hard to state
honestly, because it is true or false depending entirely on what "than" means.

**At matched accuracy on every step, checkpointing wins.** Measured on six stacks at a 1e-2
target: 4-9x in favour of checkpoints. One stored step of ours is roughly one whole checkpoint,
so an interval coarse enough to hit the accuracy target is simply cheaper. There is no tuning
that fixes this - it is what a per-step delta over every tensor costs.

**At matched accuracy on the stored steps of a 1000-step run, recording wins**: 1.7x to 5.9x
smaller than a checkpoint at every step (CNN 3.8x, RNN 5.9x, multimodal 5.5x, MoE 2.9x, SSM
1.7x). The gap grows with run length, because checkpointing costs model-size times step count
while the recorder's per-step cost falls as the phases let it skip.

Both are in `Docs/Research.json` and both benchmarks are committed. The claim on the box is now
worded as resolution - dense where the model moved - and not as a smaller file. That is the
honest version, and it is also the one a user would notice: a checkpoint interval cannot be
dense at an event and sparse elsewhere, and claim 6 measured that difference at 12/12 events
against 1/12.

**Building the benchmark found three bugs, all in my own work rather than in the product.**
1. The truth was indexed one update early, so the recorder looked like it was uniformly one
   step behind - a flattering bug for checkpoints and a real one for me.
2. `assert_python = all(...), ("message")` in the benchmark. That is a tuple assignment. It
   checked nothing, in a file whose whole purpose is not lying to yourself.
3. Offering int16 quantization at the default budget quietly made the default path lossy. The
   tests caught it (`step 1 w1 differs`), and the decision was to keep the default exact and
   make int16 opt-in rather than to update the promise.

And one real product bug that the tests could not have caught, because the code path did not
exist until this benchmark created it: **int16 quantization shares encoding id 2 with int8, and
the reader decoded by encoding id alone.** Every int16 delta came back at half length and failed
on reshape. The recorder wrote a file that its own reader rejected - the worst possible shape of
this bug, and it existed for the ten minutes between adding int16 and running the benchmark.
The width now comes from the block's value dtype, and `Tests/unit/encoders/test_quantization_path.py`
pins the window where int16 is chosen.

---

## The sparse encoder at a realistic size

An open question sat in the notes for a while: does the sparse encoder still win at 4096x4096,
or was the 256x256 table an artefact of small layers? Measured, the answer is that sparse was
never really winning - **zstd was**.

At 4096² with 1% of entries moved, exact sparse costs 842 KB and int8 quantization costs
509 KB at 1.1e-2 relative error. int8 is smaller than *exact* sparse, because a quantized delta
is mostly zeros and zstd collapses them; sparse spells out which entries are non-zero and then
pays for their values. The same ordering holds at 3% and 10% density.

Sparse's real comparison is against compressed raw float32, and there it is 1.5x at 1% and
1.35x at 10% - a real but modest saving, not the order of magnitude the phrase "sparse layer"
suggests. At matched error, sparse (exact, 842 KB) and int16 (4.2e-5, 790 KB) are within 7% of
each other.

The caveat that matters: the synthetic delta places moved entries at random positions. Real
sparse movement can be structured - whole rows, whole attention heads, a frozen block with one
trainable projection - and structured positions compress differently. This measurement cannot
speak to that, and the encoder chooser does not need it to: it measures every candidate on the
actual delta and picks the smallest one inside the budget. The lesson is about the *documentation*
claim, not the code path.

---

## What a Hugging Face callback can and cannot see

Phase 6 looked like the easy one: the plan already said "register the same optimizer post-hook
as chain 1, then read gradient norms at `on_pre_optimizer_step`". Both halves of that turned
out to be wrong, and only a real `Trainer` showed it.

**The optimizer you are handed is not an optimizer.** `optimizer.register_step_post_hook(...)`
inside `on_train_begin` raises `AttributeError: 'AcceleratedOptimizer' object has no attribute
'_optimizer_step_post_hooks'`. The object is accelerate's wrapper, which is not a
`torch.optim.Optimizer`; the method is inherited from the base class and reaches for state the
wrapper never creates. It is the *inner* optimizer (`trainer.optimizer.optimizer`) that is real.
Since that call sits inside `on_train_begin`, the failure would have killed the user's training
run - a recorder bringing down the thing it was asked to observe.

**The fallback hooked the wrong moment anyway.** The plan's second idea was `on_step_end` plus a
hook. But the loop is `optimizer.step()` -> `lr_scheduler.step()` -> `model.zero_grad()` ->
`global_step += 1` -> `on_step_end`, so at `on_step_end` the gradients are already zero and the
per-step learning rate has already moved on. The events that bracket the update are
`on_pre_optimizer_step` and `on_optimizer_step`, and they exist on every version this project
supports. So the adapter registers no optimizer hook at all: gradients and the learning rate are
read before the step, the step itself is taken after it, and `state.global_step` - not yet
incremented at that moment - is exactly the 0-based index of the update that just happened.

**The gradient norm in the file is the clipped one.** `Trainer` calls `_clip_grad_norm` before it
calls any callback, so with the default `max_grad_norm=1.0` every callback sees a norm pinned at
1.0 once the true norm exceeds it. Measured: four steps reporting `1.0, 1.0, 1.0` while the real
norms were `3.3, 3.1, 3.7`. The public alternative, `register_post_accumulate_grad_hook`, sees the
true value - but it fires *per parameter*, so a frozen or unused tensor never calls it at all,
and a norm assembled from whichever hooks happened to fire is quietly wrong in a way that looks
plausible. The honest choice is to record the clipped norm and write `grad_norm_clipped` into the
metadata, so a reader is never left to assume which number it is holding.

**The loss is the one thing a callback cannot have, and every convenient route to it lies.**
`on_log` looks like the answer and is the worst of them: it fires *after* the step has been
recorded, and it reports the mean over the whole logging window. Using it produced losses on
steps 1, 2 and 3 and none on step 0 - and, worse, a run with `logging_steps=50` would have left
49 steps in 50 with no loss while the file still claimed tier 1. The second attempt,
`compute_loss_func`, was subtler: that seam *pops the labels out of the inputs* before the model
runs, so the wrapper has to rebuild the loss from the logits - which means guessing whether the
model wants a shifted cross-entropy, a plain one, an MSE or something else. It guessed, and
produced a confidently wrong number on a small regression model.

What works is observing rather than reconstructing: wrap the trainer instance's `compute_loss`,
call the original untouched, and read the value it returns. The run stays bit-identical (the loss
curve and the final weights are compared in the tests), and a Trainer subclass that overrides
`compute_loss` the way TRL's do still works because the wrapper wraps *their* bound method.

With none of that in place, the file says tier 0 and names the way out - `loss_hook(trainer)` or
`callback.log(loss=...)` - instead of writing a plausible-looking average under the wrong step
number. A recorder whose selling point is "you can see why the weights moved" cannot afford to be
the thing that is vague about what it actually saw.

---

## Zero-code injection fails quietly, in three different ways

`archx run -- python train.py` looked like the easiest item on the roadmap: put a
`sitecustomize.py` on `PYTHONPATH`, patch `Optimizer.step`, done. It took three separate
silent failures before a single step was recorded, and every one of them left a script that
ran perfectly and produced nothing.

**Torch does not exist yet.** `sitecustomize` is imported by `site.py` at interpreter
start-up, before the user's script has imported anything. The first prototype did
`import torch` inside `try/except ImportError: pass` - so on a machine where torch is a
*site-package* (which it is, for the venv this project uses), the import failed, the except
swallowed it, and the patch was never installed. Nothing printed. The fix is a
`sys.meta_path` finder that installs the patch when `torch.optim` finishes loading, and a
real `import torch.optim` first in case the hook runs late.

**`AdamW` has no `step`.** With the finder in place, steps were still not counted. Patching
`torch.optim.Optimizer.step` captures subclasses that *inherit* it - and `AdamW` does not:
`'step' in AdamW.__dict__` is `False`, and `AdamW.step` resolves to `Adam.step`. The most
common optimizer in the world, silently invisible. The hook now walks every class in
`torch.optim` and wraps each one that defines its own `step`, marking them so a second
import cannot double-count.

**A broken `sitecustomize` is ignored.** The launcher prepends a marker line to the copied
hook. The first marker was `archx: injected by archx run` - a syntax error. CPython's
response to an exception in `sitecustomize` is to print
`Error in sitecustomize; set PYTHONVERBOSE for traceback` to stderr and then run the script
anyway. The script worked, the launcher reported "no optimizer step was observed", and the
real cause was never mentioned. The injected file is now compiled before it is written, and
`test_the_injected_hook_always_compiles` asserts the marker is a comment.

The design principle that came out of it: **the launcher checks its own result.** It does not
trust that injection worked - it opens the file afterwards and counts the steps. Zero steps
means the file is deleted and the command exits 2 with instructions. A launcher that leaves
an empty recording behind is worse than one that fails, because the empty one gets read by
`archx info` and looks like a run that never moved.

The last thing this measured is uncomfortable for the feature itself: for a Hugging Face
Trainer, the launcher is strictly worse than the callback. It records the steps (tier 0) but
the loss, the gradients, the learning rate and the parameter *names* are all out of reach -
an optimizer holds tensors, not a model, so tensors come out as `param[0][3]`. `ArchxCallback`
reads the same optimizer steps plus everything else, for one line of code. The launcher's
justification is the case where the code cannot be edited at all, and its honesty about tier 0
is what keeps it from being a trap.

---

## What the versions of this product kept getting wrong, in one place

Every phase of this project found a defect that a green test suite had been hiding, and the
pattern is consistent enough to be worth naming.

**The recorder was silently dropping history.** `_enqueue` counted a full queue as `failed`
and threw the item away: at the default `queue_size=8`, a 1000-step run stored 39 steps and
lost 13 of them. The fix is not a bigger queue - it is encoding the block on the training
thread when the queue is full, so a full queue costs memory instead of data.

**A queue reorder starved the writer.** Remembering the oldest item and re-appending it
produced a file that began at step 12 with its anchor at step 19, while the counts still
looked right.

**Coalescing an anchor into a later step wrote the old weights under a new step number.**
That is the worst shape of this bug: it looks correct in every aggregate and is wrong in
every value.

**The Hugging Face adapter's planned optimizer hook crashes.** `register_step_post_hook` on
accelerate's `AcceleratedOptimizer` raises `AttributeError`, inside `on_train_begin`.

**`on_log` attributed a window-averaged loss to the wrong step**, and let a run with
`logging_steps=50` claim tier 1 while 49 steps in 50 had no loss.

**The Lightning adapter stored one update behind**, because `on_before_optimizer_step` fires
before `optimizer.step()`. Counts, losses and step numbers were all plausible.

**A tier-2 file could not answer its own question**: 17 of 20 batches carried an empty
`region_effects` map.

**The launcher patched nothing, three separate ways**, most memorably a `sitecustomize` with a
syntax error, which CPython reports and then ignores.

**`Plan/Format_Spec.md` described a file format that had never been written.**

What these have in common: each one produced *plausible output*. Every single one would have
survived a test suite that checked shapes, counts and exit codes. What caught them was checking
values against an independent source - the live weights of the run, the Trainer's own reported
loss, the same seed twice, a real checkpoint on disk, or the file's own bytes parsed against its
own spec. The lesson is not "test more"; it is that a test which only confirms the shape of the
output cannot tell a correct recording from a confident one.

## What the data layer kept getting wrong

Four more of the same kind, found by measuring Phase 2 instead of assuming it was finished.

**A level that was documented and never written.** `full` had a paragraph in
`Plan/Architecture.md` from the first design - "every batch exactly as fed, no dedup, exact
sample order" - and no branch in the code. It wrote the same deduplicated file as `segments`;
on 60 identical batches the two files differed by 4 bytes. The lesson is narrower than
"write tests": a *level* is a promise about behaviour, and there was no test that compared two
levels, so nothing could notice that two of them were the same level.

**A stale map that counted into nothing.** Merging a duplicate segment removed it from
`segments` but left it in `_by_identity`. Because sealing sets `content = None`, the identity
fast-path could never succeed again, so the entry was pure poison: every later sighting of the
same batch incremented a segment that no longer existed in the file. The aggregate still looked
plausible - there was a segment, it had a `times_seen`, the file opened. What exposed it was
adding up `times_seen` over the *final* statement of every segment and comparing it to the
number of observations: 19 against 20.

**A measurement with no address.** `span_steps` recorded how many steps a shared measurement
covered and not which recorded step it belonged to. So a batch seen on a skipped step carried a
real `delta_norm` that pointed nowhere: 36 of 60 segments in the first measurement. This is the
subtle one, because the data was all *there* - the numbers were right, the join was simply
unaddressable. The fix was a field, and the test that keeps it fixed asserts the step exists in
the file rather than that the field is present.

**A benchmark that measured the wrong thing.** The first version of the capture-overhead bench
timed `optimizer.step()`, which included AdamW, and duly reported that capturing data *saved*
3 ms per step. It also reported `full` as cheaper than `segments` and 0.000 ms for every level
after a units slip (seconds printed as milliseconds). Three separate errors, and each one
produced a number. The only reason they were caught is that the numbers were implausible: a
negative overhead, an ordering that contradicted the code, and a zero. A benchmark whose
results are believable is not checked; implausibility is the last line of defence, which is why
the bench now also prints its spread and its catch-up count.

**The one level that promised the least did nothing.** `none` returned early from `observe`,
so it wrote no data block at all - while its module docstring said "hash and count only" and
Architecture 6.8 said the fingerprint is present "at every level". It was the cheapest level,
the one nobody would check, and it was the one that kept none of the evidence it advertised.
Found by opening a `none` recording and asking it which batches it had seen: it could not
answer, and it was not supposed to have to guess.

**A test that measured the runner's mood.** The data-overhead assertion compared one run
against another with a 20% budget, and the quantity it was chasing is 1% of the step. It
passed on Windows, then failed on Windows at a commit that changed only a test's matching
string - proof on its own that nothing under test had moved. The lesson is not "loosen the
threshold": it is that a *ratio* of two noisy measurements cannot detect a small effect, while
a *difference* between two runs that share the noise can. The assertion now asks whether the
copy costs anything at all and prints the number, because "0.03 to 0.10 ms" is a fact and
"under 20%" was a guess with a stopwatch.

**A benchmark that measures the wrong thread.** The activation-overhead bench called
`recorder.step()` inside the same timing window as the forward pass, so the writer thread was
compressing blocks on another core while the forward was being timed. It reported +208%, then
+503%, for identical code. Removing the writer from the window gave 1.38 -> 3.02 ms. The
feature had not changed; the measurement had been reading contention and calling it cost.

**A cap set by round numbers instead of measurement.** `MAX_ELEMENTS = 1_000_000` looked
generous and safe. Per element the reduction cost ~2.8 us per 1,000 values, so a million
elements per layer per forward was several milliseconds - the cap *was* the cost. At 16,384
the layer costs ~45 us and the stride still spans the tensor. The test that keeps this honest
compares the sampled mean to the exact mean on a tensor whose two halves differ by 20.

**A probe that reported tier 0 and looked broken.** The sick-model probe fed a loss and never
stepped the optimizer, so no gradients existed, and the tier ladder - which is strictly
sequential by design - correctly refused to award tier 3 on top of a missing tier 1. The code
was right and the probe was wrong. Worth remembering whenever a file "has the data but not the
tier": that is the ladder working, not a bug.

**The tests read the API and the product reads the CLI.** Every test of the capability block
called `reader.capabilities()` and got tier 3. The command a user runs first printed
`capability_tier 0`, because it read META - which carries the floor tier deliberately, being
written before the first step - and then listed every recorded field underneath it. The file was
right, the reader was right, and the sentence shown to a person was a contradiction. Nothing in
the suite could see it, because nothing in the suite was looking at the output a human sees.

The common thread with the eight defects above: none of them crashed, none of them turned a
test red, and every one of them produced output that a careful reader would have accepted. The
difference this time is that the suspicious numbers were *impossible* rather than merely
unexpected, so the measurement argued with itself.

## Scale test: the first file big enough to argue back

Three phases of work measured everything on models of 6,000-18,000 parameters, because that is
what fits in a unit test and runs in a second. The instruction was to train a million-parameter
model and hand-test the product on it, on GitHub Actions. That is 60x the parameters and 3.7x the
bytes per checkpoint, and both halves - the run and the test - produced defects that no smaller
model could have.

**A parameter budget by hand is a guess wearing a number.** The first shape was worked out on
paper: embeddings plus `4d²` of attention plus `2d²` of MLP per block, which came to 1,002,292.
`count_parameters` on the real module said **1,244,732** - 25% over the target, because the
arithmetic had no LayerNorms and no attention out-projection. The shape that actually measures
1,004,176 (d=164, three blocks, vocab 2100) was found by sweeping the real counter. The comment
in the bench keeps the wrong number, because the wrong number is the reason the code looks the
way it does.

**The scale claim survived; the file size claim needed a denominator.** The recording rebuilds all
1,004,176 values to 9.31e-10 on Linux, Windows and macOS, and costs **0.64-0.68x a checkpoint per
stored step**. The first version of the bench printed `recording / one_checkpoint` instead, which
was 7.16x at 10 stored steps and reads as "the recorder is seven times a checkpoint". Both numbers
are true; only one of them is the like-for-like comparison, and the bench now prints both with the
label that says which is which. A ratio whose denominator is a different *quantity* is the data
layer's capture-overhead mistake in another costume.

**Hand-testing found what unit tests structurally cannot.** The suite had 240 tests, all green,
and every one of them called the library. A user calls the CLI. Running the five shipped commands
against a real 1M file found, in order:

1. `archx reconstruct --step 2` exited 1 - "step 2 is not recorded" - while
   `Timeline.reconstruct(2)` returned all 1,004,176 values. The command refused the product's
   headline promise *on the product's own file*, and the test suite agreed with the command
   because the suite only ever asked for stored steps.
2. `archx diff --other` compared two files that share step numbers and no tensor names, compared
   zero values, and printed "identical at every common step" - its strongest possible verdict - on
   nothing. The emptiness check read as agreement.
3. `archx recover` closed a crashed file and left it reporting `capability_tier 0`, because the
   capability block is written at close and the crash happened before close. A 20-step tier-3 run
   became, after recovery, a file that claimed it could see nothing but a step counter.
4. `archx info` read that same crashed file while `archx reconstruct` refused it - two commands
   disagreeing about what the same bytes were, and the stricter one was the one with the answer.
5. `archx diff` labelled 8 intervals as "8 stored steps in between" when 7 steps are between.
6. `archx events` printed an empty `moved ` column for launcher recordings, which have no region
   names - a blank where the honest answer ("this file has no region names") belongs.
7. Every tier-3 file this project ever wrote said `activation_capture: false` in META while
   holding hundreds of summaries, because the field was `bool(ActivationCapture)` and that object
   is falsy until something has been captured - which is *after* the moment META is written. Two
   halves of the same file disagreeing, and nothing read the field, so nothing could notice.

**Eight statements, one shape.** Every one of them is a sentence in the product that is not true,
and none of them crashed or turned a test red. The suite tests what the code *returns*; a user
reads what the product *says*. That gap is where all eight lived, and the difference from the
earlier rounds is that this time the products were large enough for the sentences to be about
something - 1,004,176 parameters, a 134 MB file, 150 steps of history - which is exactly where a
wrong sentence stops being cosmetic.

**CI could not keep the evidence.** The Ubuntu artifact step failed with
`EACCES: permission denied, scandir '/tmp/snap-private-tmp'` because the workflow globbed
`/tmp/**/*.archx.mdl` and the bench wrote its recording to a `mkdtemp()`. The scale test was green
in every step that measured anything and had no file behind it. The recording now goes to
`Bench/out/`, the artifact step fails loudly if it is missing, and the hand test runs *inside the
bench* so the workflow asserts on the CLI's text rather than trusting that it was run.

**What is still a promise and not a measurement.** `Plan/Architecture.md` §6.11 lists tier 3 as
including eval metrics. No adapter records them, and until this round every tier-3 file said so
anyway. The file no longer claims it; the architecture still plans it. That gap is deliberate and
named, which is the only honest state for it.

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

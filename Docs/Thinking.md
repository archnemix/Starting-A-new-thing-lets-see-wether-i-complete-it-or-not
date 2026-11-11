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

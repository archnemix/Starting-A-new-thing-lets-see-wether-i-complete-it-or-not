# Plan: Open Questions

> Purpose: every decision that is the user's to make, with my recommendation and the
> evidence behind it. Nothing here is guessed in code; unanswered means "not built".

Answers so far: Q1 (language) is answered - it must work from Python, the
implementation language is my call.

---

## Blocking (needed before Phase 1 code)

**Q2. Distribution name.** PyPI `archx` is taken by an unrelated project; `archx-mdl`
is free. Options: (a) ship `archx-mdl` importing as `archx`, (b) pick another name.
*Recommend (a).*

**Q3. Minimum Python and PyTorch versions.** Evidence: torch 2.15 is current with
wheels for CPython 3.11-3.14 (including the free-threaded `cp314t` build);
`Optimizer.register_step_post_hook` exists from 2.0 (verified at v2.0.0);
`register_post_accumulate_grad_hook` exists from 2.1 (verified at v2.1.0);
`TrainerCallback.on_pre_optimizer_step` from transformers 4.46; Lightning callback
surface stable across 2.x.
*Recommend: Python >= 3.10, PyTorch >= 2.1, transformers >= 4.46 for Tier 2 on HF
(older versions get the fallback path, no error).* Confirm or narrow.

**Q4. Run the optional Rust/maturin accelerator at any point in v1?** It cannot be
built in this sandbox (no Rust toolchain, registry hosts unreachable), only on CI.
*Recommend: no - keep the Python core, add the native path only if Phase 2 numbers
show the encode step is the bottleneck.*

**Q5. Overhead budget.** Brief says < 5%. Is that median step time, or p95, and is it
measured with deltas enabled every step (worst case) or on the default adaptive
policy? *Recommend: hard gate on p95 under the default policy, plus a reported
worst-case number for every-step recording.*

**Q6. Benchmark datasets.** They must be tiny (CPU runners, 5-runner cap, and storage
in git). *Recommend: char-level tiny-Shakespeare sample for the transformer and RNN,
MNIST subset for the CNN, all generated/committed as small fixtures with fixed seeds.
Say if you want something else (e.g. CIFAR-10 subset, or a synthetic task).*

## Needed before Phase 6/8

**Q7. Framework versions to support on day one.** *Recommend: transformers >= 4.46,
Lightning 2.x latest, torch >= 2.1.* Confirm if you need older.

**Q8. Multi-GPU in v1?** Recording on one rank only is the honest v1 answer; true
multi-rank merging is v2. *Recommend: single-rank recording, hard failure-mode test
if more ranks are detected without sharding support.*

**Q9. CI runner hardware.** GitHub-hosted runners are CPU-only. All benchmarks are
CPU-only by design unless you want a GPU runner (cost). *Recommend: CPU-only
benchmarks; every accelerator claim stays unverified until a GPU run happens.*

## New, from the origin chat (`Plan/Origin_Chat.md`)

**Q17 - ANSWERED (2026-11-11): store the real data.** User: the data is stored because that
is what shows "this data had this effect on the weights at this stage", which is far more
valuable; on by default, can be opted out, and the cost must be optimised as far as
possible. Implemented as `data_level` with `segments` as the default (Architecture 6.8).
Opting out stays possible but must be explicit and is recorded in the file.

**Q18 - ANSWERED (2026-11-11): reserve the fields, build composition in v2.** Still a
reserved-null pair (`META.base_model` / `META.ancestor`) in v1, since the main thing is
the in-run record and composition is a separate feature. Confirm if that ordering is wrong.

**Q19 - ANSWERED (2026-11-11): yes, keep it.** The user confirmed the file must stay useful
as a weights file, which is also what the origin chat's GGUF lineage implies.

**Q20. Data retention and privacy.** Now that data is stored by default: for a real
training run, is storing the first N% of each shard (default: all of it) acceptable, or do
you want a cap (`data_max_bytes`) so a run can never fill the disk? *Recommendation: no cap
by default, but the recorder reports projected data size at start and supports
`data_max_bytes` with a documented degradation to `pointers` when exceeded.*

## Small, but affects structure

**Q10. Folder spellings.** `internels/` (as you wrote it) and `Docs/` keep their
current names; `Docs/Thinking.md`, `Docs/Technical.md`, `Docs/Files_Info.json` live in
`Docs/` as decided, `Roadmap.md` lives in `Plan/` next to `Architecture.md` because
the brief names it `Plan/Architecture.md` and `Roadmap.md` as the main roadmap.
Confirm or tell me to move them.

**Q11. JSON files cannot hold comments.** Rule 5 ("purpose line in every file") cannot
be satisfied literally in `.json`. Options: (a) a `"_purpose"` field as the first key,
(b) a sibling `.md` header, (c) exempt JSON. *Recommend (a).*

**Q12. Test-record and Actions folder names.** Proposed `Tests/Records/vX.Y.Z/` and
`Actions/in/`, `Actions/out/`, `Actions/triggers/`. Confirm or rename.

**Q13. Retention.** "current + last 2 versions" - does a version mean a released
version (v0.2.0) or every change? *Recommend: released version tags only; per-run
records inside a version are overwritten by newer runs of the same test.*

**Q14. Where does the project live?** Recordings must not be committed (they are
binary run data). Proposed `.gitignore` for `*.archx.mdl` plus an `examples/` output
folder that stays empty except for a README. Confirm.

**Q15. License.** Public repo, PyPI package. *Recommend: Apache-2.0 or MIT - your
call; nothing is written until you answer.*

**Q16. Public versioned format spec from day one?** *Recommend yes, `Plan/Format_Spec.md`
is the normative spec, and the file's header carries `format_major.minor`.*

# Docs: Workaround

> Purpose: things that cannot be done directly in this environment, and the exact way
> they are done instead. Written so the workaround is not rediscovered every time.

---

## W1 - No Rust toolchain, and none installable

**Problem:** no `rustc`/`cargo`/`rustup`; `apt` has no candidate; `static.rust-lang.org`,
`sh.rustup.rs`, `crates.io`, `index.crates.io`, `static.crates.io`, `docs.rs` are all
unreachable (verified 2026-11-11).

**Workaround:** the v1 core is Python. If a native accelerator is ever wanted, it is
built and benchmarked only inside a GitHub Actions job (where the official Rust setup
action has network access to the toolchain hosts), and the Python implementation stays
the reference that CI compares against.

**Do not:** write Rust files intended to be compiled locally. They could not be
compiled or tested here, so they would violate the "built and tested" rule.

## W2 - Actions logs and artifacts are unreadable

**Problem:** the token can read workflow and run metadata but gets 403 on
`actions/secrets`, `actions/variables`, `actions/permissions`; log and artifact
download endpoints are equally out of reach.

**Workaround:** the workflow itself commits everything it produced into
`Actions/out/<run-id-or-label>/` on a branch, using a commit made by the workflow's own
token (`contents: write`). Reading the result is a normal `git pull`, not an API call.
Artifacts (`actions/upload-artifact`) are never used.

**Detail that matters:** commits made by `GITHUB_TOKEN` do not re-trigger workflows,
which is what keeps the log-commit loop from recursing. If a workflow ever needs to
trigger another one, it must be an explicit `workflow_dispatch` with a separate token.

## W3 - Source reading without raw.githubusercontent.com

**Problem:** `raw.githubusercontent.com` is blocked, so `https://raw.githubusercontent.com/...`
links (the normal way to read a file or docs page) fail.

**Workaround:** read repository files through the API and decode:

```
gh api repos/<owner>/<repo>/contents/<path>?ref=<tag> --jq .content | base64 -d
```

This works for any public repo and any tag or branch, and is how the hook/API facts in
`Docs/Research.json` were verified. Docs sites (`docs.rs`, `docs.github.com`) are
blocked too, so documentation is read from the source tree itself.

## W4 - Heavy installs (PyTorch is a 500 MiB wheel)

**Problem:** the sandbox has 2 vCPU / ~3.6 GiB RAM / ~20 GB disk, and installs do not
persist between sessions.

**Workaround:** keep the always-installed footprint tiny (`numpy`, `zstandard`,
`pytest`). Anything needing `torch`, `transformers` or `lightning` runs in CI, where
the install is cached by the workflow. Locally, prefer CPU-only, tiny models, and
skip the download unless the task truly needs it.

## W5 - GitHub Actions concurrency limit

**Problem:** at most 5 runners at once, and a run per trigger file edit.

**Workaround:** one workflow, one matrix of at most 5 entries. Heavy benchmarks are a
separate workflow with a single job, dispatched by editing its own trigger file, so a
quick test run never queues behind a benchmark.

## W6 - Binary run data must not enter git

**Problem:** `.archx.mdl` files are large binaries and would bloat history; CI commits
are also capped in practice.

**Workaround:** recordings live in `Actions/out/` only for tiny proof runs (kilobytes),
and the repo ignores `*.archx.mdl` everywhere else. Larger recordings are summarised
as numbers (bytes, ratio, error) committed in their place.

## Installing torch on this sandbox (and the trap that cost an hour)

**What works:** plain `pip install torch`. Nothing else. It pulls `cuda-toolkit` and every
`nvidia-*` runtime it needs, and `import torch` then succeeds. Measured 2026-11-12: 2.15.0
(+cu132, CPU execution), ~8 GB of disk, about 70 seconds.

**What fails, and why the earlier note here was wrong:**

| Attempt | Result |
|---|---|
| `pip install --no-deps torch` | installs, then `import torch` fails: `libcublasLt.so.*[0-9] not found` |
| `pip install` the nvidia list by hand | works only if `nvidia-nvshmem-cu13` is in the list; it is easy to miss, and without it the import fails on `libnvshmem_host.so.3` |
| `--index-url https://download.pytorch.org/whl/cpu` | unreachable from here: TLS handshake closed (`SSLZeroReturnError`), no version list |
| plain `pip install torch` | **works** - `cuda-toolkit` brings the whole runtime set including nvshmem |

The first version of this note blamed the wheel and prescribed the manual nvidia list. The
real cause was `--no-deps`: with dependencies left to pip, the CUDA runtime arrives by itself.
Do not hand-install the nvidia packages; do not chase the version-conflict warnings torch's
pins produce against `setuptools` - neither affects CPU execution.

**Consequence for CI.** The GitHub runners fail to install `torch` at all (see the run records:
`pip install -e ".[test]"` with the optional torch extra is slow enough to hit the job limit,
and the wheel is large). So `Tests/integration/test_torch_adapter.py` starts with
`pytest.importorskip("torch")`: on a runner without torch it skips **and says why**, and the
93 other tests still prove the container, the CLI and the crash path. Locally, with torch
installed, the same suite is 99 tests and every adapter test runs for real.

**Cost, so it is not a surprise:** torch plus its CUDA runtimes take 8 GB of a 20 GB disk. If
space matters more than the adapter tests, skip installing torch and read the six adapter
tests as "verified on Linux with torch 2.15.0" rather than "verified everywhere".

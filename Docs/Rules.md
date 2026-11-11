# Rules

Binding rules for this repository. Read this file before making any change.
If a rule blocks the work, stop and ask; do not improvise around it.

---

## 1. Readability and structure (highest priority)

1. Every file, folder and document is clearly separated and clearly named.
2. A file holds all the code for its own functionality and nothing else.
3. Files are medium-sized: no huge files, and none so small that they are noise.
   Split code by responsibility.
4. A folder contains only what belongs to its responsibility, and has a short README.
5. Every file starts with a one-line comment (or docstring) saying what it is for.
   Empty files too. `Docs/` prose files start with an H1 title and a purpose line.
6. `internels/` holds anything that is not part of the public product: helper
   material, drafts, skills. A skill is one markdown file in `internels/skills/`,
   added only when it is actually worth having.

## 2. Process

7. Nothing new starts until the current thing is fully built and tested, or its
   bug is fixed. Nothing is left half-done. The only allowed exception is two
   pieces that cannot work without each other.
8. Each roadmap step is finished completely before the next one starts.
9. Before writing code for a file: list what it will use, look up every library
   and API it touches so the code matches current versions and correct usage,
   record the findings in `Docs/Research.json`, then update the other docs.
10. Code is clean: no filler comments, comments only where they earn their place,
    and modular enough that adding, changing or removing one thing does not ripple
    across many files.
11. After each round of work, update `Docs/Thinking.md`. Keep `Docs/Files_Info.json`,
    `Docs/Technical.md`, `Docs/Logic.md` and `Docs/Changelog.md` current. A new
    request in `Docs/Change.md` also updates `Docs/Roadmap.md`.

## 3. No guessing

12. Do not guess and do not assume. Where something is unclear, ask.
13. If the question is whether an implementation is better, you are not allowed to
    have an opinion without a test: run the test and report the measurement.

## 4. Testing

14. Test enough to matter, not everything: every test must cover one important
    behaviour ("do not test from over the code").
15. Unit tests cover one file. Run them right after creating that file, and re-run
    them only when that file changes, never for untouched files.
16. Integration tests cover two or more files working together. Run them whenever
    a file connected to the main system is added, removed or changed.
17. Every test runs on all three GitHub runners: Windows, Linux, macOS.
18. Every test run is recorded in `Tests/Records/<version>/`: which exact test ran,
    what inputs it got, what it output. Keep the current version plus the last two.

## 5. GitHub Actions (limited-token workaround)

19. Everything a workflow produces (logs, outputs, downloads, artifacts) is
    committed back into the repository. Do not use Actions artifacts, and do not
    rely on the Actions API: the token here can read runs and workflows but cannot
    read logs, artifacts, secrets or variables.
20. A run is triggered by modifying a trigger file in the repo, not by hand.
21. Inputs and outputs stay clean and readable in their own directory (`Actions/`).
22. Use Actions for work that is heavy (RAM, CPU, time) or impossible in this
    sandbox. Do small work locally. Local sandbox: 2 vCPU, ~3.5 GiB RAM.
23. At most 5 runners in any configuration (1 job x 5 matrix, 5 jobs x 1, ...).
24. When a test needs an API key or a site this sandbox cannot reach: tell the user
    the secret name, the user sets it in repository settings, and the exact name
    plus what it is for is recorded in `Docs/Secrets.md`.

## 6. Environment facts that shape the rules

25. This sandbox has no Rust toolchain and cannot reach `static.rust-lang.org`,
    `sh.rustup.rs`, `crates.io`, `index.crates.io`, `static.crates.io`, `docs.rs`,
    `raw.githubusercontent.com` or general websites. It can reach GitHub,
    `api.github.com`, `codeload.github.com`, PyPI and npm.
26. Therefore: code that must build here is Python. Anything that cannot be built
    or tested here belongs in an Actions workflow (see section 5).

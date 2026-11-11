# Tests folder

> Folder-level README. `Tests/` holds test code and the permanent record of every test
> run.

---

## Layout

```
Tests/
  unit/                 one file's behaviour in isolation
  integration/          two or more files working together
  bench/                measured experiments (storage, overhead, error)
  Records/
    README.md           naming and retention rules
    vX.Y.Z/             one folder per released version (current + last two)
```

## Rules (from `Docs/Rules.md`)

- A unit test runs right after the file it covers is created; re-run only when that
  file changes.
- Integration tests run whenever a file joined to the main system is added, removed or
  changed.
- Every test runs on Linux, Windows and macOS through GitHub Actions.
- Every run is recorded: exact test id, inputs, outputs, environment.

## What does not belong here

- Recordings and large binaries -> `Actions/out/` (tiny) or not committed (large).
- CI workflow definitions -> `.github/workflows/`.

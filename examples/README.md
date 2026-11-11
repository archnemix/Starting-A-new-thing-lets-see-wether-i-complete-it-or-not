# examples

> Folder-level README. Small, runnable examples of using archx. Empty until there is an
> API to demonstrate.

---

## What belongs here

- One folder per example: `examples/<name>/` with a `README.md` and a `run.py`.
- Each example must run on CPU in well under a minute and record into a file it creates
  next to itself (ignored by git).

## What does not belong here

- Datasets or recordings (they are generated, not stored).
- Benchmarks that measure something -> `Tests/bench/`.

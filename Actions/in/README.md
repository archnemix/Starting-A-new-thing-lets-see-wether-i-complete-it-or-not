# Actions: inputs

> Folder-level README. Files a workflow reads: fixtures, config files, matrices.

---

## What belongs here

- `*.json` matrices (e.g. the test matrix a run must execute).
- Tiny binary fixtures used by tests (a few hundred KiB at most).
- Config files the workflow passes to the code.

## Rules

- Files here are inputs, never overwritten by a run.
- Anything reproducible in code does not belong here; only data does.
- Keep it small: this is committed on every run that changes it.

## Current inputs

None yet. `Actions/scripts/run_tests.py` takes only a label argument.

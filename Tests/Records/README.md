# Test records

> Folder-level README. One folder per released version. A record is evidence: what ran, on
> what, with what inputs and outputs.

---

## What a version folder contains

```
Tests/Records/v0.1.0/
  <date>_<os>_<python>_<suite>.json     machine record, copied from the run
  <date>_<os>_<python>_<suite>.log      raw output of that run
```

Records are produced by `Actions/scripts/run_tests.py` into `Actions/out/test-runs/`
(that is the live copy), then snapshotted here when a version is released, so the record
survives the next run overwriting the live file.

## Retention

Keep the current version and the two before it. Older folders are removed in the same
commit that adds a new one - never left to rot.

## Record shape

```json
{
  "purpose": "test run record...",
  "label": "Linux-py3.11",
  "command": "python -m pytest Tests -q --tb=short",
  "exit_code": 0,
  "summary_line": "23 passed in 0.31s",
  "started_utc": "...",
  "duration_s": 0.4,
  "runner": {"os": "...", "github_os": "Linux", "python": "3.11.9", "github_sha": "..."},
  "stdout_tail": ["..."],
  "stderr_tail": []
}
```

## Honest gap

`Tests/Records/` has no version folder yet: v0.1.0 is not released, so runs are live in
`Actions/out/test-runs/`. The snapshot happens at the release commit.

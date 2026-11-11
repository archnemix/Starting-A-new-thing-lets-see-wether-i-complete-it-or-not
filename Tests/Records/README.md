# Test records

> Folder-level README. One folder per released version. A record is evidence: what ran,
> on what, with what result.

---

## Naming

```
Tests/Records/v0.2.0/
  2026-11-11T14-05-00_linux_py311_container-writer.json    <- machine record
  2026-11-11T14-05-00_linux_py311_container-writer.log     <- raw output
```

One record per test *run*, not per test case: a record contains every case that ran in
that invocation, with its inputs and outputs.

## Retention

Keep the current version and the two before it (rule 18). Older version folders are
deleted in the same commit that adds a new one - never left to rot.

Within a version, a newer record for the same test and runner replaces the older one
(the record's own `superseded` list names what it replaced, so history is not silent).

## Record shape

```json
{
  "test_id": "container-writer/roundtrip",
  "kind": "unit",
  "version": "0.2.0",
  "runner": {"os": "linux", "python": "3.11.9", "torch": "2.15.0"},
  "commit": "<sha>",
  "started_utc": "...",
  "duration_s": 1.23,
  "cases": [{"name": "...", "inputs": {}, "expected": "...", "actual": "...", "pass": true}],
  "artifacts": ["Actions/out/<label>/..."],
  "notes": ""
}
```

## What does not belong here

- Passing/failing *status* only, with no inputs or outputs - that is not a record.

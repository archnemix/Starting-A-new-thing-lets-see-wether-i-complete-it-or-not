# Actions: outputs

> Folder-level README. Everything a workflow produced, committed back into the repo.

---

## Layout

```
out/<label>/
```

`<label>` is chosen by the workflow and is stable and readable, e.g.
`latest-tests`, `latest-bench`, `release-0.3.0`. A latest-* folder is replaced wholesale
on each run (its history lives in git, not in duplicate folders).

## Contents of a label folder

| File | Purpose |
|---|---|
| `summary.md` | human summary: what ran, pass/fail counts, key numbers |
| `summary.json` | same, machine-readable, consumed by later steps |
| `*.log` | raw output of each job |
| `artifacts/` | produced files (tiny ones only) |
| `commands.txt` | exact commands, environment and versions |

## Rules

- Never rely on the Actions UI: if it is not in this folder, it did not happen.
- Do not commit secrets, tokens, or user data - hashes and counts only.
- Keep total size small (tens of MiB at most); binaries are pruned in favour of numbers.

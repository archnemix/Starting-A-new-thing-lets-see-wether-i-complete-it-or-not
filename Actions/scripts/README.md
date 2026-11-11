# Actions: scripts

> Folder-level README. Small scripts the workflows call. Kept out of the workflow YAML so
> they can be tested and read on their own.

---

## Files

| File | Used by | What it does |
|---|---|---|
| `env_probe.py` | `tests.yml` | writes one JSON per runner: OS, Python, CPU, versions - the first proof that all three runners work and can commit results |
| `commit_results.sh` | `tests.yml` | commits whatever landed in `Actions/out/`, with a rebase retry so parallel runners do not lose each other's files |

## Rules

- A script does one thing, takes its paths as arguments, and exits non-zero on failure.
- No secrets are ever read here; secrets are referenced by name only in `Docs/Secrets.md`.
- Anything that grows beyond a screen belongs in the package, not here.

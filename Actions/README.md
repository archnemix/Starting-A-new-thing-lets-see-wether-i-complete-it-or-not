# Actions folder

> Folder-level README. Clean input/output space for GitHub Actions runs. Everything a
> workflow produces lands here as a commit, because logs and artifacts are not readable
> from this sandbox.

---

## Layout

```
Actions/
  README.md
  triggers/       one tiny file per workflow; editing it starts that workflow
  in/             input files a workflow reads (fixtures, configs, matrices)
  out/            everything a run produces, one folder per label
    README.md
```

## Rules

- No Actions artifacts. No reliance on the Actions API for results.
- `out/<label>/` holds: `summary.json`, `summary.md`, raw logs, produced recordings,
  and the exact command lines used.
- A run never writes outside its own `out/<label>/` folder.
- Binary outputs are kept only when tiny; otherwise the numbers are kept and the file is
  dropped (see `Docs/Workaround.md` W6).
- Commit messages from workflows are prefixed `ci:` so they are easy to filter.

## What does not belong here

- Workflow YAML -> `.github/workflows/`.
- Test source -> `Tests/`.

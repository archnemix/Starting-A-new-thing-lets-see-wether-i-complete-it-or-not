# Actions: trigger files

> Folder-level README. Editing a file in this folder is how a workflow is started, since
> runs cannot be dispatched reliably from here.

---

## How it works

Each workflow watches exactly one file in this folder:

```
triggers/tests.txt      -> .github/workflows/tests.yml
triggers/bench.txt      -> .github/workflows/bench.yml   (when it exists)
```

Change the file (even one character), commit and push, and the workflow runs. The file's
content is free text; the last line is echoed into the run log so the intent is visible.

## Rules

- One trigger file per workflow, no sharing.
- The trigger file must never be read for configuration - configs live in `Actions/in/`.
- Commits made by the workflow itself use `GITHUB_TOKEN`, which does not re-trigger
  workflows; that is what keeps the log-commit loop from recursing.

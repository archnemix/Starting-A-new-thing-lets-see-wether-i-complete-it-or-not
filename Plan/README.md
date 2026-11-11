# Plan folder

> Folder-level README. `Plan/` holds what we intend to build and the exact decisions
> behind it, before any code exists. It is the source the roadmap and the code are
> derived from.

---

## What belongs here

- `Architecture.md` - the decided design and its non-goals.
- `Roadmap.md` - ordered steps; each finished before the next starts.
- `Format_Spec.md` - the normative `.archx.mdl` container format, versioned.
- `Open_Questions.md` - decisions that are the user's to make.

## What does not belong here

- Reasoning history -> `Docs/Thinking.md`.
- Measured results -> `Docs/Research.json` and `Tests/Records/`.
- Code -> the package directory.

## Rule

A plan change is written here first, then reflected in `Roadmap.md`, then built.

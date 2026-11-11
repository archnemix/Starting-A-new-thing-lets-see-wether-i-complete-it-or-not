# Docs: Change

> Purpose: the inbox for anything the user asks to add or change later. Every entry
> here is reviewed, then either enters `Plan/Roadmap.md` as a step or is declined with
> a reason. Nothing here is silently dropped.

Format per entry:

```
## C<n> - <short title> - <date> - status
Requested: what the user asked for, in their words
Effect on roadmap: which step it becomes / which step it changes / not now, because...
Notes: constraints, dependencies, open questions
```

---

## C1 - Library must be usable from Python; language is our call - 2026-11-11 - accepted

Requested: "the library should be made for python not for rust ... i do not care what
language it is made in but it should work in python".

Effect on roadmap: rewrites Phase 0.3 (packaging) and removes the Rust/maturin build
from v1. A native accelerator is kept as an optional later step (Open_Questions Q4).

Notes: no Rust toolchain is installable in this sandbox anyway, so a Rust core could
never have been built or tested here.

## C2 - Cross-file base reference ("model B = model A + delta") - 2026-11-11 - deferred to v2, fields reserved

Requested: the origin chat calls storing only what changed from a base model the single
most valuable format feature.

Effect on roadmap: not a v1 step. v1 reserves `META.base_model` / `META.ancestor` and the
first-anchor-optional rule so v1 files can be composed later without a format major bump.
Real composition becomes a v2 roadmap step. See Q18.

## C3 - Optionally store the training data itself - 2026-11-11 - awaiting decision

Requested: the origin chat says the data behind a weight change must be stored so a human
or AI can see what caused it. The approved brief says fingerprints only.

Effect on roadmap: adds `DATA_BLOB` (reserved id 17) as an opt-in, off-by-default section
plus a `recording.data_store` option, if approved (Q17). Fingerprints are unaffected.

## C4 - Make the origin chat local-only - 2026-11-11 - accepted

Requested: keep the chat that produced the idea, but never on GitHub, not even in a deeply
nested folder.

Effect on roadmap: none. Applied immediately: file lives in `internels/chat/`, excluded by
`.gitignore`, rule 25, and the extracted conclusions live in `Plan/Origin_Chat.md`.

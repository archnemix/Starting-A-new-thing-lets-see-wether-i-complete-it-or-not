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

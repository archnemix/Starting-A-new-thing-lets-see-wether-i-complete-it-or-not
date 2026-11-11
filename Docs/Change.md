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

## C3 - Store the training data itself - 2026-11-11 - accepted, on by default

Requested: the origin chat says the data behind a weight change must be stored so a human
or AI can see what caused it. The approved brief says fingerprints only.

Effect on roadmap: new steps 2.7-2.8 (codec benchmarks) and 3.6-3.7 (capture + conversion
evidence). Format gains `DATA_SHARD` (20), `DATA_SEGMENT` (17), `DATA_POINTER` (21) and
`data_level` in META. Compression effort shifts to float deltas, since data is stored once
and deltas repeat.

## C4 - Make the origin chat local-only - 2026-11-11 - accepted

Requested: keep the chat that produced the idea, but never on GitHub, not even in a deeply
nested folder.

Effect on roadmap: none. Applied immediately: file lives in `internels/chat/`, excluded by
`.gitignore`, rule 25, and the extracted conclusions live in `Plan/Origin_Chat.md`.

## C5 - The chat outranks the generated brief - 2026-11-11 - accepted

Requested: "follow what the chat says more than what the prompt i sent you says because the
chat is the original source of truth", and the real starting point is the 19:51 turn, not
the earlier format discussion.

Effect on roadmap: the product definition is now the recording triple (state + cause +
stage, Architecture 6.0), data content is default-on, and every place where the brief
assumed otherwise is marked as superseded rather than quietly changed. See
`Plan/Origin_Chat.md` sections 3a and 3b.

## C6 - Naming: `archx.mdl` on PyPI, npm `archx` as a signpost - 2026-11-11 - accepted

Requested: no cap on stored data by default; `archx` is the user's brand on the npm registry
too; since the clean package name is taken, the extension is used as the disambiguator
because "extension is a must to represent the tool".

Effect on roadmap: step 0.3 pins the distribution name `archx.mdl` (import/CLI stay
`archx`), and step 0.5 adds the npm signpost package. Nothing else changes name: file
extension, import, CLI and docs all keep `archx`.

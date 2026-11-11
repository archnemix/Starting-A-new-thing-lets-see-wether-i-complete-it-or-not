# Archnemix Model Life Line (`.archx.mdl`)

A recorder for neural network training. It attaches to a training run that already
exists, records how the weights change over time, stores the whole journey in one
compact file, and lets you query it later.

Think "git for model training": a few full snapshots (anchors) plus compressed deltas,
instead of only the final weights.

Status: **design only, no code yet** (waiting on plan approval).

Where to start reading:

- `Plan/Architecture.md` - what we are building and how it is structured.
- `Plan/Open_Questions.md` - decisions that are still yours to make.
- `Docs/Rules.md` - the working rules for this repository.
- `Docs/Roadmap.md` - the order of work.

Folders:

- `Docs/` - rules, docs, research log, file map.
- `Plan/` - architecture and roadmap.
- `Tests/` - test records kept per version.
- `Actions/` - clean inputs/outputs for GitHub Actions runs.
- `internels/` - non-public helper material (drafts, skills).

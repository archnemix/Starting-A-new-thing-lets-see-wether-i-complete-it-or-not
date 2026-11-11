# Archnemix Model Life Line (`.archx.mdl`)

A recorder for neural network training. It attaches to a training run that already
exists and records three things together: **how the weights changed**, **which data caused
it**, and **at which stage**. The whole journey goes into one compact, crash-safe file that
humans and other AI systems can query afterwards.

Think "git for model training": a few full snapshots (anchors) plus compressed deltas,
instead of only the final weights.

Status: **design only, no code yet** (waiting on plan approval).

Where to start reading:

- `Plan/Architecture.md` - what we are building and how it is structured.
- `Plan/Open_Questions.md` - decisions that are still yours to make.
- `Docs/Rules.md` - the working rules for this repository.
- `Plan/Roadmap.md` - the order of work.
- `Plan/Origin_Chat.md` - what the original conversation asked for.

Folders:

- `Docs/` - rules, docs, research log, file map.
- `Plan/` - architecture and roadmap.
- `Tests/` - test records kept per version.
- `Actions/` - clean inputs/outputs for GitHub Actions runs.
- `internels/` - non-public helper material (drafts, skills).

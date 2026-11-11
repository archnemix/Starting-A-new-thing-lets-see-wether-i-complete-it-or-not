# Plan: Origin Chat - what the original conversation adds

> Purpose: the durable extraction of the user's origin chat
> (`internels/chat/2026-11-10-to-11-11_origin-chat.md`, local-only), so the reasoning
> survives even though the raw export never leaves this machine.

Source: chat export "Safetensors vs GGUF model formats", 2026-11-10 20:55 to
2026-11-11 20:28. Section references below are the user turns, by timestamp.

---

## 1. How the idea actually formed (the lineage)

| Turn | What the user asked | What it contributed to the product |
|---|---|---|
| 11-10 20:55 | difference between safetensors / gguf / pt / onnx | context: formats exist to solve one pain each; ours must name its pain |
| 11-10 20:58 | "a computer directly inside an AI model" | not in the product, but sets the interest: models as artifacts, not black boxes |
| 11-10 21:00 | why 33k tok/s is possible, 2D heads, T4 training | attention to *cost per step*; the product must be cheap or it is useless |
| 11-10 21:04 | LLM for Pygame game generation | scope discipline: a reduced scope is still hard, data and verification dominate |
| 11-11 19:39 | "if we made a format, what would we focus on" | the format wishlist (section 2) |
| 11-11 19:51 | **the pivot**: store the journey, not the destination | the product. "git for AI models", weight-space programming, other AIs reading it |
| 11-11 19:52 | which weights handle which capabilities | the data -> weight -> capability map ("weight atlas") |
| 11-11 20:00 | it will be terabytes | storage must be compressed and efficient, or the product does not exist |
| 11-11 20:04 | tiered detail + store only what changed | phase-adaptive policy + delta encoding + numerical-aware compression |
| 11-11 20:11 | the rules prompt, and the closing paragraph | `.archx.mdl`, the brand, and the extra requirements in section 3 |

## 2. The format wishlist from 19:39 (what the user cares about in a format)

Ranked as the user framed it, with the v1 answer:

| Wish | v1 status |
|---|---|
| single self-contained file, everything included | **in** (META + registry + blocks) |
| index at the start, load any tensor without reading the file | **in** (per-block offsets, CRC) |
| delta/diff natively, "this is base X plus these changes" | **partial** - deltas within a run: yes. Cross-file "model B = model A + delta": not yet (see 4.2) |
| per-tensor flexible quantization | **in** for deltas (per-tensor encoding field) |
| streaming-first, answer "what is this" from the first KB | **in** (fixed header + META block first) |
| multimodal as named components | **in** (component field per tensor, per-component stats) |
| provenance and audit trail | **in** (META: adapter, versions, capability tier, settings) |
| safety: no executable content by design, validate before allocating | **in** (no pickle, length caps, CRCs) |

The chat called **delta compression with a base-model reference** "the single most
important focus". That is the one wish v1 only half-serves, and it is cheap to reserve
now.

## 3. Requirements in the closing paragraph that the brief left out

The brief (section 6) says: fingerprints only, *never the data itself*. The origin chat
says the opposite in one sentence:

> "since we woudl also be storing the trianign data so the ai or human could understand
> that this set of data implemented this set of change"

Three other things appear only in the chat:

1. **The file is also meant to be an inference/storage format eventually** (the whole
   conversation starts from safetensors/gguf; GGUF export is a listed post-v1 item).
   Consequence: the final anchor should be usable as a plain weights snapshot, without
   the history machinery. It already is - recorded here so it stays true.
2. **Parts of the file should be independently usable** ("stored as parts of the file so
   each part can have a different thing"). Consequence: per-block compression and
   per-section readability, which the format already has.
3. **Other AI systems are the intended reader**, not only humans. Consequence: machine
   readable metadata and a stable schema matter more than pretty output; the CLI prints
   `--json`.

## 4. Where the chat and the approved brief disagree (needs a decision)

### 4.1 Training data: fingerprints only, or store the data?

- Brief: "Data fingerprints: hash, token count, characteristics per segment. **Never the
  data itself.**"
- Chat: store the data so a human or AI can see which data caused which change.

Recommendation: keep the brief's rule as the **default**, and make data storage an
**opt-in section type** (`DATA_BLOB`, reserved, off by default, with the fingerprint
always present). That satisfies both without changing the byte budget: a file with
`DATA_BLOB` sections is bigger and says so in its META. **Asked as Q17.**

### 4.2 Cross-file base reference (the chat's most-wanted feature)

"Model B = model A + this delta" is a *different* feature from "this run's history", and
it is the one the user called most valuable. It is nearly free to reserve:

- META gains `base_model` (fingerprint + tensor registry hash of a parent file) and
  `ancestor` (path or content hash).
- The first anchor becomes optional when a valid base reference exists: a file can then
  be a pure delta against a parent, which is exactly how the ecosystem stores fine-tunes
  today, but with per-tensor encodings.

Not v1 scope, but the fields ship in v1 so a v1 file can later be composed. **Asked as
Q18**: include the base-reference *fields* in v1 (recommended), or defer entirely.

### 4.3 Weight atlas / edit vectors / query interface

The chat's system diagram ends at `/weight_atlas`, `/capability_map`, `/data_signatures`,
`/edit_vectors` and a query interface. The brief defers all of it. No conflict in
sequencing, but it constrains the **recording** side today: per-region and per-component
statistics must exist from v1 so an atlas can be built from v1 files later without
re-training. That is already true in the design and is now an explicit reason for it.

## 5. Equivalence worth recording (so it is never re-argued)

The chat's "Strategy 1" (accumulate tiny deltas, flush when the significance threshold is
crossed) and the brief's phase-adaptive policy look different but store the **same**
information: a delta flushed at a trigger point is exactly `state(trigger) - state(last
stored)`, which is the sum of every skipped step's delta. The only difference is *when
the tensor is copied*. Copying every step and accumulating costs more and stores nothing
more, unless per-step intermediate states are wanted.

Decision: keep copying **only at store points** (cheaper), and document that the stored
delta is the exact accumulated movement, not a single step's change. This is already what
Architecture 6.2 means; the wording is now explicit.

## 6. Numbers in the chat that are hypotheses, not findings

The chat quotes rank-8 low-rank "~244x", "20-100x from thresholding", "60-250x on top".
Those are naive arithmetic on a 4096x4096 matrix, with no measured reconstruction error.
They are **not** used anywhere in the design. Roadmap Phase 2 measures the real values,
per architecture, and `Docs/Research.json` records them then.

## 7. What changes because of this document

- `Plan/Architecture.md`: origin-chat section, base-reference reservation, data-blob
  opt-in note, "final anchor doubles as a model snapshot" property.
- `Plan/Format_Spec.md`: reserved block ids (`DATA_BLOB`, `ATLAS`, `EDIT_VECTOR`), META
  fields `base_model` / `ancestor`, and the delta wording fix.
- `Docs/Change.md`: C2 (base reference), C3 (training-data opt-in).
- `Plan/Open_Questions.md`: Q17, Q18.
- `Docs/Rules.md` rule 25: the origin chat is local-only, never pushed.

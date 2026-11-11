# Plan: Origin Chat - what the original conversation adds

> Purpose: the durable extraction of the user's origin chat
> (`internels/chat/2026-11-10-to-11-11_origin-chat.md`, local-only), so the reasoning
> survives even though the raw export never leaves this machine.

Source: chat export "Safetensors vs GGUF model formats", 2026-11-10 20:55 to
2026-11-11 20:28. Section references below are the user turns, by timestamp.

**Priority:** this chat outranks the generated brief wherever they disagree (user
instruction, 2026-11-11). The real starting point is **11-11 19:51**; turns before it are
the road that led there and explain where the brief's assumptions came from.

**Order to read, if short on time:** section 1 (lineage) -> 3a (the main-thing turn) ->
3b (the four later refinements, which are the real storage spec) -> 4 (where the brief
disagrees) -> 6 (numbers not to trust).

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

## 3a. The main-thing turn (11-11 19:51) - verbatim and unpacked

> "What if we build rather than a format a way to store model durign trainig ... after
> traingi the mdoel is stored at its final point no wya of know wha tits shape was before
> ... design a system such that durign trinanig it stores information such as whn this set
> of data was introduces it changesd like this its state changed like this with different
> level of detials now what this woudl solve or brign is that the ability to udnerstand how
> the model changes ists states based onteh data adn different states of model liek a git
> but for ai models and can later the new emerging concept of weight space programming ti
> can help other ai nderstand hwat kidn of input effect it how and let them give a direction
> on how they can do what htey can change manaulyl withotu traingn to make it better towards
> something"

Unpacked into requirements, in the order the sentence gives them:

| Phrase | Requirement |
|---|---|
| "rather than a format, a system that stores the model during training" | the product is a *recorder*, not a file spec; the format is its container |
| "after training the model is stored at its final point, no way of knowing what its shape was before" | anchors + history are mandatory; final weights alone are the thing we replace |
| "when this set of data was introduced, it changed like this, its state changed like this" | the **data -> state-change join**, per step, is the core record (Architecture 6.0) |
| "with different levels of detail" | phase-adaptive policy (6.2) - detail varies by stage and by event |
| "understand how the model changes its states based on the data" | the readable output is a *change explanation*, not a weight dump |
| "like a git but for AI models" | reconstruct any point, diff two points, annotate events |
| "weight space programming ... other AI understand what kind of input affects it how" | machine-readable metadata; other models are first-class readers |
| "give a direction ... change manually without training to make it better" | the post-v1 goal the format must not foreclose (6.10) |

The assistant's reply to this turn supplied the first architecture: a four-level capture
scheme (macro/meso/micro/semantic), the git analogy **and** its limits (weights are
continuous, attribution is probabilistic, scale is gigabytes), and the three hard problems
(storage scale, causal attribution, semantic meaning). Those limits are why this project
records *observational* evidence and says so, instead of claiming causation.

## 3b. The four refinements that followed (the real storage spec)

These turns came after the main idea and are the user's own design decisions, so they
outrank anything the brief assumed:

| Turn | User's words (condensed) | Requirement |
|---|---|---|
| 19:52 | "this set of training data ... modified these weights ... which would help other AI understand what set of weights are handling what things and how we can add new data correctly" | per-region effects and a reusable map; "add new data correctly" is the practical use case |
| 20:00 | "it can easily become for a medium model gigabytes to even terabytes ... steps can be thousands to millions, epochs differ" | storage must be sub-linear in steps and epochs; dedup is mandatory, not optional |
| 20:04 | "store the first initial changes with more detail, then when the threshold passes, then in detail for the middle, then threshold, then the end"; "rather than storing the whole weights store what changed like git"; "compression algorithms that heavily do textual data ... these are numerical" | phase-adaptive detail; delta storage; numerical-aware compression. This is 6.2-6.4 |
| 20:11 | the rules prompt, then: "we would also be storing the training data so the AI or human could understand that this set of data implemented this set of change ... stored as parts of the file so each part can have a different thing" | **data is stored** (overriding the brief), and **per-part codecs** (each block chooses its own) |

The assistant's reply to 20:04 added the concrete tooling that this project should test
rather than assume: sparsify-then-compress (index gaps are compressible), low-rank delta
factors, fp16/int8 delta quantization, zstd with numerical preprocessing, fingerprints +
characteristics for data, shared signatures across models, and event-driven storage instead
of step-driven. All of it is candidate machinery in Phase 2; none of it is adopted as fact.

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

### 4.1 Training data: fingerprints only, or store the data? - RESOLVED

- Brief: "Data fingerprints: hash, token count, characteristics per segment. **Never the
  data itself.**"
- Chat: store the data so a human or AI can see which data caused which change.
- **User decision (2026-11-11): store it.** On by default, opt-out possible, cost optimised.
  The brief's sentence is treated as an artefact of the earlier format-only discussion.

Implemented as `data_level` (`none | pointers | segments | full`, default `segments`):
deduplicated content-addressed shards, per-shard codecs, explicit and recorded opt-out.
See Architecture 6.8. The byte budget changes because the user chose value over size, and
compression effort moved to where it matters (deltas).

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

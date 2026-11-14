# Plan: Format Spec (draft v1, `format 1.0`)

> Purpose: the normative description of the `.archx.mdl` container. If the code and
> this file disagree, the file is wrong - unless this spec says otherwise.

Status: **draft**. Frozen by Roadmap step 1.1. Numbers marked *(tune)* are policy, not
format, and may change per file; numbers marked *(fixed)* are format and cannot change
within major version 1.

Conventions: all integers are little-endian; all strings are UTF-8, length-prefixed
with a `u32`; all CRCs are CRC-32 (ISO-HDLC, the zlib polynomial); sizes are in bytes.

---

## 1. File shape *(fixed)*

```
offset 0
  +----------------------------+
  | header (fixed 40 bytes)    |
  +----------------------------+
  | block                      |   <- zero or more, appended in order
  | block                      |
  | ...                        |
  +----------------------------+
  | footer (fixed 56 bytes)    |   <- present only in a cleanly closed file
  +----------------------------+
```

A file without a footer is a *crashed* file: it is still readable, and `archx recover`
rebuilds the index from the blocks that survived.

## 2. Header *(fixed, 40 bytes)*

| off | size | field | value |
|---|---|---|---|
| 0 | 8 | magic | ASCII `ARCHXMDL` |
| 8 | 2 | format_major | `1` |
| 10 | 2 | format_minor | `0` |
| 12 | 4 | flags | bit0 closed cleanly, bit1 parameters complete, bit2 recovered, bit3 sharded source |
| 16 | 4 | header_len | `40` |
| 20 | 4 | header_crc | CRC-32 of bytes 0..19 |
| 24 | 8 | created_unix_us | microseconds since epoch |
| 32 | 4 | block_count_hint | 0 if unknown (hint only, never trusted) |
| 36 | 4 | reserved | 0 |

Reader rule: refuse if `format_major != 1`; accept any `format_minor >= 0` but skip
unknown block types; refuse if the header CRC fails.

## 3. Block *(fixed 28-byte prefix + payload)*

| off | size | field | notes |
|---|---|---|---|
| 0 | 2 | block_type | see section 4 |
| 2 | 1 | codec | 0 none, 1 zstd, 2 zlib, 3 lz4 *(fixed ids; new ids only in a minor bump)* |
| 3 | 1 | flags | 0 unless the type defines them |
| 4 | 8 | step | the training step this block belongs to; `0xFFFFFFFFFFFFFFFF` = not step-bound |
| 12 | 4 | payload_raw_len | length before compression, for allocation sanity |
| 16 | 8 | payload_len | length after compression |
| 24 | 4 | payload_crc | CRC-32 of the stored (compressed) payload |
| 28 | 4 | block_crc | CRC-32 of bytes 0..23 *(guards the prefix itself)* |

**Why the step is in the prefix (v1.0, changed during implementation step 1.2):** recovery
must rebuild the index from a crashed file, and the step otherwise only exists *inside the
compressed payload*. Putting it in the prefix costs 8 bytes per block and makes
`archx recover` a pure prefix scan, with no decompression and no guessing. It also means the
reader can list the timeline of a file without decoding a single delta.

Reader rules, all mandatory:
1. `block_crc` must match before the payload length is used.
2. `payload_len` must be `<=` the remaining file size, and `payload_raw_len` must be
   `<=` a configurable cap (default 2 GiB) - this is the anti-DoS rule.
3. `payload_crc` is verified before decoding.
4. An unknown `block_type` is skipped using `payload_len`; it is never an error.
5. A bad `codec` id is an error, not a skip (we cannot know what it was).

## 4. Block types

| id | name | payload | purpose |
|---|---|---|---|
| 1 | `META` | JSON | run identity: library version, adapter, framework version, model class name, tier, recorder settings, capability map (what was unavailable and why) |
| 2 | `TENSOR_REGISTRY` | binary + JSON names | every tracked tensor: name, dtype, shape, component, region, frozen flag, tie group |
| 3 | `ANCHOR` | binary | full values for all tracked tensors at a step |
| 4 | `DELTA` | binary | per-tensor encoded delta for a step, codec per tensor |
| 5 | `EVENT` | JSON | timeline entry: step, kind, values (loss, lr, grad norms, timings) |
| 6 | `DATA` | JSON | data segment fingerprint: hash, counts, characteristics - never the data |
| 7 | `ANNOT` | JSON | user annotation: step, event name, detail |
| 8 | `INFO` | JSON | out-of-band notes, warnings, degradations |
| 9 | `INDEX` | binary | block offsets by step (see section 6) |
| 16 | `INDEX_FOOTER` | binary | optional compact index written just before the footer |
| 17 | `DATA_SEGMENT` | JSON + binary | a segment of training data: metadata, characteristics, step span, region effects (6.9) |
| 20 | `DATA_SHARD` | binary | one deduplicated, compressed unit of data content referenced by segments |
| 18 | `ATLAS` | binary | **reserved**: data -> weight -> capability map (post-v1) |
| 19 | `EDIT_VECTOR` | binary | **reserved**: direction in weight space per capability (post-v1) |

Blocks 1-8 are version 1; new types are added as integers, never by changing these.
| 21 | `DATA_POINTER` | JSON | content hash + source reference instead of content (level `pointers`, or a `sensitive` shard) |
| 23 | `ACTIVATION` | JSON | per-layer output statistics for a stored step (tier 3) |
| 24 | `EXAMPLE_GRADIENTS` | NumPy NPZ, container-compressed | optional per-example local loss gradients and caller-supplied example bytes for one optimizer step |
| 25 | `EXAMPLE_GRADIENTS_SHUFFLED` | NumPy NPZ, container-compressed | reversible byte-plane shuffle and packed example bytes, same semantics as 24 |

`EXAMPLE_GRADIENTS` (24) carries an NPZ archive without pickle. Its `metadata` is
UTF-8 JSON in a uint8 array: version, ordered tensor names, base64 example bytes,
loss parts, raw size and gradient-sum error. Arrays `g0`, `g1`, etc. are float32,
indexed first by example and then by the corresponding named tensor dimensions.
The block step names the optimizer update; values are local derivatives of the
provided loss parts, not causal influence or individually measured optimizer updates.
The enclosing block codec compresses the archive. Absence of both block 24 and 25
for a step means the example trace was not recorded, not that its gradients were zero.

`EXAMPLE_GRADIENTS_SHUFFLED` (25) is an independently decodable version-2 NPZ.
It contains packed `gN` uint8 arrays and an `examples` byte array; JSON metadata
stores the original dtype, shape, representation and each example's length. The
`adaptive_float32_v1` representation chooses raw, byte-shuffled, or sparse per
tensor. Sparse stores little-endian uint32 index gaps followed by the original
4-byte nonzero values; negative zero is treated as a nonzero bit pattern. Decoding
and splitting recover all original bits and bytes. The writer compresses both 24 and
25 and stores the smaller; it may use codec NONE when compression expands the
payload. Readers of this build support both. Older readers may skip block 25 and
therefore cannot see every trace in a new file; upgrade before analyzing it.

Reserved ids are allocated so that a file written by a future version is skipped, not
misread, by a v1 reader. `DATA_SEGMENT` (17) supersedes the previously reserved
`DATA_BLOB`; 17 keeps its meaning of "training data payload" so nothing is ambiguous.

### 4.1 Required `META` fields (v1)

```json
{
  "format": "1.0",
  "library": "archx <version>",
  "adapter": "torch | hf | lightning | launcher",
  "framework": {"name": "torch", "version": "..."},
  "model_class": "...",
  "capability_tier": 0,
  "capabilities_missing": {"grad_norms": "reason it was unavailable"},
  "recorder_settings": {"policy": "...", "region_tags": {}, "compression": "..."},
  "components": ["vision_encoder", "language_model"],
  "base_model": null,
  "ancestor": null,
  "attribution": "observational",
  "data_level": "segments",
  "data_codecs": {"text": "zstd-dict", "tokens": "varint-delta-zstd", "images": "lossless-webp"},
  "sensitive_shards": 0
}
```

`data_level` is one of `none | pointers | segments | full` (default `segments`). It must
agree with the blocks present: `segments`/`full` require at least one `DATA_SHARD`,
`pointers` requires `DATA_POINTER` blocks and no shards, `none` requires neither. A reader
that finds a mismatch reports a corrupt header rather than guessing.

`none` may still write `DATA_SEGMENT` blocks. It writes one row per unique shard with its
content hash in `fingerprint`, its characteristics, its step span and its region effects -
never the content, and never a `DATA_SHARD`. Three levels therefore answer the weight half of
the record completely and differ only in how much of the data half they can show; a reader
must not assume `data_level: none` means "no data blocks".

`segments` and `full` differ in one measurable way: `segments` deduplicates by content hash,
so an epoch over the same data costs pointers (`times_seen` > 1), and two *named* segments
over identical bytes share one shard. `full` stores one shard per observation, with no
`times_seen` above 1, and therefore preserves exact sample order and repetition. A shard in a
`full` file may carry a suffixed id (`<hash>-<n>`) so that two identical copies stay distinct;
the fingerprint of every segment is always the plain content hash. A v1 reader must key
shards by their `shard_id` field, not by recomputing the hash and assuming uniqueness.

`base_model` and `ancestor` are reserved-but-null in v1: a future version can make a file
a pure delta against a parent (Q18). `data_stored` must be `false` in v1 and must agree
with the presence or absence of `DATA_BLOB` blocks.

## 5. Payload encodings

### 5.1 `TENSOR_REGISTRY`

```
u32 tensor_count
per tensor:
  str name
  u8  dtype        (0 f32, 1 f16, 2 bf16, 3 f64, 4 i8, 5 i16, 6 i32, 7 i64, 8 bool)
  u32 ndim, then ndim * u64 dims
  u8  flags        (bit0 frozen, bit1 tied, bit2 sparse-friendly)
  str component    (may be empty)
  str region       (may be empty)
  u32 tie_group    (0 = none, otherwise shared id)
```

### 5.2 `DELTA`

```
u64 step
u32 entry_count
per entry:
  u32 tensor_id            (index into the registry)
  u8  encoding             (0 unchanged, 1 sparse, 2 dense_quant, 3 lowrank, 4 dense_raw)
  u8  value_dtype          (dtype the values are stored in)
  then encoding-specific body (below)
```

| encoding | body |
|---|---|
| 0 unchanged | nothing |
| 1 sparse | `u64 nnz`, then `nnz * u64 index` (flat, ascending), then `nnz * value_dtype` |
| 2 dense_quant | `f32 scale`, `i32 zero_point`, then `numel * value_dtype` values |
| 3 lowrank | `u16 rank`, `u32 rows`, `u32 cols`, then `U` (rows x rank) and `V` (rank x cols) in `value_dtype` |
| 4 dense_raw | `numel * value_dtype` values |

Reconstruction: `w_t = w_anchor + sum(delta_i)`, then compare against the recorded
float32 reference; the format itself never claims exactness beyond the dtype stored.

### 5.2.1 `DATA_SHARD` *(JSON header line, then the raw payload bytes)*

```
str  shard_id            (its content hash, so the id is the dedup key)
u8   codec               (0 raw, 1 zstd, 2 zstd-dict, 3 varint-delta-zstd, 4 lossless image)
u8   sensitive           (1 -> content omitted, only hash and characteristics remain)
u32  dict_id             (0 = no dictionary; otherwise an id in the run's dictionary table)
u64  raw_len
u64  stored_len
bytes payload            (empty when sensitive = 1)
u32  content_crc         (CRC-32 of the *uncompressed* content, so identity survives any codec)
f32  reconstruction_error (0.0 for lossless; measured value if a lossy codec was used)
```

The block payload is a JSON object on one line (`shard_id`, `codec`, `raw_len`,
`content_crc32`, `reconstruction_error`), then `\n`, then the stored bytes. The compression is
zstd, which keeps the content hash meaningful and lets a reader verify identity without
decompressing twice.

A shard is written once per run. Seeing the same content again never writes it again; the
segment that mentions it just points at the existing shard id.

### 5.2.2 `DATA_SEGMENT` *(actually shipped as UTF-8 JSON, not a packed layout)*

The earlier version of this section described a packed structure. The implementation writes a
JSON object, and the specification now says what the code does - a spec that disagrees with
the writer is worse than no spec, because it is the thing a second implementation would trust.

```json
{
  "segment_id": "data-1",
  "shard_id": "...",              // content hash, or "" when the shard is a DATA_POINTER
  "source": "...",                // path, URL or hub id; may be empty
  "fingerprint": "...",           // hash of the shard content
  "characteristics": {},          // language, domain, complexity, counts, dedup ratio
  "first_step": 0, "last_step": 0,
  "times_seen": 1,
  "region_effects": {},           // region -> {delta_norm, grad_norm_max, loss_delta}
  "component_mask": 0,
  "delta_reconstruction_error": 0.0,
  "span_segments": 1,             // how many batches share this effect measurement
  "span_steps": 1,                // how many steps it was measured across
  "effect_step": 3                // the stored step at which the movement was measured,
                                  // or -1 when this data has no measured movement
}
```

`span_segments` and `span_steps` exist because a batch observed on a *skipped* step has no
movement of its own: the weights did not change on that step. Such a batch is attributed with
everything observed since the previous store point, and these two fields say how many batches
and steps that was. Without them an empty `region_effects` is ambiguous - it could mean "this
batch caused nothing", which is a claim the recorder is not entitled to make.

`effect_step` closes the remaining gap: `span_steps` said how many steps a measurement
covered but never *which* recorded step it belonged to, so a batch seen on a skipped step was
attributed to movement the reader could not locate. `effect_step` is the store point that
followed the batch and is always a step present in an `EVENT` block. It is `-1` when no
movement was measured. It is an additive v1 change: a reader that does not know the field
still reads `first_step`/`last_step` as before.

A segment is **re-stated** rather than rewritten: the file is append-only, so a growing segment
is written again with its new `last_step`, and the last statement about an id is the true one.

### 5.2.2a `ACTIVATION` *(JSON)*

One block per layer per stored step. Not a tensor: a fixed-size summary, so the size of a
recording is a function of the layer count and the step count, and never of the batch size.

```json
{"step": 8, "layer": "2", "shape": [64, 64], "dtype": "float32", "forwards": 1,
 "count": 4096, "mean": -0.0191, "std": 0.2551, "min": -0.916, "max": 0.884,
 "zero_fraction": 0.0, "saturated_fraction": 0.0, "saturation_limit": 4.0, "sampled": 4096}
```

- `layer` is the module's name from `named_modules()`, so `"2"` in a `Sequential` and
  `"encoder.layer.0.attention"` in a transformer both read correctly.
- `forwards` says how many forward passes were folded into this summary. A step that runs the
  model twice is recorded as `2`, not as one of them silently winning.
- `count` may be smaller than the tensor: above 16,384 elements the reduction is
  **stride-sampled** across the whole tensor (never truncated from the start), and `sampled`
  records how many elements the moments are over.
- `zero_fraction` is the fraction of units exactly 0.0 (a dead ReLU);
  `saturated_fraction` is the fraction with `|x| >= saturation_limit`.
- `nonfinite` appears only when NaN/Inf was seen; those values are counted and excluded from
  the moments rather than poisoning them.
- Blocks are written after the step's `DELTA` and before its `EVENT`, so a reader that has
  reached the event has the whole chain: movement, then what the forward pass saw.

Tier 3 requires an `ACTIVATION` block behind the claim; a file that lists `activations` in
`capabilities_present` but holds none is corrupt.

### 5.2.3 `EVENT` *(JSON)*

```json
{"step": 3, "loss": 1.23, "grad_norm": 4.5, "lr": 0.01,
 "regions": {"body": 0.0235}, "segment_id": "data-4", "seconds": 1700000000.0}
```

Written for every **stored** step in adaptive mode. In full-history mode step 0 is an
initial anchor, not an update, so events start at step 1. Full-history events contain
`batch_status` and ordered `batches` references with a 128-bit BLAKE2b content hash, source, name,
format and caller-provided loss reduction. Structured batch v1 is a NumPy archive
containing a JSON tree and non-object arrays; it is stored as a normal data shard.
References may have no available content in pointer mode or after a data cap.
`loss`, `grad_norm` and `lr` are `null` when the adapter
could not see them - never a substitute value. `regions` holds per-region relative movement,
which is what the policy's event trigger and the segment join both use.

### 5.2.4 `ANNOT` *(JSON)*

```json
{"event": "stage_2", "detail": "unfroze the encoder"}
```

A user annotation: a stage boundary, an unfreeze, a curriculum change. Separate from `EVENT` on
purpose, so a user's note can never be mistaken for a measurement.

### 5.2.5 `INFO` *(JSON)*

Two kinds, both `{"event": ...}`:

```json
{"event": "capabilities", "capability_tier": 2, "capabilities": [...],
 "capabilities_missing": {"activations": "not provided by this adapter"}}
{"event": "recorder_fault", "error": "...", "steps_stored": 12}
{"event": "JOIN_INCOMPLETE", "data_level": "none",
 "cannot_answer": ["data_content", "data_characteristics_join", "segment_spans"],
 "degraded_from_cap": false,
 "detail": "the weights and deltas are complete; only the data side is partial"}
{"event": "unsynced_log", "segment_id": "data-9", "loss": 0.31,
 "detail": "log() was called with no optimizer step afterwards, so this data was observed
            but never trained on. It is stored with no region effects."}
```

`JOIN_INCOMPLETE` is written for `none` and `pointers` (and for any run whose byte cap
degraded it to `pointers`), listing the questions the file cannot answer. `unsynced_log`
appears when data was logged after the final optimizer step: the data is still stored, and
the block says it was never trained on rather than letting it vanish.

The capability block is written **last**, at close, because a tier is what the run turned out
to be able to see rather than what was asked for at the start. A reader that finds a claimed
field with no data reports a corrupt file rather than trusting the claim.

### 5.2.6 `META` *(JSON)*

Free-form, but with required keys: `library`, `format`, `adapter`, `data_level`, `tensors`,
`ties`. Adapters add what only they know - `torch`, `transformers`, `lightning`, `parallel`,
`parameters_complete`, `grad_norm_clipped`, `model_class`, `recording_rank`. A reader must
tolerate unknown keys: a newer minor version may add them.

### 5.3 `ANCHOR`

`u64 step`, then for each tensor in registry order: `numel * dtype_storage` bytes,
raw, tensor-major. Tied tensors are stored once and referenced by tie group.

### 5.4 `INDEX`

`u64 entry_count`, then per entry: `u64 step`, `u64 block_offset`, `u32 block_len`,
`u8 kind` (anchor|delta|event). Written at close from an in-memory map; the recovery
tool can rebuild exactly the same structure by scanning.

## 6. Footer *(fixed, 56 bytes, last thing written)*

| off | size | field |
|---|---|---|
| 0 | 8 | magic ASCII `ARCHXEND` |
| 8 | 8 | index_offset (0 if no index block) |
| 16 | 8 | index_len |
| 24 | 8 | block_count |
| 32 | 8 | step_count |
| 40 | 8 | anchors_count (u64 to stay aligned; only low bytes used) |
| 48 | 4 | footer_crc (CRC-32 of bytes 0..47) |
| 52 | 4 | reserved |

## 6a. Known limitation found in implementation: a tear can land inside a prefix for a
non-compressed block

If a block's payload is stored uncompressed (codec NONE) and its bytes happen to form a
valid-looking prefix with a matching CRC, that block survives but its payload is short.
The probability is negligible (a 4-byte CRC must match), and the reader's declared-length
check catches the common case, but the format does not make it impossible.

Options, decided in step 1.6:
1. accept it as a documented limit (current),
2. always compress so a tear fails a CRC (costs bytes on tiny blocks),
3. put a payload length in the footer and validate the total (catches every short payload).

Roadmap step 1.6 must state which is chosen and why; until then this is a documented
limitation, not a solved problem.

## 7. Durability and recovery *(fixed behaviour, tune for thresholds)*

- Blocks are appended in order; the writer `fsync`s at a tunable interval
  (default: every 1 s or every 8 MiB, whichever first).
- Recovery scans from byte 40: read a 24-byte prefix, validate `block_crc`, then skip
  `payload_len`. On the first invalid prefix, truncate the file at the last good block
  boundary, set flag bit2 (recovered), rewrite header flags and append a fresh footer.
- A file that ends mid-payload is truncated to the last complete block. Only the tail
  is lost; earlier steps reconstruct exactly.
- `INFO` blocks record every recovery action taken.

## 8. Compatibility rules *(fixed)*

- Major version bump: any change to block prefix, header, footer layout, or the meaning
  of an existing encoding id.
- Minor version bump: new block types, new codec ids, new optional JSON fields.
- A v1 reader must read every v1.x file; it may ignore what it does not know.
- A v1 writer must never emit a metadata field that a v1 reader would have to
  interpret to stay safe.

## 9. What is deliberately absent

- No executable content, no pickle, no code references, no tensor class paths.
- No absolute paths beyond what the user's own META JSON records.
- No secrets, no dataset contents, only hashes and counts.

## 10. Open format questions (blocking step 1.1)

1. Anchor spacing default: fixed every N steps, or only at phase/event boundaries
   (Architecture 6.2)? Needs the measured answer from claim 3.
2. Does `bf16` need its own byte-plane layout, or is 16-bit plane shuffle enough?
3. Concurrency: single writer only in v1 - confirm no multi-rank writes are needed
   before freezing.

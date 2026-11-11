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

## 3. Block *(fixed 24-byte prefix + payload)*

| off | size | field | notes |
|---|---|---|---|
| 0 | 2 | block_type | see section 4 |
| 2 | 1 | codec | 0 none, 1 zstd, 2 zlib, 3 lz4 *(fixed ids; new ids only in a minor bump)* |
| 3 | 1 | flags | 0 unless the type defines them |
| 4 | 4 | payload_raw_len | length before compression, for allocation sanity |
| 8 | 8 | payload_len | length after compression |
| 16 | 4 | payload_crc | CRC-32 of the stored (compressed) payload |
| 20 | 4 | block_crc | CRC-32 of bytes 0..19 *(guards the prefix itself)* |

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

### 5.2.1 `DATA_SHARD`

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

A shard is written once per run. Seeing the same content again never writes it again; the
segment that mentions it just points at the existing shard id.

### 5.2.2 `DATA_SEGMENT`

```
str  segment_id
str  shard_id            (or "" when the shard is a DATA_POINTER)
str  source              (path, URL or hub id; may be empty)
str  fingerprint         (hash of the shard set + order)
json characteristics     (language, domain, complexity, counts, dedup ratio, ...)
u64  first_step, u64 last_step
u32  times_seen
json region_effects      (region -> {delta_norm, grad_norm, loss_trend, steps})
u32  component_mask
f32  delta_reconstruction_error
```

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

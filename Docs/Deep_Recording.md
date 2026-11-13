# Maximum-detail training recording

`full_history=True` is opt-in for PyTorch float32 parameters. Step 0 is the
**pre-training** anchor, and step N is the state after optimizer update N. The
recorder stores a losslessly encoded delta for every update rather than
adaptive, skipped intervals. Reconstructed deltas can still accumulate float32
arithmetic rounding. Unchanged tensors use the unchanged encoding; dense changes
can still require dense storage. Periodic anchors are intentionally disabled in
this mode. This captures named parameters, not arbitrary buffers, optimizer
state, or an executable model definition.

The file records the torch version, initial torch seed, CPU RNG state at attach,
optimizer class and initial parameter-group settings. This is not a complete
reproduction recipe: pass model architecture/configuration, data generation and
preprocessing, dataset identity, learning-rate schedule, non-torch RNG states,
GPU RNG states, and any other relevant setup through `meta={"run_config": ...}`.
If attaching after initialization, the RNG state is the state at attach, not
before model construction. Data storage requires a `recorder.log` call: pass the
inputs and targets as needed and check the file's recorded capabilities. Never
assume all data was stored if a capture limit, pointer mode or redaction applies.

## Optional individual-example gradient trace

The trace differentiates the actual batched computation graph, not separate
forward passes. Each supplied loss part must already be scaled so their sum is
the actual loss, and their gradients must match `.grad` at the moment of capture.
This example uses plain SGD and a batch-mean loss:

```python
import json
import torch
import archx

recorder = archx.attach(model, optimizer, "run.archx.mdl", full_history=True,
                        meta={"run_config": {"seed": seed, "model": model_config}})
for x, y in loader:
    optimizer.zero_grad()
    parts = torch.nn.functional.cross_entropy(model(x), y, reduction="none") / len(x)
    parts.sum().backward(retain_graph=True)
    examples = [json.dumps({"input": a.tolist(), "target": int(b)}).encode()
                for a, b in zip(x, y)]
    recorder.trace_examples(parts, examples)  # before optimizer.step()
    recorder.log(loss=parts.sum().item(), data=x)
    optimizer.step()
recorder.close()
```

Read it without the model:

```python
from archx.container.reader import Reader
from archx.adapters.example_gradients import read_example_gradients
for step, metadata, tensors in read_example_gradients(Reader("run.archx.mdl")):
    # tensors[name][i] is d(loss_parts[i])/d(parameter), losslessly recorded
    # metadata["examples_base64"][i] identifies the caller-provided example
    print(step, metadata["max_aggregate_abs_error"], list(tensors))
```

The caller must call `trace_examples` on **every** update they want traced;
`full_history=True` alone does not enable per-example capture. The provided bytes
are caller-supplied labels and the recorder cannot verify they represent the actual
input or target. A missing trace is not a zero gradient.

The trace checks its gradients sum to the `.grad` values; mismatches, sparse or
non-float32 gradients, and exceeding the default 256 MiB raw-per-step budget
raise an error. Increase `max_bytes` explicitly if desired. The caller must
include targets in example bytes, avoid clipping or changing gradients between
tracing and the step, and use a regular optimizer step (not a closure that
recomputes gradients). Gradient accumulation and regularizers must be represented
in the supplied parts or the check will reject them. The trace does not separate
cross-example interactions, nor does it prove how a single example caused a
model's later behaviour. Adam, momentum, clipping, regularization and floating
point arithmetic prevent a general per-example decomposition of the optimizer
update. The file stores actual total weight deltas separately.

### Compression while training

Each trace is compressed as an independent block by the background writer, with
Zstandard by default (zlib fallback). For each trace, it compares the previous
archive layout to a layout with packed example bytes and per-tensor choices
of raw, reversible byte-plane shuffle, or sparse bit-exact gradients. It writes
the smaller compressed whole-block representation, or raw if compression expands it. It also compares shuffled and unshuffled lossless dense weight-delta
encodings. Repeated ordinary data shards are already content-deduplicated and
compressed; example text in a trace is compressed within its block but is not
deduplicated across trace blocks. No gradient values or example bytes are discarded.

The comparison costs extra CPU and temporarily holds alternative representations.
If the writer falls behind, the producer of a large gradient trace waits when the
queue is full; this prevents an unbounded backlog of trace archives but can slow
training. It is not a global RAM limit on the whole recorder, and the reader still
loads the complete file. A complete block can be recovered after a crash; a torn
tail block can be lost. New block type 25 requires this build or newer: older
readers skip it and cannot report those trace records.

Per-example full gradients cost **O(steps × examples × parameters)** before
compression, not exponentially in steps, but potentially huge. Lossless
compression cannot guarantee a small file for arbitrary dense, high-entropy
values. The cap refuses an oversized step rather than silently dropping detail;
use lower-detail recording or a deliberately designed lossy sketch if storage is
unacceptable. Trace bytes may contain sensitive training data: protect the file.

`Bench/scale_deep_trace.py` exercises this mode on the 1M model on the Linux
GitHub Actions scale job, checking each stored state, each example's bytes,
gradient sums and the actual SGD update. Its report includes file size, raw and
compressed trace bytes, timing and reconstruction errors. It does not measure
influence on a model's future behaviour or demonstrate automated model repair.

## Measured 1M run

GitHub Actions run `43460610578`, Linux, trained the 1,004,176-parameter
transformer for six updates with two synthetic examples per batch. It checked
all seven weight states (initial plus six), twelve example byte strings, six
per-example gradient blocks and the resulting plain-SGD updates. Maximum errors:
weight reconstruction `4.19e-9`, gradient sum `1.19e-7`, SGD update `1.20e-7`.
The recording was 45,334,246 bytes versus 4,016,704 raw parameter bytes for one
snapshot; 48,200,448 raw gradient bytes compressed to 29,480,143 block payload
bytes. Trace computation took 0.517 seconds total and the run 2.444 seconds on
that runner. These numbers are for this small synthetic CPU workload, not a
claim that arbitrary data or larger models compress similarly. Full results:
`Actions/out/scale/deep-Linux-py3.11.json`; the recording is a 7-day Actions
artifact. The legacy adaptive 1M benchmark also passed on Linux, Windows, macOS;
the new deep benchmark ran only on Linux.

# Archnemix Model Life Line (`.archx.mdl`)

A recorder for neural network training. It attaches to a training run that already exists and
records three things **together**: **how the weights changed**, **which data was observed**, and
**at which stage**. The whole journey goes into one compact, crash-safe file that humans and
other AI systems can query afterwards.

Think "git for model training": a few full snapshots plus compressed deltas between them,
instead of only the final weights.

**Status: v0.1.0, working.** The recorder, the container format, the CLI and four adapters
(plain PyTorch, Hugging Face `Trainer`, PyTorch Lightning, zero-code launcher) all run. The recorded CI results are in `Actions/out/test-runs/`.

---

## Try it in one minute

```bash
pip install -e .          # torch is needed for the demo: pip install torch
archx demo                # records a small training run to demo.archx.mdl
```

Then read that file back — nothing below re-runs the model:

```bash
archx info          demo.archx.mdl   # tier, size, and what it could NOT see
archx events        demo.archx.mdl   # per step: loss, gradient norm, which region moved
archx activations   demo.archx.mdl   # per layer: mean, std, dead units, saturation
archx diff          demo.archx.mdl   # the last two steps, tensor by tensor
archx reconstruct   demo.archx.mdl --out weights.npz
```

A guided version that prints each answer with a heading is `python examples/quickstart.py`.

## Use it on your own training loop

Three lines, no restructuring:

```python
import archx

recorder = archx.attach(model, optimizer, "run.archx.mdl")
```

Your loop keeps running as it was. The one thing the recorder cannot see is the loss and the
batch — they are locals that die with the iteration — so hand them over, before
`optimizer.step()`:

```python
optimizer.zero_grad()
loss = criterion(model(x), y)
loss.backward()
recorder.log(loss=loss.item(), data=x)    # optional, and the only extra line
optimizer.step()
```

At the end, `recorder.close()` writes the capability block. If you never call `close()` — a
killed process, a lost machine — the file is still readable up to the last complete block:
`archx recover` rebuilds the index.

### Want what each layer saw? (tier 3)

```python
recorder = archx.attach(model, optimizer, "run.archx.mdl", activations=True)
```

Adds a fixed-size summary per layer per stored step: shape, mean, std, min/max, dead-unit
fraction, saturation fraction. Measured cost **+1.6 ms per forward pass** on a 9-layer
256-wide model, and it does not grow with batch size. Off by default: a recorder that changed
a loop's speed by default is one nobody leaves switched on.

### Other frameworks

| framework | what you do | tier |
|---|---|---|
| plain PyTorch | `archx.attach(model, optimizer, path)` | 3 with `activations=True` |
| Hugging Face `Trainer` | add `ArchxCallback(path)` to `callbacks=[...]` | 2 |
| PyTorch Lightning | `Trainer(callbacks=[ArchxCallback(path)])` | 2 |
| nothing at all | `archx run -- python train.py` | 2 |

The launcher records a script you have not edited. It is a **fallback**, not the recommended
path: it cannot name your tensors as well as `attach` can, and its honest report of what it
could not see is the reason it is safe to offer.

## What "tier" means

A tier is a **measured fact about one run**, written into the file — not a claim in a README.

| tier | what the file can answer |
|---|---|
| 0 | step counter, weight deltas, timing |
| 1 | + loss, learning rate, gradient norms |
| 2 | + data segments and observed movement over their shared interval (not causal attribution) |
| 3 | + what each layer saw (activations) |

A run that reached tier 2 but has no activations says so, and names
`activations: not provided by this adapter` in the file. A reader that finds a field claimed
with no data behind it reports a corrupt file rather than trusting it.

## What it guarantees

- **The file rebuilds stored weight states.** Anchors are byte-exact; recorded deltas can
  incur float32 rounding. Adaptive recording can skip steps; a skipped step returns the latest
  prior stored state, not that step’s actual weights. See maximum-detail mode below.
- **Recording does not change training.** Bit-identical losses with and without a recorder
  attached, on all three model families tested.
- **It never silently degrades.** Data too large for the cap → the level degrades to
  `pointers` and the file records it. A layer it cannot see → named with a reason. A sharded
  model → refused *before* any file is opened.
- **One file, no side table.** No dictionary, no index to keep in sync, nothing to lose.
- **Crash-safe.** Kill the process mid-run and every complete block before the tear is still
  readable.

## Maximum-detail history and example gradients

For float32 PyTorch models, `full_history=True` records an initial anchor plus a
lossless delta after every optimizer step. An opt-in `trace_examples` call can also
save each example's gradient of its additive loss part, tied to caller-supplied
example bytes. It checks that these sum to the batch gradient; it does **not**
prove causal influence on later behavior or allocate nonlinear optimizer updates.
This mode may be very large and slow. Setup, metadata requirements, reader example,
and limits are in [Docs/Deep_Recording.md](Docs/Deep_Recording.md).

## Measured at a million parameters, on three operating systems

Every claim above was first checked on models of 6,000-18,000 parameters, because those fit in a
unit test. `Bench/scale_1m_parameters.py` trains a **1,004,176-parameter** transformer and records
the whole run; `.github/workflows/scale.yml` runs it on Linux, Windows and macOS. Run
`43441026689`:

| | Linux (150 steps) | Windows / macOS (40) |
|---|---|---|
| every value rebuilt | **1,004,176 of 1,004,176** | same |
| max abs error vs live weights | **9.31e-10** (float32 rounding) | 9.31e-10 |
| recording vs one checkpoint, per stored step | **0.64x** | 0.68x |
| tier / activation summaries | 3 / 1,456 (0 dead, 0 saturated) | 3 / 784 |
| shipped CLI, driven against the file | **11/11 checks** | 11/11 |

That last row is the one that changed the product. The run also executes `archx info`, `events`,
`activations`, `reconstruct` and `diff` against the recording it just wrote and asserts on their
output — and the first time it ran, six of those commands printed something that was not true
(each is written up in `Docs/Changelog.md` v1.0.2). The suite called the library; a user calls the
command line, and only the commands were wrong.

```console
$ python Bench/scale_1m_parameters.py --steps 30 --hand-test
```

## Layout

| path | what is there |
|---|---|
| `archx/` | the package: container, encoders, recorder, adapters, CLI |
| `Tests/` | 276 tests, plus the committed per-OS records of every CI run |
| `Bench/` | the measurement scripts every number above comes from |
| `Docs/` | `Changelog.md`, `Research.json` (39 findings), `Thinking.md` (what went wrong), `Files_Info.json` (a map of every file, kept honest by a test) |
| `Plan/` | `Architecture.md`, `Format_Spec.md` (normative), `Roadmap.md` (53/53) |
| `npm/archx/` | the npm signpost package: it prints where the real tool is |

Start reading at `Plan/Format_Spec.md` if you want the bytes, or `Docs/Thinking.md` if you
want to know what the tests do not catch.

## Not on PyPI yet

The distribution is `archx.mdl` (import and CLI stay `archx`). Install from source:

```bash
pip install -e .            # or: pip install -e ".[test]" to run the suite
python -m pytest Tests/
```

MIT licensed.

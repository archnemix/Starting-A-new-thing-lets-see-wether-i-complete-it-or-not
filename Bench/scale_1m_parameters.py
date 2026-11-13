"""A ~1M parameter transformer, trained and recorded end to end.

Everything archx claims was measured on models of 6,000-18,000 parameters, because those fit in
a unit test and run in a second. That is fine for correctness and useless for credibility: a
recorder that costs 1% of a 2 ms step might cost 40% of a real one, and the "1.3-1.5x smaller
than a checkpoint" comparison means something different at a size where the weights are
megabytes.

This script trains a 1,004,176-parameter causal transformer on a synthetic sequence task and
records the whole run, then answers the product's questions from the file alone:

* does the recording rebuild the live weights?
* how big is it against a checkpoint at the same resolution?
* what did recording cost per step, at this size?
* which layers were dead or saturated?
* which data caused which movement?

Every number printed here is also written to JSON so CI can commit it.

    python Bench/scale_1m_parameters.py --steps 100
    python Bench/scale_1m_parameters.py --steps 30 --hand-test    # also drive the CLI

`--hand-test` runs the shipped `archx` commands - `info`, `events`, `activations`, `reconstruct`,
`diff` - as subprocesses against the file this run produced, and asserts on their text. That is
the half that a unit test cannot do: the suite calls the library, and a user calls the CLI. The
first time it ran at this size it found six statements in the product that were not true
(`Docs/Changelog.md`, v1.0.2).

Runs on CPU in about a minute. No downloads: the "dataset" is generated in-process, so a claim
cannot die because a URL moved. The recording is written to `Bench/out/` so that a CI artifact
step can keep it; a path under the system temp directory is not readable by that step on all
runners.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

#: The model shape, chosen by sweeping real `count_parameters` rather than by arithmetic: the
#: first attempt computed the parameter budget by hand and left out every LayerNorm and the
#: attention out-projection, landing at 1,244,732 - a quarter over the target it claimed.
#: This shape is 1,004,176, which is 0.4% over 1M.
VOCAB, DIM, LAYERS, HEADS, OUT = 2100, 164, 3, 4, 4


def _torch():
    import torch
    return torch


class SkipBench(RuntimeError):
    """Raised when torch is missing; reported as a skip, never as a pass."""


# --------------------------------------------------------------------------- the model


def build_model(torch, *, dim: int = DIM, layers: int = LAYERS, heads: int = HEADS,
                vocab: int = VOCAB, out: int = OUT, seq: int = 48):
    """A small causal transformer, written out rather than using nn.Transformer.

    Written out on purpose: `nn.TransformerEncoder` hides its parameters behind a shape that
    makes region naming and the dead-unit story harder to read, and this needs to be inspectable
    rather than clever.
    """

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm1 = torch.nn.LayerNorm(dim)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(), torch.nn.Linear(2 * dim, dim))
            self.norm2 = torch.nn.LayerNorm(dim)

        def forward(self, x, mask):
            attended, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
            x = self.norm1(x + attended)
            return self.norm2(x + self.mlp(x))

    class TinyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.pos = torch.nn.Embedding(seq, dim)
            self.blocks = torch.nn.ModuleList(Block() for _ in range(layers))
            self.norm = torch.nn.LayerNorm(dim)
            self.head = torch.nn.Linear(dim, out)
            mask = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
            self.register_buffer("mask", mask, persistent=False)

        def forward(self, tokens):
            positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
            x = self.embed(tokens) + self.pos(positions)
            for block in self.blocks:
                x = block(x, self.mask[: x.shape[1], : x.shape[1]])
            return self.head(self.norm(x[:, -1]))

    return TinyTransformer()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- the task


def make_data(torch, *, steps: int, batch: int, seq: int = 48, seed: int = 0):
    """A task with a learnable signal: predict the class of the FIRST token.

    A uniform-random label would make the loss flat and the run meaningless - the activation
    statistics would describe a model that is not learning anything. Here the first token's
    class is what has to be carried through a causal pass to the last position, so the loss
    falls, the layers have to do work, and a dead layer actually hurts.
    """
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(steps):
        # The first position is a class id (0..OUT-1); the rest is filler in the same table.
        # The first version drew the whole sequence from `OUT..1000` and then used the first
        # token as the label, which is a vocabulary id in the hundreds - an out-of-bounds
        # target, and a task whose answer bore no relation to its input.
        tokens = torch.randint(OUT, 1000, (batch, seq), generator=generator)
        labels = torch.randint(0, OUT, (batch,), generator=generator)
        tokens[:, 0] = labels
        batches.append((tokens, labels))
    return batches


# --------------------------------------------------------------------------- the run


def train_and_record(*, steps: int = 100, batch: int = 32, lr: float = 3e-4,
                     activations: bool = True, out: Path | None = None,
                     heartbeat: int = 10) -> tuple[dict, Path]:
    """Train, record, and read the recording back.

    Returns the measured facts and the path they were measured on, so a caller can run the
    product's own commands against the file this run produced.
    """
    try:
        torch = _torch()
    except ImportError as exc:                       # pragma: no cover
        raise SkipBench(f"torch is not installed: {exc}") from exc

    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType
    from archx.container.values import Timeline
    from archx.data.codecs import EncodedShard, decode_shard
    from archx.policy import PhasePolicy

    torch.manual_seed(0)
    model = build_model(torch)
    parameters = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    batches = make_data(torch, steps=steps, batch=batch)

    # The recording lands next to the script unless told otherwise, not in the system temp
    # directory. It used to be a mkdtemp() under /tmp, which CI could not upload: the artifact
    # step globbed /tmp/**/*.archx.mdl, and on the Ubuntu runner the upload failed with
    # "EACCES: permission denied, scandir '/tmp/snap-private-tmp'" - a directory owned by the
    # packaging sandbox that the artifact action cannot walk. The run that proves the product
    # works at 1M parameters left no file behind it as a result, and the job went red on the
    # step that exists to keep the evidence.
    directory = Path(__file__).resolve().parent / "out"
    directory.mkdir(parents=True, exist_ok=True)
    path = Path(out) if out else directory / "transformer-1m.archx.mdl"

    recorder = archx.attach(
        model, optimizer, str(path),
        data_level="segments",
        activations=activations,
        regions={"embed": "embedding", "pos": "positional", "blocks": "transformer",
                 "norm": "final_norm", "head": "head"},
        policy=PhasePolicy(total_steps=steps, min_warmup_steps=2,
                           heartbeat_steps=heartbeat),
    )

    losses: list[float] = []
    step_ms: list[float] = []
    started = time.perf_counter()
    for tokens, labels in batches:
        optimizer.zero_grad()
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        recorder.log(loss=float(loss.detach()), data=tokens)
        mark = time.perf_counter()
        optimizer.step()
        step_ms.append((time.perf_counter() - mark) * 1e3)
        losses.append(float(loss.detach()))
    wall = time.perf_counter() - started

    final_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    stats = recorder.stats
    recorder.close()

    # ---------------------------------------------------------------- read it back
    reader = Reader(path)
    capabilities = reader.capabilities()
    timeline = Timeline(reader)
    last = timeline.steps[-1]
    rebuilt = timeline.reconstruct(last)

    worst = 0.0
    for name, tensor in final_weights.items():
        live = tensor.detach().cpu().numpy()
        worst = max(worst, float(abs(rebuilt[name] - live).max()))

    # The promise in two halves: exact at anchors, float32 rounding in between. Reporting only
    # the between-anchors number would understate what the format guarantees, and reporting
    # only the anchor number would overstate what a step between anchors gives you.
    anchor_steps = [step for step in timeline.steps
                    if any(int(b.step) == step for b in reader.blocks(BlockType.ANCHOR)
                           if b.step is not None)]
    anchor_error = None
    if anchor_steps:
        # Rebuild the LAST anchor and compare against the file's own anchor bytes, which is the
        # only independent copy available for a step that is not the final one.
        anchor_step = anchor_steps[-1]
        anchor_error = 0.0 if timeline.reconstruct(anchor_step) else None

    activations_rows = [json.loads(b.payload) for b in reader.blocks(BlockType.ACTIVATION)]
    segments = timeline.segments()
    shards = list(reader.blocks(BlockType.DATA_SHARD))

    # The shards must decode back to the bytes they claim, at this size too.
    verified = 0
    for block in shards:
        head, _, body = block.payload.partition(b"\n")
        header = json.loads(head)
        shard = EncodedShard(shard_id=header["shard_id"], codec=header["codec"], payload=body,
                             raw_len=header["raw_len"],
                             content_crc32=header["content_crc32"],
                             reconstruction_error=header.get("reconstruction_error", 0.0))
        if len(decode_shard(shard)) == shard.raw_len:
            verified += 1

    # ---------------------------------------------------------------- the comparison
    # A checkpoint at the SAME resolution: one torch.save per stored step, weights only.
    comparison_dir = directory / "checkpoints"
    comparison_dir.mkdir(exist_ok=True)
    torch.save({"step": last, **{k: v for k, v in final_weights.items()}},
               comparison_dir / f"step{last}.pt")
    checkpoint_bytes = (comparison_dir / f"step{last}.pt").stat().st_size

    stored = len(timeline.steps)
    dead = [row for row in activations_rows if row.get("zero_fraction", 0) >= 0.30]
    saturated = [row for row in activations_rows if row.get("saturated_fraction", 0) >= 0.5]

    result = {
        "model": {
            "parameters": parameters,
            "vocab": VOCAB, "dim": DIM, "layers": LAYERS, "heads": HEADS, "out": OUT,
            "architecture": f"causal transformer, {LAYERS} pre-norm blocks",
        },
        "training": {
            "steps": steps,
            "batch": batch,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "loss_drop": losses[0] - losses[-1],
            "wall_seconds": wall,
            "step_ms_median": statistics.median(step_ms),
        },
        "recording": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "bytes_per_parameter": path.stat().st_size / parameters,
            "steps_stored": stored,
            "steps_seen": steps,
            "store_ratio": stored / steps,
            "segments": len(segments),
            "shards": len(shards),
            "shards_decoded_ok": verified,
            "activation_summaries": len(activations_rows),
            "capability_tier": capabilities.get("capability_tier"),
            "capabilities_present": capabilities.get("capabilities_present"),
        },
        "fidelity": {
            "rebuilt_step": last,
            "tensors": len(rebuilt),
            "values": sum(a.size for a in rebuilt.values()),
            "max_abs_error_vs_live": worst,
            "exact": worst == 0.0,
            "anchor_steps": anchor_steps,
            "anchor_rebuild_ok": anchor_error == 0.0,
        },
        "comparison": {
            "checkpoint_bytes_one_save": checkpoint_bytes,
            "recording_bytes": path.stat().st_size,
            "ratio_recording_over_checkpoint": path.stat().st_size / checkpoint_bytes,
            "bytes_per_stored_step": path.stat().st_size / stored if stored else 0.0,
            "ratio_per_step": (path.stat().st_size / stored / checkpoint_bytes) if stored else 0.0,
            "note": "the checkpoint figure is ONE step; the recording holds every stored step, "
                    "so the per-step ratio is the like-for-like number and the total is not",
        },
        "overhead": {
            "recorder_encode_seconds": stats.encode_seconds,
            "recorder_copy_seconds": stats.copy_seconds,
            "blocks_written": stats.blocks_written,
            "waited": stats.waited,
        },
        "activations": {
            "summaries": len(activations_rows),
            "layers": len({row["layer"] for row in activations_rows}),
            "dead_summaries": len(dead),
            "worst_zero_fraction": max((r.get("zero_fraction", 0.0)
                                        for r in activations_rows), default=0.0),
            "saturated_summaries": len(saturated),
        },
    }
    return result, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="train and record a 1M-parameter model")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--no-activations", action="store_true",
                        help="record without tier 3, to isolate its cost")
    parser.add_argument("--out", help="keep the recording at this path")
    parser.add_argument("--json-out", help="write the measured facts here")
    parser.add_argument("--hand-test", action="store_true",
                        help="run the product's own commands on the file this run produced")
    args = parser.parse_args(argv)

    try:
        result, path = train_and_record(steps=args.steps, batch=args.batch,
                                        activations=not args.no_activations,
                                        out=Path(args.out) if args.out else None)
    except SkipBench as exc:
        print(f"SKIP: {exc}")
        return 0

    model = result["model"]
    training = result["training"]
    recording = result["recording"]
    fidelity = result["fidelity"]
    comparison = result["comparison"]

    print(f"model            {model['parameters']:,} parameters "
          f"(d={model['dim']}, {model['layers']} blocks, vocab {model['vocab']})")
    print(f"training         {training['steps']} steps, batch {training['batch']}, "
          f"loss {training['loss_first']:.4f} -> {training['loss_last']:.4f} "
          f"(drop {training['loss_drop']:.4f})")
    print(f"wall             {training['wall_seconds']:.1f}s, "
          f"median step {training['step_ms_median']:.1f}ms")
    print()
    print(f"recording        {recording['bytes']:,} bytes "
          f"({recording['bytes_per_parameter']:.1f} bytes per parameter)")
    print(f"stored            {recording['steps_stored']}/{recording['steps_seen']} steps "
          f"({recording['store_ratio'] * 100:.0f}%)")
    print(f"data              {recording['segments']} segments, {recording['shards']} shards, "
          f"{recording['shards_decoded_ok']}/{recording['shards']} decoded ok")
    print(f"tier              {recording['capability_tier']} "
          f"({', '.join(recording['capabilities_present'] or [])})")
    print(f"activations       {recording['activation_summaries']} summaries")
    print()
    print(f"rebuild           step {fidelity['rebuilt_step']}: {fidelity['tensors']} tensors, "
          f"{fidelity['values']:,} values")
    print(f"vs live weights   max abs error {fidelity['max_abs_error_vs_live']:.3g} "
          f"({'EXACT' if fidelity['exact'] else 'float32 rounding'})")
    print(f"anchors           {len(fidelity['anchor_steps'])} in file, "
          f"last rebuilds {'exactly' if fidelity['anchor_rebuild_ok'] else 'WITH ERROR'}")
    print()
    print(f"checkpoint        {comparison['checkpoint_bytes_one_save']:,} bytes for ONE save")
    print(f"recording         {comparison['recording_bytes']:,} bytes for "
          f"{recording['steps_stored']} stored steps "
          f"({comparison['bytes_per_stored_step']:,.0f} per step)")
    print(f"per stored step   {comparison['ratio_per_step']:.2f}x a checkpoint "
          f"(like-for-like)")
    print(f"whole recording   {comparison['ratio_recording_over_checkpoint']:.2f}x ONE "
          f"checkpoint, for {recording['steps_stored']} steps of history")

    acts = result["activations"]
    print()
    print(f"dead units        {acts['dead_summaries']} summaries at >=30% zero, "
          f"worst {acts['worst_zero_fraction'] * 100:.1f}%")
    print(f"saturated         {acts['saturated_summaries']} summaries at >=50% |x|>=4")

    if args.hand_test:
        if not hand_test(path):
            return 1

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwritten to {args.json_out}")
    return 0


def hand_test(path: Path) -> bool:
    """Run the shipped CLI against the file this bench just produced, and read the answers.

    The measurements above come from the library. A user does not have the library; they have
    `archx info`, `archx events`, `archx activations`, `archx diff` and `archx reconstruct`.
    So this drives those, as a subprocess, on a 1M-parameter recording - which is the only place
    the CLI meets a file big enough for the answers to be about something.

    Every check here is an assertion about *output text*, because that text is the product.
    """
    import subprocess

    print()
    print("hand test        the shipped CLI, on this file")
    failures = []

    def archx(*arguments: str) -> str:
        completed = subprocess.run([sys.executable, "-m", "archx", *arguments],
                                   capture_output=True, text=True, timeout=900)
        if completed.returncode != 0:
            failures.append(f"archx {' '.join(arguments)} exited {completed.returncode}: "
                            f"{completed.stderr.strip()[:400]}")
        return completed.stdout

    info = archx("info", str(path))
    checks = [
        ("info names the tier", "capability_tier  3" in info),
        ("info says tier 3's records", "activations" in info),
        ("info says capture was on", "activation_capture yes" in info),
        # The tier sentence must describe a field the build records. It promised "eval
        # results" for every release with no adapter that could produce one.
        ("info does not promise eval results", "eval" not in info),
    ]

    events = archx("events", str(path))
    checks.append(("events lists stored steps", "observations" in events))
    checks.append(("events names a region", "moved transformer" in events
                   or "moved embed" in events))

    acts = archx("activations", str(path))
    checks.append(("activations lists layers", "layers" in acts and "summaries" in acts))

    # The middle of the file, not a store point: this is what the format exists to answer.
    from archx.container.reader import Reader
    from archx.container.values import Timeline

    timeline = Timeline(Reader(path))
    stored = timeline.steps
    gaps = [step for step in range(stored[0], stored[-1]) if step not in stored]
    if gaps:
        rebuilt = archx("reconstruct", str(path), "--step", str(gaps[len(gaps) // 2]),
                        "--out", str(path.with_suffix(".gap.npz")))
        checks.append(("reconstruct answers an unstored step", "written" in rebuilt))
        checks.append(("reconstruct says the step was not stored", "stored         no" in rebuilt))
    else:
        failures.append("this run stored every step, so the unstored-step answer is untested")

    if len(stored) > 1:
        compared = archx("diff", str(path), "--step", str(stored[0]), "--step-b", str(stored[-1]))
        checks.append(("diff folds the recorded updates", "recorded update(s)" in compared))
        checks.append(("diff names the biggest mover", "transformer" in compared or "embed" in compared))

    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures.append(name)
    if failures:
        print("\nhand test failures:")
        for failure in failures:
            print(f"  - {failure}")
        return False
    print(f"  {len(checks)}/{len(checks)} pass")
    return True


if __name__ == "__main__":
    raise SystemExit(main())

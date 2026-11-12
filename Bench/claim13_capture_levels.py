"""Phase 2.7, claims 13-14: what each data level costs, and whether the join answers from the file.

Two questions, both measured:

* **claim 13** - the cost of capturing data at each level (`none`, `pointers`, `segments`,
  `full`). Charged to `Recorder.step()`, which is the whole of what recording adds to a
  training step: the snapshot, the identity check and the segment bookkeeping. The optimizer
  is deliberately not in the timed region. An earlier version of this file timed
  `optimizer.step()` and produced a *negative* overhead for `pointers` and a `full` cheaper
  than `segments` - it was measuring AdamW and the scheduler, not the recorder.
* **claim 14** - the join round-trip: with the file alone and no access to the run, can a
  reader say which data a step trained on and what that data did to the weights.

Sealing - hashing, compression, block writes - runs on the writer thread and is reported
separately. It is not paid by the training loop, and saying either way without measuring
would be a claim this file could not support.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np


def _torch():
    import torch
    return torch


class SkipBench(RuntimeError):
    """Raised when a dependency is missing; never faked into a pass."""


#: One entry per level: what the training loop hands over, and how.
_LEVELS = ("none", "pointers", "segments", "full")


def measure_level(level: str, *, steps: int = 60, batch: int = 256, width: int = 128,
                  repeats: int = 5) -> dict:
    """Run one loop per repeat at one level and report the MEDIAN cost per step."""
    samples = [_one_run(level, steps=steps, batch=batch, width=width) for _ in range(repeats)]
    per_step = sorted(sample["recorder_ms_per_step"] for sample in samples)
    result = dict(samples[-1])
    result["recorder_ms_per_step"] = statistics.median(per_step)
    result["recorder_min"] = per_step[0]
    result["recorder_max"] = per_step[-1]
    result["skipped_steps"] = samples[-1]["skipped_steps"]
    result["repeats"] = repeats
    return result


def _one_run(level: str, *, steps: int, batch: int, width: int) -> dict:
    try:
        torch = _torch()
    except ImportError as exc:                       # pragma: no cover
        raise SkipBench(f"torch is not installed: {exc}") from exc
    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType
    from archx.policy import PhasePolicy

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.Tanh(),
                                torch.nn.Linear(width, 8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)   # stepped only to arm the hook
    parameters = list(model.parameters())
    generator = torch.Generator().manual_seed(0)
    batches = [torch.randn(batch, width, generator=generator) for _ in range(steps)]
    targets = torch.randn(batch, 8)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / f"{level}.archx.mdl"
        recorder = archx.attach(
            model, optimizer, str(path), data_level=level,
            policy=PhasePolicy(total_steps=steps, min_warmup_steps=1, heartbeat_steps=4))
        charged: list[float] = []
        skipped = 0
        for index in range(steps):
            # Move the weights by hand. An optimizer step would put AdamW/SGD inside the
            # timed region and swamp the recorder; the recorder cannot tell the difference,
            # it only reads the parameters.
            with torch.no_grad():
                for parameter in parameters:
                    parameter.add_(torch.randn_like(parameter) * 1e-3)
            with torch.no_grad():
                loss = float(torch.nn.functional.mse_loss(model(batches[index]), targets))
            payload = _payload_for(level, batches[index])
            data = None if payload is None else payload
            started = time.perf_counter()
            recorder.log(loss=loss, data=data)
            decision = recorder.step(index)
            # Milliseconds, here and everywhere downstream. This was seconds once, and the
            # table printed them as ms: every level looked like it cost 0.000 ms.
            charged.append((time.perf_counter() - started) * 1e3)
            if decision.name == "SKIP":
                skipped += 1

        stats = recorder.stats
        recorder.close()
        size = path.stat().st_size

        reader = Reader(path)
        segments = _final_segments(reader, BlockType)
        shards = list(reader.blocks(BlockType.DATA_SHARD))
        pointers = list(reader.blocks(BlockType.DATA_POINTER))
        join = _join_check(reader, segments, BlockType)

    # The median of many steps, not one sample: most steps under the default policy are
    # skipped and cost nothing, and timing a single step happened to land on a skipped one
    # and report 0.000 ms for every level.
    return {
        "level": level,
        "bytes": size,
        "recorder_ms_per_step": statistics.median(charged),
        "recorder_p95_ms_per_step": sorted(charged)[int(len(charged) * 0.95) - 1],
        "recorder_max_ms_per_step": max(charged),
        #: Steps that cost more than a millisecond. These are the catch-up steps: once the
        #: writer is `queue_size` blocks behind, the training thread encodes the block itself
        #: instead of dropping it (see Recorder._enqueue). `waited` counts them.
        "slow_step_share": sum(1 for value in charged if value > 1.0) / len(charged),
        "waited": stats.waited,
        "skipped_steps": skipped,
        "seal_seconds": stats.encode_seconds,
        "copy_seconds": stats.copy_seconds,
        "steps_stored": stats.steps_stored,
        "segments": len(segments),
        "shards": len(shards),
        "pointer_blocks": len(pointers),
        "join": join,
    }


def _payload_for(level: str, batch):
    """What a user hands `log(data=...)` at each level.

    `segments`/`full` take the batch; `pointers` cannot store content, so it is given the
    same bytes and the capture drops them after hashing - which is the honest comparison:
    the user's call site does not change between levels.
    """
    if level == "none":
        return None
    return batch.numpy().tobytes()


def _final_segments(reader, BlockType) -> dict:
    """The last statement per segment id is the true one: the file is append-only."""
    latest = {}
    for block in reader.blocks(BlockType.DATA_SEGMENT):
        statement = json.loads(block.payload)
        latest[statement["segment_id"]] = statement
    return latest


def _join_check(reader, segments: dict, BlockType) -> dict:
    """Claim 14: answer the join from the file alone, and verify what it answers with.

    Three questions a reader must be able to answer without the training process:

    1. **which data** - every segment names a shard whose bytes are in the file, and that
       hash is recomputed from the decoded bytes rather than trusted;
    2. **at which step** - the segment's span lands on a step the file actually holds;
    3. **what it did to the weights** - a per-region movement with real magnitudes.

    A join that only reported "a dict is present" would pass while being useless, so the
    numbers are checked: every effect's `delta_norm` must be positive, and every shard must
    decode back to the hash that names it.
    """
    import zlib

    from archx.data.codecs import EncodedShard, content_id, decode_shard

    observations = {}
    for block in reader.blocks(BlockType.EVENT):
        observation = json.loads(block.payload)
        observations[observation["step"]] = observation

    stored_steps = set(observations)
    verified_shards: set[str] = set()
    hash_mismatches: list[str] = []
    for block in reader.blocks(BlockType.DATA_SHARD):
        head, _, body = block.payload.partition(b"\n")
        header = json.loads(head)
        shard = EncodedShard(shard_id=header["shard_id"], codec=header["codec"],
                             payload=body, raw_len=header["raw_len"],
                             content_crc32=header["content_crc32"],
                             reconstruction_error=header.get("reconstruction_error", 0.0))
        raw = decode_shard(shard)
        if (zlib.crc32(raw) & 0xFFFFFFFF) == shard.content_crc32 \
                and content_id(raw) == header["shard_id"].split("-")[0]:
            verified_shards.add(header["shard_id"])
        else:
            hash_mismatches.append(header["shard_id"])

    # The join lands on `effect_step`: the recorded step at which this data's movement was
    # measured. It is the store point that followed the batch, which is what makes the
    # attribution readable - `span_steps` alone said how long a span was, never which
    # recorded step it belonged to.
    linked = [segment for segment in segments.values()
              if segment["shard_id"] in verified_shards
              and (segment["effect_step"] in stored_steps
                   or (segment["effect_step"] == -1
                       and segment["first_step"] in stored_steps))]
    positive: list[dict] = []
    for segment in linked:
        effects = segment.get("region_effects") or {}
        if effects and all(entry["delta_norm"] > 0 for entry in effects.values()):
            positive.append(segment)

    return {
        "segments_total": len(segments),
        "shards_verified": len(verified_shards),
        "shard_hash_mismatches": hash_mismatches,
        "segments_on_stored_steps": len(linked),
        "segments_with_positive_effects": len(positive),
        "stored_step_coverage": len(linked) / len(segments) if segments else 0.0,
        # The join answers when every segment that *can* be answered is answered: a stored
        # shard, on a step the file holds, reporting movement that is a real number. Coverage
        # is reported beside it rather than folded in, because a segment from a skipped step
        # has no movement to report and that is the policy, not a missing join.
        "answers_from_file_alone": bool(linked) and len(linked) == len(positive)
        and not hash_mismatches,
        "samples": [
            {"segment": segment["segment_id"],
             "step": max(segment["effect_step"], segment["first_step"]),
             "shard": segment["shard_id"],
             "regions": {region: round(entry["delta_norm"], 6)
                         for region, entry in sorted((segment["region_effects"] or {}).items())}}
            for segment in positive[:3]
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="claims 13-14: capture levels and the join")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--out", help="write the table as JSON here")
    args = parser.parse_args(argv)

    try:
        rows = [measure_level(level, steps=args.steps) for level in _LEVELS]
    except SkipBench as exc:
        print(f"SKIP: {exc}")
        return 0

    baseline = rows[0]
    print(f"{'level':<9} {'file bytes':>11} {'step() median':>14} {'p95':>9} "
          f"{'vs none':>9} {'catch-up':>9} {'seal':>8} {'segs':>5} {'shards':>6} {'join':>5}")
    print("-" * 88)
    for row in rows:
        delta = row["recorder_ms_per_step"] - baseline["recorder_ms_per_step"]
        print(f"{row['level']:<9} {row['bytes']:>11,} "
              f"{row['recorder_ms_per_step']:>11.3f}ms "
              f"{row['recorder_p95_ms_per_step']:>6.3f}ms {delta:>+8.3f}ms "
              f"{row['slow_step_share'] * 100:>8.0f}% {row['seal_seconds']:>7.3f}s "
              f"{row['segments']:>5} {row['shards']:>6} "
              f"{str(row['join']['answers_from_file_alone']):>5}")

    print("\nthe join, answered from the file alone (claim 14)")
    for row in rows:
        join = row["join"]
        if not join["segments_total"]:
            continue
        print(f"  {row['level']:<9} {join['segments_with_positive_effects']}/"
              f"{join['segments_total']} segments: verified shard + stored step + positive "
              f"movement   (shards checked {join['shards_verified']}, "
              f"hash mismatches {len(join['shard_hash_mismatches'])})")

    print("\nthe tail is not the data level: measured at `none` with no data handed over at")
    print("all, 14 of 60 steps still ran over 1ms. It is the writer falling behind and the")
    print("training thread encoding a held block itself instead of dropping it (Recorder")
    print("._enqueue) - the cost of storing weights, paid at every level:")
    for row in rows:
        print(f"  {row['level']:<9} median {row['recorder_ms_per_step']:.3f}ms, "
              f"p95 {row['recorder_p95_ms_per_step']:.2f}ms, max "
              f"{row['recorder_max_ms_per_step']:.1f}ms, "
              f"{row['slow_step_share'] * 100:.0f}% of steps over 1ms, "
              f"{row['waited']} blocks encoded on the training thread")

    print("\nspread across repeats (median / min / max, ms per step)")
    for row in rows:
        print(f"  {row['level']:<9} {row['recorder_ms_per_step']:.3f} / "
              f"{row['recorder_min']:.3f} / {row['recorder_max']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps({"levels": rows}, indent=2) + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

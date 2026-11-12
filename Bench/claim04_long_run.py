"""Claim 4, second half: the cost of a recording over a long run.

The short sweep in `claim04_storage_vs_checkpoints.py` answers "which is cheaper at a given
accuracy", and for a 40-step run the answer favours checkpoints: a stored step of ours is
about one whole checkpoint, so a coarse interval is hard to beat.

That is the wrong shape of question for a training run, which has thousands of steps. There,
checkpointing every step costs the model's size times the step count, while the recorder's cost
per step falls as the phases let it skip. This measures the crossover instead of assuming it.

Reported per architecture: the model's own size, the bytes for a checkpoint at every step, the
recorder's bytes at the same accuracy target, and the ratio. A ratio above 1 would mean the
recorder lost, and it is printed as it comes out.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .claim_arch_sweep import DATASET_SIZE, SkipBench, _torch, batches, build_models
from .claim04_storage_vs_checkpoints import forward


def measure(name: str, steps: int, budget: float) -> dict:
    import archx
    from archx.container.reader import Reader
    from archx.policy import PhasePolicy

    torch = _torch()
    model, regions = build_models()[name]
    one_checkpoint = sum(value.nbytes for value in model.state_dict().values())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    dataset = list(batches(name, DATASET_SIZE))

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.archx.mdl"
        recorder = archx.attach(model, optimizer, path=str(path), regions=regions,
                                max_rel_error=budget,
                                policy=PhasePolicy(total_steps=steps))
        for step in range(steps):
            inputs, targets = dataset[step % DATASET_SIZE]
            optimizer.zero_grad()
            loss = criterion(forward(model, inputs), targets)
            loss.backward()
            recorder.log(loss=float(loss.item()), data=inputs)
            optimizer.step()
        recorder.close()

        reader = Reader(path)
        from archx.container.schema import BlockType

        stored = len(list(reader.blocks(BlockType.DELTA))) + len(list(reader.blocks(BlockType.ANCHOR)))
        recorded = path.stat().st_size

    every_step = one_checkpoint * steps
    return {
        "architecture": name,
        "steps": steps,
        "one_checkpoint_bytes": one_checkpoint,
        "checkpoint_every_step_bytes": every_step,
        "recorder_bytes": recorded,
        "stored_steps": stored,
        "ratio": recorded / every_step if every_step else None,
        "shrink_vs_one_checkpoint_per_step": every_step / recorded if recorded else None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--budget", type=float, default=1e-2)
    parser.add_argument("--architectures", default="cnn,rnn,multimodal,ssm_like,moe_ish")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        _torch()
    except SkipBench as exc:
        print(f"skipped: {exc}")
        return 2

    names = [name.strip() for name in args.architectures.split(",") if name.strip()]
    rows = [measure(name, args.steps, args.budget) for name in names]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    print(f"{args.steps} steps, accuracy target {args.budget:g} per delta\n")
    header = (f"{'architecture':<14}{'one ckpt':>10}{'every step':>13}{'stored':>8}"
              f"{'recorder':>11}{'ratio':>8}{'saving':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['architecture']:<14}{row['one_checkpoint_bytes']:>10}"
              f"{row['checkpoint_every_step_bytes']:>13}{row['stored_steps']:>8}"
              f"{row['recorder_bytes']:>11}{row['ratio']:>7.2f}x"
              f"{row['shrink_vs_one_checkpoint_per_step']:>8.1f}x")
    print("\n'ratio' is recorder bytes / a checkpoint at every step; below 1.0 means recording"
          "\nis the smaller option. Comparing against a coarser checkpoint interval is a"
          "\ndifferent question and it is answered in claim04_storage_vs_checkpoints.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

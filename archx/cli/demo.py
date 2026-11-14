"""`archx demo`: produce a real recording you can poke at, with no dataset and no setup.

The first thing anyone wants after installing a recorder is a file. Waiting for them to write
a training loop first means the product is judged on its documentation. This trains a small
model on data it generates, records it at tier 3, and then tells the reader what to type next.

It deliberately writes the file and *does not* analyse it: the point is to leave the reader
with a path and the commands to run on it.
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_demo(out: str = "demo.archx.mdl", *, steps: int = 20, quiet: bool = False) -> int:
    """Record a small run to `out`. Returns a shell exit code."""

    def say(message: str) -> None:
        if not quiet:
            print(message)

    try:
        import torch
    except ImportError:
        print("archx: this demo needs torch.\n  pip install torch", file=sys.stderr)
        return 1

    import archx

    path = Path(out)
    if path.exists():
        # The recorder refuses to overwrite a recording, and a demo should not be the one
        # place that surprises someone by destroying a file they already had.
        print(f"archx: {out} already exists; choose another name with --out", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.randn(64, 32, generator=generator),
                torch.randint(0, 4, (64,), generator=generator)) for _ in range(steps)]

    recorder = archx.attach(
        model, optimizer, str(path),
        data_level="segments",
        activations=True,
        regions={"0": "input_projection", "2": "hidden", "4": "head"},
    )
    losses = []
    for inputs, targets in batches:
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        recorder.log(loss=float(loss.detach()), data={"inputs": inputs, "targets": targets},
                     source="synthetic demo", loss_reduction="mean")
        optimizer.step()
        losses.append(float(loss.detach()))
    recorder.close()

    size = path.stat().st_size
    say(f"recorded {steps} steps to {path} ({size:,} bytes)")
    say(f"loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    say("")
    say("now read it:")
    for command in (
        f"archx info          {path}",
        f"archx events        {path}",
        f"archx activations   {path}",
        f"archx diff          {path}",
        f"archx history       {path} --tensor 0.weight --row 0 --data",
        f"archx reconstruct   {path} --out weights.npz",
    ):
        say(f"  {command}")
    return 0

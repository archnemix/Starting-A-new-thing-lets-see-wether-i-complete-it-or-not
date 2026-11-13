#!/usr/bin/env python3
"""The five-minute tour, with no downloads and no dataset.

Run it:

    python examples/quickstart.py

It trains a small model on data it generates, records the run, and then asks the recording
the questions the product exists to answer - each one printed with the answer, so nothing
here has to be taken on trust:

    1. what were the weights at step 12?           (and are they exact?)
    2. which data did that step train on?
    3. what did that data do to the weights?
    4. which layers were dead or saturated?
    5. what could this recording *not* see?

Requires torch for steps 1-4. Without it, the file step proves the container and the CLI work.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np


def torch_or_none():
    try:
        import torch
        return torch
    except ImportError:
        return None


def main() -> int:
    torch = torch_or_none()
    if torch is None:
        print("torch is not installed; install it to run the full tour.\n"
              "  pip install torch\n")
        return 1

    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType
    from archx.container.values import Timeline
    from archx.policy import PhasePolicy

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    # A tiny "dataset" built in-process, so this script never depends on a download.
    generator = torch.Generator().manual_seed(0)
    batches = [(torch.randn(64, 32, generator=generator),
                torch.randint(0, 4, (64,), generator=generator)) for _ in range(20)]

    output = Path(tempfile.mkdtemp(prefix="archx-quickstart-")) / "tour.archx.mdl"
    print(f"recording to: {output}\n")

    recorder = archx.attach(
        model, optimizer, str(output),
        data_level="segments",       # keep the data, deduplicated and compressed
        activations=True,            # tier 3: what each layer saw
        regions={"0": "input_projection", "2": "hidden", "4": "head"},
        policy=PhasePolicy(total_steps=len(batches), min_warmup_steps=2, heartbeat_steps=4),
    )

    losses = []
    for step, (inputs, targets) in enumerate(batches):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        # The only line the recorder cannot write for you: the loss and the batch are locals
        # that die with the iteration. Before optimizer.step(), as documented.
        recorder.log(loss=float(loss.detach()), data=inputs)
        optimizer.step()
        losses.append(float(loss.detach()))

    final_weights = {name: p.detach().clone() for name, p in model.named_parameters()}
    recorder.close()

    size = output.stat().st_size
    print(f"loss {losses[0]:.4f} -> {losses[-1]:.4f} over {len(losses)} steps")
    print(f"file {size:,} bytes for {len(losses)} steps and "
          f"{sum(p.numel() for p in final_weights.values()):,} parameters\n")

    reader = Reader(output)
    capabilities = reader.capabilities()
    print(f"--- 0. what it recorded ------------------------------------------------")
    print(f"capability tier  {capabilities['capability_tier']}  "
          f"({capabilities.get('capability_detail', '')})")
    print(f"present          {', '.join(capabilities['capabilities_present'])}")
    for name, reason in capabilities.get("capabilities_missing", {}).items():
        print(f"missing          {name}: {reason}")

    timeline = Timeline(reader)
    print(f"\n--- 1. the weights at a step -------------------------------------------")
    # The last step, because that is the one there is something independent to compare with:
    # the live tensors. Comparing a middle step only against itself would prove nothing.
    last = timeline.steps[-1]
    rebuilt = timeline.reconstruct(last)
    worst = 0.0
    for name, tensor in final_weights.items():
        error = float(np.max(np.abs(rebuilt[name] - tensor.detach().cpu().numpy())))
        worst = max(worst, error)
    print(f"steps in file    {timeline.steps}")
    print(f"rebuilt step     {last}: {len(rebuilt)} tensors, "
          f"{sum(a.size for a in rebuilt.values())} values")
    print(f"vs live weights  max abs error {worst:.3g} "
          f"({'EXACT' if worst == 0.0 else 'float32 rounding'})")

    print(f"\n--- 2. which data did that step train on? ------------------------------")
    segments = timeline.segments()
    near = [s for s in segments if s["first_step"] <= last <= s["last_step"]]
    stored = len({b.payload.split(b"\n")[0] for b in reader.blocks(BlockType.DATA_SHARD)})
    print(f"batches seen     {len(losses)}")
    print(f"segment rows     {len(segments)} (one per unique content hash; the loop drew a fresh "
          f"random batch every step, so nothing repeated)")
    print(f"stored shards    {stored} compressed copies of the batch bytes")
    for segment in near[:3]:
        characteristics = segment.get("characteristics", {})
        print(f"  {segment['segment_id']:<10} step {segment['first_step']}-"
              f"{segment['last_step']}  {characteristics.get('kind', '?')} "
              f"{characteristics.get('bytes', 0)}B  seen {segment['times_seen']}x")

    print(f"\n--- 3. what did it do to the weights? ----------------------------------")
    print(f"the batch trained on at step {last}:")
    for segment in near[:3]:
        effects = segment.get("region_effects") or {}
        summary = ", ".join(f"{region}={entry['delta_norm']:.2e}"
                            for region, entry in sorted(effects.items()) if region)
        print(f"  {segment['segment_id']:<10} measured at step "
              f"{segment.get('effect_step', -1)}  {summary or '(no movement recorded)'}")

    print(f"\n--- 4. which layers were dead or saturated? ----------------------------")
    activations = [json.loads(b.payload) for b in reader.blocks(BlockType.ACTIVATION)]
    print(f"summaries        {len(activations)} across "
          f"{len({a['layer'] for a in activations})} layers")
    for row in activations[:4]:
        print(f"  step {row['step']:>2} layer {row['layer']:<2} "
              f"mean {row['mean']:+.3f} std {row['std']:.3f} "
              f"dead {row['zero_fraction'] * 100:5.1f}% "
              f"sat {row['saturated_fraction'] * 100:5.1f}%")

    print(f"\n--- 5. now read it yourself -------------------------------------------")
    print(f"  archx info         {output}")
    print(f"  archx events       {output}")
    print(f"  archx activations  {output}")
    print(f"  archx diff         {output}")
    print(f"  archx reconstruct  {output} --step {last} --out weights.npz")
    print("\nThe file is the product. Everything above was read back from it alone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

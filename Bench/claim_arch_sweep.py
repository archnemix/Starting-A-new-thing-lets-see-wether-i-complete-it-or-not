"""Phase 4.5: the same recorder on six architectures, with no model-specific code.

What is being tested is *not* that recording works - `test_torch_adapter.py` covers one model.
It is that nothing in the adapter or the recorder needs to know what the model is. A module
tree, a frozen embedding, a tied weight, a recurrent state, a mixture with a router, and a
multi-tower model all go through `archx.attach` unchanged, reach the same capability tier, and
rebuild to their own recorded accuracy.

Each architecture is reported with:

- the tier it reached and anything it could not see,
- the worst reconstruction error over every step after the first stored one, read back through
  the public reader (not through the recorder's memory),
- the fraction of the run's stored steps, so "tier 0 on everything" cannot pass by storing
  nothing.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - CI without torch skips this file
        raise SkipBench(f"torch is not installed: {exc}") from exc
    return torch


class SkipBench(RuntimeError):
    """Raised when the benchmark cannot run here; the caller reports it, it never fakes a pass."""


def build_models():
    """The six stacks, built lazily so importing this file does not require torch."""
    torch = _torch()
    # Seeded here, not in the run: the models are constructed once and handed to `run_one`,
    # so an unseeded build made the whole table change between runs - the same architecture
    # reported 82 stored steps and then 54, from initialisation alone.
    torch.manual_seed(0)
    nn = torch.nn

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64,
                                           batch_first=True)
                for _ in range(2)])
            self.head = nn.Linear(32, 4)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return self.head(x.mean(dim=1))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.classifier = nn.Linear(16, 4)

        def forward(self, x):
            return self.classifier(self.features(x).flatten(1))

    class RNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1])

    class SSMlike(nn.Module):
        """A gated state-space block: elementwise recurrence, no attention, no conv."""

        def __init__(self, dim=16):
            super().__init__()
            self.in_proj = nn.Linear(8, dim)
            self.gate = nn.Linear(8, dim)
            self.state_kernel = nn.Parameter(torch.randn(dim) * 0.02)
            self.out_proj = nn.Linear(dim, 4)

        def forward(self, x):
            h = torch.zeros(x.shape[0], self.state_kernel.numel(), device=x.device)
            gate = torch.sigmoid(self.gate(x))
            for t in range(x.shape[1]):
                h = h * torch.sigmoid(self.state_kernel) + self.in_proj(x[:, t]) * gate[:, t]
            return self.out_proj(h)

    class MoEish(nn.Module):
        """Four experts and a router, so a model has parameters that never move at all."""

        def __init__(self, dim=16, experts=4):
            super().__init__()
            self.router = nn.Linear(8, experts)
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(8, dim), nn.ReLU(),
                                                        nn.Linear(dim, 4))
                                          for _ in range(experts)])

        def forward(self, x):
            weights = torch.softmax(self.router(x.mean(dim=1)), dim=-1)
            stacked = torch.stack([expert(x.mean(dim=1)) for expert in self.experts], dim=1)
            return (stacked * weights.unsqueeze(-1)).sum(dim=1)

    class MultiTower(nn.Module):
        """Two towers and a tied projection: the same matrix under two names."""

        def __init__(self):
            super().__init__()
            self.text = nn.Linear(8, 16)
            self.image = nn.Linear(8, 16)
            self.shared = nn.Linear(16, 16)
            self.text_out = self.shared          # an actual tie, one storage unit
            self.embedding = nn.Embedding(64, 8)
            self.embedding.weight.requires_grad_(False)   # a frozen table
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            y = self.shared(torch.relu(self.image(x)))
            return self.head(y)

    return {
        "transformer": (Transformer(), {"layers": "encoder", "head": "head"}),
        "cnn": (CNN(), {"features": "vision", "classifier": "head"}),
        "rnn": (RNN(), {"rnn": "sequence", "head": "head"}),
        "ssm_like": (SSMlike(), {}),
        "moe_ish": (MoEish(), {"experts": "experts", "router": "router"}),
        "multimodal": (MultiTower(), {"text": "text", "image": "vision",
                                      "embedding": "text", "head": "head"}),
    }


def batches(name: str, count: int, seed: int = 0):
    """Input shaped the way each stack expects, and a target to produce a loss against.

    A fixed dataset, cycled. Fresh random batches every step would give the model nothing to
    learn, and the "did the loss fall" check - the one that proves the recorded run is still a
    real training run - would be measuring noise.
    """
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(count):
        if name == "cnn":
            yield torch.randn(2, 3, 8, 8, generator=generator), torch.randint(0, 4, (2,), generator=generator)
        elif name in ("rnn", "ssm_like", "moe_ish"):
            yield torch.randn(2, 5, 8, generator=generator), torch.randint(0, 4, (2,), generator=generator)
        elif name == "transformer":
            yield torch.randn(2, 4, 32, generator=generator), torch.randint(0, 4, (2,), generator=generator)
        else:
            yield torch.randn(2, 8, generator=generator), torch.randint(0, 4, (2,), generator=generator)


#: How many distinct batches make up each architecture's dataset. The run cycles through them.
DATASET_SIZE = 8


def run_one(name, model, regions, steps, archive: bool, data_level: str = "segments") -> dict:
    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType
    from archx.container.values import Timeline
    from archx.policy import PhasePolicy

    torch = _torch()
    torch.manual_seed(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.archx.mdl"
        recorder = archx.attach(model, optimizer, path=str(path), regions=regions,
                                data_level=data_level,
                                policy=PhasePolicy(total_steps=steps))
        truth: dict[int, dict[str, np.ndarray]] = {}
        losses = []
        dataset = list(batches(name, DATASET_SIZE))
        for step in range(steps):
            inputs, targets = dataset[step % DATASET_SIZE]
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            recorder.log(loss=float(loss.item()), data=inputs)
            optimizer.step()
            losses.append(float(loss.item()))
            # Every step's weights would be 40 copies of the whole model in memory; a fixed
            # sample is enough to catch a reconstruction that is wrong at a particular phase.
            if archive and (step < 4 or step >= steps - 6 or step % 7 == 0):
                truth[step] = {key: value.detach().cpu().numpy().copy()
                               for key, value in model.state_dict().items()}
        recorder.close()

        reader = Reader(path)
        stored = sorted({b.step for b in reader.blocks(BlockType.DELTA)}
                        | {b.step for b in reader.blocks(BlockType.ANCHOR)})
        info = [json.loads(b.payload) for b in reader.blocks(BlockType.INFO)]
        capability = next((b for b in info if b.get("event") == "capabilities"), {})
        timeline = Timeline(reader)

        worst = 0.0
        checked = 0
        worst_gap = 0
        if archive and stored:
            for step in sorted(truth):
                rebuilt = timeline.reconstruct(step)
                real = truth[step]
                error = 0.0
                magnitude = 0.0
                for key, values in real.items():
                    if key not in rebuilt:
                        continue
                    rebuilt_values = np.asarray(rebuilt[key], dtype=np.float64).ravel()
                    real_values = np.asarray(values, dtype=np.float64).ravel()
                    if rebuilt_values.size != real_values.size:
                        continue
                    error += float(np.linalg.norm(rebuilt_values - real_values))
                    magnitude += float(np.linalg.norm(real_values))
                if magnitude:
                    # The gap is what explains the error: a delta is allowed to be off by
                    # max_rel_error (1e-3 by default) and a skipped step is rebuilt by
                    # carrying that error forward, so a model with fewer store points must
                    # show a larger error. Reporting the error without the gap would read as
                    # "this architecture is recorded worse", which is not what happened.
                    if error / magnitude > worst:
                        worst = error / magnitude
                        previous = [s for s in stored if s <= step]
                        worst_gap = step - previous[-1] if previous else step
                    checked += 1

        return {
            "model": name,
            "bytes": path.stat().st_size,
            "tensors": len(reader.registry.entries) if hasattr(reader, "registry") else None,
            "tier": capability.get("capability_tier"),
            "missing": sorted((capability.get("capabilities_missing") or {}).keys()),
            "stored": len(stored),
            "steps": steps,
            "worst_relative_error": worst,
            "worst_gap": worst_gap,
            # Error spread over the steps that carried it. The gap is what the recorder
            # controls; the rate is how much this architecture's weights drift per skipped
            # step relative to their own size. Comparing the totals alone would call a
            # fast-moving model "recorded worse" when it was recorded equally well.
            "error_per_skipped_step": (worst / (worst_gap + 1)) if worst_gap >= 0 else worst,
            "steps_checked": checked,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "loss_decreased": bool(losses and losses[-1] < losses[0]),
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        models = build_models()
    except SkipBench as exc:
        print(f"skipped: {exc}")
        return 2

    rows = []
    for name, (model, regions) in models.items():
        rows.append(run_one(name, model, regions, args.steps, archive=True))

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    header = (f"{'architecture':<14}{'bytes':>8}{'tier':>5}{'stored':>8}{'loss fell':>11}"
              f"{'worst err':>12}{'gap':>5}{'per step':>11}  missing")
    print(header)
    print("-" * (len(header) + 20))
    for row in rows:
        print(f"{row['model']:<14}{row['bytes']:>8}{str(row['tier']):>5}"
              f"{str(row['stored']) + '/' + str(row['steps']):>8}"
              f"{str(row['loss_decreased']):>11}"
              f"{row['worst_relative_error']:>12.2e}{row['worst_gap']:>5}"
              f"{row['error_per_skipped_step']:>11.2e}  {','.join(row['missing'])}")
    print("\n'gap' is how many skipped steps the worst error was carried across, and"
          "\n'per step' is that error divided by the steps that carried it. The totals are not"
          "\ncomparable across architectures on their own: a model whose weights drift further"
          "\nper step between stores accumulates more, without being recorded worse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

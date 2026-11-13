"""Phase 8.3, claim 15: what tier 3 costs, and whether a forward-hook summary finds anything.

Two questions:

* **claim 15** - the price of activation capture, charged to the forward pass where it is
  actually paid. It is off by default for exactly this reason, and the number decides whether
  that default is right.
* **the value question** - a recorder that costs time and reports nothing useful is a tax. So
  a model is deliberately made sick (a ReLU layer with a large negative bias, a tanh driven
  into saturation) and the activation stream must be the thing that shows it, without seeing
  the loss curve.

Sampling is bounded by `MAX_ELEMENTS` per layer per forward, so the cost is a function of the
layer count and not the batch size. That claim is measured too: the same model at two batch
sizes, because "bounded" is a claim about the second one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path


def _torch():
    import torch
    return torch


class SkipBench(RuntimeError):
    """Raised when torch is missing; never faked into a number."""


def _model(torch, *, hidden: int = 256, depth: int = 4, sick: bool = False):
    """A small stack. `sick` kills half the ReLU units with a large negative bias."""
    layers = []
    width = hidden
    for index in range(depth):
        linear = torch.nn.Linear(width, hidden)
        if sick and index == 1:
            with torch.no_grad():
                linear.bias.fill_(-3.0)      # pushes half the pre-activations below zero
        layers += [linear, torch.nn.ReLU() if index % 2 == 0 else torch.nn.Tanh()]
    layers.append(torch.nn.Linear(hidden, 4))
    return torch.nn.Sequential(*layers)


def _forward_median_ms(*, activations: bool, batch: int, hidden: int, repeats: int = 3,
                       sick: bool = False) -> float:
    """Median wall time of one forward pass, with or without an activation capture.

    The forward is timed **with no writer thread running**: `step()` is not called inside the
    measurement window. The first version of this bench called it, so the writer thread was
    compressing blocks on the other core while the forward was being timed, and the same code
    reported +208% and then +503% on consecutive runs. That number was the machine, not the
    feature. Writing is measured separately, end to end, further down.
    """
    try:
        torch = _torch()
    except ImportError as exc:                       # pragma: no cover
        raise SkipBench(f"torch is not installed: {exc}") from exc
    import archx
    from archx.policy import PhasePolicy

    torch.manual_seed(0)
    model = _model(torch, hidden=hidden, sick=sick)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    inputs = torch.randn(batch, hidden)

    with tempfile.TemporaryDirectory() as directory:
        recorder = archx.attach(
            model, optimizer, str(Path(directory) / "acts.archx.mdl"),
            data_level="none", activations=activations,
            policy=PhasePolicy(total_steps=40, min_warmup_steps=1, heartbeat_steps=8))
        timings = []
        try:
            for _ in range(8):                    # warm-up, discarded
                with torch.no_grad():
                    model(inputs)
            for _ in range(40):
                started = time.perf_counter()
                with torch.no_grad():
                    model(inputs)
                timings.append((time.perf_counter() - started) * 1e3)
        finally:
            recorder.close()
    timings.sort()
    return timings[len(timings) // 2]


def measure(repeats: int = 3) -> dict:
    """The cost table, and the detection probe."""
    plains, withs = [], []
    for _ in range(repeats):
        plains.append(_forward_median_ms(activations=False, batch=256, hidden=256))
        withs.append(_forward_median_ms(activations=True, batch=256, hidden=256))
    plain = statistics.median(plains)
    captured = statistics.median(withs)

    # The cost must plateau once a layer exceeds MAX_ELEMENTS (256 x 256 = 65,536, so the cap
    # binds above a batch of 64). Three batch sizes, 4x apart: the *added* cost should grow
    # until the cap binds and then stop.
    scaling = {}
    for batch in (64, 256, 1024):
        plains_b = [_forward_median_ms(activations=False, batch=batch, hidden=256)
                    for _ in range(repeats)]
        caps_b = [_forward_median_ms(activations=True, batch=batch, hidden=256)
                  for _ in range(repeats)]
        scaling[batch] = {"plain_ms": statistics.median(plains_b),
                          "captured_ms": statistics.median(caps_b)}
        scaling[batch]["added_ms"] = (scaling[batch]["captured_ms"]
                                      - scaling[batch]["plain_ms"])
    wide_plain = scaling[1024]["plain_ms"]
    wide_captured = scaling[1024]["captured_ms"]

    detection = _detection_probe()
    end_to_end = _end_to_end_step_ms(repeats=repeats)
    return {
        "end_to_end": end_to_end,
        "forward": {
            "batch": 256,
            "plain_ms": plain,
            "captured_ms": captured,
            "added_ms": captured - plain,
            "added_percent": (captured - plain) / plain * 100 if plain else 0.0,
        },
        "batch_scaling": {str(batch): row for batch, row in scaling.items()},
        "detection": detection,
    }


def _end_to_end_step_ms(*, repeats: int = 3) -> dict:
    """A whole training step with the recorder running, which is what a user actually pays.

    Separate from the forward measurement on purpose: here the writer thread is doing its work
    at the same time, exactly as in a real run, and the difference includes that contention.
    """
    torch = _torch()
    import archx
    from archx.policy import PhasePolicy

    results = {}
    for name, activations in (("plain", False), ("activations", True)):
        medians = []
        for _ in range(repeats):
            torch.manual_seed(0)
            model = _model(torch, hidden=256, depth=4)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            inputs = torch.randn(256, 256)
            with tempfile.TemporaryDirectory() as directory:
                recorder = archx.attach(
                    model, optimizer, str(Path(directory) / "e2e.archx.mdl"),
                    data_level="none", activations=activations,
                    policy=PhasePolicy(total_steps=40, min_warmup_steps=1, heartbeat_steps=4))
                timings = []
                try:
                    for _ in range(40):
                        started = time.perf_counter()
                        with torch.no_grad():
                            model(inputs)
                        optimizer.step()
                        recorder.step(0)
                        timings.append((time.perf_counter() - started) * 1e3)
                finally:
                    recorder.close()
            timings.sort()
            medians.append(timings[len(timings) // 2])
        results[name] = statistics.median(medians)
    results["added_ms"] = results["activations"] - results["plain"]
    results["added_percent"] = (
        results["added_ms"] / results["plain"] * 100 if results["plain"] else 0.0)
    return results


def _detection_probe() -> dict:
    """Can the activation stream see a sick model that the loss curve looks fine on?"""
    import json as _json

    import archx
    from archx.container.reader import Reader
    from archx.container.schema import BlockType
    from archx.policy import PhasePolicy

    torch = _torch()
    torch.manual_seed(0)
    model = _model(torch, hidden=128, depth=4, sick=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    inputs = torch.randn(128, 128)
    targets = torch.randint(0, 4, (128,))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sick.archx.mdl"
        recorder = archx.attach(model, optimizer, str(path), data_level="none",
                                activations=True,
                                policy=PhasePolicy(total_steps=4, min_warmup_steps=1,
                                                   heartbeat_steps=1))
        losses = []
        for _ in range(4):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()                       # gradient norms, so the tier ladder holds
            losses.append(float(loss.detach()))
            recorder.log(loss=losses[-1], data=inputs)
            optimizer.step()                      # fires the hook: gradient norms, tier 1
        recorder.close()

        reader = Reader(path)
        rows = [_json.loads(block.payload) for block in reader.blocks(BlockType.ACTIVATION)]
        tiers = reader.capabilities()

    dead = [row for row in rows if row.get("zero_fraction", 0) >= 0.49]
    hot = [row for row in rows if row.get("saturated_fraction", 0) >= 0.5]
    loss_spread = max(losses) - min(losses)
    return {
        "summaries": len(rows),
        "layers": len({row["layer"] for row in rows}),
        "tier": tiers.get("capability_tier"),
        "dead_layer_summaries": len(dead),
        "worst_zero_fraction": max((row.get("zero_fraction", 0) for row in rows), default=0.0),
        "saturated_summaries": len(hot),
        "loss_spread": loss_spread,
        "loss_first": losses[0] if losses else 0.0,
        "loss_last": losses[-1] if losses else 0.0,
        # The claim: the activation stream flags the damage while the loss shows nothing.
        "loss_alone_would_not_show_it": loss_spread == 0.0,
        "found_without_the_loss": bool(dead),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="claim 15: the cost of tier 3")
    parser.add_argument("--out", help="write the table as JSON here")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        results = measure(repeats=args.repeats)
    except SkipBench as exc:
        print(f"SKIP: {exc}")
        return 0

    forward = results["forward"]
    print(f"{'forward pass':<22} {'median':>10} {'added':>10}")
    print("-" * 44)
    print(f"{'plain':<22} {forward['plain_ms']:>9.2f}ms")
    print(f"{'with activation capture':<22} {forward['captured_ms']:>9.2f}ms "
          f"{forward['added_ms']:>+9.2f}ms ({forward['added_percent']:+.1f}%)")

    e2e = results["end_to_end"]
    print(f"\nwhole training step (writer thread running, as in a real run)")
    print(f"  without activations {e2e['plain']:.2f}ms -> with {e2e['activations']:.2f}ms "
          f"({e2e['added_ms']:+.2f}ms, {e2e['added_percent']:+.1f}%)")

    print("\nthe cost plateaus once a layer exceeds MAX_ELEMENTS (stride-sampled, not copied)")
    for batch, row in results["batch_scaling"].items():
        print(f"  batch {batch:>4}: plain {row['plain_ms']:>6.2f}ms -> "
              f"captured {row['captured_ms']:>6.2f}ms  (+{row['added_ms']:.2f}ms)")

    detection = results["detection"]
    print("\ndoes the activation stream earn its cost?")
    print(f"  tier reached                 {detection['tier']}")
    print(f"  summaries / layers           {detection['summaries']} / {detection['layers']}")
    print(f"  dead-unit fraction (worst)   {detection['worst_zero_fraction'] * 100:.1f}%")
    print(f"  summaries flagged dead       {detection['dead_layer_summaries']}")
    print(f"  summaries flagged saturated  {detection['saturated_summaries']}")
    print(f"  loss {detection['loss_first']:.4f} -> {detection['loss_last']:.4f} "
          f"(spread {detection['loss_spread']:.3g})")
    print(f"  found without the loss       {detection['found_without_the_loss']}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

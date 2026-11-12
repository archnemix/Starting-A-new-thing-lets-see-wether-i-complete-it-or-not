"""Phase 7.3, claim 8: does the adapter depend on a torch version, and where is the floor?

The mechanism is `optimizer.register_step_post_hook`, which the plain-torch adapter relies on.
Claim 8 says reliability across torch versions; this benchmark produces the evidence instead of
the assurance.

Two layers, because they answer different questions:

* **static** - read the wheel *without installing it* and ask whether the API the adapter
  needs exists in that release. This can cover many versions cheaply and is what establishes
  the floor.
* **dynamic** - install one older torch into a throwaway environment and actually record a
  run, reading it back. Only this can catch a signature change or a behavioural difference.

Run `--versions 2.0.1 2.4.1 2.15.0` for the static table. `--dynamic` additionally runs one
recording per installable version that is available locally; when a version cannot be installed
here (disk, index) it is reported as unmeasured, never as passing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: The API the plain-torch adapter cannot work without.
REQUIRED = ("register_step_post_hook",)
#: Nice to have; reported but not fatal, so the floor is honest about what breaks where.
OPTIONAL = ("register_step_pre_hook",)


def wheel_for(version: str, directory: Path) -> tuple[Path | None, str]:
    """Download a wheel without installing it. Returns (path or None, why it failed).

    The reason is returned rather than swallowed: "not measured" is a claim about this
    machine, and a reader is entitled to know whether the index had no such version or the
    download ran out of disk halfway through.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "-q",
             "--dest", str(directory), f"torch=={version}"],
            capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "timed out after 900s"
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return None, f"pip exited {completed.returncode}: {tail[-1] if tail else 'no output'}"
    # A local version suffix (`2.4.1+cu121`) means the filename does not start with the
    # plain version: globbing `torch-2.4.1-*` misses it and the version was reported as "not
    # measured" while its wheel was sitting in the directory.
    matches = [path for path in sorted(directory.glob("torch-*.whl"))
               if path.name.split("-")[1].split("+")[0] == version]
    return (matches[0], "") if matches else (None, "pip reported success but wrote no wheel")




def static_check(wheel: Path) -> dict:
    """Look for the required API in `torch/optim/optimizer.py` inside the wheel."""
    try:
        with zipfile.ZipFile(wheel) as archive:
            source = archive.read("torch/optim/optimizer.py").decode("utf-8", "replace")
    except (KeyError, zipfile.BadZipFile) as exc:
        return {"wheel": wheel.name, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "wheel": wheel.name,
        "required": {name: (name in source) for name in REQUIRED},
        "optional": {name: (name in source) for name in OPTIONAL},
        "supported": all(name in source for name in REQUIRED),
    }


def dynamic_check(version: str) -> dict:
    """Record a real run in an environment that already has this torch, and read it back.

    Deliberately run in a subprocess: importing two torch builds in one process is not
    supported, and a crash here must not take the benchmark with it.
    """
    script = r"""
import json, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import archx
from archx.container.reader import Reader
from archx.container.values import Timeline

torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh(), torch.nn.Linear(6, 2))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
path = sys.argv[2]
recorder = archx.attach(model, optimizer, path, max_rel_error=0.0)
x = torch.randn(8, 4)
for step in range(5):
    optimizer.zero_grad()
    loss = torch.nn.functional.mse_loss(model(x), torch.randn(8, 2))
    loss.backward()
    recorder.log(loss=float(loss.detach()), lr=0.01)
    optimizer.step()
recorder.close()

timeline = Timeline(Reader(path))
live = {name: parameter.detach().numpy()
        for name, parameter in model.named_parameters(remove_duplicate=False)}
rebuilt = timeline.reconstruct(timeline.steps[-1])
print(json.dumps({
    "torch": torch.__version__,
    "steps": timeline.steps,
    "reconstruct_error": max(float(np.abs(rebuilt[k] - live[k]).max()) for k in live),
    "has_post_hook": hasattr(torch.optim.Optimizer, "register_step_post_hook"),
}))
"""
    with tempfile.TemporaryDirectory() as directory:
        recording = str(Path(directory) / "matrix.archx.mdl")
        completed = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve().parents[1]), recording],
            capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        return {"version": version, "error": completed.stderr.strip().splitlines()[-1:] or
                ["failed"], "supported": False}
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"version": version, "error": ["no parseable output"], "supported": False}
    payload["supported"] = (payload["steps"] == [0, 1, 2, 3, 4]
                            and payload["reconstruct_error"] < 1e-6)
    payload["version"] = payload.pop("torch")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="torch version matrix for claim 8")
    parser.add_argument("--versions", nargs="+", default=["2.0.1", "2.4.1", "2.15.0"])
    parser.add_argument("--dynamic", action="store_true",
                        help="also record a real run in an environment with that torch")
    parser.add_argument("--out", help="write the table as JSON here")
    args = parser.parse_args(argv)

    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for version in args.versions:
            for stale in Path(directory).glob("torch-*.whl"):
                stale.unlink()                  # never answer for the previous version's wheel
            wheel, why = wheel_for(version, Path(directory))
            if wheel is None:
                rows.append({"version": version, "supported": None, "note": why})
                continue
            row = static_check(wheel)
            row["version"] = version
            rows.append(row)

    print(f"{'torch':<10} {'register_step_post_hook':>24}  {'supported':>10}")
    print("-" * 50)
    for row in rows:
        found = row.get("required", {}).get("register_step_post_hook")
        mark = {True: "yes", False: "NO", None: "not measured"}[found]
        note = f"  ({row['note']})" if row.get("note") else ""
        print(f"{row['version']:<10} {mark:>24}  {str(row['supported']):>10}{note}")

    supported = [row["version"] for row in rows if row["supported"]]
    unsupported = [row["version"] for row in rows if row["supported"] is False]
    if supported:
        print(f"\nsupported: {', '.join(supported)}")
    if unsupported:
        print(f"NOT supported: {', '.join(unsupported)} - the adapter must say so, not fail "
              f"with an AttributeError mid-run")

    if args.dynamic:
        print("\ndynamic: a real recording, read back")
        for version in args.versions:
            result = dynamic_check(version)
            status = ("ok" if result.get("supported")
                      else f"FAILED: {result.get('error')}")
            print(f"  {version:<10} {status}")
            for row in rows:
                if row["version"] == version:
                    row["dynamic"] = result

    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows}, indent=2) + "\n")
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

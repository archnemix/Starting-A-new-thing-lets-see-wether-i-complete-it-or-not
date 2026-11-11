"""Run the test suite and write a machine-readable record of the run.

The record is what makes a test run evidence rather than an opinion: exact test ids,
inputs, outputs, versions, and the commit. Written to Actions/out/test-runs/.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str]) -> int:
    label = argv[1] if len(argv) > 1 else f"{os.environ.get('RUNNER_OS', 'local')}"
    out_dir = ROOT / "Actions" / "out" / "test-runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now(timezone.utc)
    cmd = [sys.executable, "-m", "pytest", "Tests", "-q", "-ra", "--tb=short"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = datetime.now(timezone.utc)

    summary, counts = _summarise(proc.stdout or "")
    record = {
        "purpose": "test run record written by Actions/scripts/run_tests.py",
        "label": label,
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "summary_line": summary,
        "counts": counts,
        "started_utc": started.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 3),
        "runner": {
            "os": platform.platform(),
            "github_os": os.environ.get("RUNNER_OS", "local"),
            "github_arch": os.environ.get("RUNNER_ARCH", ""),
            "python": platform.python_version(),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
        },
        "stdout_tail": (proc.stdout or "").strip().splitlines()[-40:],
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-20:],
    }
    target = out_dir / f"{label}.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    print(f"record: {target.relative_to(ROOT)}")
    return proc.returncode


def _summarise(stdout: str) -> tuple[str, dict[str, int]]:
    """Best-effort summary of a pytest run.

    pytest -q prints only dots when everything passes, so the dot line is parsed instead
    of invented: len(dots) is the number of tests that ran.
    """
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    summary = ""
    for raw in stdout.splitlines():
        line = raw.strip()
        for key, needle in (("passed", " passed"), ("failed", " failed"),
                            ("errors", " error"), ("skipped", " skipped")):
            if needle in line and any(ch.isdigit() for ch in line):
                summary = line.strip("= ")
        stripped = "".join(line.split("[")[0].split()).rstrip("%0123456789 ")
        if stripped and set(stripped) <= set(".sFEx") and len(stripped) > 3:
            counts["passed"] += stripped.count(".")
            counts["skipped"] += stripped.count("s")
            counts["failed"] += stripped.count("F")
            counts["errors"] += stripped.count("E")
    if not summary:
        summary = (f"{counts['passed']} passed, {counts['failed']} failed, "
                   f"{counts['errors']} errors, {counts['skipped']} skipped")
    return summary, counts


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

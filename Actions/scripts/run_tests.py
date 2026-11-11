"""Run the test suite and write a machine-readable record of the run.

The record is what makes a run evidence rather than an opinion, so it does not parse
pytest's console output: pytest writes JUnit XML and this reads it, which gives exact counts
and the name of every test that ran.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _parse_junit(path: Path) -> dict:
    """Exact counts and per-test outcomes from the JUnit XML pytest wrote."""
    if not path.is_file():
        return {"available": False}
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "total": 0}
    failures = []
    names = []
    for suite in suites:
        for case in suite.findall("testcase"):
            counts["total"] += 1
            name = f"{case.get('classname', '')}::{case.get('name', '')}"
            names.append(name)
            if case.find("failure") is not None:
                counts["failed"] += 1
                failures.append({"test": name, "detail": (case.findtext("failure") or "")[:800]})
            elif case.find("error") is not None:
                counts["errors"] += 1
                failures.append({"test": name, "detail": (case.findtext("error") or "")[:800]})
            elif case.find("skipped") is not None:
                counts["skipped"] += 1
            else:
                counts["passed"] += 1
    return {"available": True, "counts": counts, "failures": failures, "tests": names}


def main(argv: list[str]) -> int:
    label = argv[1] if len(argv) > 1 else os.environ.get("RUNNER_OS", "local")
    out_dir = ROOT / "Actions" / "out" / "test-runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    junit = out_dir / f"{label}.junit.xml"

    started = datetime.now(timezone.utc)
    cmd = [sys.executable, "-m", "pytest", "Tests", "-q", "-ra", "--tb=short",
           f"--junitxml={junit}"]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    finished = datetime.now(timezone.utc)

    parsed = _parse_junit(junit)
    counts = parsed.get("counts", {})
    summary = (f"{counts.get('total', 0)} tests: {counts.get('passed', 0)} passed, "
               f"{counts.get('failed', 0)} failed, {counts.get('errors', 0)} errors, "
               f"{counts.get('skipped', 0)} skipped")

    record = {
        "purpose": "test run record written by Actions/scripts/run_tests.py",
        "label": label,
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "summary_line": summary,
        "counts": counts,
        "failures": parsed.get("failures", []),
        "tests": parsed.get("tests", []),
        "started_utc": started.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 3),
        "runner": {
            "os": platform.platform(),
            "github_os": os.environ.get("RUNNER_OS", "local"),
            "github_arch": os.environ.get("RUNNER_ARCH", ""),
            "python": platform.python_version(),
            "pytest": _pytest_version(),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
        },
        "stdout_tail": (proc.stdout or "").strip().splitlines()[-40:],
        "stderr_tail": (proc.stderr or "").strip().splitlines()[-20:],
    }
    target = out_dir / f"{label}.json"
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    print(f"record: {target.relative_to(ROOT)}  ({summary})")
    return proc.returncode


def _pytest_version() -> str:
    try:
        import pytest

        return pytest.__version__
    except ImportError:  # pragma: no cover
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

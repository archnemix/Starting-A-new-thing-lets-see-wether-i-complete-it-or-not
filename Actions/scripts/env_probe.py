"""Write one small JSON describing the runner this executed on.

Used by .github/workflows/tests.yml to prove three things at once: the runner works,
the repo is checked out, and results can be committed back into the repository.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: env_probe.py <output.json>", file=sys.stderr)
        return 2

    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "purpose": "runner environment record written by Actions/scripts/env_probe.py",
        "written_utc": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "os_name": os.name,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": sys.version.split()[0],
            "github_os": os.environ.get("RUNNER_OS", ""),
            "github_arch": os.environ.get("RUNNER_ARCH", ""),
            "github_ref": os.environ.get("GITHUB_REF_NAME", ""),
            "github_sha": os.environ.get("GITHUB_SHA", ""),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
        },
        "checks": {
            "repo_checked_out": Path("README.md").is_file(),
            "can_write_output": True,
        },
    }

    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

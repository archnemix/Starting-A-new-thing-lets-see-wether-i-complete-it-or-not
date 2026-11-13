"""The file map says it must be updated whenever a file is created, changed or removed.

`Docs/Files_Info.json` opens with exactly that instruction, and by the time anyone checked it
listed 126 of 170 tracked files: every benchmark from claims 3 to 8, the architecture sweep,
`archx/cli/diff.py`, `Actions/triggers/scale.txt` and several test modules were invisible to it.
A map that is trusted and wrong is worse than no map, and the drift is invisible from inside the
repository because nothing reads the map except a person.

So the rule is a test. It fails when a tracked file has no entry, which is the moment someone
would otherwise have to remember.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "Docs" / "Files_Info.json"

#: Not in the map on purpose: these are outputs, not sources, and one entry per committed run
#: record would be a list of CI timestamps rather than a description of the repository.
EXEMPT_PREFIXES = ("Actions/out/",)


def test_every_tracked_file_has_an_entry():
    entries = set(json.loads(MAP.read_text())["files"])
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split()

    missing = [path for path in tracked
               if path not in entries and not path.startswith(EXEMPT_PREFIXES)]
    assert not missing, (
        "these tracked files have no entry in Docs/Files_Info.json, which says it must be "
        f"updated whenever a file is created: {missing}")


def test_no_entry_describes_a_file_that_no_longer_exists():
    entries = set(json.loads(MAP.read_text())["files"])
    tracked = set(subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                                 text=True, check=True).stdout.split())

    stale = sorted(path for path in entries if path not in tracked)
    assert not stale, f"Docs/Files_Info.json describes files that are not in the repo: {stale}"


def test_every_entry_has_the_four_fields_the_rules_list():
    files = json.loads(MAP.read_text())["files"]

    for path, entry in files.items():
        for field in ("purpose", "inputs", "outputs", "handles"):
            assert field in entry, f"{path} has no {field}"
        # Empty lists are legitimate - a licence file takes no input and produces none - so
        # this checks the shape of the answer, not that every file does something.
        assert isinstance(entry["inputs"], list), path
        assert isinstance(entry["outputs"], list), path
        assert entry["purpose"].strip() and entry["handles"].strip(), path

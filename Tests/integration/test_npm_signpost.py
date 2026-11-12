"""The npm package is a signpost: it must install, explain itself, and fail.

Skipped - loudly, with a reason - when node is not installed. It is never faked into a pass:
the point of these tests is that a wrong install cannot look like a working one, and a test
that passes without running node would defeat exactly that.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "npm" / "archx"
CLI = PACKAGE / "bin" / "archx.js"

#: Resolved once. `node` is a real executable on every platform; `npm` is a `.cmd` shim on
#: Windows and is deliberately not used here - see `test_the_node_suite_itself_passes`.
node = shutil.which("node")
pytestmark = pytest.mark.skipif(node is None, reason="node is not installed")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([node, str(CLI), *args], capture_output=True, text=True,
                          timeout=60, check=False)


def test_the_package_exists_and_declares_the_archx_binary():
    manifest = json.loads((PACKAGE / "package.json").read_text())
    assert manifest["name"] == "archx"
    assert manifest["bin"] == {"archx": "bin/archx.js"}
    assert "!win32" not in json.dumps(manifest), (
        "a signpost must install on every platform: blocking Windows would send a Windows "
        "user nowhere instead of telling them where the real tool is")
    assert manifest.get("private") is not True, "it is meant to be publishable"
    assert CLI.exists()


def test_using_it_fails_loudly_and_names_the_real_install():
    result = _run("train.py", "--out", "run.archx.mdl")
    assert result.returncode != 0, "a signpost that exits 0 makes a wrong install look right"
    assert "pip install archx-mdl" in result.stderr
    assert "is a signpost" in result.stderr
    assert result.stdout == "", "there is no output to produce"


def test_asking_about_the_package_succeeds():
    version = _run("--version")
    assert version.returncode == 0
    assert "npm signpost" in version.stdout
    assert "pip install archx-mdl" in version.stdout

    helped = _run("--help")
    assert helped.returncode == 0
    assert "pip install archx-mdl" in helped.stdout


def test_the_node_suite_itself_passes():
    """The package's own gate, run directly.

    `node --test`, not `npm test`: on Windows `npm` is a `.cmd` shim that `subprocess` cannot
    execute without a shell, and the first version of this test failed there with
    `FileNotFoundError: [WinError 2]` while passing on Linux and macOS. Shelling out for the
    privilege of resolving a script name is not worth a platform conditional - the same file
    runs either way, and the package manifest and CI still use `npm test` for users.
    """
    test_file = PACKAGE / "test" / "cli.test.js"
    assert test_file.exists()
    result = subprocess.run([node, "--test", str(test_file)], cwd=PACKAGE,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, f"node --test failed:\n{result.stdout}\n{result.stderr}"
    # The reporter prefix is not stable across node versions: 22 prints `ℹ fail 0`, 20 prints
    # `# fail 0`, and 18 prints `# fail 0` too. Matching a literal prefix made this test fail
    # on macOS while every one of the 8 node tests passed - the exit code is the real signal,
    # and this is a readability check on top of it.
    assert re.search(r"fail[ \t]+0\b", result.stdout), result.stdout
    assert re.search(r"pass[ \t]+([1-9]\d*)", result.stdout), "no tests ran"

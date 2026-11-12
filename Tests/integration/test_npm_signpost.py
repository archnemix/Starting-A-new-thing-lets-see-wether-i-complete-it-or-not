"""The npm package is a signpost: it must install, explain itself, and fail.

Skipped - loudly, with a reason - when node is not installed. It is never faked into a pass:
the point of these tests is that a wrong install cannot look like a working one, and a test
that passes without running node would defeat exactly that.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "npm" / "archx"
CLI = PACKAGE / "bin" / "archx.js"

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
    """`npm test` is the package's own gate; running it here keeps one command honest."""
    if shutil.which("npm") is None:
        pytest.skip("npm is not installed")
    result = subprocess.run(["npm", "test", "--silent"], cwd=PACKAGE, capture_output=True,
                            text=True, timeout=300, check=False)
    assert result.returncode == 0, f"npm test failed:\n{result.stdout}\n{result.stderr}"
    assert "# fail 0" in result.stdout, result.stdout

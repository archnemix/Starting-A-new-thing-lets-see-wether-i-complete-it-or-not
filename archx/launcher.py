"""Zero-code recording: `archx run -- python train.py`.

The mechanism is `sitecustomize`. `site.py` imports a module by that name during interpreter
start-up, and `PYTHONPATH` is searched before site-packages, so a copy placed on it wins. The
launcher writes `archx/_launcher_hook.py` into a temporary directory as `sitecustomize.py`,
prepends that directory to `PYTHONPATH`, runs the command, and then **checks whether anything
was recorded**.

That last part is the whole design. The first prototype patched `Optimizer.step` from a
`sitecustomize` that did `import torch` inside a `try/except`; torch is not importable that
early, so the patch silently never happened and the script ran to completion looking fine.
Two measured facts came out of chasing that:

* the patch has to be installed from a `sys.meta_path` finder that fires when `torch.optim`
  finishes importing, because at `sitecustomize` time torch does not exist yet;
* patching the base class alone captures **zero** steps for `AdamW`, which resolves to
  `Adam.step`, so every optimizer class that defines its own `step` is wrapped.

What the launcher deliberately cannot do, and says in the file:

* it never sees the loss or the batch - those are locals inside the user's loop - so the
  recording reports tier 0 and names `archx.attach` as the way to reach tier 1-2;
* parameter *names* are not recoverable from an optimizer, so tensors are recorded as
  `param[group][position]` and the metadata says `names: unavailable`.

It never leaves a plausible-looking empty recording behind: if no optimizer step was seen,
the file is deleted and the call raises with instructions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The module the interpreter imports by name; our hook ships as this file.
HOOK_FILENAME = "sitecustomize.py"
#: Where the hook lives inside the package, and the marker line used to recognise a stale copy.
HOOK_SOURCE = Path(__file__).with_name("_launcher_hook.py")
MARKER = "# archx: injected by `archx run` - safe to delete"


class LauncherError(RuntimeError):
    """The launcher could not do what it promised; nothing was recorded."""


def build_env(recording_path: str | os.PathLike[str],
              base_env: dict[str, str] | None = None) -> tuple[dict[str, str], Path]:
    """Write the injected `sitecustomize.py` and return (environment, temp directory).

    The caller owns the directory and should remove it with `cleanup` once the run is over.
    """
    directory = Path(tempfile.mkdtemp(prefix="archx-launcher-"))
    target = directory / HOOK_FILENAME
    # The path reaches the hook through the environment, not through string substitution:
    # a Windows path with backslashes and spaces in a generated literal is a bug waiting to
    # happen, and this file must behave identically on all three runners.
    source = HOOK_SOURCE.read_text(encoding="utf-8")
    injected = f"{MARKER}\n" + source
    # Compile before writing. A broken `sitecustomize` fails at interpreter start-up, where
    # Python reports "Error in sitecustomize" and carries on as if nothing happened - the
    # hook silently does nothing while the command runs to completion. That is exactly how a
    # marker line that was not a comment cost an hour, so it cannot be shipped again.
    try:
        compile(injected, str(target), "exec")
    except SyntaxError as exc:                          # pragma: no cover - a bug in this file
        raise LauncherError(f"the injected hook does not compile: {exc}") from exc
    target.write_text(injected, encoding="utf-8")

    env = dict(base_env if base_env is not None else os.environ)
    env["ARCHX_LAUNCHER_PATH"] = str(Path(recording_path).resolve())
    # The launcher's own interpreter must not patch itself when it imports archx; the child
    # is a fresh process with this variable absent.
    env.pop("ARCHX_LAUNCHER_ACTIVE", None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(directory) + (os.pathsep + existing if existing else "")
    return env, directory


def cleanup(directory: Path) -> None:
    """Remove the temporary hook directory. A leftover temp directory is not an error."""
    shutil.rmtree(directory, ignore_errors=True)


def steps_in(path: Path) -> int:
    """How many store points the child actually wrote. Zero means nothing was captured."""
    try:
        from .container.reader import Reader
        from .container.values import Timeline
        return len(Timeline(Reader(path)).steps)
    except Exception:                                  # noqa: BLE001 - a broken file is zero
        return 0


def run(command: list[str], recording_path: str | os.PathLike[str], *,
        base_env: dict[str, str] | None = None) -> int:
    """Run `command` with recording injected, and prove something was recorded.

    Returns the child's exit code. Raises `LauncherError` when no optimizer step was seen,
    after deleting the empty file, so a caller cannot mistake it for a recording.
    """
    if not command:
        raise LauncherError("nothing to run: pass the command after `--`")
    path = Path(recording_path)
    if path.exists():
        raise LauncherError(f"{path} already exists; refusing to overwrite a recording")

    env, directory = build_env(path, base_env=base_env)
    try:
        completed = subprocess.run(command, env=env)
    finally:
        cleanup(directory)

    observed = steps_in(path) if path.exists() else 0
    if observed == 0:
        if path.exists():
            path.unlink()
        raise LauncherError(
            "no optimizer step was observed, so no recording was written.\n"
            "  The launcher can only see torch.optim steps in this process tree. Use\n"
            "  `archx.attach(model, optimizer, path=...)` for a plain loop, or\n"
            "  `ArchxCallback(path)` for a Hugging Face Trainer.\n"
            f"  The command exited with status {completed.returncode}.")
    return completed.returncode


def main(argv: list[str], *, recording: str | None = None) -> int:
    """`archx run -- <command...>`."""
    command = argv[argv.index("--") + 1:] if "--" in argv else list(argv)
    target = recording or os.environ.get("ARCHX_RECORDING", "run.archx.mdl")
    try:
        code = run(command, target)
    except LauncherError as exc:
        print(f"archx: {exc}", file=sys.stderr)
        return 2
    print(f"archx: recorded {steps_in(Path(target))} steps into {target}")
    print("archx: this recording has no loss, data or parameter names (adapter=launcher); "
          "`archx.attach` or `ArchxCallback` gives a fuller one.")
    return code

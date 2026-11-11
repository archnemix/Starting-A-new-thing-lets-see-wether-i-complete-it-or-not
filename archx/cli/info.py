"""`archx info`: say what a recording contains without loading its weights."""

from __future__ import annotations

import json
import sys

from ..container.reader import Reader
from ..errors import ArchxError


def run_info(path: str, *, as_json: bool = False) -> int:
    """Print a summary of one recording. Returns a shell exit code."""
    try:
        reader = Reader(path)
    except ArchxError as exc:
        print(f"archx: cannot read {path}: {exc}", file=sys.stderr)
        return 1

    info = reader.info()
    if as_json:
        print(json.dumps(info, indent=2, sort_keys=True))
        return 0

    meta = info["meta"]
    print(f"file           {info['path']}")
    print(f"format         {info['format']}")
    print(f"size           {info['size_bytes']} bytes")
    print(f"closed         {info['closed']}" + ("  (crashed: run `archx recover`)" if not info["closed"] else ""))
    print(f"recovered      {info['recovered']}")
    print(f"blocks         {info['blocks']}  types={info['block_types']}")
    if meta:
        for key in ("adapter", "capability_tier", "data_level", "attribution"):
            if key in meta:
                print(f"{key:<14} {meta[key]}")
    return 0

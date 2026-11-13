"""`archx info`: say what a recording contains without loading its weights."""

from __future__ import annotations

import json
import sys

from ..container.reader import Reader
from ..errors import ArchxError


def run_info(path: str, *, as_json: bool = False) -> int:
    """Print a summary of one recording. Returns a shell exit code."""
    try:
        # A crashed recording is exactly when someone runs `archx info`, so a torn tail is
        # read, not refused: the CLI says the file is unclosed and points at `archx recover`.
        reader = Reader(path, allow_torn=True)
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
    state = "yes" if info["closed"] else "no  (crashed: run `archx recover`)"
    print(f"closed         {state}")
    print(f"recovered      {info['recovered']}")
    print(f"blocks         {info['blocks']}  types={info['block_types']}")
    capabilities = info.get("capabilities") or {}
    if meta:
        for key in ("adapter", "data_level", "attribution"):
            if key in meta:
                print(f"{key:<16} {meta[key]}")
        # Whether activation summaries are actually in the file, answered from the blocks.
        # Files written before this line existed said `activation_capture: false` in META
        # while holding hundreds of summaries - a header contradicting its own body - so a
        # stored `false` is not repeated here as fact.
        if capabilities.get("capabilities_present") and "activations" in capabilities[
                "capabilities_present"]:
            print(f"{'activation_capture':<16} yes")
            if reader.meta_activation_capture_disagrees():
                print(f"{'warning':<16} META says activation_capture false, but this file "
                      "holds activation summaries; its META block predates the fix and is "
                      "the wrong half of the contradiction")
        # The *achieved* tier comes from the capabilities block written at close, not from
        # META. META carries 0 - the floor, true by construction at open, before a single step
        # has happened - and this command printed that instead. A tier-3 recording therefore
        # reported `capability_tier 0` while listing every field it had recorded in the very
        # next line: a file contradicting itself on screen, in the command people run first.
        tier = capabilities.get("capability_tier")
        if tier is None:
            tier = meta.get("capability_tier")
            print(f"{'capability_tier':<16} {tier}  (from META: the run did not close)")
        else:
            print(f"{'capability_tier':<16} {tier}")
    # What this build had to skip or work around. A reader that quietly drops a block it does
    # not understand looks identical to a reader that read everything, and the difference
    # matters when the file was written by a newer version.
    if capabilities.get("capabilities_present"):
        print(f"{'records':<16} {', '.join(capabilities['capabilities_present'])}")
    for name, reason in (capabilities.get("capabilities_missing") or {}).items():
        print(f"{'missing':<16} {name} ({reason})")
    for warning in info.get("warnings") or []:
        print(f"{'warning':<16} {warning}")
    return 0

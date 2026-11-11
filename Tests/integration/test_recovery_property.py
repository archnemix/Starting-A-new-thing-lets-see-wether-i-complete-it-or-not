"""Property test: for any block sequence and any tear point, recovery keeps a prefix.

No hypothesis dependency: a fixed set of seeds gives reproducible, reviewable failures.
The invariant being defended: recovery never invents data and never reorders it.
"""

from __future__ import annotations

import json
import random

import pytest

from archx.container.reader import Reader
from archx.container.recovery import recover_in_place
from archx.container.schema import BlockType
from archx.container.writer import Writer


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_any_tear_point_leaves_a_valid_prefix(tmp_path, seed):
    rng = random.Random(seed)
    path = tmp_path / f"run-{seed}.archx.mdl"

    written_steps: list[int] = []
    with Writer(path) as w:
        w.write_block(BlockType.META, json.dumps({"seed": seed}).encode())
        for step in range(rng.randint(3, 12)):
            payload = bytes(rng.randrange(256) for _ in range(rng.randint(0, 300)))
            w.write_block(BlockType.DELTA, payload, step=step)
            written_steps.append(step)

    data = path.read_bytes()
    tear = rng.randrange(1, len(data) - 40)  # never inside the header
    path.write_bytes(data[:tear])

    report = recover_in_place(path)
    assert report.bytes_truncated >= 0

    reader = Reader(path)
    assert reader.container.footer is not None, "recovery must produce a closed file"
    meta = reader.meta()
    # A tear may land inside the META payload itself; then META is gone. What must never
    # happen is META coming back wrong.
    assert meta in ({}, {"seed": seed})
    surviving = [b.step for b in reader.blocks(BlockType.DELTA)]
    assert surviving == written_steps[: len(surviving)], "survivors are a prefix, in order"
    index = [step for step, _o, _l in Reader.index_block_payload(reader.container)]
    assert index == surviving, "the rebuilt index matches the blocks that survived"

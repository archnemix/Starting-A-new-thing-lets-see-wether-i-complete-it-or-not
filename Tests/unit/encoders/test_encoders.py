"""Unit tests for the delta encoders: exactness where promised, measured error elsewhere.

The point of these tests is not coverage, it is the contract: "unchanged" and "sparse" must
be lossless, quantized and low-rank must report the error they actually introduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from archx.container.compress import compress
from archx.container.schema import Codec
from archx.encoders import DenseQuant, DenseRaw, LowRank, Sparse, Unchanged, choose_encoder
from archx.encoders.base import relative_error
from archx.encoders.plane import shuffle, unshuffle


def compress_output(data: bytes) -> int:
    return len(compress(data, Codec.ZSTD))


def roundtrip(encoder, delta):
    encoded = encoder.encode(delta)
    assert encoded is not None
    rebuilt = encoder.decode(encoded.body, delta.shape, encoded.value_dtype, encoded.nnz_or_rank)
    return encoded, rebuilt


def test_unchanged_is_detected_and_free():
    delta = np.zeros((8, 8), dtype=np.float32)
    encoded, rebuilt = roundtrip(Unchanged(), delta)
    assert encoded.nbytes == 0
    assert encoded.rel_error == 0.0
    assert np.array_equal(rebuilt, delta)


def test_unchanged_refuses_a_delta_that_moved():
    assert Unchanged().encode(np.ones((4,), dtype=np.float32)) is None


def test_sparse_is_lossless_and_scales_with_movement_not_size():
    rng = np.random.default_rng(7)
    big = np.zeros(100_000, dtype=np.float32)
    big[rng.choice(100_000, 50, replace=False)] = rng.normal(0, 1e-3, 50)

    encoded, rebuilt = roundtrip(Sparse(threshold=0.0), big)
    assert np.array_equal(rebuilt, big), "sparse must be exact"
    assert encoded.nbytes < big.nbytes / 10, "50 of 100k values must cost far less than the tensor"
    assert encoded.nnz_or_rank == 50


def test_sparse_refuses_dense_input_where_indices_cost_more_than_values():
    dense = np.ones(1000, dtype=np.float32) * 1e-3
    assert Sparse().encode(dense) is None


def test_dense_quantisation_reports_its_real_error():
    rng = np.random.default_rng(1)
    delta = rng.normal(0, 1e-3, (64, 64)).astype(np.float32)
    encoded, rebuilt = roundtrip(DenseQuant(bits=8), delta)
    assert encoded.nbytes < delta.nbytes
    measured = relative_error(delta, rebuilt)
    assert measured == pytest.approx(encoded.rel_error, rel=1e-6), "reported error must be the real one"
    assert measured < 0.02, "int8 on a dense small delta should stay near 1%"


def test_lowrank_wins_on_structured_movement_and_refuses_noise():
    rng = np.random.default_rng(3)
    structured = (rng.normal(size=(128, 4)) @ rng.normal(size=(4, 128))).astype(np.float32) * 1e-3
    encoded, rebuilt = roundtrip(LowRank(max_rank=16, max_rel_error=1e-6), structured)
    assert encoded.nnz_or_rank == 4, "a rank-4 movement must be found at rank 4"
    assert encoded.nbytes < structured.nbytes / 4
    assert relative_error(structured, rebuilt) < 1e-6

    noise = rng.normal(0, 1e-3, (128, 128)).astype(np.float32)
    assert LowRank(max_rank=8, max_rel_error=1e-3).encode(noise) is None, \
        "low-rank must refuse what it cannot represent"


def test_choice_prefers_the_smallest_encoding_inside_the_budget():
    rng = np.random.default_rng(5)
    sparse_delta = np.zeros(4096, dtype=np.float32)
    # varied magnitudes: with identical values a quantizer's rounding can cancel exactly
    sparse_delta[rng.choice(4096, 10, replace=False)] = rng.normal(0, 1e-3, 10)

    candidates = [Sparse(), DenseQuant(bits=8), LowRank()]
    picked = choose_encoder(sparse_delta, candidates, max_rel_error=1e-6)
    assert picked is not None
    assert picked.encoding == Sparse.encoding_id
    assert picked.rel_error == 0.0


def test_dense_raw_is_the_exact_fallback_when_nothing_else_fits():
    rng = np.random.default_rng(11)
    noise = rng.normal(0, 1e-5, (128, 128)).astype(np.float32)
    assert Sparse().encode(noise) is None
    assert LowRank(max_rank=16, max_rel_error=1e-3).encode(noise) is None

    encoded, rebuilt = roundtrip(DenseRaw(), noise)
    assert encoded.rel_error == 0.0
    assert np.array_equal(rebuilt, noise), "the fallback must be bit exact"


def test_plane_shuffle_is_reversible_and_does_not_change_bytes():
    rng = np.random.default_rng(13)
    data = rng.integers(0, 256, 4000, dtype=np.uint8).tobytes()
    for itemsize in (2, 4, 8):
        assert unshuffle(shuffle(data, itemsize), itemsize) == data
        assert sorted(shuffle(data, itemsize)) == sorted(data), "a permutation can only reorder"


def test_plane_shuffle_measured_behaviour_is_recorded_not_assumed():
    # Bench/claim_plane_shuffle.py found a small gain for dense deltas and a real loss for
    # sparse ones. This test pins the reason: a fully zero tensor is shuffle-invariant, but a
    # mostly-zero one is not - shuffling scatters the zero runs zstd was exploiting.
    zeros = np.zeros(4096, dtype=np.float32).tobytes()
    assert compress_output(shuffle(zeros, 4)) == compress_output(zeros)

    rng = np.random.default_rng(17)
    sparse_delta = np.zeros(256 * 256, dtype=np.float32)
    sparse_delta[rng.choice(sparse_delta.size, sparse_delta.size // 100, replace=False)] = 1e-3
    raw = sparse_delta.tobytes()
    assert compress_output(shuffle(raw, 4)) > compress_output(raw), \
        "shuffling must stay off for zero-heavy deltas"

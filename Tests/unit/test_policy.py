"""Unit tests for the phase policy: dense at the ends, event-driven in the middle."""

from __future__ import annotations

import numpy as np
import pytest

from archx.policy import Decision, Phase, PhasePolicy, Triggers


def test_short_run_is_all_warmup_then_cooldown_never_stuck():
    policy = PhasePolicy(total_steps=100)
    assert policy.phase_at(0) is Phase.WARMUP
    assert policy.phase_at(50) is Phase.MIDDLE
    assert policy.phase_at(99) is Phase.COOLDOWN


def test_unknown_total_only_leaves_warmup_after_the_minimum():
    policy = PhasePolicy(total_steps=None)
    assert policy.phase_at(0) is Phase.WARMUP
    assert policy.phase_at(19) is Phase.WARMUP
    assert policy.phase_at(20) is Phase.MIDDLE


def test_first_step_is_an_anchor():
    policy = PhasePolicy(total_steps=1000)
    assert policy.decide(0) is Decision.ANCHOR


def test_warmup_stores_every_step_and_middle_mostly_skips():
    policy = PhasePolicy(total_steps=10_000)
    assert policy.decide(1) is Decision.STORE
    decisions = [policy.decide(step) for step in range(500, 560)]
    assert decisions.count(Decision.SKIP) > decisions.count(Decision.STORE)


def test_heartbeat_bounds_the_skipped_gap():
    policy = PhasePolicy(total_steps=10_000, heartbeat_steps=5)
    decisions = [policy.decide(step) for step in range(500, 530)]
    assert Decision.STORE in decisions
    longest_gap = 0
    gap = 0
    for decision in decisions:
        gap = 0 if decision is not Decision.SKIP else gap + 1
        longest_gap = max(longest_gap, gap)
    assert longest_gap < 5


def test_a_loss_drop_triggers_storage_in_the_middle():
    policy = PhasePolicy(total_steps=10_000)
    policy.decide(500)
    policy.observe(loss=1.0)
    assert policy.decide(501, loss=0.9) is Decision.STORE


def test_a_gradient_spike_triggers_storage_once_a_baseline_exists():
    policy = PhasePolicy(total_steps=10_000)
    policy.decide(500)
    for i in range(10):
        policy.observe(grad_norm=1.0 + 0.01 * i)
    assert policy.decide(510, grad_norm=5.0) is Decision.STORE


def test_a_big_region_move_triggers_storage():
    policy = PhasePolicy(total_steps=10_000)
    policy.decide(500)
    assert policy.decide(501, region_movement=0.5) is Decision.STORE


def test_explicit_events_and_anchors_win_over_the_phase():
    policy = PhasePolicy(total_steps=10_000, anchor_every=250)
    policy.decide(500)
    assert policy.decide(501, event=True) is Decision.STORE
    assert policy.decide(750) is Decision.ANCHOR


def test_a_gradient_spike_is_seen_even_when_history_is_flat():
    """A flat baseline must not switch the trigger off.

    The run this trigger exists for is a quiet stretch followed by a spike, and a quiet
    stretch has zero variance. Guarding the z-score with `std > 0` disabled the trigger in
    precisely that case, so the spike that should have been recorded was skipped.
    """
    policy = PhasePolicy(total_steps=10_000)
    policy.decide(500)
    for _ in range(10):
        policy.observe(grad_norm=1.0)
    assert policy.decide(510, grad_norm=5.0) is Decision.STORE


def test_an_infinite_threshold_disables_the_trigger_even_when_history_is_flat():
    """Infinity is how a caller says "never react"; the flat-baseline path must honour it."""
    policy = PhasePolicy(total_steps=10_000,
                         triggers=Triggers(grad_norm_zscore=float("inf")))
    policy.decide(500)
    for _ in range(10):
        policy.observe(grad_norm=1.0)
    assert policy.decide(510, grad_norm=50.0) is Decision.SKIP


def test_a_region_move_observed_after_a_store_triggers_the_next_step():
    """`decide` cannot see movement it has not measured yet, so `observe` must carry it.

    Movement is only measurable between two stored steps, which means it arrives one step
    after the decision it should have influenced. Passing it to `observe` is what makes the
    trigger reachable at all - it used to be dead code.
    """
    policy = PhasePolicy(total_steps=10_000)
    policy.decide(500)
    policy.observe(region_movement={"block0": 0.5})
    assert policy.decide(501) is Decision.STORE


def test_region_movement_is_a_fraction_of_the_tensor_not_an_absolute_norm():
    """The threshold is named `region_move_fraction`, so the quantity must be a fraction.

    An absolute Frobenius norm is not comparable between layers or architectures, and a
    256x256 layer of weights around 0.1 drifts by an absolute ~2.7 every step at 1e-2 noise.
    Compared against a 0.02 threshold, every step of that run reads as an event.
    """
    from Tests.unit.test_recorder import FakeModel
    from archx.formats.registry import TensorRegistry
    from archx.recorder import _movement_by_region

    model = FakeModel(seed=3, size=64)
    registry = model.registry()
    before = {k: v.copy() for k, v in model.params.items()}
    after = {k: v + 0.01 for k, v in model.params.items()}   # 0.01 absolute

    movement = _movement_by_region(registry, before, after)
    assert movement, "a change must be reported"

    # Every tensor in this fake model sits in one region. `w1` has a magnitude to be a
    # fraction of; `b1` is all zeros, so it has no baseline and must be left out rather than
    # reported as 100% movement.
    expected = 0.0
    for key in ("w1",):
        delta = np.full(before[key].shape, 0.01, dtype=np.float32)
        expected += float(np.linalg.norm(delta)) / float(np.linalg.norm(before[key]))
    assert sum(movement.values()) == pytest.approx(expected, rel=1e-5), (
        "movement must be reported relative to each tensor's own magnitude")
    assert sum(movement.values()) < 1.0


def test_a_zero_initialised_tensor_does_not_report_permanent_movement():
    """Biases start at zero. A fraction of zero is not a fraction, and treating it as one
    makes the trigger fire on every step for thousands of steps as it decays like 1/sqrt(n).
    """
    from Tests.unit.test_recorder import FakeModel
    from archx.recorder import _movement_by_region

    model = FakeModel(seed=3, size=64)
    registry = model.registry()
    before = {k: v.copy() for k, v in model.params.items()}
    assert not before["b1"].any()
    after = {k: v + 0.01 for k, v in model.params.items()}

    assert _movement_by_region(registry, before, after), "w1 still moved"
    assert _movement_by_region(registry, before, {**before, "b1": before["b1"]}) == {}, (
        "an unchanged model reports nothing")

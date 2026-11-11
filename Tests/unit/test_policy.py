"""Unit tests for the phase policy: dense at the ends, event-driven in the middle."""

from __future__ import annotations

from archx.policy import Decision, Phase, PhasePolicy


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

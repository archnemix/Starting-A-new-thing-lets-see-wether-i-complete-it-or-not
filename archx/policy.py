"""What to store at a given step: the phase-adaptive policy.

Three phases with different density, plus event triggers that cut through the middle phase.
Thresholds are per region, never global, because layer dynamics differ by orders of
magnitude between architectures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Phase(str, Enum):
    """Where a step sits in the run."""

    WARMUP = "warmup"
    MIDDLE = "middle"
    COOLDOWN = "cooldown"


class Decision(str, Enum):
    """The three things a policy can say about a step."""

    STORE = "store"       # record deltas now
    SKIP = "skip"         # do not record; the next stored delta absorbs it
    ANCHOR = "anchor"     # store a full snapshot (implies STORE)


@dataclass
class Triggers:
    """Event thresholds. All are per region unless stated otherwise."""

    grad_norm_zscore: float = 4.0
    loss_drop_fraction: float = 0.05
    region_move_fraction: float = 0.02


@dataclass
class PhasePolicy:
    """Default policy: dense at the ends, event-driven in the middle.

    `total_steps` may be a plan (known schedule) or an estimate (learned live by
    `observe_step`). Unknown totals are handled honestly: the policy starts in warmup and
    only leaves it once something is known.
    """

    total_steps: int | None = None
    warmup_fraction: float = 0.01
    cooldown_fraction: float = 0.01
    min_warmup_steps: int = 20
    heartbeat_steps: int = 100
    anchor_every: int | None = None
    triggers: Triggers = field(default_factory=Triggers)

    _max_step_seen: int = 0
    _last_loss: float | None = None
    _grad_history: list[float] = field(default_factory=list)
    _steps_since_store: int = 0
    #: Movement measured between the last two stored steps. It can only be seen after a
    #: store, so the trigger fires one step late - but with a real measurement rather than
    #: a guess. Measuring it on skipped steps would mean reading every weight every step,
    #: which is the cost the policy exists to avoid.
    _last_movement: float | None = None

    # ------------------------------------------------------------- phases

    def phase_at(self, step: int, total_steps: int | None = None) -> Phase:
        """Which phase a step belongs to. Warmup always wins if the run is short."""
        total = total_steps if total_steps is not None else self.total_steps
        if total:
            # The fraction is the policy; min_warmup_steps is only a floor for tiny runs, so
            # a 1% warmup of 10,000 steps is 100, not 20.
            warmup_steps = min(total, max(1, int(total * self.warmup_fraction)))
            warmup_steps = max(warmup_steps, min(self.min_warmup_steps, total))
            cooldown_start = total - max(1, int(total * self.cooldown_fraction))
        else:
            warmup_steps = self.min_warmup_steps
            cooldown_start = None
        if step < warmup_steps:
            return Phase.WARMUP
        if cooldown_start is not None and step >= cooldown_start:
            return Phase.COOLDOWN
        return Phase.MIDDLE

    # ------------------------------------------------------------- decisions

    def decide(self, step: int, *, loss: float | None = None,
               grad_norm: float | None = None,
               region_movement: float | None = None,
               event: bool = False,
               total_steps: int | None = None) -> Decision:
        """Decide what to do with this step, given what the hook could see."""
        phase = self.phase_at(step, total_steps)
        self._steps_since_store += 1

        if self.anchor_every and step > 0 and step % self.anchor_every == 0:
            return self._store(Decision.ANCHOR)
        if step == 0:
            return self._store(Decision.ANCHOR)
        if event:
            return self._store(Decision.STORE)
        if phase in (Phase.WARMUP, Phase.COOLDOWN):
            return self._store(Decision.STORE)
        if self._steps_since_store >= self.heartbeat_steps:
            return self._store(Decision.STORE)
        if self._is_significant(loss, grad_norm, region_movement):
            return self._store(Decision.STORE)
        return Decision.SKIP

    def _is_significant(self, loss, grad_norm, region_movement) -> bool:
        t = self.triggers
        if loss is not None and self._last_loss is not None and self._last_loss > 0:
            if (self._last_loss - loss) / self._last_loss >= t.loss_drop_fraction:
                return True
        if t.grad_norm_zscore != float("inf") and grad_norm is not None \
                and len(self._grad_history) >= 8:
            mean = sum(self._grad_history) / len(self._grad_history)
            var = sum((g - mean) ** 2 for g in self._grad_history) / len(self._grad_history)
            std = var ** 0.5
            if std == 0:
                # A perfectly flat history makes every deviation infinitely many sigmas.
                # Guarding this with `std > 0` used to disable the trigger in exactly the
                # run it exists for: the quiet stretch before a spike. A threshold of
                # infinity is how a caller switches the trigger off, so it must still win.
                if grad_norm != mean:
                    return True
            elif (grad_norm - mean) / std >= t.grad_norm_zscore:
                return True
        movement = region_movement if region_movement is not None else self._last_movement
        if movement is not None and movement >= t.region_move_fraction:
            return True
        return False

    def observe(self, *, loss: float | None = None, grad_norm: float | None = None,
                region_movement: dict[str, float] | None = None) -> None:
        """Feed the statistics that later decisions depend on. Call after deciding."""
        if region_movement:
            self._last_movement = max(region_movement.values(), default=0.0)
        if loss is not None:
            self._last_loss = loss
        if grad_norm is not None:
            self._grad_history.append(grad_norm)
            if len(self._grad_history) > 64:
                self._grad_history.pop(0)

    def _store(self, decision: Decision) -> Decision:
        self._steps_since_store = 0
        return decision

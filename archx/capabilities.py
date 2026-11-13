"""What this run could actually see, recorded in the file instead of claimed in a README.

The four tiers come from Architecture 6.11. A tier is a *measured* fact about one run, not a
request: if the recording has no gradient norms because the adapter could not reach them, the
file says tier 1 is unreachable and names the field that is missing. A reader that finds a
capability claimed but its data absent reports a corrupt file rather than trusting the claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Tier -> the fields it promises. Order matters: a tier is reachable only if the one below it is.
TIER_FIELDS = {
    0: ("steps", "deltas"),
    1: ("loss", "grad_norms"),
    2: ("data_segments",),
    3: ("activations",),
}

#: What each tier means, as this build can honestly say it. Tier 3 used to read "activation
#: statistics, eval results", and no adapter has ever recorded an eval result - so every
#: tier-3 file carried a promise in its own capability block that nothing in the build could
#: keep. The planned field is named in Architecture §6.11 and in the changelog; it is not
#: claimed here until there is a block to back it.
TIER_MEANING = {
    0: "step counter, weight deltas, timing",
    1: "loss, learning rate, gradient norms",
    2: "data segments and their effects on the weights",
    3: "activation statistics",
}


@dataclass
class Capabilities:
    """The tier this run reached, and why anything above it was unavailable."""

    available: set[str] = field(default_factory=set)
    missing: dict[str, str] = field(default_factory=dict)

    def have(self, name: str) -> bool:
        return name in self.available

    def offer(self, name: str, reason: str = "") -> None:
        """Mark a field available. An offered field is never silently dropped later.

        The earlier denial is cleared, not left behind: a field listed as both present and
        missing is a file that contradicts itself.
        """
        self.available.add(name)
        self.missing.pop(name, None)

    def deny(self, name: str, reason: str) -> None:
        """Say that a field is unavailable, and why - the reason goes into the file."""
        self.available.discard(name)
        self.missing[name] = reason

    def resolve(self) -> "Capabilities":
        """Settle the tier and give every absent field a reason.

        A tier's fields cannot be present if the lower tier's are not: without this, a run
        that saw data but no gradient norms would claim tier 2 while its own file shows tier 1
        was never reached - a claim a reader would have to disbelieve. And every field that is
        simply absent is named with a reason, so "tier 2" is a statement with its gaps
        attached rather than a promise the reader has to take on faith.
        """
        for tier in sorted(TIER_FIELDS):
            fields = TIER_FIELDS[tier]
            blocked = [lower for lower in TIER_FIELDS if lower < tier
                       and not all(f in self.available for f in TIER_FIELDS[lower])]
            for name in fields:
                if name in self.available:
                    # A field the run really did record is never taken away because a
                    # different field in its tier is missing. It used to be: a loop that
                    # handed over gradient norms but no loss produced a file saying gradient
                    # norms were unavailable, which is the one thing this module exists to
                    # get right - the file would have understated what it recorded.
                    continue
                if blocked:
                    reason = f"requires tier {min(blocked)}, which this run did not reach"
                else:
                    reason = "not provided by this adapter"
                self.missing.setdefault(name, reason)
        return self

    @property
    def tier(self) -> int:
        """Highest tier whose every field is present."""
        reached = 0
        for tier in sorted(TIER_FIELDS):
            if all(name in self.available for name in TIER_FIELDS[tier]):
                reached = tier
            else:
                break
        return reached

    def to_dict(self) -> dict:
        """The META fields `capability_tier` and `capabilities_missing`."""
        return {
            "capability_tier": self.tier,
            "capabilities_missing": dict(sorted(self.missing.items())),
            "capability_detail": TIER_MEANING[self.tier],
            # What is actually in the file, listed separately from the tier. A tier is a
            # threshold; this is the inventory, and it is what lets a reader see that a run
            # below tier 2 still has gradient norms for every step.
            "capabilities_present": sorted(self.available),
        }

"""The capability block is the file's own honesty report, so it must not understate the run."""

from __future__ import annotations

from archx.capabilities import Capabilities


def test_a_field_the_run_recorded_is_never_reported_missing():
    """A loop that hands over gradient norms but not loss.

    This used to report `grad_norms` as unavailable: `resolve()` discarded every field of an
    unreached tier, so the file understated what it had. A reader would have been told to
    ignore gradient norms that are in every EVENT block.
    """
    capabilities = Capabilities()
    for name in ("steps", "deltas", "grad_norms"):
        capabilities.offer(name)
    capabilities.resolve()

    assert capabilities.have("grad_norms")
    assert "grad_norms" not in capabilities.missing
    assert "loss" in capabilities.missing
    assert capabilities.tier == 0, "tier 1 still needs the loss, so the tier stays at 0"
    assert capabilities.to_dict()["capabilities_present"] == ["deltas", "grad_norms", "steps"]


def test_a_missing_lower_tier_explains_why_higher_fields_were_lost():
    capabilities = Capabilities()
    for name in ("steps", "deltas", "loss"):
        capabilities.offer(name)          # no gradient norms, no data
    capabilities.resolve()

    assert capabilities.tier == 0
    assert "grad_norms" in capabilities.missing
    assert "not provided" in capabilities.missing["grad_norms"]
    assert "requires tier 1" in capabilities.missing["data_segments"]


def test_offering_a_field_after_denial_promotes_it_back():
    capabilities = Capabilities()
    capabilities.deny("loss", "the loop never called log(loss=...)")
    assert not capabilities.have("loss")

    capabilities.offer("loss")
    capabilities.resolve()
    assert capabilities.have("loss")
    assert "loss" not in capabilities.missing

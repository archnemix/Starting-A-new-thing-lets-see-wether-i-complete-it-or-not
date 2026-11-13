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


def test_tier_three_does_not_promise_eval_results():
    """The capability block is the file's statement about itself, so every word counts.

    Tier 3 read "activation statistics, eval results" in every file this build wrote, and no
    adapter has ever recorded an eval result - the field is planned (Architecture §6.11) and
    unimplemented. A promise in the file that nothing in the build can keep is exactly what
    this module exists to prevent, so the claim is gone until there is a block to back it.
    """
    from archx.capabilities import TIER_MEANING

    capabilities = Capabilities()
    for name in ("steps", "deltas", "loss", "grad_norms", "data_segments", "activations"):
        capabilities.offer(name)
    assert capabilities.tier == 3
    for detail in (TIER_MEANING[3], capabilities.resolve().to_dict()["capability_detail"]):
        assert "eval" not in detail, detail
        assert "activation statistics" in detail


#: How each recorded field is worded in TIER_MEANING. Written out rather than split from the
#: field name because the prose is for people: "grad_norms" reads as "gradient norms" there.
#: Adding a word to a tier's description means adding it here, deliberately.
FIELD_WORDING = {
    "steps": "step counter",
    "deltas": "weight deltas",
    "loss": "loss",
    "learning_rate": "learning rate",
    "grad_norms": "gradient norms",
    "data_segments": "data segments",
    "activations": "activation statistics",
}

#: Prose that is allowed to describe a tier without naming a field. Kept to what is true of
#: the recording itself, so nothing here can turn into a claim about data it does not hold.
EXTRA_PROSE = {"timing", "and their effects on the weights"}


def test_every_tier_meaning_describes_exactly_the_fields_that_tier_promises():
    """A tier's prose may not name anything the tier does not record.

    Tier 3 said "activation statistics, eval results" for the whole life of this build, and no
    adapter recorded an eval result: every tier-3 file promised, in its own capability block,
    something the build could not produce. This is the mechanical form of that review - build
    the sentence from the fields, then look at what is left over.
    """
    from archx.capabilities import TIER_FIELDS, TIER_MEANING

    for tier, fields in TIER_FIELDS.items():
        meaning = TIER_MEANING[tier]
        remaining = meaning
        # learning_rate is part of tier 1's promise and not a TIER_FIELDS entry, so it is
        # named on the tier that carries it rather than tolerated as leftover prose.
        expected = (*fields, "learning_rate") if tier == 1 else fields
        for field in expected:
            assert FIELD_WORDING[field] in meaning, (
                f"tier {tier} does not describe {field!r} anywhere: {meaning!r}")
            remaining = remaining.replace(FIELD_WORDING[field], "")
        for phrase in EXTRA_PROSE:
            remaining = remaining.replace(phrase, "")
        leftover = remaining.replace(",", " ").split()
        assert not leftover, (
            f"tier {tier} describes {leftover}, which is not a field it records: {meaning!r}")

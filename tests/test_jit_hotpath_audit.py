from pcf.jit import FingerprintAudit, HotPathDetector


def _hot_detector(*, min_audit_observations=3):
    detector = HotPathDetector(min_calls=3, min_stability=2 / 3, min_audit_observations=min_audit_observations)
    for stable in (True, True, True):
        detector.observe("fp-a", stable=stable)
    return detector


def _clean_audit(*, count=3):
    audit = FingerprintAudit()
    for i in range(count):
        audit.observe_fingerprint("fp-a", semantic_id="sem-a", behavior_id="beh-a", observed_at=float(i))
    return audit


def test_hot_path_is_not_compile_eligible_without_fingerprint_audit():
    detector = _hot_detector()
    decision = detector.evaluate_compile_eligibility("fp-a", FingerprintAudit().summary())

    assert detector.is_hot("fp-a")
    assert not decision.eligible
    assert decision.reasons == ("audit_missing",)
    assert decision.audit_observations == 0


def test_clean_audit_with_depth_allows_hot_path_compilation():
    detector = _hot_detector()
    audit = _clean_audit().summary()

    decision = detector.evaluate_compile_eligibility("fp-a", audit)
    assert decision.eligible
    assert decision.reasons == ()
    assert decision.calls == 3
    assert decision.audit_observations == 3
    assert detector.is_compile_eligible("fp-a", audit)


def test_thin_audit_cannot_unlock_compilation():
    detector = _hot_detector(min_audit_observations=4)
    decision = detector.evaluate_compile_eligibility("fp-a", _clean_audit(count=3).summary())

    assert not decision.eligible
    assert decision.reasons == ("insufficient_audit_observations",)


def test_collision_blocks_compile_eligibility_even_when_hot_and_well_sampled():
    detector = _hot_detector()
    audit = _clean_audit()
    audit.observe_fingerprint("fp-a", semantic_id="sem-b", behavior_id="beh-b", observed_at=10.0)

    decision = detector.evaluate_compile_eligibility("fp-a", audit.summary())
    assert not decision.eligible
    assert "fingerprint_collision" in decision.reasons


def test_semantic_drift_blocks_compile_eligibility():
    detector = _hot_detector()
    audit = _clean_audit()
    audit.observe_fingerprint("fp-a", semantic_id="sem-a", behavior_id="beh-b", observed_at=10.0)

    decision = detector.evaluate_compile_eligibility("fp-a", audit.summary())
    assert not decision.eligible
    assert "semantic_drift" in decision.reasons


def test_fingerprint_split_blocks_compile_eligibility():
    detector = _hot_detector()
    audit = _clean_audit()
    audit.observe_fingerprint("fp-b", semantic_id="sem-a", behavior_id="beh-a", observed_at=10.0)

    decision = detector.evaluate_compile_eligibility("fp-a", audit.summary())
    assert not decision.eligible
    assert "fingerprint_split" in decision.reasons


def test_volume_and_stability_still_gate_even_with_clean_audit():
    detector = HotPathDetector(min_calls=4, min_stability=0.75, min_audit_observations=3)
    for stable in (True, False, True):
        detector.observe("fp-a", stable=stable)

    decision = detector.evaluate_compile_eligibility("fp-a", _clean_audit().summary())
    assert not decision.eligible
    assert decision.reasons == ("insufficient_calls", "insufficient_stability")

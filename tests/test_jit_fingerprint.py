import pytest

from pcf.jit import FingerprintAudit, TaskSignature


def signature(**kwargs):
    defaults = dict(
        operation="sales.territory.classify",
        input_schema_id="lead-v3",
        output_schema_id="territory-v2",
        policy_version="policy-7",
        authority_scope="sales-read",
    )
    defaults.update(kwargs)
    return TaskSignature(**defaults)


def test_fingerprint_is_deterministic_and_mapping_order_independent():
    first = signature(stable_features={"region": "US", "segment": "ENT"})
    second = signature(stable_features={"segment": "ENT", "region": "US"})
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith("sha256:")


def test_semantically_relevant_contract_changes_change_fingerprint():
    base = signature()
    assert base.fingerprint != signature(policy_version="policy-8").fingerprint
    assert base.fingerprint != signature(authority_scope="sales-write").fingerprint
    assert base.fingerprint != signature(output_schema_id="territory-v3").fingerprint
    assert base.fingerprint != signature(tool_contract_id="crm-read-v1").fingerprint


def test_collision_is_same_fingerprint_with_multiple_adjudicated_semantics():
    audit = FingerprintAudit()
    task = signature()
    audit.observe(task, semantic_id="territory-classification", behavior_id="v1", observed_at=1)
    audit.observe(task, semantic_id="account-ownership", behavior_id="v1", observed_at=2)

    summary = audit.summary()
    assert summary.collision_fingerprints == 1
    assert summary.collision_rate == 1
    assert not summary.clean
    assert set(summary.collision_examples[task.fingerprint]) == {
        "territory-classification",
        "account-ownership",
    }


def test_drift_is_same_fingerprint_and_semantic_with_multiple_behaviors():
    audit = FingerprintAudit()
    task = signature()
    audit.observe(task, semantic_id="territory-classification", behavior_id="rules-v1", observed_at=1)
    audit.observe(task, semantic_id="territory-classification", behavior_id="rules-v2", observed_at=2)

    summary = audit.summary()
    assert summary.drift_fingerprints == 1
    assert summary.drift_rate == 1
    key = f"{task.fingerprint}|territory-classification"
    assert summary.drift_examples[key] == ("rules-v1", "rules-v2")


def test_split_measures_same_semantic_identity_across_multiple_fingerprints():
    audit = FingerprintAudit()
    first = signature(stable_features={"region": "US"})
    second = signature(stable_features={"region": "CA"})
    audit.observe(first, semantic_id="territory-classification", behavior_id="v1", observed_at=1)
    audit.observe(second, semantic_id="territory-classification", behavior_id="v1", observed_at=2)

    summary = audit.summary()
    assert summary.split_semantics == 1
    assert summary.split_rate == 1
    assert set(summary.split_examples["territory-classification"]) == {first.fingerprint, second.fingerprint}


def test_clean_audit_has_zero_collision_and_drift_rates():
    audit = FingerprintAudit()
    audit.observe(signature(), semantic_id="territory", behavior_id="v1", observed_at=1)
    audit.observe(
        signature(operation="lead.score"),
        semantic_id="lead-score",
        behavior_id="v1",
        observed_at=2,
    )
    summary = audit.summary()
    assert summary.clean
    assert summary.collision_rate == 0
    assert summary.drift_rate == 0


def test_direct_fingerprint_observation_supports_external_fingerprinters():
    audit = FingerprintAudit()
    observation = audit.observe_fingerprint(
        "sha256:external",
        semantic_id="external-task",
        behavior_id="behavior-v1",
        observed_at=1,
    )
    assert observation.fingerprint == "sha256:external"
    assert audit.summary().observations == 1


def test_invalid_signature_and_observation_inputs_fail_closed():
    with pytest.raises(ValueError):
        signature(operation="")
    with pytest.raises(ValueError, match="unsupported"):
        signature(fingerprint_version="0.2")
    audit = FingerprintAudit()
    with pytest.raises(ValueError, match="TaskSignature"):
        audit.observe("not-a-signature", semantic_id="x", behavior_id="y", observed_at=1)

"""The release gate accepts bound production evidence and rejects claim laundering."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_release_gate.py"
spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def evidence():
    arms = {name: {"conversations": 2, "requests": 2, "failures": 0} for name in gate.ARMS}
    gates = {name: {"passed": True, "estimate": .5, "limit": .8, "comparison": "frozen CI boundary"}
             for name in gate.GATES}
    digest = "sha256:" + "0" * 64
    return {"evidence_version": 1, "pilot_id": "pilot-1", "source": "production", "decision": "pass",
            "preregistered_at": "2026-09-01T00:00:00Z", "started_at": "2026-09-02T00:00:00Z",
            "ended_at": "2026-09-20T00:00:00Z", "provider": "provider", "model": "model",
            "deployment_revision": "sha256:deployment", "minimum_conversations_per_arm": 2,
            "actual_prices_per_million_tokens": {"uncached_input": 2, "cache_write": 2.5,
                                                   "cache_read": .2, "output": 10},
            "arms": arms,
            "blind_review": {"blinded": True, "independent_reviewers": 2, "rubric_version": "quality-v1",
                             "minimum_reviewed_conversations_per_arm": 1,
                             "reviewed_conversations_by_arm": {name: 1 for name in gate.ARMS},
                             "inter_rater_agreement": .94, "disagreements_adjudicated": True},
            "gates": gates, "artifacts": {"pilot_log_sha256": digest, "blind_review_log_sha256": digest,
                                           "analysis_sha256": digest},
            "approvals": [{"reviewer": "owner", "role": "pilot-owner",
                            "approved_at": "2026-09-21T00:00:00Z"},
                           {"reviewer": "quality", "role": "independent-quality-reviewer",
                            "approved_at": "2026-09-21T01:00:00Z"}]}


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def bind(tmp_path: Path, doc: dict) -> Path:
    rows = [{"conversation": f"{arm}-{i}", "request_id": f"{arm}-{i}", "arm": arm, "success": True}
            for arm in gate.ARMS for i in range(2)]
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text("".join(json.dumps(row) + "\n" for row in rows))
    blind = tmp_path / "blind-review.json"
    blind.write_text(json.dumps(doc["blind_review"], sort_keys=True))
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"gates": doc["gates"]}, sort_keys=True))
    doc["artifacts"] = {"pilot_log_sha256": _digest(pilot), "blind_review_log_sha256": _digest(blind),
                        "analysis_sha256": _digest(analysis)}
    return tmp_path


def test_complete_production_evidence_passes(tmp_path):
    doc = evidence()
    assert gate.validate_evidence(doc, bind(tmp_path, doc)) == {
        "pilot_id": "pilot-1", "decision": "pass", "provider": "provider", "model": "model", "conversations": 8}


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(source="synthetic"), "source production"),
    (lambda d: d.update(decision="pending"), "decision pass"),
    (lambda d: d["arms"]["placed"].update(conversations=1), "sample minimum"),
    (lambda d: d["blind_review"].update(blinded=False), "must be blinded"),
    (lambda d: d["gates"]["quality_noninferiority"].update(passed=False), "did not pass"),
    (lambda d: d["approvals"].__setitem__(1, {**d["approvals"][1], "reviewer": "owner"}), "distinct"),
])
def test_incomplete_or_synthetic_evidence_is_rejected(tmp_path, mutation, match):
    doc = evidence()
    artifact_dir = bind(tmp_path, doc)
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        gate.validate_evidence(doc, artifact_dir)


def test_gate_boolean_cannot_override_a_failed_numeric_threshold(tmp_path):
    doc = evidence()
    doc["gates"]["latency"].update(estimate=.9, limit=.8, passed=True)
    artifact_dir = bind(tmp_path, doc)
    with pytest.raises(ValueError, match="exceeds"):
        gate.validate_evidence(doc, artifact_dir)


def test_bound_artifact_hashes_and_arm_counts_are_verified(tmp_path):
    doc = evidence()
    artifact_dir = bind(tmp_path, doc)
    (artifact_dir / "pilot.jsonl").write_text('{"conversation":"x","arm":"placed","success":true}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        gate.validate_evidence(doc, artifact_dir)

    artifact_dir = bind(tmp_path, doc)
    doc["arms"]["placed"]["requests"] = 3
    with pytest.raises(ValueError, match="does not match arm summary"):
        gate.validate_evidence(doc, artifact_dir)


def test_unbound_evidence_is_never_accepted():
    with pytest.raises(ValueError, match="bound release artifacts"):
        gate.validate_evidence(evidence())

"""The release gate accepts complete production evidence and rejects claim laundering."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "check_release_gate.py"
spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def evidence():
    arms = {name: {"conversations": 100, "requests": 800, "failures": 2} for name in gate.ARMS}
    gates = {name: {"passed": True, "estimate": .5, "limit": .8, "comparison": "frozen CI boundary"}
             for name in gate.GATES}
    digest = "sha256:" + "a" * 64
    return {"evidence_version": 1, "pilot_id": "pilot-1", "source": "production", "decision": "pass",
            "preregistered_at": "2026-09-01T00:00:00Z", "started_at": "2026-09-02T00:00:00Z",
            "ended_at": "2026-09-20T00:00:00Z", "provider": "provider", "model": "model",
            "deployment_revision": "sha256:deployment", "minimum_conversations_per_arm": 100,
            "actual_prices_per_million_tokens": {"uncached_input": 2, "cache_write": 2.5,
                                                   "cache_read": .2, "output": 10},
            "arms": arms,
            "blind_review": {"blinded": True, "independent_reviewers": 2, "rubric_version": "quality-v1",
                             "minimum_reviewed_conversations_per_arm": 25,
                             "reviewed_conversations_by_arm": {name: 25 for name in gate.ARMS},
                             "inter_rater_agreement": .94, "disagreements_adjudicated": True},
            "gates": gates, "artifacts": {"pilot_log_sha256": digest, "blind_review_log_sha256": digest,
                                             "analysis_sha256": digest},
            "approvals": [{"reviewer": "owner", "role": "pilot-owner",
                            "approved_at": "2026-09-21T00:00:00Z"},
                           {"reviewer": "quality", "role": "independent-quality-reviewer",
                            "approved_at": "2026-09-21T01:00:00Z"}]}


def test_complete_production_evidence_passes():
    assert gate.validate_evidence(evidence()) == {"pilot_id": "pilot-1", "decision": "pass",
                                                  "provider": "provider", "model": "model",
                                                  "conversations": 400}


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(source="synthetic"), "source production"),
    (lambda d: d.update(decision="pending"), "decision pass"),
    (lambda d: d["arms"]["placed"].update(conversations=99), "sample minimum"),
    (lambda d: d["blind_review"].update(blinded=False), "must be blinded"),
    (lambda d: d["gates"]["quality_noninferiority"].update(passed=False), "did not pass"),
    (lambda d: d["approvals"].__setitem__(1, {**d["approvals"][1], "reviewer": "owner"}), "distinct"),
])
def test_incomplete_or_synthetic_evidence_is_rejected(mutation, match):
    doc = copy.deepcopy(evidence())
    mutation(doc)
    with pytest.raises(ValueError, match=match):
        gate.validate_evidence(doc)

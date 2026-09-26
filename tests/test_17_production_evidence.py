"""Behavioral regressions for live history, four-arm pilots and measured audit inputs."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


domain = load("live_domain_sessions")
pilot = load("pilot_analysis")
audit = load("cache_audit")


def config(**kwargs):
    return SimpleNamespace(**dict(provider="openai", model="gpt-5.6", turns=10, read_multiplier=.1,
                                  violation_tokens=80, effort="low", thinking="default", **kwargs))


def test_actual_wrong_answer_is_replayed_without_repair_and_capture_matches_send(monkeypatch):
    scenario = domain.SCENARIOS["tax"]
    observed = []

    def call(provider, client, request, cfg):
        observed.append(copy.deepcopy(domain.placement.request_payload(provider, request, cfg)))
        # An invented wrong value and a unique marker make ground-truth substitution detectable.
        return {"usage": {"input_tokens": 100}}, "$999999 MODEL_REPLY", {"output_tokens": 5}

    monkeypatch.setattr(domain.placement, "call", call)
    cfg = config(history_mode="model-text", capture_requests=True)
    rows = domain.session("tax", "fixed-tail", "test", cfg, client=object())
    assert "MODEL_REPLY" not in json.dumps(observed[0])
    assert "$999999 MODEL_REPLY" in json.dumps(observed[1])
    assert scenario.reply(0, scenario.question(0)[1]) not in json.dumps(observed[1])
    assert [r["request"] for r in rows] == observed
    assert all(not r["correct"] for r in rows)
    assert any(r["prior_error_exposed"] for r in rows)
    assert any(r["repeated_prior_error"] for r in rows)
    assert all(r["request_ts"] > 0 for r in rows)


def test_actual_history_cannot_be_invented_offline():
    with pytest.raises(ValueError, match="requires responses"):
        domain.session("tax", "placed", "test", config(history_mode="model-text"))


def test_fixed_baseline_places_declared_volatile_records_from_turn_zero():
    rows = domain.session("tax", "fixed-tail", "test", config())
    expected = domain.SCENARIOS["tax"].volatile()
    assert expected
    assert all(set(r["tail"]) == expected for r in rows)
    assert "reference" not in expected


def test_history_modes_cannot_be_pooled(tmp_path):
    old = SCRIPTS.parent / "results/2026-09-25/domain-gpt-5.6-60turn-echo.json"
    changed = json.loads(old.read_text())
    changed["meta"]["history_mode"] = "model-text"
    new = tmp_path / "new.json"
    new.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="history_mode"):
        domain.analyze([str(old), str(new)])


def test_actual_history_analysis_allows_different_stale_exposure(tmp_path):
    old = SCRIPTS.parent / "results/2026-09-25/domain-gpt-5.6-60turn-echo.json"
    saved = json.loads(old.read_text())
    saved["meta"]["history_mode"] = "model-text"
    row = saved["scenarios"]["tax"][0]["placed"][0]
    row["stale_trap"] = not row["stale_trap"]
    new = tmp_path / "new.json"
    new.write_text(json.dumps(saved))
    assert domain.analyze([str(new)])["gpt-5.6"]["pairs"]


def pilot_rows():
    return [{"conversation": f"{arm}-{i}", "arm": arm, "request_id": f"{arm}-{i}",
             "cached": 0, "written": 0, "uncached": 100 if arm == "front-tuned" else 50,
             "output_tokens": 10, "correct": True, "latency_s": 1.0}
            for arm in pilot.PILOT_ARMS for i in range(3)]


def test_four_arm_assignment_and_all_comparisons():
    assigned = [pilot.assign_multi(f"c{i}", "a") for i in range(10000)]
    assert set(assigned) == set(pilot.PILOT_ARMS)
    assert all(2200 < assigned.count(a) < 2800 for a in pilot.PILOT_ARMS)
    assert pilot.assign_multi("c", "a") == pilot.assign_multi("c", "a")
    report = pilot.analyze(pilot_rows(), samples=30)
    assert len(report["comparisons"]) == 6
    assert report["comparisons"]["placed vs front-tuned"]["input_cost_ratio"]["treatment_over_baseline"] == .5
    assert report["comparisons"]["placed vs fixed-tail"]["input_cost_ratio"]["treatment_over_baseline"] == 1


def test_pilot_rejects_missing_arms_contamination_and_bad_measurements():
    rows = pilot_rows()
    with pytest.raises(ValueError, match="requires exactly"):
        pilot.analyze(rows[:-3])
    rows[-1]["conversation"] = rows[0]["conversation"]
    with pytest.raises(ValueError, match="multiple arms"):
        pilot.analyze(rows)
    for key, value in (("correct", "false"), ("uncached", -1), ("latency_s", float("nan"))):
        rows = pilot_rows()
        rows[0][key] = value
        with pytest.raises(ValueError):
            pilot.analyze(rows)
    rows = pilot_rows()
    rows.append(rows[0])
    with pytest.raises(ValueError, match="duplicate request_id"):
        pilot.analyze(rows)


def test_zero_cost_does_not_manufacture_a_savings_ratio():
    rows = pilot_rows()
    for r in rows:
        r["uncached"] = r["output_tokens"] = 0
    report = pilot.analyze(rows, samples=10)
    for comparison in report["comparisons"].values():
        assert comparison["input_cost_ratio"] == {"treatment_over_baseline": None, "ci95": None}


def audit_row(conversation="c", ts=0):
    return {"conversation": conversation, "ts": ts, "request": {"model": "m", "instructions": "x" * 4000,
            "input": "q"}, "usage": {"cached": 0, "written": 1000, "uncached": 0}}


def test_audit_uses_observed_write_cost_and_caps_diagnostics():
    rows = [audit_row(), audit_row(ts=1)]
    result = audit.audit(rows, write=2, read=.1)
    assert result["totals"]["billed_input_units"] == 4000
    assert result["totals"]["estimated_miss_opportunity_units"] <= 1900
    assert result["log_quality"]["rows_without_cache_scope"] == 2
    assert "not recoverable savings" in audit.report(result)


def test_cross_conversation_reuse_requires_matching_explicit_scope_and_model():
    rows = [audit_row("a"), audit_row("b", 1)]
    assert not audit.audit(rows)["events"]
    for r in rows:
        r["cache_scope"] = "project-route"
    assert audit.audit(rows)["events"][0]["kind"] == "unread"
    rows[1]["request"]["model"] = "other"
    assert not audit.audit(rows)["events"]


def test_audit_expiry_includes_exact_ttl_and_rejects_invalid_logs():
    rows = [audit_row(), audit_row(ts=300)]
    assert audit.audit(rows, ttl=300)["events"][0]["kind"] == "expired"
    rows[1]["ts"] = -1
    with pytest.raises(ValueError, match="out of order"):
        audit.audit(rows)
    rows = [audit_row()]
    rows[0]["usage"]["cached"] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        audit.audit(rows)


def test_annotation_scoring_counts_errors_and_requires_independent_labels():
    rows = [audit_row(), audit_row(ts=1)]
    result = audit.audit(rows)
    with pytest.raises(ValueError, match="no independent annotations"):
        audit.validate_labels(rows, result)
    rows[0]["annotation"] = {"kinds": []}
    rows[1]["annotation"] = {"kinds": ["expired"]}  # deliberately disagrees with the detector
    report = audit.validate_labels(rows, result)
    assert report["classes"]["unread"]["fp"] == 1
    assert report["classes"]["expired"]["fn"] == 1


def test_model_history_audit_requires_captures_and_preserves_them(tmp_path):
    used = {"cached": 0, "written": 100, "uncached": 0}
    saved = {"meta": {"provider": "openai", "model": "gpt-5.6", "turns": 1, "read_multiplier": .1,
                      "violation_tokens": 80, "history_mode": "model-text"},
             "scenarios": {"tax": [{"placed": [used]}]}}
    path = tmp_path / "run.json"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="captured requests"):
        audit.from_domain(str(path), "placed")
    used.update(request={"model": "m", "input": "actual reply in history"}, request_ts=123)
    path.write_text(json.dumps(saved))
    row, = audit.from_domain(str(path), "placed")
    assert row["request"] == used["request"] and row["ts"] == 123
    assert row["source"] == "captured-synthetic"

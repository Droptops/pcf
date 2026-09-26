"""Pilot assignment and analysis (scripts/pilot_analysis.py), offline."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("pilot", os.path.join(SCRIPTS, "pilot_analysis.py"))
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def turn(conversation, arm, cached, written, uncached, output=10, correct=True, latency=1.0,
         success=True, blind_acceptable=None):
    return {"conversation": conversation, "arm": arm, "cached": cached, "written": written, "uncached": uncached,
            "output_tokens": output, "latency_s": latency, "correct": correct, "success": success,
            "blind_acceptable": blind_acceptable}


def test_assignment_is_stable_and_respects_the_share():
    assert pilot.assign("conv-1") == pilot.assign("conv-1")
    share = sum(pilot.assign(f"c{i}", .2) == "treatment" for i in range(10000)) / 10000
    assert .18 < share < .22
    assert pilot.assign("c", 0) == "baseline" and pilot.assign("c", 1) == "treatment"
    assert {pilot.assign(f"c{i}", .5, salt="a") == pilot.assign(f"c{i}", .5, salt="b") for i in range(50)} == {
        True, False}
    with pytest.raises(ValueError):
        pilot.assign("c", 1.5)


def test_costs_are_per_conversation_under_the_given_multipliers():
    rows = [turn("b1", "baseline", 0, 100, 0), turn("b1", "baseline", 100, 0, 10),
            turn("t1", "treatment", 0, 100, 0, correct=False), turn("t1", "treatment", 100, 0, 5)]
    report = pilot.analyze(rows, write=2, read=.5, output=1, samples=50)
    assert report["arms"]["baseline"]["input_cost_per_conversation"] == 260
    assert report["arms"]["treatment"]["total_cost_per_conversation"] == 275
    assert report["arms"]["treatment"]["error_rate"] == .5
    assert report["error_rate_difference"]["treatment_minus_baseline"] == .5


def test_failure_and_blind_review_rates_are_conversation_bootstrapped():
    rows = [turn("b1", "baseline", 0, 1, 0, blind_acceptable=True),
            turn("b2", "baseline", 0, 1, 0, blind_acceptable=True),
            turn("t1", "treatment", 0, 1, 0, blind_acceptable=False, success=False),
            turn("t2", "treatment", 0, 1, 0, blind_acceptable=True)]
    report = pilot.analyze(rows, samples=50)
    assert report["arms"]["treatment"]["blind_error_rate"] == .5
    assert report["arms"]["treatment"]["failure_rate"] == .5
    assert report["blind_error_rate_difference"]["treatment_minus_baseline"] == .5
    assert report["failure_rate_difference"]["treatment_minus_baseline"] == .5


def test_a_conversation_in_both_arms_is_rejected():
    with pytest.raises(ValueError, match="both arms"):
        pilot.analyze([turn("c", "baseline", 0, 1, 0), turn("c", "treatment", 0, 1, 0)])
    with pytest.raises(ValueError, match="no conversations"):
        pilot.analyze([turn("c", "baseline", 0, 1, 0)])


def test_committed_60_turn_claude_run_as_a_pilot_log():
    rows = pilot.from_domain(str(SCRIPTS.parent / "results/2026-09-25/domain-claude-sonnet-5-60turn.json"))
    report = pilot.analyze(rows, samples=300)
    assert report["arms"]["baseline"]["conversations"] == report["arms"]["treatment"]["conversations"] == 12
    ratio = report["input_cost_ratio"]
    assert ratio["treatment_over_baseline"] == pytest.approx(.534, abs=.001)
    low, high = ratio["ci95"]
    assert low < ratio["treatment_over_baseline"] < high < 1
    assert report == pilot.analyze(rows, samples=300)  # seeded

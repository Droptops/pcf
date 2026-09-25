"""Offline checks of the synthetic domain workloads and their grader (scripts/domain_scenarios.py)."""
from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, SCRIPTS)
from domain_scenarios import SCENARIOS, answer_value, normalize  # noqa: E402

spec = importlib.util.spec_from_file_location("domain", os.path.join(SCRIPTS, "live_domain_sessions.py"))
domain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(domain)


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_grader_reads_the_expected_value_from_typical_replies(key):
    scenario = SCENARIOS[key]
    for turn in range(24):
        ask, value = scenario.question(turn)
        for reply in (scenario.reply(turn, value), f"**{value}**", f"{value}, per the record.", f"It's {value}."):
            assert answer_value(ask, reply) == normalize(ask.kind, value), (key, turn, reply)


@pytest.mark.parametrize("key", sorted(SCENARIOS))
def test_history_holds_stale_values_that_later_records_replace(key):
    scenario = SCENARIOS[key]
    said, traps = {}, 0
    for turn in range(24):
        ask, value = scenario.question(turn)
        traps += ask.text in said and normalize(ask.kind, said[ask.text]) != normalize(ask.kind, value)
        said[ask.text] = value
    assert traps >= 8


def test_grader_takes_the_first_value_and_ignores_times_and_ids():
    clinical = SCENARIOS["clinical"]
    potassium = clinical.asks[3]
    assert answer_value(potassium, "At 06:00 the potassium was 3.4 mmol/L.") == "3.4"
    helpdesk = SCENARIOS["helpdesk"].asks[2]
    assert answer_value(helpdesk, "Non-compliant (it was compliant yesterday).") == "noncompliant"
    claims = SCENARIOS["claims"].asks[3]
    assert answer_value(claims, "He owes $1,012.30 now; last month it was $975.00.") == "1012.30"


def test_offline_sessions_place_memory_as_each_layout_says():
    cfg = SimpleNamespace(provider="openai", model="gpt-5.6", turns=8, read_multiplier=0.1, violation_tokens=80)
    rows = {arm: domain.session("tax", arm, "offline", cfg) for arm in domain.ARMS}
    everything = {"reference", *SCENARIOS["tax"].modules}
    assert all(not r["tail"] for r in rows["front"]) and all(not r["tail"] for r in rows["front-tuned"])
    assert all(set(r["tail"]) == everything for r in rows["tail"])
    assert "reference" not in rows["placed"][-1]["tail"] and "balance" in rows["placed"][-1]["tail"]
    assert [r["expected"] for r in rows["front"]] == [r["expected"] for r in rows["placed"]]

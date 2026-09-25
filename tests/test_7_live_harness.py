"""The live placement harness grades strictly and runs offline; paid calls are never made here."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "live_memory_placement.py"
spec = importlib.util.spec_from_file_location("live_memory_placement", SCRIPT)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


@pytest.mark.parametrize("answer, lead", [
    ("4", "4"), ("4 (T-4102)", "4"), ("8 — T-4106", "8"), ("Spanish (T-4103)", "spanish"),
    ("Unlimited Plus", "unlimited plus"), ("The answer is SMS. Order 9040 is on schedule.", "sms"),
    ("**Premium**", "premium"), ("email.", "email"),
])
def test_lead_value(answer, lead):
    assert live.lead_value(answer) == lead


def test_grade_is_exact_and_flags_stale_and_copied_replies():
    assert not live.grade("Unlimited Plus", "Unlimited", None, "template", 60)["correct"]
    # A copied template contains order numbers that include the expected count; the lead value decides.
    copied = "The answer is 3. Order 9042 is on schedule. Order 9043 is on schedule."
    assert live.grade(copied, "4", "3", "template", 60) == {
        "lead": "3", "correct": False, "gave_stale": True, "violation": True}
    assert not live.grade("Spanish", "Spanish", "Spanish", "template", 60)["gave_stale"]  # not a trap
    assert live.grade("x " * 200, "x", None, "varied", 60)["violation"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_offline_run_plans_every_arm(provider):
    out = subprocess.run([sys.executable, str(SCRIPT), "--provider", provider, "--turns", "12", "--arms",
                          *live.ARMS], capture_output=True, text=True, check=True).stdout
    run = json.loads(out)["runs"][0]
    assert set(run) == {*live.ARMS, "summary"} and all(len(run[arm]) == 12 for arm in live.ARMS)
    assert run["placed-spacer"][-1]["tail"][-1] == "notice" and run["front"][-1]["tail"] == []
    assert all(row["est_tokens"] > 0 for row in run["tail"])

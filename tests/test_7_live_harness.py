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


@pytest.mark.parametrize("answer, turn, value", [
    ("4", 0, "4"), ("4 (T-4102)", 0, "4"), ("You have 10 open tickets, per ticket record T-4108.", 8, "10"),
    ("8 — T-4106", 0, "8"), ("The answer is 3. Order 9042 is on schedule.", 0, "3"), ("T-4104", 0, ""),
    ("Your current plan is **Premium**.", 1, "premium"), ("Unlimited Plus", 1, "unlimited plus"),
    ("You are on Unlimited.", 1, "unlimited"), ("The answer is SMS. Survey 7010 was sent.", 2, "sms"),
    ("SMS[T-4106]", 2, "sms"), ("Spanish [T-4119]", 3, "spanish"), ("Español", 3, ""), ("", 1, ""),
    ("Earlier I said 2, 6, 10 open tickets. **Open tickets:** 14 (per record). Directly: **14**", 0, "14"),
    ("**Premium** (you moved from Unlimited)", 1, "premium"),
    ("**SMS**; I will no longer use **email**.", 2, "sms"),
])
def test_answer_value(answer, turn, value):
    assert live.answer_value(answer, turn) == value


def test_grade_is_exact_and_flags_stale_and_copied_replies():
    assert not live.grade("Unlimited Plus", 1, "Unlimited", None, "template", 60)["correct"]
    # A copied template contains order numbers that include the expected count; ids are not answers.
    copied = "The answer is 3. Order 9042 is on schedule. Order 9043 is on schedule."
    assert live.grade(copied, 0, "4", "3", "template", 60) == {
        "value": "3", "correct": False, "gave_stale": True, "violation": True}
    assert not live.grade("Spanish", 3, "Spanish", "Spanish", "template", 60)["gave_stale"]  # not a trap
    assert live.grade("x " * 200, 3, "x", None, "varied", 60)["violation"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_offline_run_plans_every_arm(provider):
    out = subprocess.run([sys.executable, str(SCRIPT), "--provider", provider, "--turns", "12", "--arms",
                          *live.ARMS], capture_output=True, text=True, check=True).stdout
    run = json.loads(out)["runs"][0]
    assert set(run) == {*live.ARMS, "summary"} and all(len(run[arm]) == 12 for arm in live.ARMS)
    assert run["placed-spacer"][-1]["tail"][-1] == "notice" and run["front"][-1]["tail"] == []
    assert all(row["est_tokens"] > 0 for row in run["tail"])


@pytest.mark.parametrize("model, name", [("claude-haiku-4-5", "anthropic/claude-haiku-4.5"),
                                         ("claude-sonnet-5", "anthropic/claude-sonnet-5"),
                                         ("claude-opus-5-5", "anthropic/claude-opus-5.5")])
def test_openrouter_model_names(model, name):
    assert live.openrouter_model(model) == name

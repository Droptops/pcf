from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "live_safe_memory_matrix.py"
spec = importlib.util.spec_from_file_location("live_safe_memory_matrix", SCRIPT)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


def test_safe_fixture_is_explicitly_synthetic():
    profile = safe._safe_profile()
    assert profile["name"] == "Benchmark User"
    assert profile["region"] == "Test Region"
    assert all("Synthetic location slot" in value for value in profile["addresses"])
    assert "Dana Ruiz" not in str(profile)
    assert "Main Street" not in str(profile)


def test_safe_fixture_preserves_memory_change_rates():
    assert safe._safe_preferences(0) == safe._safe_preferences(5)
    assert safe._safe_preferences(5) != safe._safe_preferences(6)
    assert safe._safe_notes(0) == safe._safe_notes(2)
    assert safe._safe_notes(2) != safe._safe_notes(3)
    assert safe._safe_account(0) != safe._safe_account(1)
    assert safe._safe_profile() == safe._safe_profile()


def test_provider_error_code_uses_only_structured_code():
    exc = SimpleNamespace(body={"error": {"code": "invalid_prompt", "message": "sensitive raw body"}})
    assert safe._provider_error_code(exc) == "invalid_prompt"


def test_provider_error_code_falls_back_to_exception_type():
    assert safe._provider_error_code(ValueError("raw provider detail")) == "ValueError"


def test_anthropic_gets_exact_value_contract_but_openai_does_not():
    safe.apply_safe_fixture()
    anthropic = safe._system_text("anthropic", "abc123")
    openai = safe._system_text("openai", "abc123")
    assert safe.ANTHROPIC_OUTPUT_CONTRACT in anthropic
    assert safe.ANTHROPIC_OUTPUT_CONTRACT not in openai
    assert "Benchmark session abc123" in anthropic
    assert "Benchmark session abc123" in openai


def test_safe_grade_splits_filler_copy_from_length_violation():
    safe.apply_safe_fixture()
    expected = str(safe._safe_account(0)["open_tickets"])

    copied = safe._grade_safe(
        "2 Synthetic log entry 7 was recorded for benchmark continuity.",
        0,
        expected,
        None,
        "varied",
        60,
    )
    assert copied["correct"] is True
    assert copied["copied_filler"] is True
    assert copied["too_long"] is False
    assert copied["format_violation"] is True
    assert copied["contract_output"] == "2"
    assert copied["contract_repaired"] is True

    long = safe._grade_safe("2 " + "x" * 300, 0, expected, None, "varied", 60)
    assert long["correct"] is True
    assert long["copied_filler"] is False
    assert long["too_long"] is True
    assert long["format_violation"] is True
    assert long["contract_output"] == "2"
    assert long["contract_repaired"] is True

    clean = safe._grade_safe("2", 0, expected, None, "varied", 60)
    assert clean["correct"] is True
    assert clean["copied_filler"] is False
    assert clean["too_long"] is False
    assert clean["format_violation"] is False
    assert clean["contract_repaired"] is False


def test_summary_preserves_legacy_violation_and_split_counters():
    rows = [
        {
            "cached": 10,
            "written": 2,
            "uncached": 3,
            "output_tokens": 1,
            "correct": True,
            "stale_trap": False,
            "gave_stale": False,
            "violation": True,
            "copied_filler": True,
            "too_long": False,
            "format_violation": True,
            "contract_repaired": True,
        },
        {
            "cached": 10,
            "written": 2,
            "uncached": 3,
            "output_tokens": 1,
            "correct": True,
            "stale_trap": True,
            "gave_stale": False,
            "violation": False,
            "copied_filler": False,
            "too_long": False,
            "format_violation": False,
            "contract_repaired": False,
        },
    ]
    cfg = SimpleNamespace(read_multiplier=0.1, output_multiplier=5.0)
    summary = safe._summarize(rows, cfg, 1.25)
    assert summary["violations"] == "1/2"
    assert summary["copied_filler"] == "1/2"
    assert summary["too_long"] == "0/2"
    assert summary["format_violation"] == "1/2"
    assert summary["contract_repaired"] == "1/2"


def test_sanitized_misses_expose_coordinates_not_raw_answer():
    runs = [
        {
            "placed": [
                {
                    "turn": 1,
                    "expected": "Gold",
                    "value": "silver",
                    "contract_output": "silver",
                    "correct": False,
                    "stale_trap": False,
                    "copied_filler": False,
                    "too_long": True,
                    "format_violation": True,
                    "answer": "RAW MODEL RESPONSE MUST NOT LEAK",
                }
            ],
            "summary": {},
        }
    ]
    misses = safe._sanitized_misses(runs, ["placed"])
    assert misses == [
        {
            "repeat": 1,
            "arm": "placed",
            "turn": 1,
            "question_type": "current_plan",
            "expected": "Gold",
            "extracted": "silver",
            "stale_trap": False,
            "copied_filler": False,
            "too_long": True,
            "format_violation": True,
        }
    ]
    assert "answer" not in misses[0]

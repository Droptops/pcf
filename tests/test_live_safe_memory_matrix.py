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

"""The live tool-loop script runs offline and never marks tail memory; paid calls are never made here."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "live_tool_loop.py"


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_offline_tool_loop_marks_system_and_history_only(provider):
    out = json.loads(subprocess.run([sys.executable, str(SCRIPT), "--provider", provider, "--iterations", "4"],
                                    capture_output=True, text=True, check=True).stdout)
    assert out["ok"] and len(out["rows"]) == 8
    for row in out["rows"]:
        assert "s" in row["marked"] and not {"profile", "cart", "session", "notes"} & set(row["marked"])

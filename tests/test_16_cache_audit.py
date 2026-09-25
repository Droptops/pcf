"""The cache audit (scripts/cache_audit.py) on hand-built logs and on committed runs, offline."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("cache_audit", os.path.join(SCRIPTS, "cache_audit.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
RESULTS = SCRIPTS.parent / "results/2026-09-25"


def request(record: dict, history: list[str], question: str) -> dict:
    """An Anthropic-shaped request: a JSON record in front, then history, then the question."""
    memory = json.dumps({"kind": "memory", "source": "account", "data": record})
    messages = [{"role": "user", "content": [{"type": "text", "text": memory, "cache_control": {"type": "x"}}]}]
    messages += [{"role": "user" if i % 2 == 0 else "assistant", "content": h} for i, h in enumerate(history)]
    return {"model": "m", "system": "policy " * 50, "messages": messages + [{"role": "user", "content": question}]}


def row(conversation, req, cached, written, uncached, ts=None):
    return {"conversation": conversation, "ts": ts, "request": req,
            "usage": {"cached": cached, "written": written, "uncached": uncached}}


def test_a_changed_field_is_named_and_charged_for_what_followed_it():
    history = ["question " * 200, "answer " * 200]
    rows = [row("c", request({"plan": "basic", "balance": 10}, history, "q1"), 0, 1000, 10),
            row("c", request({"plan": "basic", "balance": 12}, history + ["q1", "a1"], "q2"), 60, 950, 10)]
    result = audit.audit(rows)
    (changed,) = [e for e in result["events"] if e["kind"] == "changed"]
    assert changed["field"] == "memory 'account' balance"
    assert changed["rebilled_tokens"] > 800
    assert result["causes"][0]["cause"] == "memory 'account' balance"


def test_markers_and_settings_do_not_count_as_changes():
    a = request({"plan": "basic"}, [], "q")
    b = json.loads(json.dumps(a))
    b["messages"][0]["content"][0]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    b["max_tokens"] = 99
    assert audit.flatten(a) == audit.flatten(b)


def test_an_expired_gap_and_an_unread_prefix_are_told_apart():
    same = request({"plan": "basic"}, ["x " * 300, "y " * 300], "q")
    rows = [row("c", same, 0, 1000, 0, ts=0), row("c", same, 0, 1000, 0, ts=10), row("c", same, 0, 1000, 0, ts=1000)]
    kinds = [e["kind"] for e in audit.audit(rows, ttl=300)["events"]]
    assert kinds == ["unread", "expired"]


def test_rebuilt_probes_show_the_first_request_anchor_bug_and_its_fix():
    before = audit.audit(audit.from_fleet(str(RESULTS / "fleet-probe-anthropic-before-anchor-fix.json"),
                                          "placed-shared"), ttl=300)
    after = audit.audit(audit.from_fleet(str(RESULTS / "fleet-probe-anthropic.json"), "placed-shared"), ttl=300)
    unread = [e for e in before["events"] if e["kind"] == "unread"]
    assert any(not e["same_conversation"] and e["reusable_tokens"] > 3000 and e["cached"] == 0 for e in unread)
    assert not [e for e in after["events"] if e["kind"] == "unread"]


def test_typescript_placer_fixture_is_current():
    fixture_spec = importlib.util.spec_from_file_location("fixture", os.path.join(SCRIPTS, "export_placer_fixture.py"))
    exporter = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(exporter)
    committed = json.loads((SCRIPTS.parent / "ts/test/fixtures/placer.json").read_text())
    assert committed == exporter.fixture(), "run python scripts/export_placer_fixture.py"

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pytest

SCRIPT = pathlib.Path(__file__).parents[1] / "scripts" / "live_anthropic_workspace.py"
spec = importlib.util.spec_from_file_location("live_anthropic_workspace", SCRIPT)
workspace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workspace)


def _fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda **kwargs: kwargs))


def _clear(monkeypatch):
    for name in (*workspace.live.ANTHROPIC_KEYS, *workspace.WORKSPACE_ID_NAMES):
        monkeypatch.delenv(name, raising=False)


def test_workspace_header_is_added_for_org_scoped_key(monkeypatch):
    _clear(monkeypatch)
    _fake_sdk(monkeypatch)
    monkeypatch.setenv("PCF_ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("PCF_ANTHROPIC_WORKSPACE_ID", "wrkspc_test")

    assert workspace.make_client() == {
        "api_key": "key",
        "base_url": "https://api.anthropic.com",
        "default_headers": {"anthropic-workspace-id": "wrkspc_test"},
    }


def test_single_workspace_key_does_not_get_synthetic_header(monkeypatch):
    _clear(monkeypatch)
    _fake_sdk(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "key")

    assert workspace.make_client() == {
        "api_key": "key",
        "base_url": "https://api.anthropic.com",
    }


def test_workspace_id_fallback_is_supported(monkeypatch):
    _clear(monkeypatch)
    _fake_sdk(monkeypatch)
    monkeypatch.setenv("PCF_ANTHROPIC_API_KEY", "key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_fallback")

    client = workspace.make_client()
    assert client["default_headers"] == {"anthropic-workspace-id": "wrkspc_fallback"}


def test_missing_anthropic_key_fails_before_sdk_use(monkeypatch):
    _clear(monkeypatch)
    with pytest.raises(SystemExit, match="PCF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY"):
        workspace.make_client()

#!/usr/bin/env python3
"""Bounded direct-Anthropic runner with optional workspace-scoped org key support.

This is a thin live wrapper around ``live_memory_placement``. Anthropic org-level API
keys may require the ``anthropic-workspace-id`` header; single-workspace keys do not.
The workspace id is read from PCF_ANTHROPIC_WORKSPACE_ID or ANTHROPIC_WORKSPACE_ID.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import live_memory_placement as live  # noqa: E402

WORKSPACE_ID_NAMES = ("PCF_ANTHROPIC_WORKSPACE_ID", "ANTHROPIC_WORKSPACE_ID")


def make_client():
    key = next((os.environ[name] for name in live.ANTHROPIC_KEYS if os.environ.get(name)), None)
    if not key:
        raise SystemExit("direct Anthropic run requires PCF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY")
    workspace_id = next((os.environ[name] for name in WORKSPACE_ID_NAMES if os.environ.get(name)), None)
    import anthropic

    kwargs = {"api_key": key, "base_url": "https://api.anthropic.com"}
    if workspace_id:
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
    return anthropic.Anthropic(**kwargs)


def run(cfg) -> dict:
    cfg.provider = "anthropic"
    cfg.anthropic_route = "direct"
    cfg.model = cfg.model or "claude-sonnet-5"
    cfg.thinking = "disabled"
    cfg.effort = "low"
    cfg.read_multiplier = 0.1
    cfg.output_multiplier = 5.0
    cfg.violation_tokens = 60
    cfg.workers = 1

    client = make_client()
    writes = live.make_compiler(
        cfg.provider,
        cfg.model,
        effort=cfg.effort,
        thinking=cfg.thinking,
    ).descriptor.cache_write_multiplier
    jobs = [(index, arm) for index in range(cfg.repeats) for arm in cfg.arms]
    nonces = live.session_nonces(jobs, True)
    done = {job: live.session(job[1], nonces[job], cfg, client) for job in jobs}
    runs = []
    for index in range(cfg.repeats):
        one = {arm: done[(index, arm)] for arm in cfg.arms}
        one["summary"] = {arm: live.summarize(one[arm], cfg, writes) for arm in cfg.arms}
        runs.append(one)
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": live.git_sha(),
        "provider": "anthropic",
        "model": cfg.model,
        "anthropic_route": "direct",
        "turns": cfg.turns,
        "repeats": cfg.repeats,
        "arms": cfg.arms,
        "history_style": cfg.history_style,
        "thinking": cfg.thinking,
        "effort": cfg.effort,
        "write_multiplier": writes,
        "read_multiplier": cfg.read_multiplier,
        "output_multiplier": cfg.output_multiplier,
        "paid": True,
        "nonce_scope": "session",
        "workers": cfg.workers,
        "workspace_header_configured": any(os.environ.get(name) for name in WORKSPACE_ID_NAMES),
    }
    return {"meta": meta, "runs": runs, "aggregate": live.aggregate(runs, cfg.arms)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=live.ARMS, required=True)
    parser.add_argument("--model")
    parser.add_argument("--history-style", choices=("template", "varied"), default="varied")
    cfg = parser.parse_args()
    if cfg.turns < 1 or cfg.repeats < 1:
        raise SystemExit("turns and repeats must be positive")
    print(json.dumps(run(cfg), indent=2))

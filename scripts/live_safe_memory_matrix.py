#!/usr/bin/env python3
"""Synthetic-safe paid memory-placement matrix for longer live provider validation.

This preserves the legacy harness's memory change rates, answer domains, placement logic,
grading, and billing accounting while replacing real-looking customer/profile text with
explicit benchmark data. Provider failures are re-raised with sanitized location metadata
(provider/arm/turn/status/code) and never include prompt text, model output, or secrets.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import live_memory_placement as live  # noqa: E402

WORKSPACE_ID_NAMES = ("PCF_ANTHROPIC_WORKSPACE_ID", "ANTHROPIC_WORKSPACE_ID")
FIXTURE_ID = "synthetic-safe-v1"
QUESTION_TYPES = ("open_ticket_count", "current_plan", "contact_channel", "preferred_language")
ANTHROPIC_OUTPUT_CONTRACT = (
    "Benchmark response contract: return exactly one answer value and nothing else. "
    "For counts, return digits only. For plan, contact-channel, and language questions, return only the exact "
    "configured value. Do not explain, quote, cite, summarize, or repeat synthetic history."
)


def _safe_profile() -> dict:
    return {
        "name": "Benchmark User",
        "preferred_language": "Spanish",
        "member_since": 2019,
        "region": "Test Region",
        "devices": [f"Synthetic device {k}: model X{k}, benchmark year 20{15 + k}, active" for k in range(8)],
        "addresses": [f"Synthetic location slot {k}" for k in range(4)],
    }


def _safe_preferences(turn: int) -> dict:
    return {"contact_channel": live.CHANNELS[(turn // 6) % len(live.CHANNELS)], "quiet_hours": "21:00-08:00"}


def _safe_notes(turn: int) -> dict:
    version = turn // 3
    return {
        "agent_notes": [f"Synthetic note {k}: follow up on reference REF-{700 + k}." for k in range(12)],
        "current_plan": live.PLANS[version % len(live.PLANS)],
    }


def _safe_account(turn: int) -> dict:
    return {"open_tickets": turn + 2, "latest_ticket": f"TEST-{4100 + turn}"}


def apply_safe_fixture() -> None:
    live.POLICIES = [
        f"Benchmark rule {i}: for synthetic topic {i}, inspect the synthetic record, cite the relevant test "
        f"reference, and keep the reply under three sentences."
        for i in range(60)
    ]
    live.profile = _safe_profile
    live.preferences = _safe_preferences
    live.notes = _safe_notes
    live.account = _safe_account
    live.QUESTIONS = [
        (
            "What is the current open-ticket count in this synthetic record? Answer with just the number.",
            lambda t: str(_safe_account(t)["open_tickets"]),
        ),
        (
            "Which plan is currently configured in this synthetic record? Answer with just the plan name.",
            lambda t: _safe_notes(t)["current_plan"],
        ),
        (
            "Which contact channel is currently configured in this synthetic record? Answer with one word.",
            lambda t: _safe_preferences(t)["contact_channel"],
        ),
        (
            "Which preferred language is currently configured in this synthetic record? Answer in English with one word.",
            lambda t: _safe_profile()["preferred_language"],
        ),
    ]
    live.VARIED_FILLER = [
        "Synthetic log entry {n} was recorded for benchmark continuity.",
        "Synthetic reference {n} was checked during the benchmark.",
        "Synthetic event {n} was appended to the test history.",
        "Synthetic checkpoint {n} completed in the test session.",
    ]


def _provider_error_code(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            if isinstance(code, str) and code:
                return code[:80]
    return type(exc).__name__


def _system_text(provider: str, nonce: str) -> str:
    contract = f"\n{ANTHROPIC_OUTPUT_CONTRACT}" if provider == "anthropic" else ""
    return f"Benchmark session {nonce}.{contract}\n" + "\n".join(live.POLICIES)


def _grade_safe(
    answer: str,
    turn: int,
    expected: str,
    stale: str | None,
    style: str,
    violation_tokens: int,
) -> dict:
    """Preserve the legacy grade while splitting format failure causes.

    `contract_output` is a deterministic extraction of the bounded answer value. It is
    diagnostic/fallback evidence only; raw provider format compliance remains separately visible.
    """
    base = live.grade(answer, turn, expected, stale, style, violation_tokens)
    fillers = [live.TEMPLATE_FILLER] if style == "template" else [
        filler.split("{n}")[1].strip() for filler in live.VARIED_FILLER
    ]
    copied_filler = any(filler.lower() in answer.lower() for filler in fillers)
    too_long = math.ceil(len(answer) / 4) > violation_tokens
    format_violation = copied_filler or too_long
    if base["violation"] != format_violation:
        raise AssertionError("safe fixture format diagnostics drifted from legacy grading")
    contract_output = base["value"]
    return {
        **base,
        "copied_filler": copied_filler,
        "too_long": too_long,
        "format_violation": format_violation,
        "contract_output": contract_output,
        "contract_repaired": format_violation and bool(contract_output),
    }


def _summarize(rows: list[dict], cfg, write_multiplier: float) -> dict:
    summary = live.summarize(rows, cfg, write_multiplier)
    total = len(rows)
    for key in ("copied_filler", "too_long", "format_violation", "contract_repaired"):
        summary[key] = f"{sum(bool(row[key]) for row in rows)}/{total}"
    return summary


def _aggregate(runs: list[dict], arms) -> dict:
    aggregate = live.aggregate(runs, arms)
    for arm in arms:
        if arm not in aggregate:
            continue
        summaries = [run["summary"][arm] for run in runs]
        for key in ("copied_filler", "too_long", "format_violation", "contract_repaired"):
            pairs = [summary[key].split("/") for summary in summaries]
            aggregate[arm][key] = f"{sum(int(a) for a, _ in pairs)}/{sum(int(b) for _, b in pairs)}"
    return aggregate


def _sanitized_misses(runs: list[dict], arms) -> list[dict]:
    misses = []
    for repeat_index, run in enumerate(runs, start=1):
        for arm in arms:
            for row in run[arm]:
                if row.get("correct"):
                    continue
                misses.append(
                    {
                        "repeat": repeat_index,
                        "arm": arm,
                        "turn": row["turn"],
                        "question_type": QUESTION_TYPES[row["turn"] % len(QUESTION_TYPES)],
                        "expected": row["expected"],
                        "extracted": row.get("contract_output", ""),
                        "stale_trap": bool(row["stale_trap"]),
                        "copied_filler": bool(row.get("copied_filler")),
                        "too_long": bool(row.get("too_long")),
                        "format_violation": bool(row.get("format_violation")),
                    }
                )
    return misses


def _make_client(provider: str):
    if provider == "openai":
        return live.make_client("openai")
    key = next((os.environ[name] for name in live.ANTHROPIC_KEYS if os.environ.get(name)), None)
    if not key:
        raise SystemExit("direct Anthropic run requires PCF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY")
    workspace_id = next((os.environ[name] for name in WORKSPACE_ID_NAMES if os.environ.get(name)), None)
    import anthropic

    kwargs = {"api_key": key, "base_url": "https://api.anthropic.com"}
    if workspace_id:
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
    return anthropic.Anthropic(**kwargs)


def _session(arm: str, nonce: str, cfg, client) -> list[dict]:
    compiler = live.make_compiler(cfg.provider, cfg.model, effort=cfg.effort, thinking=cfg.thinking)
    placer = live.MemoryPlacer(
        compiler.tokenizer,
        write_multiplier=compiler.descriptor.cache_write_multiplier,
        read_multiplier=cfg.read_multiplier,
    )
    system = live.Segment("s", "system", _system_text(cfg.provider, nonce))
    history, rows, said = [], [], {}
    prefix_tokens = compiler.tokenizer.count(system.content)
    for turn in range(cfg.turns):
        front, tail = live.arrange(arm, placer, live.memory(turn), history, prefix_tokens)
        text, expected = live.question(turn)
        ctx = live.Context(
            [system, *front, *history, *tail, live.Segment("u", "user", text, stable=False)],
            cache_namespace=f"{arm}-{nonce}",
        )
        compiled = compiler.compile(ctx)
        estimate = compiler.warmth(ctx, live.PrefixCache(compiler.descriptor.ttl_seconds), 0.0)
        stale = said.get(text)
        row = {
            "turn": turn,
            "tail": [segment.id for segment in tail],
            "expected": expected,
            "stale": stale,
            "stale_trap": stale is not None and stale != expected,
            "est_tokens": compiled.total_tokens,
            "est_written": estimate.cache_creation_tokens,
        }
        try:
            response, answer, out = live.call(cfg.provider, client, compiled.request, cfg)
        except Exception as exc:
            status = live._status_code(exc)
            status_text = str(status) if status is not None else "unknown"
            code = _provider_error_code(exc)
            raise RuntimeError(
                f"provider_call_failed provider={cfg.provider} arm={arm} turn={turn} "
                f"status={status_text} code={code}"
            ) from None
        usage, answer = compiler.usage_from_response(response), answer.strip()
        row.update(
            cached=usage.cache_read_input_tokens,
            written=usage.cache_creation_input_tokens,
            uncached=usage.input_tokens,
            answer=answer,
            **out,
            **_grade_safe(answer, turn, expected, stale, cfg.history_style, cfg.violation_tokens),
        )
        rows.append(row)
        said[text] = expected
        history.append(live.history_turn(turn, text, expected, cfg.history_style))
    return rows


def run(cfg) -> dict:
    apply_safe_fixture()
    cfg.anthropic_route = "direct"
    cfg.model = cfg.model or ("gpt-5.6" if cfg.provider == "openai" else "claude-sonnet-5")
    cfg.thinking = "disabled" if cfg.provider == "anthropic" else "default"
    cfg.effort = "low"
    cfg.read_multiplier = 0.1
    cfg.output_multiplier = 5.0
    cfg.violation_tokens = 60
    cfg.workers = 1

    client = _make_client(cfg.provider)
    writes = live.make_compiler(
        cfg.provider,
        cfg.model,
        effort=cfg.effort,
        thinking=cfg.thinking,
    ).descriptor.cache_write_multiplier
    jobs = [(index, arm) for index in range(cfg.repeats) for arm in cfg.arms]
    nonces = live.session_nonces(jobs, True)
    done = {job: _session(job[1], nonces[job], cfg, client) for job in jobs}
    runs = []
    for index in range(cfg.repeats):
        one = {arm: done[(index, arm)] for arm in cfg.arms}
        one["summary"] = {arm: _summarize(one[arm], cfg, writes) for arm in cfg.arms}
        runs.append(one)
    meta = {
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": live.git_sha(),
        "provider": cfg.provider,
        "model": cfg.model,
        "anthropic_route": "direct" if cfg.provider == "anthropic" else None,
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
        "fixture": FIXTURE_ID,
        "output_contract": "anthropic-exact-value-v1" if cfg.provider == "anthropic" else "question-local-v1",
        "workspace_header_configured": (
            any(os.environ.get(name) for name in WORKSPACE_ID_NAMES) if cfg.provider == "anthropic" else None
        ),
    }
    return {
        "meta": meta,
        "runs": runs,
        "aggregate": _aggregate(runs, cfg.arms),
        "misses": _sanitized_misses(runs, cfg.arms),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    parser.add_argument("--turns", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=live.ARMS, required=True)
    parser.add_argument("--model")
    parser.add_argument("--history-style", choices=("template", "varied"), default="varied")
    cfg = parser.parse_args()
    if cfg.turns < 1 or cfg.repeats < 1:
        raise SystemExit("turns and repeats must be positive")
    print(json.dumps(run(cfg), indent=2))

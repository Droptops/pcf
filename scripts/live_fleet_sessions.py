#!/usr/bin/env python3
"""The fleet cache test in docs/FLEET_CACHE.md. Offline (SimEngine and a clock) by default; paid calls need --run.

Arms, on the synthetic domain scenarios:
  tuned-private - the tuned front layout with a per-session nonce in the system text and a per-session namespace
  tuned-shared  - the same layout; system text, reference and namespace identical across the arm's sessions
  placed-shared - MemoryPlacer with the same shared prefix; one placer per session
  placed-cold   - MemoryPlacer, and a gap of ttl_seconds + 30 before every turn after the first
Warm arms run 1 warmup session and then --sessions measured sessions back to back, per scenario. The system text
carries a run id and the arm name, so arms and earlier runs never share a prefix; within an arm it is identical.
Cold sessions each get a private prefix, so they can run concurrently without warming each other.
--analyze FILE... applies the pre-registered pass rules to saved runs.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from pcf import Context, Segment  # noqa: E402
from pcf.cache import PrefixCache  # noqa: E402
from pcf.families.sim import SimEngine  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402
from domain_scenarios import SCENARIOS, normalize  # noqa: E402
from live_domain_sessions import arrange, grade, memory, placement  # noqa: E402

ARMS = ("tuned-private", "tuned-shared", "placed-shared", "placed-cold")
WARM = ARMS[:3]


def session(key: str, arm: str, tag: str, cfg, client=None, engine=None, clock=None, sink=None) -> list[dict]:
    """One session. Paid when client is set; otherwise simulated on engine, advancing clock[0].

    The scripted sessions of a scenario are identical, so a per-session segment follows the system text and the
    reference: without it a shared prefix would reach into the previous session's history, which real users
    do not repeat."""
    scenario = SCENARIOS[key]
    compiler = placement.make_compiler(cfg.provider, cfg.model)
    ttl = compiler.descriptor.ttl_seconds
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=compiler.descriptor.cache_write_multiplier,
                          read_multiplier=cfg.read_multiplier)
    system = Segment("s", "system", f"Fleet {tag}.\n" + scenario.system())
    user = Segment("session", "memory", f"Session {uuid.uuid4().hex[:12]}.", provenance="session")
    probe = Context([system, memory(scenario, 0)[0], Segment("u", "user", "?", stable=False)])
    prefix_tokens = compiler.compile(probe).total_tokens
    cold = arm == "placed-cold"
    history, rows, said = [], [], {}
    for turn in range(cfg.turns):
        gap = 0 if turn == 0 else (ttl + 30 if cold else 1)
        if client is not None and cold and turn:
            time.sleep(gap)
        if clock is not None:
            clock[0] += gap
        mem = memory(scenario, turn)
        if arm.startswith("placed"):
            front, tail = placer.split(mem, history, cold=cold and turn > 0)
        else:
            front, tail = arrange("front-tuned", placer, mem, history, scenario.volatile())
        ask, expected = scenario.question(turn)
        shared = [s for s in front if s.id == "reference"]
        own = [s for s in front if s.id != "reference"]
        ctx = Context([system, *shared, user, *own, *history, *tail, Segment("u", "user", ask.text, stable=False)],
                      cache_namespace=tag)
        stale = said.get(ask.text)
        row = {"turn": turn, "gap_s": gap, "module": ask.module, "tail": [s.id for s in tail],
               "expected": expected, "stale": stale,
               "stale_trap": stale is not None and normalize(ask.kind, stale) != normalize(ask.kind, expected)}
        if client is not None:
            compiled = compiler.compile(ctx)
            started = time.perf_counter()
            response, answer, out = placement.call(cfg.provider, client, compiled.request, cfg)
            out["latency_s"] = round(time.perf_counter() - started, 3)
            usage, answer = compiler.usage_from_response(response), answer.strip()
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, answer=answer, **out,
                       **grade(ask, answer, expected, stale, cfg.violation_tokens))
        else:
            usage, compiled = engine.run(ctx, clock[0])
            if sink is not None:
                sink.append(compiled.request)
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, output_tokens=0)
        if turn == 0:
            row["prefix_tokens"] = prefix_tokens
        rows.append(row)
        said[ask.text] = expected
        history.append(Segment(f"h{turn}", "history", [{"role": "user", "content": ask.text},
                                                        {"role": "assistant",
                                                         "content": scenario.reply(turn, expected)}]))
    return rows


def billed(rows: list[dict], write: float, read: float) -> float:
    return math.fsum(r["uncached"] + write * r["written"] + read * r["cached"] for r in rows)


def analyze(paths: list[str]) -> dict:
    """Pass rules of docs/FLEET_CACHE.md, per model. Pools measured sessions across files of the same model."""
    by_model: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            saved = json.load(f)
        meta = saved["meta"]
        model = by_model.setdefault(meta["model"], {"meta": meta, "cells": {}, "sources": []})
        model["sources"].append(os.path.basename(path))
        for key, arms in saved["cells"].items():
            for arm, cell in arms.items():
                model["cells"].setdefault(key, {}).setdefault(arm, []).extend(cell["measured"])
    out = {}
    for name, model in by_model.items():
        meta, cells = model["meta"], model["cells"]
        write, read = meta["write_multiplier"], meta["read_multiplier"]

        def mean_cost(key, arm, turns):
            sessions = cells.get(key, {}).get(arm, [])
            return statistics.mean(billed(s[:turns], write, read) for s in sessions) if sessions else None

        report = {"sources": model["sources"], "simulated": not meta["paid"], "cells": {}}
        for key, arms in cells.items():
            for arm, sessions in arms.items():
                rows = [r for s in sessions for r in s]
                turn0 = [s[0]["cached"] / s[0]["prefix_tokens"] for s in sessions]
                cell = {"sessions": len(sessions), "turns": len(sessions[0]),
                        "billed_input_mean": round(mean_cost(key, arm, len(sessions[0]))),
                        "turn0_read_share_median": round(statistics.median(turn0), 3),
                        "turns_with_no_read": f"{sum(r['cached'] == 0 for r in rows)}/{len(rows)}"}
                if len(sessions[0]) > 24 and arm in WARM:
                    cell["billed_input_mean_24"] = round(mean_cost(key, arm, 24))
                if "correct" in rows[0]:
                    cell["correct"] = f"{sum(r['correct'] for r in rows)}/{len(rows)}"
                    cell["gave_stale"] = sum(r["gave_stale"] for r in rows)
                    cell["output_tokens_mean"] = round(statistics.mean(sum(r["output_tokens"] for r in s)
                                                                       for s in sessions))
                    latency = sorted(r["latency_s"] for r in rows)
                    cell["latency_s"] = {"p50": latency[len(latency) // 2],
                                         "p90": latency[int(len(latency) * .9)]}
                report["cells"][f"{key}/{arm}"] = cell
        rules = {}
        warm_keys = [k for k in cells if {"tuned-shared", "placed-shared"} <= set(cells[k])]
        turns = min((len(cells[k]["tuned-shared"][0]) for k in warm_keys), default=0)
        if warm_keys:
            for n in sorted({24, turns}):
                if n > turns:
                    continue
                ratio = (math.fsum(mean_cost(k, "placed-shared", n) for k in warm_keys) /
                         math.fsum(mean_cost(k, "tuned-shared", n) for k in warm_keys))
                rules[f"placed_over_tuned_shared_{n}"] = round(ratio, 3)
            rules["rule1_long_session_pass"] = (turns >= 60 and rules[f"placed_over_tuned_shared_{turns}"] <= .6
                                                if turns >= 60 else None)
            shares = {arm: statistics.median(s[0]["cached"] / s[0]["prefix_tokens"]
                                             for k in cells for s in cells[k].get(arm, []))
                      for arm in WARM if any(arm in cells[k] for k in cells)}
            rules["turn0_read_share"] = {arm: round(v, 3) for arm, v in shares.items()}
            rules["rule2_fleet_pass"] = (shares.get("tuned-shared", 0) >= .8 and
                                         shares.get("placed-shared", 0) >= .8 and
                                         shares.get("tuned-private", 1) < .2)
        cold = cells.get("benefits", {}).get("placed-cold")
        if cold and cells["benefits"].get("tuned-shared"):
            n = len(cold[0])
            ratio = mean_cost("benefits", "placed-cold", n) / mean_cost("benefits", "tuned-shared", n)
            rules[f"cold_placed_over_warm_tuned_shared_{n}"] = round(ratio, 3)
            rules["rule3_gap_pass"] = ratio > .85 if meta["provider"] == "anthropic" else None
        report["rules"] = rules
        out[name] = report
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="make paid calls")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--model")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=["benefits", "clinical"])
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(WARM))
    parser.add_argument("--turns", type=int, default=60)
    parser.add_argument("--sessions", type=int, default=8, help="measured sessions per arm and scenario")
    parser.add_argument("--warmup", type=int, default=1, help="unscored sessions before the measured ones")
    parser.add_argument("--cold-turns", type=int, help="turns per cold session (default: --turns)")
    parser.add_argument("--thinking", choices=("default", "disabled"), default="default")
    parser.add_argument("--effort", default="low", help="openai reasoning effort")
    parser.add_argument("--read-multiplier", type=float, default=0.1)
    parser.add_argument("--output-multiplier", type=float, default=5.0)
    parser.add_argument("--violation-tokens", type=int, default=80)
    parser.add_argument("--analyze", nargs="+", metavar="FILE", help="apply the pass rules to saved runs and exit")
    cfg = parser.parse_args()
    if cfg.analyze:
        print(json.dumps(analyze(cfg.analyze), indent=2))
        return
    cfg.model = cfg.model or ("gpt-5.6" if cfg.provider == "openai" else "claude-sonnet-5")
    client = None
    if cfg.run:
        key = "OPENAI_API_KEY" if cfg.provider == "openai" else "OPENROUTER_API_KEY"
        if not os.environ.get(key):
            raise SystemExit(f"--run --provider {cfg.provider} requires {key}")
        if cfg.provider == "openai":
            import openai
            client = openai.OpenAI()
        else:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ[key], base_url="https://openrouter.ai/api")
    compiler = placement.make_compiler(cfg.provider, cfg.model)
    ttl = compiler.descriptor.ttl_seconds
    run_id = uuid.uuid4().hex[:10] if client else "offline"

    def chain(job):
        """The warmup and measured sessions of one warm arm, back to back; or one cold session."""
        key, arm, index = job
        try:
            if arm == "placed-cold":
                sub = argparse.Namespace(**{**vars(cfg), "turns": cfg.cold_turns or cfg.turns})
                engine, clock = (None, None) if client else (SimEngine(compiler, PrefixCache(ttl)), [0.0])
                return [session(key, arm, f"{run_id}-{arm}-{index}", sub, client, engine, clock)]
            tag = f"{run_id}-{arm}"
            engine, clock = (None, None) if client else (SimEngine(compiler, PrefixCache(ttl)), [0.0])
            return [session(key, arm, f"{tag}-{i}" if arm == "tuned-private" else tag, cfg, client, engine, clock)
                    for i in range(cfg.warmup + cfg.sessions)]
        except Exception as exc:  # e.g. quota exhausted: report the chain as failed, keep the others
            return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    jobs = [(k, arm, 0) for k in cfg.scenarios for arm in cfg.arms if arm != "placed-cold"]
    jobs += [(k, "placed-cold", i) for k in cfg.scenarios if "placed-cold" in cfg.arms for i in range(cfg.sessions)]
    with ThreadPoolExecutor(len(jobs)) as pool:
        done = dict(zip(jobs, pool.map(chain, jobs)))
    failed = [{"scenario": j[0], "arm": j[1], "index": j[2], "error": r["error"]}
              for j, r in done.items() if isinstance(r, dict)]
    cells: dict = {}
    for (key, arm, _), result in done.items():
        if isinstance(result, dict):
            continue
        cell = cells.setdefault(key, {}).setdefault(arm, {"warmup": [], "measured": []})
        if arm == "placed-cold":
            cell["measured"].extend(result)
        else:
            cell["warmup"], cell["measured"] = result[:cfg.warmup], result[cfg.warmup:]
    meta = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_sha": placement.git_sha(), "provider": cfg.provider, "model": cfg.model, "run_id": run_id,
            "ttl_seconds": ttl, "cold_gap_seconds": ttl + 30, "turns": cfg.turns,
            "cold_turns": cfg.cold_turns or cfg.turns, "sessions": cfg.sessions, "warmup": cfg.warmup,
            "arms": cfg.arms, "scenarios": cfg.scenarios, "thinking": cfg.thinking, "effort": cfg.effort,
            "write_multiplier": compiler.descriptor.cache_write_multiplier, "read_multiplier": cfg.read_multiplier,
            "output_multiplier": cfg.output_multiplier, "violation_tokens": cfg.violation_tokens,
            "paid": client is not None, "data": "synthetic; see scripts/domain_scenarios.py",
            "failed_chains": failed}
    print(json.dumps({"meta": meta, "cells": cells}, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} session chains failed")


if __name__ == "__main__":
    main()

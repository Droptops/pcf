#!/usr/bin/env python3
"""Memory placement on synthetic domain workloads. Offline by default; paid calls need --run.

Runs the six scenarios in scripts/domain_scenarios.py (healthcare, government, enterprise) under seven layouts:
  front       - reference and record modules before history, stable first, all left `stable` (a naive layout)
  front-tuned - the same order, with the modules that change marked `stable=False` (the best front layout)
  tail        - all memory after history, just before the question
  placed      - MemoryPlacer picks front or tail per module from observed change rates and cache prices
  echo        - memory frozen in front as it was on turn 0 (the cached prefix never changes), and the current
                version of the module the question asks about repeated before the question, marked current
  echo-all    - the same, repeating every module that changes (an application rarely knows which one is asked)
  fixed-tail  - declared volatile modules after history from turn zero, stable modules in front
--history-mode model-text replays each arm's actual visible responses instead of scripted replies. Questions and
records remain synthetic; this mode does not replay hidden reasoning or execute tools.
Replies are graded by the first value asserted from the question's answer set. Costs use the same units as
scripts/live_memory_placement.py. --analyze FILE... prints per-scenario totals and exact McNemar tests pairing
each turn across layouts; --regrade FILE re-grades a saved run with the current grader, offline.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import math
import os
import statistics
import time
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
from pcf import Context, Segment  # noqa: E402
from pcf.cache import PrefixCache  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402
from domain_scenarios import SCENARIOS, answer_value, normalize  # noqa: E402

spec = importlib.util.spec_from_file_location("placement", os.path.join(HERE, "live_memory_placement.py"))
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)

ARMS = ("front", "front-tuned", "tail", "placed", "echo", "echo-all", "fixed-tail")


def memory(scenario, turn: int) -> list[Segment]:
    mods = [("reference", scenario.reference)] + [(name, fn(turn)) for name, fn in scenario.modules.items()]
    return [Segment(name, "memory", data, provenance=name) for name, data in mods]


def arrange(arm: str, placer: MemoryPlacer, mem: list[Segment], history: list[Segment], volatile: set[str]):
    if arm == "fixed-tail":
        return ([s for s in mem if s.id not in volatile],
                [Segment(s.id, "memory", s.content, False, provenance=s.provenance)
                 for s in mem if s.id in volatile])
    if arm == "front-tuned":
        return [Segment(s.id, "memory", s.content, s.id not in volatile, provenance=s.provenance) for s in mem], []
    if arm == "placed":
        return placer.split(mem, history)
    if arm == "tail":
        return [], [Segment(s.id, "memory", s.content, False, provenance=s.provenance) for s in mem]
    return mem, []


def grade(ask, answer: str, expected: str, stale: str | None, violation_tokens: int) -> dict:
    value, want = answer_value(ask, answer), normalize(ask.kind, expected)
    trap = stale is not None and normalize(ask.kind, stale) != want
    return {"value": value, "correct": value == want,
            "gave_stale": trap and value == normalize(ask.kind, stale),
            "violation": math.ceil(len(answer) / 4) > violation_tokens}


def session(key: str, arm: str, nonce: str, cfg, client=None, sink: list | None = None) -> list[dict]:
    """One session; `sink`, when given, receives each compiled request."""
    scenario = SCENARIOS[key]
    history_mode = getattr(cfg, "history_mode", "scripted")
    if history_mode not in ("scripted", "model-text"):
        raise ValueError("unknown history mode")
    if history_mode == "model-text" and client is None:
        raise ValueError("model-text history requires responses; use --run")
    compiler = placement.make_compiler(cfg.provider, cfg.model)
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=compiler.descriptor.cache_write_multiplier,
                          read_multiplier=cfg.read_multiplier)
    system = Segment("s", "system", f"Session {nonce}-{key}-{arm}.\n" + scenario.system())
    history, rows, said, prior = [], [], {}, {}
    for turn in range(cfg.turns):
        ask, expected = scenario.question(turn)
        if arm.startswith("echo"):
            front = memory(scenario, 0)
            wanted = scenario.volatile() if arm == "echo-all" else {ask.module} & scenario.volatile()
            tail = [Segment(f"{s.id}-now", "memory", s.content, False, provenance=f"{s.id} (current)")
                    for s in memory(scenario, turn) if s.id in wanted]
        else:
            front, tail = arrange(arm, placer, memory(scenario, turn), history, scenario.volatile())
        ctx = Context([system, *front, *history, *tail, Segment("u", "user", ask.text, stable=False)],
                      cache_namespace=f"{key}-{arm}-{nonce}")
        started = time.perf_counter()
        compiled = compiler.compile(ctx)
        compile_ms = round(1000 * (time.perf_counter() - started), 2)
        if sink is not None:
            sink.append(compiled.request)
        estimate = compiler.warmth(ctx, PrefixCache(compiler.descriptor.ttl_seconds), 0.0)
        stale = said.get(ask.text)
        row = {"turn": turn, "module": ask.module, "tail": [s.id for s in tail], "expected": expected,
               "stale": stale, "stale_trap": stale is not None and normalize(ask.kind, stale) !=
               normalize(ask.kind, expected), "est_tokens": compiled.total_tokens,
               "est_written": estimate.cache_creation_tokens, "compile_ms": compile_ms}
        row["history_mode"] = history_mode
        if getattr(cfg, "capture_requests", False):
            row["request"] = placement.request_payload(cfg.provider, compiled.request, cfg)
            row["request_ts"] = time.time()
        if client is not None:
            started = time.perf_counter()
            response, answer, out = placement.call(cfg.provider, client, compiled.request, cfg)
            out["latency_s"] = round(time.perf_counter() - started, 3)  # request to full response, not streamed
            usage, answer = compiler.usage_from_response(response), answer.strip()
            row.update(cached=usage.cache_read_input_tokens, written=usage.cache_creation_input_tokens,
                       uncached=usage.input_tokens, answer=answer, **out,
                       **grade(ask, answer, expected, stale, cfg.violation_tokens))
            previous = prior.get(ask.text)
            row["prior_error_exposed"] = bool(history_mode == "model-text" and previous and not previous[1])
            row["repeated_prior_error"] = bool(row["prior_error_exposed"] and not row["correct"]
                                                and row["value"] and row["value"] == previous[0])
            prior[ask.text] = (row["value"], row["correct"])
        rows.append(row)
        reply = answer if history_mode == "model-text" else scenario.reply(turn, expected)
        said[ask.text] = (answer_value(ask, reply) or None) if history_mode == "model-text" else expected
        history.append(Segment(f"h{turn}", "history", [{"role": "user", "content": ask.text},
                                                        {"role": "assistant",
                                                         "content": reply}]))
    return rows


def regrade(path: str) -> dict:
    """Re-grade every row of a saved run with the current grader; summaries and aggregates are recomputed."""
    saved = json.load(open(path))
    meta = saved["meta"]
    prices = argparse.Namespace(read_multiplier=meta["read_multiplier"], output_multiplier=meta["output_multiplier"])
    changes = []
    for key, runs in saved["scenarios"].items():
        scenario = SCENARIOS[key]
        for i, run in enumerate(runs):
            for arm in meta["arms"]:
                prior = {}
                for row in run[arm]:
                    ask, expected = scenario.question(row["turn"])
                    old = row["correct"]
                    row.update(grade(ask, row["answer"], expected, row["stale"], meta["violation_tokens"]))
                    if meta.get("history_mode", "scripted") == "model-text":
                        previous = prior.get(ask.text)
                        row["prior_error_exposed"] = bool(previous and not previous[1])
                        row["repeated_prior_error"] = bool(row["prior_error_exposed"] and not row["correct"]
                                                           and row["value"] and row["value"] == previous[0])
                        prior[ask.text] = (row["value"], row["correct"])
                    if old != row["correct"]:
                        changes.append({"scenario": key, "run": i, "arm": arm, "turn": row["turn"], "now": row["correct"]})
            run["summary"] = {arm: placement.summarize(run[arm], prices, meta["write_multiplier"]) for arm in meta["arms"]}
        saved["aggregate"][key] = placement.aggregate(runs, meta["arms"])
    saved["meta"] = {**meta, "regraded": True, "regrade_changes": changes}
    return saved


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value for b and c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def analyze(paths: list[str]) -> dict:
    """Pool compatible repetitions before producing reports; never overwrite an earlier scenario."""
    grouped, seen = {}, set()
    fields = ("provider", "turns", "thinking", "effort", "write_multiplier", "read_multiplier",
              "output_multiplier", "violation_tokens", "data")
    for path in paths:
        real = os.path.realpath(path)
        if real in seen:
            raise ValueError(f"duplicate input file: {path}")
        seen.add(real)
        with open(path) as handle:
            saved = json.load(handle)
        meta = saved["meta"]
        model = meta["model"]
        config = {k: meta[k] for k in fields}
        config["history_mode"] = meta.get("history_mode", "scripted")
        config["arms"] = sorted(meta["arms"])
        per = grouped.setdefault(model, {"config": config, "runs": {}, "sources": []})
        if config != per["config"]:
            different = [k for k in config if config[k] != per["config"][k]]
            raise ValueError(f"incompatible runs for {model}: {', '.join(different)}")
        per["sources"].append(path)
        prices = argparse.Namespace(**config)
        for key, runs in saved["scenarios"].items():
            for run in runs:
                for arm in config["arms"]:
                    if len(run[arm]) != config["turns"]:
                        raise ValueError(f"incomplete session: {model}/{key}/{arm}")
                # Recompute summaries from the observations, not a possibly stale saved aggregate.
                run["summary"] = {arm: placement.summarize(run[arm], prices, config["write_multiplier"])
                                  for arm in config["arms"]}
            per["runs"].setdefault(key, []).extend(runs)

    out = {}
    for model, per in grouped.items():
        arms = per["config"]["arms"]
        scenarios = {key: placement.aggregate(runs, arms) for key, runs in per["runs"].items()}
        pairs = {}
        for key, runs in per["runs"].items():
            for run in runs:
                for a, b in (("front", "tail"), ("front", "placed"), ("front-tuned", "tail"),
                             ("front-tuned", "placed"), ("tail", "placed"), ("echo", "placed"),
                             ("echo-all", "placed"), ("front-tuned", "echo"), ("front-tuned", "echo-all"),
                             ("fixed-tail", "placed"), ("front-tuned", "fixed-tail")):
                    if a not in arms or b not in arms:
                        continue
                    p = pairs.setdefault(f"{a} vs {b}", {"all": [0, 0], "stale_traps": [0, 0]})
                    for ra, rb in zip(run[a], run[b], strict=True):
                        paired = ("turn", "expected", "stale_trap") if per["config"]["history_mode"] == "scripted" \
                            else ("turn", "expected")
                        if any(ra[k] != rb[k] for k in paired):
                            raise ValueError(f"unpaired observations: {model}/{key}/{a}/{b}")
                        for slot in ("all", "stale_traps") if ra["stale_trap"] and rb["stale_trap"] else ("all",):
                            if ra["correct"] and not rb["correct"]:
                                p[slot][0] += 1
                            elif rb["correct"] and not ra["correct"]:
                                p[slot][1] += 1
        for p in pairs.values():
            for slot, (b, c) in list(p.items()):
                p[slot] = {"only_first_correct": b, "only_second_correct": c, "p": round(mcnemar(b, c), 4)}
        totals, all_repeats = {}, {}
        prices = argparse.Namespace(**per["config"])
        for arm in arms:
            agg = [s[arm] for s in scenarios.values()]
            totals[arm] = {"billed_input_units": sum(s["billed_input_units"]["mean"] for s in agg),
                           "billed_total_units": sum(s["billed_total_units"]["mean"] for s in agg),
                           "correct": _sum_frac(s["correct"] for s in agg),
                           "correct_on_stale_traps": _sum_frac(s["correct_on_stale_traps"] for s in agg),
                           "gave_stale": sum(s["gave_stale"] for s in agg)}
            rows = [r for runs in per["runs"].values() for run in runs for r in run[arm]]
            all_repeats[arm] = placement.summarize(rows, prices, per["config"]["write_multiplier"])
            totals[arm]["no_value_given"] = sum(not r["value"] for r in rows)
            totals[arm]["prior_error_exposures"] = sum(r.get("prior_error_exposed", False) for r in rows)
            totals[arm]["repeated_prior_errors"] = sum(r.get("repeated_prior_error", False) for r in rows)
            latencies = sorted(r["latency_s"] for r in rows if "latency_s" in r)
            if latencies:
                totals[arm]["latency_s"] = {"median": round(statistics.median(latencies), 2),
                                            "p90": latencies[int(0.9 * (len(latencies) - 1))],
                                            "n": len(latencies)}
        out[model] = {"sources": per["sources"], "config": per["config"], "scenarios": scenarios,
                      "repeats_by_scenario": {k: len(v) for k, v in per["runs"].items()}, "pairs": pairs,
                      "totals_of_scenario_means": totals, "totals_all_repeats": all_repeats,
                      "cost_basis": "totals_of_scenario_means averages cost over repetitions within each scenario; "
                                    "accuracy counts cover all repetitions. totals_all_repeats sums both."}
    return out


def _sum_frac(values) -> str:
    pairs = [v.split("/") for v in values]
    return f"{sum(int(a) for a, _ in pairs)}/{sum(int(b) for _, b in pairs)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make paid calls (needs OPENAI_API_KEY or an Anthropic "
                        "key, see --anthropic-route; any value works when a proxy injects the real key)")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default="openai")
    parser.add_argument("--anthropic-route", choices=placement.ANTHROPIC_ROUTES, default="direct",
                        help="direct: Anthropic's API (PCF_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY); openrouter: "
                        "OpenRouter pinned to Anthropic (OPENROUTER_API_KEY), as in the runs published up to 2026-09-26")
    parser.add_argument("--model", help="default: gpt-5.6 (openai) or claude-sonnet-5 (anthropic)")
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=sorted(SCENARIOS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--history-mode", choices=("scripted", "model-text"), default="scripted",
                        help="model-text feeds each arm's actual visible replies into its subsequent requests")
    parser.add_argument("--capture-requests", action="store_true", help="save outbound requests for audit replay")
    parser.add_argument("--thinking", choices=("default", "disabled"), default="default")
    parser.add_argument("--effort", default="low", help="openai reasoning effort")
    parser.add_argument("--read-multiplier", type=float, default=0.1)
    parser.add_argument("--output-multiplier", type=float, default=5.0)
    parser.add_argument("--violation-tokens", type=int, default=80)
    parser.add_argument("--workers", type=int, default=1, help="sessions run concurrently (paid runs only)")
    parser.add_argument("--analyze", nargs="+", metavar="FILE", help="summarize saved runs and exit")
    parser.add_argument("--regrade", metavar="FILE", help="re-grade a saved run offline and print it")
    cfg = parser.parse_args()
    if cfg.regrade:
        print(json.dumps(regrade(cfg.regrade), indent=2))
        raise SystemExit
    if cfg.analyze:
        print(json.dumps(analyze(cfg.analyze), indent=2))
        raise SystemExit
    cfg.model = cfg.model or ("gpt-5.6" if cfg.provider == "openai" else "claude-sonnet-5")
    if cfg.history_mode == "model-text" and not cfg.run:
        parser.error("--history-mode model-text requires --run; no responses exist in an offline dry run")
    client = None
    if cfg.run:
        client = placement.make_client(cfg.provider, cfg.anthropic_route)
    writes = placement.make_compiler(cfg.provider, cfg.model).descriptor.cache_write_multiplier
    repeats = cfg.repeats if client else 1
    nonces = {(k, i): uuid.uuid4().hex[:12] if client else "offline" for k in cfg.scenarios for i in range(repeats)}
    run_git_sha = placement.git_sha()
    jobs = [(k, i, arm) for k in cfg.scenarios for i in range(repeats) for arm in cfg.arms]
    def attempt(job):
        try:
            return session(job[0], job[2], nonces[(job[0], job[1])], cfg, client)
        except Exception as exc:  # e.g. quota exhausted: keep the sessions that finished
            return {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}

    with ThreadPoolExecutor(max(1, cfg.workers if client else 1)) as pool:
        done = dict(zip(jobs, pool.map(attempt, jobs)))
    failed = {j: r["error"] for j, r in done.items() if isinstance(r, dict)}
    complete = [k for k in cfg.scenarios if not any(j[0] == k for j in failed)]
    scenarios, aggregate = {}, {}
    for k in complete:
        runs = []
        for i in range(repeats):
            run = {arm: done[(k, i, arm)] for arm in cfg.arms}
            run["summary"] = {arm: placement.summarize(run[arm], cfg, writes) for arm in cfg.arms}
            runs.append(run)
        scenarios[k] = runs
        aggregate[k] = placement.aggregate(runs, cfg.arms)
    meta = {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_sha": run_git_sha, "provider": cfg.provider, "model": cfg.model,
            "anthropic_route": cfg.anthropic_route if cfg.provider == "anthropic" else None, "turns": cfg.turns,
            "history_mode": cfg.history_mode, "capture_requests": cfg.capture_requests,
            "repeats": repeats, "arms": cfg.arms, "scenarios": cfg.scenarios, "thinking": cfg.thinking,
            "effort": cfg.effort, "write_multiplier": writes, "read_multiplier": cfg.read_multiplier,
            "output_multiplier": cfg.output_multiplier, "violation_tokens": cfg.violation_tokens,
            "paid": client is not None, "data": "synthetic; see scripts/domain_scenarios.py",
            "failed_sessions": [{"scenario": j[0], "repeat": j[1], "arm": j[2], "error": e} for j, e in failed.items()],
            "scenarios_complete": complete}
    print(json.dumps({"meta": meta, "scenarios": scenarios, "aggregate": aggregate}, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} sessions failed; scenarios with a failed session are left out")

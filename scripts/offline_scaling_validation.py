#!/usr/bin/env python3
"""Deterministic cache-model experiments, NOT live billing or answer-quality measurements.

Runs native provider compilers through SimEngine on scripted domain histories. Counts are
canonical-native-JSON token estimates. Virtual time models expiry without sleeping. No API calls.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pcf import Context, Segment, PrefixCache  # noqa: E402
from pcf.families.openai_adapter import OpenAICompiler  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.families.sim import SimEngine  # noqa: E402
from pcf.placement import MemoryPlacer  # noqa: E402
from domain_scenarios import SCENARIOS  # noqa: E402
from live_domain_sessions import memory, arrange  # noqa: E402

PROFILES = ('openai-30m', 'anthropic-5m', 'anthropic-1h')


def compiler_for(profile):
    return (OpenAICompiler('gpt-5.6') if profile == 'openai-30m' else
            AnthropicCompiler('claude-sonnet-5', ttl='1h' if profile == 'anthropic-1h' else '5m'))


def run_session(job):
    profile, scenario_key, arm, turns, schedule = job
    compiler = compiler_for(profile)
    ttl, write = compiler.descriptor.ttl_seconds, compiler.descriptor.cache_write_multiplier
    engine = SimEngine(compiler, PrefixCache(ttl))
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=write, read_multiplier=.1)
    scenario = SCENARIOS[scenario_key]
    system = Segment('s', 'system', scenario.system())
    history, rows, now = [], [], 0
    for turn in range(turns):
        gap = 0 if turn == 0 else 30
        if schedule.startswith('expired-every-') and turn and turn % int(schedule.rsplit('-', 1)[1]) == 0:
            gap = ttl + 1
        elif schedule == '600s-every-4' and turn and turn % 4 == 0:
            gap = 600
        now += gap
        mem = memory(scenario, turn)
        front, tail = (placer.split(mem, history, cold=gap >= ttl) if arm == 'placed' else
                       arrange(arm, placer, mem, history, scenario.volatile()))
        ask, expected = scenario.question(turn)
        ctx = Context([system, *front, *history, *tail, Segment('u', 'user', ask.text, stable=False)],
                      cache_namespace='isolated-session')
        usage, compiled = engine.run(ctx, now)
        cost = usage.cache_read_input_tokens * .1 + usage.cache_creation_input_tokens * write + usage.input_tokens
        rows.append({'turn': turn, 'gap_seconds': gap, 'expired_gap': gap >= ttl,
                     'tail': [s.id for s in tail], 'estimated_tokens': compiled.total_tokens,
                     'usage': asdict(usage), 'input_cost_units': round(cost, 4)})
        history.append(Segment(f'h{turn}', 'history', [{'role': 'user', 'content': ask.text},
                          {'role': 'assistant', 'content': scenario.reply(turn, expected)}]))
    return {'profile': profile, 'scenario': scenario_key, 'arm': arm, 'turns': turns,
            'schedule': schedule, 'ttl_seconds': ttl, 'write_multiplier': write, 'rows': rows}


def fleet(profile, mode, sessions=20):
    compiler = compiler_for(profile)
    ttl, write = compiler.descriptor.ttl_seconds, compiler.descriptor.cache_write_multiplier
    engine = SimEngine(compiler, PrefixCache(ttl))
    reference = memory(SCENARIOS['claims'], 0)[0]
    rows = []
    for i in range(sessions):
        system = 'Use the authorized synthetic record. Answer the question.'
        if mode == 'unique-system':
            system += f' Session {i}.'
        namespace = 'authorized-shared-workflow' if mode != 'isolated-namespace' else f'session-{i}'
        ctx = Context([Segment('s', 'system', system), reference,
                       Segment('record', 'memory', {'synthetic_user': i, 'balance': i * 7}, False),
                       Segment('u', 'user', 'What is the balance?', False)], cache_namespace=namespace)
        usage, _ = engine.run(ctx, i * 2)
        rows.append({'session': i, 'usage': asdict(usage), 'input_cost_units': round(
            usage.cache_read_input_tokens * .1 + usage.cache_creation_input_tokens * write + usage.input_tokens, 4)})
    return {'profile': profile, 'mode': mode, 'sessions': sessions, 'rows': rows}


def summarize(runs):
    grouped = {}
    for run in runs:
        checkpoints = (24, 60, 120) if run['schedule'] == 'warm' else (60,)
        for n in checkpoints:
            key = (run['profile'], run['schedule'], n)
            costs = grouped.setdefault(key, {}).setdefault(run['arm'], [])
            costs.extend(r['input_cost_units'] for r in run['rows'][:n])
    summary = []
    for (p, s, n), arms in sorted(grouped.items()):
        # Correctly rounded sums, so the summary is identical on every supported Python version.
        t = {arm: math.fsum(costs) for arm, costs in arms.items()}
        summary.append({'profile': p, 'schedule': s, 'turns': n, 'summed_scenario_costs': t,
                        'placed_saving_fraction': 1 - t['placed'] / t['front-tuned']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    jobs = []
    for p in PROFILES:
        schedules = ['warm', '600s-every-4']
        if p != 'anthropic-1h':
            schedules += [f'expired-every-{k}' for k in (10, 4, 2, 1)]
        for schedule in schedules:
            for scenario in sorted(SCENARIOS):
                for arm in ('front-tuned', 'placed'):
                    jobs.append((p, scenario, arm, 120 if schedule == 'warm' else 60, schedule))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = []
        for i, result in enumerate(pool.map(run_session, jobs), 1):
            runs.append(result)
            if i % 12 == 0:
                print(f'completed {i}/{len(jobs)} sessions', file=sys.stderr, flush=True)
    fleets = [fleet(p, m) for p in PROFILES for m in ('unique-system', 'isolated-namespace', 'shared-prefix')]
    result = {'meta': {'mode': 'offline-simulation', 'paid_calls': 0,
                       'source_git_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'units': 'canonical native JSON heuristic token units; read 0.1x; profile-specific writes',
                       'scope': 'Scripted short histories, virtual time, deterministic local prefix store. No answer '
                                'generation, live cache behavior, actual tokenization, real latency, or dollars verified.',
                       'expiry': '30-second normal intervals; declared long-gap schedules; placer receives cold=True '
                                 'when gap >= profile TTL; expiry is modeled, not inferred from a miss-rate multiplier.',
                       'fleet': 'Sequential synthetic first requests in one ideal cache store; shared content is '
                                'reference-only. No cross-workspace sharing or live provider hit guarantee.'},
              'summary': summarize(runs), 'runs': runs, 'fleet': fleets}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(result, indent=2) + '\n').encode()
    args.output.write_bytes(gzip.compress(payload, mtime=0) if args.output.suffix == '.gz' else payload)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()

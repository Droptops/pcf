#!/usr/bin/env python3
"""Held-out calibration of Jev (via OpenRouter) for one candidate model. Offline by default; paid calls need --run.

Contexts come from the placement harness (scripts/live_memory_placement.py): template and varied history, arms
front / tail / placed, --repeats sessions each; or, with --contexts question-bank, from the paired question bank
(scripts/live_question_bank.py), whose conflict items form the tail slice. The candidate model answers every context once; its answer is
graded by value, which gives the label. Jev scores every context once; scores are memoized by exact request body,
so validating at several thresholds re-uses them and the validated source keeps the fingerprint a live router
uses. The tail slice is the stale-history traps whose answer sits in front memory, the hardest cases in these
runs. A retest re-scores --retest contexts --retest-repeats times to measure score noise and decision flips.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf.cache import PrefixCache  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.router import (Candidate, ConfidenceUnavailable, JevConfidenceSource, ValidationSample,  # noqa: E402
                        openrouter_transport)
from pcf.segments import Context, Segment, canonical_bytes  # noqa: E402

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


placement = _load("placement", "live_memory_placement.py")
bank = _load("question_bank", "live_question_bank.py")

JEV_MODEL = "typesafe/jev-1.13-20260917"


class Settings:  # the attributes placement.session() reads
    provider, read_multiplier, history_style, turns = "anthropic", 0.1, "template", 20


def contexts(repeats: int, turns: int):
    """Yield (meta, Context) for every turn of every scripted session, without calling any model."""
    for style in ("template", "varied"):
        for repeat in range(repeats):
            for arm in ("front", "tail", "placed"):
                nonce = f"cal{repeat}-{style}-{uuid.uuid4().hex[:6]}"
                compiler = AnthropicCompiler("claude-haiku-4-5")
                placer = placement.MemoryPlacer(compiler.tokenizer, write_multiplier=1.25, read_multiplier=0.1)
                system = Segment("s", "system", f"Session {nonce}-{arm}.\n" + "\n".join(placement.POLICIES))
                history, said = [], {}
                for turn in range(turns):
                    mem = placement.memory(turn)
                    front, tail = placement.arrange(arm, placer, mem, history)
                    text, expected = placement.question(turn)
                    ctx = Context([system, *front, *history, *tail, Segment("u", "user", text, stable=False)])
                    stale = said.get(text)
                    trap = stale is not None and stale != expected
                    module = {0: "account", 1: "notes", 2: "preferences", 3: "profile"}[turn % 4]
                    far = module in {s.id for s in front}
                    yield ({"style": style, "repeat": repeat, "arm": arm, "turn": turn, "expected": expected,
                            "stale": stale, "trap": trap, "tail": trap and far}, ctx)
                    said[text] = expected
                    history.append(placement.history_turn(turn, text, expected, style))


def bank_contexts(per_type: int, compiler):
    """Question-bank items under every arm; the conflict items are the tail slice."""
    for i, (kind, final, style) in enumerate(bank.item_specs(per_type)):
        for arm in bank.ARMS:
            ctx, expected = bank.build(kind, final, style, arm, compiler, f"cal-qb{i}")
            yield ({"source": "question-bank", "type": kind, "final_turn": final, "style": style, "arm": arm,
                    "turn": final, "expected": expected, "tail": kind == "conflict"}, ctx)


def memoized(transport):
    cache = {}

    def send(body):
        key = hashlib.sha256(canonical_bytes(body)).hexdigest()
        if key not in cache:
            cache[key] = transport(body)
        return cache[key]
    return send


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--candidate", default="claude-haiku-4-5")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--turns", type=int, default=20)
    parser.add_argument("--retest", type=int, default=60)
    parser.add_argument("--retest-repeats", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--contexts", choices=("placement", "question-bank"), default="placement")
    parser.add_argument("--per-type", type=int, default=50, help="question-bank items per type")
    args = parser.parse_args()
    if args.contexts == "placement":
        rows = list(contexts(args.repeats, args.turns))
    else:
        rows = list(bank_contexts(args.per_type, AnthropicCompiler(args.candidate)))
    summary = {"contexts": len(rows), "tail": sum(m["tail"] for m, _ in rows)}
    if not args.run:
        print(json.dumps({"offline": True, **summary}, indent=1))
        raise SystemExit
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("--run requires OPENROUTER_API_KEY")
    import anthropic
    client = anthropic.Anthropic(api_key=key, base_url="https://openrouter.ai/api")
    compiler = AnthropicCompiler(args.candidate, max_tokens=1024)
    candidate = Candidate(compiler, PrefixCache(compiler.descriptor.ttl_seconds), 1.0, 0.1, is_fallback=True)

    def answer(item):
        meta, ctx = item
        request = compiler.compile(ctx).request
        response = client.messages.create(**{**request, "model": placement.openrouter_model(request["model"])},
                                          extra_body={"provider": {"order": ["Anthropic"], "allow_fallbacks": False}})
        text = "".join(getattr(b, "text", "") for b in response.content).strip()
        if meta.get("source") == "question-bank":
            return {**meta, "answer": text, "label": int(bank.grade(meta["type"], meta["turn"], text, meta["expected"]))}
        value = placement.answer_value(text, meta["turn"])
        return {**meta, "answer": text, "value": value, "label": int(value == meta["expected"].lower())}

    source = JevConfidenceSource(memoized(openrouter_transport(key)), model=JEV_MODEL)

    def score(item):
        try:
            return source.p_sufficient(item[1], candidate)
        except ConfidenceUnavailable:  # e.g. Jev's max_tokens_exceeded above ~32.8k of its input tokens
            return None

    with ThreadPoolExecutor(args.workers) as pool:
        graded = list(pool.map(answer, rows))
        scores = list(pool.map(score, rows))
    for row, s in zip(graded, scores):
        row["jev"] = s
    scored = [(row, ctx) for row, (_, ctx) in zip(graded, rows) if row["jev"] is not None]
    samples = [ValidationSample(ctx, candidate, row["label"], row["tail"]) for row, ctx in scored]
    records = {}
    for threshold in (0.7, 0.8, 0.9):
        record = source.validate(samples, dataset_id=f"{args.contexts}-{args.candidate}", threshold=threshold)
        records[str(threshold)] = record.to_json()
    # Retest: fresh (unmemoized) scores for a spread of contexts.
    fresh = JevConfidenceSource(openrouter_transport(key), model=JEV_MODEL)
    scorable = [item for item, row in zip(rows, graded) if row["jev"] is not None]
    picks = scorable[:: max(1, len(scorable) // args.retest)][: args.retest]
    with ThreadPoolExecutor(args.workers) as pool:
        retest = [list(pool.map(lambda item: fresh.p_sufficient(item[1], candidate), [p] * args.retest_repeats))
                  for p in picks]
    sds = [statistics.pstdev(r) for r in retest]
    flips = sum(min(r) < 0.8 <= max(r) for r in retest)
    labels = [r["label"] for r, _ in scored]
    scores = [r["jev"] for r, _ in scored]
    summary["unscorable"] = len(graded) - len(scored)
    result = {
        "meta": {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "git_sha": placement.git_sha(), "candidate": args.candidate, "jev_model": JEV_MODEL,
                 "contexts": args.contexts, "repeats": args.repeats, "turns": args.turns,
                 "route": "OpenRouter, candidate pinned to Anthropic"},
        "summary": {**summary, "label_rate": round(sum(labels) / len(labels), 3),
                    "tail_label_rate": round(sum(r["label"] for r in graded if r["tail"]) /
                                             max(1, sum(r["tail"] for r in graded)), 3),
                    "auc": auc(scores, labels),
                    "score_mean": round(statistics.mean(scores), 3), "score_min": min(scores),
                    "score_max": max(scores),
                    "retest": {"contexts": len(retest), "repeats": args.retest_repeats,
                               "sd_mean": round(statistics.mean(sds), 4), "sd_max": round(max(sds), 4),
                               "decisions_flipping_at_0.8": flips}},
        "validation": records,
        "rows": graded,
        "retest_scores": retest,
    }
    print(json.dumps(result, indent=1))

#!/usr/bin/env python3
"""Held-out calibration of Jev (via OpenRouter) for one candidate model. Offline by default; paid calls need --run.

Contexts come from the placement harness (scripts/live_memory_placement.py): template and varied history, arms
front / tail / placed, --repeats sessions each; or, with --contexts question-bank, from the paired question bank
(scripts/live_question_bank.py), whose conflict items form the tail slice. The candidate model answers every context once; its answer is
graded by value and the required answer format, which together give the label. Jev scores every context once; scores are memoized by exact request body,
so validating at several thresholds re-uses them and the validated source keeps the fingerprint a live router
uses. The tail slice is the stale-history traps whose answer sits in front memory, the hardest cases in these
runs. A retest re-scores --retest contexts --retest-repeats times to measure score noise and decision flips.

Jev is asked whether an answer is acceptable: correct, following the application instructions and satisfying the
request. Each row therefore carries three labels: value_correct (the graded value), instruction_compliant (full-answer matching
against the question-specific format, with normalization documented in LABELING_SPEC) and label
(both). --relabel FILE recomputes them for a saved question-bank run and re-validates from its recorded scores,
offline.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import json
import re
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pcf.cache import PrefixCache  # noqa: E402
from pcf.families.anthropic_adapter import AnthropicCompiler  # noqa: E402
from pcf.router import (Candidate, ConfidenceUnavailable, JevConfidenceSource, ValidationSample,  # noqa: E402
                        openrouter_transport)
from pcf.router.jev_adapter import build_request  # noqa: E402
from pcf.segments import Context, Segment, canonical_bytes  # noqa: E402

def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


placement = _load("placement", "live_memory_placement.py")
bank = _load("question_bank", "live_question_bank.py")

JEV_MODEL = "typesafe/jev-1.13-20260917"
LABELING = "acceptance-v3"
LABELING_SPEC = ("Case-insensitive full-answer matching after trimming outer whitespace, an optional terminal "
                 "period and one enclosing bold pair. Conflict/channel/language: one ASCII word; ticket count: "
                 "digits; plan: a known plan name; cross-module: digits, comma, known plan. No extra prose.")


def instruction_compliant(answer: str, *, kind: str, turn: int) -> bool:
    """Enforce the question's answer shape separately from whether its value is correct."""
    text = answer.strip()
    if text.endswith("."):
        text = text[:-1]
    if text.startswith("**") and text.endswith("**"):
        text = text[2:-2]
    plans = "(?:" + "|".join(re.escape(p) for p in placement.PLANS) + ")"
    if kind == "cross":
        pattern = rf"[0-9]+, *{plans}"
    elif kind == "conflict":
        pattern = r"[A-Za-z]+"
    elif kind in {"far", "placement"}:
        pattern = {0: r"[0-9]+", 1: plans, 2: r"[A-Za-z]+", 3: r"[A-Za-z]+"}[turn % 4]
    else:
        raise ValueError(f"unsupported question type: {kind!r}")
    return re.fullmatch(pattern, text, flags=re.IGNORECASE) is not None


def labels(value_correct: bool, answer: str, *, kind: str, turn: int) -> dict:
    compliant = instruction_compliant(answer, kind=kind, turn=turn)
    return {"value_correct": int(value_correct), "instruction_compliant": int(compliant),
            "label": int(value_correct and compliant), "labeling": LABELING}


class Settings:  # the attributes placement.session() reads
    provider, read_multiplier, history_style, turns = "anthropic", 0.1, "template", 20


def contexts(repeats: int, turns: int):
    """Yield (meta, Context) for every turn of every scripted session, without calling any model."""
    for style in ("template", "varied"):
        for repeat in range(repeats):
            for arm in ("front", "tail", "placed"):
                nonce = "calibration"  # no per-session tag: repeated scripted contexts must stay identical
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
            ctx, expected = bank.build(kind, final, style, arm, compiler, "calibration")
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


def relabel(path: str) -> dict:
    """Relabel a saved question-bank run and re-validate from its recorded Jev scores, without network calls.

    Contexts are rebuilt from each row's item fields and checked against the recorded token estimate; scores are
    replayed through JevConfidenceSource so validation identity matches a live source."""
    saved = json.load(open(path))
    meta = saved["meta"]
    if meta.get("contexts") != "question-bank":
        raise SystemExit("--relabel supports question-bank runs")
    compiler = AnthropicCompiler(meta["candidate"], max_tokens=1024)
    candidate = Candidate(compiler, PrefixCache(compiler.descriptor.ttl_seconds), 1.0, 0.1, is_fallback=True)
    rows, replay, mismatched = [], {}, 0
    for row in saved["rows"]:
        ctx, expected = bank.build(row["type"], row["final_turn"], row["style"], row["arm"], compiler, "calibration")
        mismatched += compiler.compile(ctx).total_tokens != row["est_tokens"]
        row = {**row, **labels(bank.grade(row["type"], row["final_turn"], row["answer"], expected),
                                row["answer"], kind=row["type"], turn=row["final_turn"])}
        rows.append((row, ctx))
        if row["jev"] is not None:
            body = build_request(ctx, candidate.model_id, model=meta["jev_model"])
            replay[hashlib.sha256(canonical_bytes(body)).hexdigest()] = row["jev"]
    if mismatched:
        raise SystemExit(f"{mismatched} rebuilt contexts differ from the recorded run; cannot replay its scores")

    def transport(body):
        return {"model": meta["jev_model"], "answers": {"sufficient": {"type": "noul",
                                           "noul": replay[hashlib.sha256(canonical_bytes(body)).hexdigest()]}}}

    source = JevConfidenceSource(transport, model=meta["jev_model"])
    scored = [(row, ctx) for row, ctx in rows if row["jev"] is not None]
    samples = [ValidationSample(ctx, candidate, row["label"], row["tail"]) for row, ctx in scored]
    records = {str(t): source.validate(samples, dataset_id=f"question-bank-{meta['candidate']}-{LABELING}",
                                       threshold=t).to_json() for t in (0.7, 0.8, 0.9)}
    scores, all_rows = [r["jev"] for r, _ in scored], [r for r, _ in rows]

    def rate(key, subset):
        return round(sum(r[key] for r in subset) / len(subset), 3) if subset else None

    summary = {"labeling": LABELING, "contexts": len(all_rows), "scorable": len(scored),
               "value_correct_rate": rate("value_correct", all_rows),
               "instruction_compliant_rate": rate("instruction_compliant", all_rows),
               "label_rate": rate("label", all_rows), "scorable_label_rate": rate("label", [r for r, _ in scored]),
               "tail_label_rate": rate("label", [r for r in all_rows if r["tail"]]),
               "auc_value_correct": auc(scores, [r["value_correct"] for r, _ in scored]),
               "auc_label": auc(scores, [r["label"] for r, _ in scored]),
               "score_min": min(scores), "score_max": max(scores),
               "ece": {t: record["ece"] for t, record in records.items()},
               "passed": {t: record["passed"] for t, record in records.items()}}
    return {"meta": {**meta, "relabeled_from": path, "labeling": LABELING, "labeling_spec": LABELING_SPEC,
                     "relabel_script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
                     "relabeled": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                     "relabel_git_sha": placement.git_sha()},
            "summary": summary, "validation": records, "rows": all_rows}


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
    parser.add_argument("--relabel", metavar="FILE", help="relabel a saved question-bank run offline and print it")
    args = parser.parse_args()
    if args.relabel:
        print(json.dumps(relabel(args.relabel), indent=1))
        raise SystemExit
    if args.contexts == "placement":
        rows = list(contexts(args.repeats, args.turns))
    else:
        rows = list(bank_contexts(args.per_type, AnthropicCompiler(args.candidate)))
    built, seen, rows = len(rows), set(), [r for r in rows]
    rows = [r for r in rows if not (r[1].prefix_chain()[-1] in seen or seen.add(r[1].prefix_chain()[-1]))]
    summary = {"contexts": len(rows), "duplicates_removed": built - len(rows), "tail": sum(m["tail"] for m, _ in rows)}
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
            return {**meta, "answer": text,
                    **labels(bank.grade(meta["type"], meta["turn"], text, meta["expected"]), text,
                             kind=meta["type"], turn=meta["turn"])}
        value = placement.answer_value(text, meta["turn"])
        return {**meta, "answer": text, "value": value, **labels(value == meta["expected"].lower(), text,
                                                                              kind="placement", turn=meta["turn"])}

    source = JevConfidenceSource(memoized(openrouter_transport(key)), model=JEV_MODEL)

    def score(item):
        try:
            return source.p_sufficient(item[1], candidate), None
        except ConfidenceUnavailable as exc:  # e.g. Jev's max_tokens_exceeded above ~32.8k of its input tokens
            cause = exc.__cause__
            detail = cause.read().decode("utf-8", "replace")[:300] if hasattr(cause, "read") else repr(cause)
            return None, f"{exc}: {detail}"

    with ThreadPoolExecutor(args.workers) as pool:
        graded = list(pool.map(answer, rows))
        scores = list(pool.map(score, rows))
    for row, (s, error), (_, ctx) in zip(graded, scores, rows):
        row["jev"] = s
        row["est_tokens"] = compiler.compile(ctx).total_tokens
        if error:
            row["jev_error"] = error
    scored = [(row, ctx) for row, (_, ctx) in zip(graded, rows) if row["jev"] is not None]
    samples = [ValidationSample(ctx, candidate, row["label"], row["tail"]) for row, ctx in scored]
    records = {}
    for threshold in (0.7, 0.8, 0.9):
        record = source.validate(samples, dataset_id=f"{args.contexts}-{args.candidate}-{LABELING}", threshold=threshold)
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
    summary["scorable_label_rate"] = round(sum(labels) / len(labels), 3)
    summary["unscorable_label_rate"] = (round(sum(r["label"] for r in graded if r["jev"] is None) /
                                         summary["unscorable"], 3) if summary["unscorable"] else None)
    result = {
        "meta": {"date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                 "git_sha": placement.git_sha(), "labeling": LABELING, "labeling_spec": LABELING_SPEC,
                 "candidate": args.candidate, "jev_model": JEV_MODEL,
                 "contexts": args.contexts, "repeats": args.repeats, "turns": args.turns,
                 "route": "OpenRouter, candidate pinned to Anthropic"},
        "summary": {**summary, "label_rate": round(sum(r["label"] for r in graded) / len(graded), 3),
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

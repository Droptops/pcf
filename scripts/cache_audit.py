#!/usr/bin/env python3
"""Find what broke the prompt cache, from request logs a team already has. Prototype; offline.

Input is JSON Lines, one provider request per line, exactly as sent, with the usage the provider returned:
  {"conversation": "c-1", "ts": 1727280000.0, "request": {...}, "usage": {"cached": 0, "written": 0, "uncached": 0}}
`ts` (seconds) is optional; without it no gap is treated as expired. Anthropic Messages and OpenAI Responses
requests are read in prefix order (tools, system or instructions, then messages or input); cache markers and
sampling settings are ignored. No PCF types are needed, and nothing in the request path changes.

For each request the audit finds the longest prefix it shares with the conversation's previous request (for a
conversation's first request, with any earlier request in the window) and estimates the tokens that prefix is
worth. It then reports three kinds of loss, in uncached-input-token units at --write/--read:
  changed   - the previous request's prompt after the first changed field was billed again; named by that field
  unread    - a reusable prefix was not read (no cache marker covers it, or it is below the provider minimum)
  expired   - the gap since the previous request exceeded --ttl
Token counts are estimated from the provider's input total in proportion to bytes; they are approximate.

  cache_audit.py LOG.jsonl [--ttl 300] [--json]
  cache_audit.py --from-domain RESULT.json [--arm front-tuned] > LOG.jsonl   rebuild a log from a saved run
  cache_audit.py --from-fleet RESULT.json [--arm placed-shared] > LOG.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

PREFIX_KEYS = ("tools", "system", "instructions", "messages", "input")
IGNORED = {"cache_control", "prompt_cache_breakpoint"}


def leaves(value, path=""):
    """(path, text) for every scalar in document order; a string holding JSON is opened and walked too."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in IGNORED:
                yield from leaves(item, f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaves(item, f"{path}[{i}]")
    elif isinstance(value, str) and value[:1] in "{[":
        try:
            inner = json.loads(value)
        except ValueError:
            yield path, value
        else:
            yield from leaves(inner, f"{path}#")
    else:
        yield path, value if isinstance(value, str) else json.dumps(value)


def flatten(request: dict) -> list[tuple[str, str]]:
    return [leaf for key in PREFIX_KEYS if key in request for leaf in leaves(request[key], key)]


def size(text: str) -> int:
    return len(text.encode())


def common_prefix(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> tuple[int, int]:
    """(bytes shared, index of the first differing leaf)."""
    shared = 0
    for i, ((pa, ta), (pb, tb)) in enumerate(zip(a, b)):
        if (pa, ta) != (pb, tb):
            if pa == pb:
                shared += size(os.path.commonprefix([ta, tb]))
            return shared, i
        shared += size(ta)
    return shared, min(len(a), len(b))


def label(request: dict, path: str) -> str:
    """A name a person recognizes for the leaf at `path`."""
    if "#" in path:
        outer, inner = path.split("#", 1)
        node = request
        for part in re.findall(r"[^.\[\]]+|\[\d+\]", outer):
            node = node[int(part[1:-1])] if part.startswith("[") else node[part]
        doc = json.loads(node)
        name = doc.get("source") if isinstance(doc, dict) else None
        field = re.sub(r"\[\d+\]", "[]", inner.lstrip(".")) or "(document)"
        if name:
            return f"memory '{name}' {field.removeprefix('data.')}"
        return f"JSON field {field}"
    head = path.split(".")[0].split("[")[0]
    if head in ("system", "instructions"):
        return "system prompt"
    if head == "tools":
        return "tool definitions"
    match = re.match(r"(messages|input)\[(\d+)\]", path)
    if match:
        message = request[match.group(1)][int(match.group(2))]
        return f"{message.get('role', 'item')} message"
    return path


def audit(rows: list[dict], ttl: float | None = None, write: float = 1.25, read: float = 0.1,
          window: int = 200, min_tokens: int = 128) -> dict:
    flat, last, events = [], {}, []
    totals = {"requests": len(rows), "conversations": len({r["conversation"] for r in rows}),
              "input_tokens": 0, "cached_tokens": 0}
    for i, row in enumerate(rows):
        u = row["usage"]
        tokens = u["cached"] + u["written"] + u["uncached"]
        totals["input_tokens"] += tokens
        totals["cached_tokens"] += u["cached"]
        cur = flatten(row["request"])
        flat.append(cur)
        prev_i = last.get(row["conversation"])
        if prev_i is None:  # first request: the best earlier request in the window, in any conversation
            best = max(range(max(0, i - window), i), key=lambda j: common_prefix(flat[j], cur)[0], default=None)
            prev_i = best
        last[row["conversation"]] = i
        if prev_i is None:
            continue
        prev = rows[prev_i]
        shared, at = common_prefix(flat[prev_i], cur)
        cur_bytes = sum(size(t) for _, t in cur) or 1
        reusable = tokens * shared / cur_bytes
        gap = row["ts"] - prev["ts"] if row.get("ts") is not None and prev.get("ts") is not None else None
        pu = prev["usage"]
        prev_tokens = pu["cached"] + pu["written"] + pu["uncached"]
        prev_bytes = sum(size(t) for _, t in flat[prev_i]) or 1
        busted = prev_tokens * (1 - shared / prev_bytes)
        base = {"index": i, "conversation": row["conversation"], "reusable_tokens": round(reusable),
                "cached": u["cached"], "same_conversation": prev["conversation"] == row["conversation"]}
        if ttl is not None and gap is not None and gap > ttl:
            events.append({**base, "kind": "expired", "gap_s": round(gap, 1),
                           "lost_units": round(max(0.0, reusable - u["cached"]) * (write - read), 1)})
            continue
        if reusable - u["cached"] > max(min_tokens, .1 * reusable):
            events.append({**base, "kind": "unread",
                           "lost_units": round((reusable - u["cached"]) * (write - read), 1)})
        if base["same_conversation"] and busted > min_tokens and at < len(flat[prev_i]):
            events.append({**base, "kind": "changed", "field": label(prev["request"], flat[prev_i][at][0]),
                           "rebilled_tokens": round(busted), "lost_units": round(busted * (write - read), 1)})
    by_cause: dict[str, dict] = {}
    for e in events:
        key = e["field"] if e["kind"] == "changed" else e["kind"]
        cause = by_cause.setdefault(key, {"kind": e["kind"], "requests": 0, "lost_units": 0.0})
        cause["requests"] += 1
        cause["lost_units"] += e["lost_units"]
    lost = math.fsum(e["lost_units"] for e in events)
    billed = totals["input_tokens"] - (1 - read) * totals["cached_tokens"]  # approximate: writes at 1x
    return {"totals": {**totals, "cached_share": round(totals["cached_tokens"] / max(1, totals["input_tokens"]), 3),
                       "estimated_lost_units": round(lost), "billed_input_units_approx": round(billed)},
            "causes": sorted(({"cause": k, **v, "lost_units": round(v["lost_units"])} for k, v in by_cause.items()),
                             key=lambda c: -c["lost_units"]),
            "events": events}


def report(result: dict) -> str:
    t = result["totals"]
    lines = [f"{t['requests']} requests in {t['conversations']} conversations; "
             f"{t['cached_share']:.0%} of input tokens read from cache.",
             f"Cost attributed to cache misses: {t['estimated_lost_units']:,} units, against billed input of about "
             f"{t['billed_input_units_approx']:,}. It bounds what a better layout could save; data that really "
             "changed has to be billed again anyway.", "", "Top causes:"]
    words = {"changed": "changed, re-billing what followed it", "unread": "reusable prefix not read from cache",
             "expired": "cache expired during an idle gap"}
    for c in result["causes"][:10]:
        what = f"{c['cause']} {words['changed']}" if c["kind"] == "changed" else words[c["kind"]]
        lines.append(f"  {c['lost_units']:>10,} units  {c['requests']:>5} requests  {what}")
    return "\n".join(lines)


def _harness():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:0] = [os.path.join(here, "..", "src"), here]


def from_domain(path: str, arm: str) -> list[dict]:
    """Rebuild a saved domain run's requests (same content up to the session nonce) with its recorded usage."""
    _harness()
    from types import SimpleNamespace
    import live_domain_sessions as domain
    saved = json.load(open(path))
    meta, rows = saved["meta"], []
    cfg = SimpleNamespace(provider=meta["provider"], model=meta["model"], turns=meta["turns"],
                          read_multiplier=meta["read_multiplier"], violation_tokens=meta["violation_tokens"])
    for key, runs in saved["scenarios"].items():
        for r, run in enumerate(runs):
            sink: list = []
            domain.session(key, arm, f"audit{r}", cfg, sink=sink)
            for turn, (request, used) in enumerate(zip(sink, run[arm])):
                rows.append({"conversation": f"{key}-{r}", "ts": r * 1e5 + turn * 5.0, "request": request,
                             "usage": {k: used[k] for k in ("cached", "written", "uncached")}})
    return rows


def from_fleet(path: str, arm: str) -> list[dict]:
    """Rebuild a saved fleet run's requests, in order, with recorded usage and each turn's recorded gap."""
    _harness()
    from types import SimpleNamespace
    from pcf.cache import PrefixCache
    from pcf.families.sim import SimEngine
    import live_fleet_sessions as fleet
    saved = json.load(open(path))
    meta, rows, clock = saved["meta"], [], 0.0
    for key, arms in saved["cells"].items():
        cell = arms[arm]
        sessions = cell["warmup"] + cell["measured"]
        cfg = SimpleNamespace(provider=meta["provider"], model=meta["model"], turns=len(sessions[0]),
                              read_multiplier=meta["read_multiplier"], violation_tokens=meta["violation_tokens"])
        compiler = fleet.placement.make_compiler(cfg.provider, cfg.model)
        tag = f"{meta['run_id']}-{arm}"
        for s, recorded in enumerate(sessions):
            sink: list = []
            engine = SimEngine(compiler, PrefixCache(compiler.descriptor.ttl_seconds))  # requests only
            fleet.session(key, arm, f"{tag}-{s}" if arm in ("tuned-private", "placed-cold") else tag, cfg,
                          engine=engine, clock=[0.0], sink=sink)
            for request, used in zip(sink, recorded):
                clock += used["gap_s"] or 1.0
                rows.append({"conversation": f"{key}-{arm}-{s}", "ts": clock, "request": request,
                             "usage": {k: used[k] for k in ("cached", "written", "uncached")}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?")
    parser.add_argument("--ttl", type=float, help="cache lifetime in seconds (for example 300 for 5 minutes)")
    parser.add_argument("--write", type=float, default=1.25)
    parser.add_argument("--read", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="print the full result, events included")
    parser.add_argument("--from-domain", metavar="RESULT")
    parser.add_argument("--from-fleet", metavar="RESULT")
    parser.add_argument("--arm")
    args = parser.parse_args()
    if args.from_domain or args.from_fleet:
        rows = (from_domain(args.from_domain, args.arm or "front-tuned") if args.from_domain
                else from_fleet(args.from_fleet, args.arm or "placed-shared"))
        for row in rows:
            print(json.dumps(row))
        return
    if not args.log:
        parser.error("give a LOG, --from-domain or --from-fleet")
    with open(args.log) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = audit(rows, args.ttl, args.write, args.read)
    print(json.dumps(result, indent=2) if args.json else report(result))


if __name__ == "__main__":
    main()

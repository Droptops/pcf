#!/usr/bin/env python3
"""Find what broke the prompt cache, from request logs a team already has. Prototype; offline.

Input is JSON Lines, one provider request per line, exactly as sent, with the usage the provider returned:
  {"conversation": "c-1", "ts": 1727280000.0, "request": {...}, "usage": {"cached": 0, "written": 0, "uncached": 0}}
`ts` (seconds) is optional; without it no gap is treated as expired. Anthropic Messages and OpenAI Responses
requests are read in prefix order (tools, system or instructions, then messages or input); cache markers and
sampling settings are ignored. No PCF types are needed, and nothing in the request path changes.

For each request the audit finds its shared prefix with the previous compatible request. Cross-conversation
matching requires an explicit cache_scope. Token positions are byte-proportional estimates. Diagnostics are:
  changed   - a changed field potentially invalidates the previous suffix; all changed leaves are listed
  unread    - estimated reusable input was not read; the provider's underlying cause is unknown
  expired   - the gap reached the configured TTL (a timing hypothesis, not proof of provider eviction)
Usage-derived input cost includes write premiums. Diagnostic opportunity is capped against non-read spending;
it is not measured or recoverable savings. --validate-labels scores independent operator annotations.

  cache_audit.py LOG.jsonl [--ttl 300] [--json]
  cache_audit.py --from-domain RESULT.json [--arm front-tuned] > LOG.jsonl   rebuild a log from a saved run
  cache_audit.py --from-fleet RESULT.json [--arm placed-shared] > LOG.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

PREFIX_KEYS = ("tools", "system", "instructions", "messages", "input")
IGNORED = {"cache_control", "prompt_cache_breakpoint"}


def validate_log(rows: list[dict]) -> dict:
    """Reject unusable measurements before doing attribution; never silently repair logs."""
    seen, clocks = set(), {}
    for i, row in enumerate(rows):
        if not isinstance(row.get("conversation"), str) or not row["conversation"]:
            raise ValueError(f"row {i}: conversation must be a nonempty string")
        if not isinstance(row.get("request"), dict) or not any(k in row["request"] for k in PREFIX_KEYS):
            raise ValueError(f"row {i}: missing provider request")
        for key in ("cached", "written", "uncached"):
            value = row.get("usage", {}).get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"row {i}: usage.{key} must be a nonnegative integer")
        for key in ("request_id", "cache_scope"):
            if row.get(key) is not None and (not isinstance(row[key], str) or not row[key]):
                raise ValueError(f"row {i}: {key} must be a nonempty string")
        if row.get("request_id") is not None:
            identity = (row.get("cache_scope"), row["request_id"])
            if identity in seen:
                raise ValueError(f"row {i}: duplicate request_id")
            seen.add(identity)
        ts = row.get("ts")
        if ts is not None:
            if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
                raise ValueError(f"row {i}: ts must be finite seconds")
            scope = (row.get("cache_scope"), row["conversation"])
            if ts < clocks.get(scope, -math.inf):
                raise ValueError(f"row {i}: timestamps are out of order within a conversation")
            clocks[scope] = ts
    return {"valid_rows": len(rows), "rows_without_timestamps": sum(r.get("ts") is None for r in rows),
            "rows_without_cache_scope": sum(not r.get("cache_scope") for r in rows),
            "rows_without_request_id": sum(not r.get("request_id") for r in rows)}


def cache_identity(row: dict) -> tuple:
    req = row["request"]
    return (row.get("cache_scope"), req.get("model"), req.get("prompt_cache_key"),
            json.dumps(req.get("extra_body", {}).get("provider"), sort_keys=True))


def changed_fields(previous: list, current: list, request: dict) -> list[str]:
    """All differing previous leaves, including removals; these are suspects, not independent costs."""
    now = dict(current)
    return list(dict.fromkeys(label(request, path) for path, value in previous if now.get(path) != value))


def leaves(value, path=()):
    """(key path, text) for every scalar in document order; a string holding JSON is opened and walked too.

    A key path is a tuple of dict keys, list indexes and "#", which marks the step into a JSON string."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in IGNORED:
                yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from leaves(item, (*path, i))
    elif isinstance(value, str) and value[:1] in "{[":
        try:
            inner = list(leaves(json.loads(value), (*path, "#")))
        except ValueError:
            inner = []
        yield from inner or [(path, value)]  # an empty or invalid document is one leaf
    else:
        yield path, value if isinstance(value, str) else json.dumps(value)


def flatten(request: dict) -> list[tuple[tuple, str]]:
    return [leaf for key in PREFIX_KEYS if key in request for leaf in leaves(request[key], (key,))]


def size(text: str) -> int:
    return len(text.encode())


def common_prefix(a: list[tuple[tuple, str]], b: list[tuple[tuple, str]]) -> tuple[int, int]:
    """(bytes shared, index of the first differing leaf)."""
    shared = 0
    for i, ((pa, ta), (pb, tb)) in enumerate(zip(a, b)):
        if (pa, ta) != (pb, tb):
            if pa == pb:
                shared += size(os.path.commonprefix([ta, tb]))
            return shared, i
        shared += size(ta)
    return shared, min(len(a), len(b))


def _field(keys) -> str:
    return ".".join("[]" if isinstance(k, int) else str(k) for k in keys).replace(".[]", "[]")


def label(request: dict, path: tuple) -> str:
    """A name a person recognizes for the leaf at `path`."""
    if "#" in path:
        cut = path.index("#")
        node = request
        for key in path[:cut]:
            node = node[key]
        doc = json.loads(node)
        name = doc.get("source") if isinstance(doc, dict) else None
        field = _field(path[cut + 1:]) or "(document)"
        if name:
            return f"memory '{name}' {field.removeprefix('data.')}"
        return f"JSON field {field}"
    head = path[0]
    if head in ("system", "instructions"):
        return "system prompt"
    if head == "tools":
        return "tool definitions"
    if head in ("messages", "input") and len(path) > 1 and isinstance(path[1], int):
        message = request[head][path[1]]
        return f"{message.get('role', 'item')} message" if isinstance(message, dict) else "item"
    return _field(path)


def audit(rows: list[dict], ttl: float | None = None, write: float = 1.25, read: float = 0.1,
          window: int = 200, min_tokens: int = 128) -> dict:
    quality = validate_log(rows)
    if not all(math.isfinite(x) for x in (write, read)) or not 0 <= read < write:
        raise ValueError("need finite prices with 0 <= read < write")
    if ttl is not None and (not math.isfinite(ttl) or ttl <= 0):
        raise ValueError("ttl must be finite and positive")
    if type(window) is not int or window < 1 or type(min_tokens) is not int or min_tokens < 0:
        raise ValueError("window must be positive and min_tokens nonnegative integers")
    flat, last, events = [], {}, []
    totals = {"requests": len(rows), "conversations": len({r["conversation"] for r in rows}),
              "input_tokens": 0, "cached_tokens": 0, "written_tokens": 0, "uncached_tokens": 0}
    for i, row in enumerate(rows):
        u = row["usage"]
        tokens = u["cached"] + u["written"] + u["uncached"]
        totals["input_tokens"] += tokens
        totals["cached_tokens"] += u["cached"]
        totals["written_tokens"] += u["written"]
        totals["uncached_tokens"] += u["uncached"]
        cur = flatten(row["request"])
        flat.append(cur)
        identity = cache_identity(row)
        key = (row["conversation"], identity)
        prev_i = last.get(key)
        if prev_i is None:  # first request: the best earlier request in the window, in any conversation
            candidates = [j for j in range(max(0, i - window), i)
                          if row.get("cache_scope") and cache_identity(rows[j]) == identity
                          and (row.get("ts") is None or rows[j].get("ts") is None or rows[j]["ts"] <= row["ts"])
                          and (ttl is None or row.get("ts") is None or rows[j].get("ts") is None
                               or row["ts"] - rows[j]["ts"] < ttl)]
            best = max(candidates, key=lambda j: common_prefix(flat[j], cur)[0], default=None)
            prev_i = best
        last[key] = i
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
                "cached": u["cached"], "same_conversation": prev["conversation"] == row["conversation"],
                "evidence": "heuristic", "token_basis": "bytes scaled to provider input usage"}
        # Bound diagnostic estimates by this request's measured non-read spending. This is still not savings.
        budget = max(0.0, u["written"] * (write - read) + u["uncached"] * max(0.0, 1 - read))
        if ttl is not None and gap is not None and gap >= ttl:
            events.append({**base, "kind": "expired", "gap_s": round(gap, 1),
                           "lost_units": round(min(budget, max(0.0, reusable - u["cached"]) * (write - read)), 1)})
            continue
        if reusable - u["cached"] > max(min_tokens, .1 * reusable):
            estimate = min(budget, (reusable - u["cached"]) * (write - read))
            events.append({**base, "kind": "unread",
                           "lost_units": round(estimate, 1)})
            budget -= estimate
        if base["same_conversation"] and busted > min_tokens and at < len(flat[prev_i]):
            events.append({**base, "kind": "changed", "field": label(prev["request"], flat[prev_i][at][0]),
                           "changed_fields": changed_fields(flat[prev_i], cur, prev["request"]),
                           "rebilled_tokens": round(busted),
                           "lost_units": round(min(budget, busted * (write - read)), 1)})
    by_cause: dict[str, dict] = {}
    for e in events:
        key = e["field"] if e["kind"] == "changed" else e["kind"]
        cause = by_cause.setdefault(key, {"kind": e["kind"], "requests": 0, "lost_units": 0.0})
        cause["requests"] += 1
        cause["lost_units"] += e["lost_units"]
    lost = math.fsum(e["lost_units"] for e in events)
    billed = totals["uncached_tokens"] + write * totals["written_tokens"] + read * totals["cached_tokens"]
    return {"totals": {**totals, "cached_share": round(totals["cached_tokens"] / max(1, totals["input_tokens"]), 3),
                       "estimated_miss_opportunity_units": round(lost), "billed_input_units": round(billed, 4)},
            "log_quality": quality,
            "claim_boundary": "Usage and cost at supplied prices are observed. Attribution is heuristic; "
                              "miss opportunity is not recoverable savings or a causal estimate. "
                              "Event lost_units is a legacy name for this diagnostic estimate.",
            "causes": sorted(({"cause": k, **v, "lost_units": round(v["lost_units"])} for k, v in by_cause.items()),
                             key=lambda c: -c["lost_units"]),
            "events": events}


def report(result: dict) -> str:
    t = result["totals"]
    lines = [f"{t['requests']} requests in {t['conversations']} conversations; "
             f"{t['cached_share']:.0%} of input tokens read from cache.",
             f"Input cost from usage at supplied prices: {t['billed_input_units']:,} units.",
             f"Heuristic miss opportunity: {t['estimated_miss_opportunity_units']:,} units; not recoverable savings.",
             "Token attribution is byte-proportional. Marker, eviction and routing causes are not proven.",
             "", "Top suspected causes:"]
    words = {"changed": "is the first changed field", "unread": "estimated reusable prefix was not read",
             "expired": "idle gap reached the configured TTL"}
    for c in result["causes"][:10]:
        what = f"{c['cause']} {words['changed']}" if c["kind"] == "changed" else words[c["kind"]]
        lines.append(f"  {c['lost_units']:>10,} units  {c['requests']:>5} requests  {what}")
    return "\n".join(lines)


def validate_labels(rows: list[dict], result: dict) -> dict:
    """Score diagnostic predictions against independently supplied annotations, never generated labels.

    An annotation contains kinds (including [] for no miss) and, for changed, first_changed_field.
    The operator must establish labels independently; this function cannot authenticate their provenance.
    """
    counts = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ("changed", "unread", "expired")}
    annotated, field_total, field_correct = 0, 0, 0
    by_index: dict[int, list] = {}
    for event in result["events"]:
        by_index.setdefault(event["index"], []).append(event)
    for i, row in enumerate(rows):
        annotation = row.get("annotation")
        if annotation is None:
            continue
        kinds = annotation.get("kinds")
        if not isinstance(kinds, list) or any(k not in counts for k in kinds) or len(kinds) != len(set(kinds)):
            raise ValueError(f"row {i}: annotation.kinds must list unique diagnostic kinds")
        if "changed" in kinds and not isinstance(annotation.get("first_changed_field"), str):
            raise ValueError(f"row {i}: changed annotation needs first_changed_field")
        annotated += 1
        predicted = {e["kind"] for e in by_index.get(i, [])}
        for kind, c in counts.items():
            c["tp"] += kind in kinds and kind in predicted
            c["fp"] += kind not in kinds and kind in predicted
            c["fn"] += kind in kinds and kind not in predicted
        if "changed" in kinds:
            field_total += 1
            field_correct += any(e.get("field") == annotation["first_changed_field"]
                                 for e in by_index.get(i, []) if e["kind"] == "changed")
    if not annotated:
        raise ValueError("no independent annotations supplied; validation cannot be claimed")
    for c in counts.values():
        c["precision"] = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else None
        c["recall"] = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else None
    return {"annotated_rows": annotated, "unannotated_rows": len(rows) - annotated, "classes": counts,
            "first_field": {"correct": field_correct, "total": field_total},
            "claim_boundary": "Agreement with supplied annotations only; neither causal savings nor production "
                              "validation is established by synthetic or reconstructed logs."}


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
            captured = all("request" in used for used in run[arm])
            if meta.get("history_mode", "scripted") != "scripted" and not captured:
                raise ValueError("model-text runs require captured requests; scripted reconstruction is invalid")
            sink: list = []
            if not captured:
                domain.session(key, arm, f"audit{r}", cfg, sink=sink)
            for turn, used in enumerate(run[arm]):
                request = used["request"] if captured else sink[turn]
                rows.append({"conversation": f"{key}-{r}",
                             "ts": used.get("request_ts") if captured else r * 1e5 + turn * 5.0,
                             "request": request, "request_id": f"{key}-{r}-{arm}-{turn}",
                             "source": "captured-synthetic" if captured else "reconstructed-synthetic",
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
        if arm not in arms:
            raise SystemExit(f"{path}: scenario {key} has no arm {arm!r}; it has {', '.join(arms)}")
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
                             "cache_scope": f"synthetic-{key}-{arm}", "source": "reconstructed-synthetic",
                             "usage": {k: used[k] for k in ("cached", "written", "uncached")}})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?")
    parser.add_argument("--ttl", type=float, help="cache lifetime in seconds (for example 300 for 5 minutes)")
    parser.add_argument("--write", type=float, default=1.25)
    parser.add_argument("--read", type=float, default=0.1)
    parser.add_argument("--json", action="store_true", help="print the full result, events included")
    parser.add_argument("--validate-labels", action="store_true", help="score independently annotated log rows")
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
    if args.validate_labels:
        print(json.dumps(validate_labels(rows, result), indent=2))
        return
    print(json.dumps(result, indent=2) if args.json else report(result))


if __name__ == "__main__":
    main()

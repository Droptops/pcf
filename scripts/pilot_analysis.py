#!/usr/bin/env python3
"""Assignment and analysis for the production pilot in docs/PILOT.md.

Log one JSON object per request (JSON Lines), from provider usage:
  {"conversation": "c-123", "arm": "front-tuned" | "echo-all" | "fixed-tail" | "placed",
   "cached": 0, "written": 0, "uncached": 0,
   "output_tokens": 0, "latency_s": 1.9, "ttft_s": 0.4, "correct": true}
`ttft_s` and `correct` are optional; `correct` is the automatic record check where the source of truth is known
(null or absent when it is not). Costs are in uncached-input-token units under --write/--read/--output
multipliers; pass the provider's current price ratios, not the synthetic runs' assumptions.

  pilot_analysis.py --assign CONVERSATION_ID                         print the four-arm assignment
  pilot_analysis.py --assign ID --legacy-two-arm --treatment-share .5 preserve old two-arm assignment
  pilot_analysis.py LOG.jsonl                                          analyze a pilot log
  pilot_analysis.py --from-domain RESULT.json > LOG.jsonl              a synthetic log from a domain run

Intervals come from a bootstrap over conversations, never turns: answers and cache state within a conversation
are correlated. The bootstrap is seeded, so the report is reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from itertools import combinations

ARMS = ("baseline", "treatment")
PILOT_ARMS = ("front-tuned", "echo-all", "fixed-tail", "placed")


def assign_multi(conversation: str, salt: str = "pcf-pilot") -> str:
    """Equal allocation to the four prespecified layouts; one assignment per conversation."""
    digest = hashlib.sha256(f"{salt}:{conversation}".encode()).digest()
    return PILOT_ARMS[int.from_bytes(digest[:8], "big") * len(PILOT_ARMS) // 2 ** 64]


def validate_rows(rows: list[dict]) -> None:
    seen = set()
    for row in rows:
        if not isinstance(row.get("conversation"), str) or not row["conversation"]:
            raise ValueError("conversation must be a nonempty string")
        for key in ("cached", "written", "uncached", "output_tokens", "latency_s", "ttft_s"):
            value = row.get(key)
            if key in ("latency_s", "ttft_s") and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            if key in ("cached", "written", "uncached", "output_tokens") and int(value) != value:
                raise ValueError(f"{key} must be an integer token count")
        if row.get("correct") is not None and type(row["correct"]) is not bool:
            raise ValueError("correct must be boolean or null")
        if row.get("request_id") is not None:
            identity = (row.get("cache_scope"), row["request_id"])
            if identity in seen:
                raise ValueError("duplicate request_id")
            seen.add(identity)


def assign(conversation: str, treatment_share: float = 0.5, salt: str = "pcf-pilot") -> str:
    """The arm for a conversation: stable for its whole life, independent of time and traffic."""
    if not 0 <= treatment_share <= 1:
        raise ValueError("treatment_share must be in [0, 1]")
    digest = hashlib.sha256(f"{salt}:{conversation}".encode()).digest()
    return "treatment" if int.from_bytes(digest[:8], "big") / 2 ** 64 < treatment_share else "baseline"


def conversations(rows: list[dict], write: float, read: float, output: float) -> dict[str, list[dict]]:
    """Per arm, one summary per conversation."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if row["arm"] not in ARMS:
            raise ValueError(f"unknown arm {row['arm']!r}")
        grouped.setdefault((row["arm"], row["conversation"]), []).append(row)
    seen: dict[str, str] = {}
    for arm, conversation in grouped:
        if seen.setdefault(conversation, arm) != arm:
            raise ValueError(f"conversation {conversation!r} appears in both arms; assign whole conversations")
    arms = {}
    for (arm, _), turns in grouped.items():
        checked = [t["correct"] for t in turns if t.get("correct") is not None]
        input_cost = math.fsum(t["uncached"] + write * t["written"] + read * t["cached"] for t in turns)
        arms.setdefault(arm, []).append({
            "turns": len(turns), "input_cost": input_cost,
            "total_cost": input_cost + output * math.fsum(t["output_tokens"] for t in turns),
            "checked": len(checked), "errors": sum(not c for c in checked),
            "latency": [t["latency_s"] for t in turns if t.get("latency_s") is not None],
            "ttft": [t["ttft_s"] for t in turns if t.get("ttft_s") is not None]})
    return arms


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def mean_cost(convs: list[dict], key: str) -> float:
    return math.fsum(c[key] for c in convs) / len(convs)


def error_rate(convs: list[dict]) -> float | None:
    checked = sum(c["checked"] for c in convs)
    return sum(c["errors"] for c in convs) / checked if checked else None


def bootstrap(base: list[dict], treat: list[dict], stat, samples: int, seed: int) -> list[float] | None:
    rng, values = random.Random(seed), []
    for _ in range(samples):
        b = [rng.choice(base) for _ in base]
        t = [rng.choice(treat) for _ in treat]
        value = stat(b, t)
        if value is not None:
            values.append(value)
    if len(values) < samples * .9:
        return None
    return [round(quantile(values, .025), 4), round(quantile(values, .975), 4)]


def _analyze_pair(rows: list[dict], write: float = 1.25, read: float = 0.1, output: float = 5.0,
            samples: int = 2000, seed: int = 0) -> dict:
    arms = conversations(rows, write, read, output)
    missing = [a for a in ARMS if a not in arms]
    if missing:
        raise ValueError(f"no conversations in arm(s): {', '.join(missing)}")
    base, treat = arms["baseline"], arms["treatment"]
    report = {"multipliers": {"write": write, "read": read, "output": output},
              "unit": "conversation; intervals are 95% bootstrap intervals over conversations", "arms": {}}
    for arm, convs in arms.items():
        turns = [t for c in convs for t in c["latency"]]
        ttft = [t for c in convs for t in c["ttft"]]
        report["arms"][arm] = {
            "conversations": len(convs), "turns": sum(c["turns"] for c in convs),
            "input_cost_per_conversation": round(mean_cost(convs, "input_cost"), 1),
            "total_cost_per_conversation": round(mean_cost(convs, "total_cost"), 1),
            "checked_answers": sum(c["checked"] for c in convs), "error_rate": error_rate(convs),
            "latency_s": {"p50": quantile(turns, .5), "p90": quantile(turns, .9)},
            "ttft_s": {"p50": quantile(ttft, .5), "p90": quantile(ttft, .9)} if ttft else None}
    for key in ("input_cost", "total_cost"):
        def ratio(b, t, k=key):
            denominator = mean_cost(b, k)
            return mean_cost(t, k) / denominator if denominator else None
        point = ratio(base, treat)
        report[f"{key}_ratio"] = {"treatment_over_baseline": None if point is None else round(point, 4),
                                 "ci95": bootstrap(base, treat, ratio, samples, seed)}

    def diff(b, t):
        rb, rt = error_rate(b), error_rate(t)
        return None if rb is None or rt is None else rt - rb
    point = diff(base, treat)
    report["error_rate_difference"] = {"treatment_minus_baseline": None if point is None else round(point, 4),
                                       "ci95": bootstrap(base, treat, diff, samples, seed + 1)}

    def p90(b, t):
        tb, tt = [x for c in b for x in c["latency"]], [x for c in t for x in c["latency"]]
        return quantile(tt, .9) - quantile(tb, .9) if tb and tt else None
    point = p90(base, treat)
    report["latency_p90_difference_s"] = {"treatment_minus_baseline": None if point is None else round(point, 3),
                                          "ci95": bootstrap(base, treat, p90, samples, seed + 2)}
    return report


def analyze(rows: list[dict], write: float = 1.25, read: float = 0.1, output: float = 5.0,
            samples: int = 2000, seed: int = 0) -> dict:
    """Four-arm comparisons; preserve the original report for legacy two-arm logs.

    Intervals are descriptive and unadjusted for multiple comparisons. No automatic ship decision is made.
    """
    validate_rows(rows)
    for value in (write, read, output):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError("price multipliers must be finite and nonnegative")
    if type(samples) is not int or samples < 1:
        raise ValueError("samples must be a positive integer")
    names = {r["arm"] for r in rows}
    if names <= set(ARMS):
        return _analyze_pair(rows, write, read, output, samples, seed)
    if names != set(PILOT_ARMS):
        raise ValueError(f"four-arm pilot requires exactly {PILOT_ARMS}; got {sorted(names)}")
    seen = {}
    for row in rows:
        if seen.setdefault(row["conversation"], row["arm"]) != row["arm"]:
            raise ValueError("conversation appears in multiple arms; assign whole conversations")
    report = {"multipliers": {"write": write, "read": read, "output": output},
              "unit": "conversation", "arms": {}, "comparisons": {},
              "inference": "Descriptive 95% conversation bootstrap intervals; no multiplicity adjustment or ship gate."}
    for baseline, treatment in combinations(PILOT_ARMS, 2):
        pair = [{**r, "arm": "baseline" if r["arm"] == baseline else "treatment"}
                for r in rows if r["arm"] in (baseline, treatment)]
        result = _analyze_pair(pair, write, read, output, samples, seed)
        report["arms"][baseline] = result["arms"]["baseline"]
        report["arms"][treatment] = result["arms"]["treatment"]
        report["comparisons"][f"{treatment} vs {baseline}"] = {
            "baseline": baseline, "treatment": treatment,
            **{k: v for k, v in result.items() if k.endswith(("_ratio", "_difference", "_difference_s"))}}
    return report


def from_domain(path: str, baseline: str = "front-tuned", treatment: str = "placed", *,
                four_arm: bool = False) -> list[dict]:
    """A synthetic pilot log from a saved domain run: each session is one conversation."""
    with open(path) as f:
        saved = json.load(f)
    rows = []
    for key, runs in saved["scenarios"].items():
        for i, run in enumerate(runs):
            for arm, name in ([(a, a) for a in PILOT_ARMS] if four_arm else
                              [(baseline, "baseline"), (treatment, "treatment")]):
                for r in run[arm]:
                    rows.append({"conversation": f"{key}-{i}-{name}", "arm": name, "cached": r["cached"],
                                 "written": r["written"], "uncached": r["uncached"],
                                 "output_tokens": r["output_tokens"], "latency_s": r.get("latency_s"),
                                 "correct": r["correct"]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", help="JSON Lines, one request per line")
    parser.add_argument("--assign", metavar="CONVERSATION_ID")
    parser.add_argument("--treatment-share", type=float, help="legacy two-arm treatment allocation (default .5)")
    parser.add_argument("--legacy-two-arm", action="store_true", help="use the original two-arm assignment")
    parser.add_argument("--four-arm", action="store_true", help="export all four layouts from a domain run")
    parser.add_argument("--salt", default="pcf-pilot", help="change per pilot so assignments are independent")
    parser.add_argument("--from-domain", metavar="RESULT_JSON")
    parser.add_argument("--write", type=float, default=1.25, help="cache write price / uncached input price")
    parser.add_argument("--read", type=float, default=0.1, help="cache read price / uncached input price")
    parser.add_argument("--output", type=float, default=5.0, help="output price / uncached input price")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.treatment_share is not None and not args.legacy_two_arm:
        parser.error("--treatment-share requires --legacy-two-arm; four-arm allocation is equal")
    if args.assign is not None:
        share = .5 if args.treatment_share is None else args.treatment_share
        print(assign(args.assign, share, args.salt) if args.legacy_two_arm
              else assign_multi(args.assign, args.salt))
    elif args.from_domain:
        for row in from_domain(args.from_domain, four_arm=args.four_arm):
            print(json.dumps(row))
    elif args.log:
        with open(args.log) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        print(json.dumps(analyze(rows, args.write, args.read, args.output, args.samples, args.seed), indent=2))
    else:
        parser.error("give a LOG, --assign or --from-domain")
        sys.exit(2)


if __name__ == "__main__":
    main()

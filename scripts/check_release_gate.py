#!/usr/bin/env python3
"""Fail a release unless real production-pilot evidence satisfies the frozen release contract."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import re
import tomllib
from pathlib import Path

ARMS = ("front-tuned", "echo-all", "fixed-tail", "placed")
GATES = ("placed_vs_fixed_tail_input_cost", "placed_vs_front_tuned_input_cost",
         "quality_noninferiority", "blind_quality_noninferiority", "latency", "failure_rate")
HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _keys(value: dict, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")


def _positive(value, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a finite positive number")
    return float(value)


def _time(value, label: str) -> datetime.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def validate_evidence(doc: dict) -> dict:
    fields = {"evidence_version", "pilot_id", "source", "decision", "preregistered_at", "started_at", "ended_at",
              "provider", "model", "deployment_revision", "minimum_conversations_per_arm",
              "actual_prices_per_million_tokens", "arms", "blind_review", "gates", "artifacts", "approvals"}
    _keys(doc, fields, "evidence")
    if doc["evidence_version"] != 1 or doc["source"] != "production" or doc["decision"] != "pass":
        raise ValueError("release evidence must be version 1, source production, decision pass")
    for key in ("pilot_id", "provider", "model", "deployment_revision"):
        if not isinstance(doc[key], str) or not doc[key].strip():
            raise ValueError(f"{key} must be a nonempty string")
    preregistered, started, ended = [_time(doc[key], key) for key in
                                     ("preregistered_at", "started_at", "ended_at")]
    if not preregistered < started < ended:
        raise ValueError("pilot timestamps must satisfy preregistered_at < started_at < ended_at")
    minimum = doc["minimum_conversations_per_arm"]
    if type(minimum) is not int or minimum < 1:
        raise ValueError("minimum_conversations_per_arm must be a positive integer")
    prices = doc["actual_prices_per_million_tokens"]
    _keys(prices, {"uncached_input", "cache_write", "cache_read", "output"}, "actual prices")
    for key, value in prices.items():
        _positive(value, f"price {key}")
    if prices["cache_read"] >= prices["uncached_input"]:
        raise ValueError("cache_read must cost less than uncached_input")
    _keys(doc["arms"], set(ARMS), "arms")
    for arm, value in doc["arms"].items():
        _keys(value, {"conversations", "requests", "failures"}, f"arm {arm}")
        for key in ("conversations", "requests", "failures"):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError(f"arm {arm} {key} must be a nonnegative integer")
        if value["conversations"] < minimum or value["requests"] < value["conversations"]:
            raise ValueError(f"arm {arm} does not meet the preregistered sample minimum")
        if value["failures"] > value["requests"]:
            raise ValueError(f"arm {arm} failures exceed requests")
    blind = doc["blind_review"]
    _keys(blind, {"blinded", "independent_reviewers", "rubric_version", "minimum_reviewed_conversations_per_arm",
                  "reviewed_conversations_by_arm", "inter_rater_agreement", "disagreements_adjudicated"},
          "blind review")
    if blind["blinded"] is not True or blind["disagreements_adjudicated"] is not True:
        raise ValueError("blind review must be blinded and all disagreements adjudicated")
    if type(blind["independent_reviewers"]) is not int or blind["independent_reviewers"] < 2:
        raise ValueError("blind review needs at least two independent reviewers")
    if not isinstance(blind["rubric_version"], str) or not blind["rubric_version"]:
        raise ValueError("blind review rubric_version must be nonempty")
    review_min = blind["minimum_reviewed_conversations_per_arm"]
    if type(review_min) is not int or review_min < 1:
        raise ValueError("minimum reviewed conversations must be positive")
    _keys(blind["reviewed_conversations_by_arm"], set(ARMS), "blind-review arms")
    if any(type(n) is not int or n < review_min for n in blind["reviewed_conversations_by_arm"].values()):
        raise ValueError("each arm must meet the blind-review sample minimum")
    agreement = blind["inter_rater_agreement"]
    if isinstance(agreement, bool) or not isinstance(agreement, (int, float)) or not 0 <= agreement <= 1:
        raise ValueError("inter_rater_agreement must be in [0, 1]")
    _keys(doc["gates"], set(GATES), "gates")
    for name, gate in doc["gates"].items():
        _keys(gate, {"passed", "estimate", "limit", "comparison"}, f"gate {name}")
        if gate["passed"] is not True:
            raise ValueError(f"release gate {name} did not pass")
        for key in ("estimate", "limit"):
            if isinstance(gate[key], bool) or not isinstance(gate[key], (int, float)) or not math.isfinite(gate[key]):
                raise ValueError(f"gate {name} {key} must be finite")
        if not isinstance(gate["comparison"], str) or not gate["comparison"]:
            raise ValueError(f"gate {name} comparison must be nonempty")
    _keys(doc["artifacts"], {"pilot_log_sha256", "blind_review_log_sha256", "analysis_sha256"}, "artifacts")
    if any(not isinstance(value, str) or not HASH.fullmatch(value) for value in doc["artifacts"].values()):
        raise ValueError("artifact identities must be sha256:<64 lowercase hex>")
    approvals = doc["approvals"]
    if not isinstance(approvals, list) or len(approvals) < 2:
        raise ValueError("release evidence needs at least two human approvals")
    identities, roles = set(), set()
    for approval in approvals:
        _keys(approval, {"reviewer", "role", "approved_at"}, "approval")
        if not isinstance(approval["reviewer"], str) or not approval["reviewer"]:
            raise ValueError("approval reviewer must be nonempty")
        identities.add(approval["reviewer"])
        roles.add(approval["role"])
        _time(approval["approved_at"], "approved_at")
    if len(identities) < 2 or not {"pilot-owner", "independent-quality-reviewer"} <= roles:
        raise ValueError("approvals need distinct pilot-owner and independent-quality-reviewer identities")
    return {"pilot_id": doc["pilot_id"], "decision": "pass", "provider": doc["provider"],
            "model": doc["model"], "conversations": sum(x["conversations"] for x in doc["arms"].values())}


def project_version(path: str = "pyproject.toml") -> str:
    with open(path, "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", help="sanitized production-pilot evidence JSON")
    parser.add_argument("--tag", help="release tag; must exactly match the project version with a v prefix")
    parser.add_argument("--pyproject", default="pyproject.toml")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    if not evidence.is_file():
        raise SystemExit(f"release blocked: missing {evidence}")
    try:
        summary = validate_evidence(json.loads(evidence.read_text()))
        version = project_version(args.pyproject)
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(f"tag {args.tag!r} does not match project version v{version}")
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"release blocked: {exc}") from exc
    print(json.dumps({"release_gate": "pass", "project_version": version, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()

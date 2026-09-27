#!/usr/bin/env python3
"""Fail a release unless real production-pilot evidence satisfies the frozen release contract."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path

ARMS = ("front-tuned", "echo-all", "fixed-tail", "placed")
GATES = ("placed_vs_fixed_tail_input_cost", "placed_vs_front_tuned_input_cost",
         "quality_noninferiority", "blind_quality_noninferiority", "latency", "failure_rate")
HASH = re.compile(r"sha256:[0-9a-f]{64}")
ARTIFACT_FILES = {"pilot_log_sha256": "pilot.jsonl", "blind_review_log_sha256": "blind-review.json",
                  "analysis_sha256": "analysis.json"}


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


def _artifact_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound_artifacts(doc: dict, artifact_dir: str | Path) -> tuple[list[dict], dict, dict]:
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    loaded = {}
    for field, filename in ARTIFACT_FILES.items():
        path = root / filename
        if not path.is_file():
            raise ValueError(f"missing bound release artifact: {path}")
        if _artifact_hash(path) != doc["artifacts"][field]:
            raise ValueError(f"artifact hash mismatch for {filename}")
        if filename.endswith(".jsonl"):
            try:
                loaded[field] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON Lines artifact: {filename}") from exc
        else:
            try:
                loaded[field] = json.loads(path.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON artifact: {filename}") from exc
    return loaded["pilot_log_sha256"], loaded["blind_review_log_sha256"], loaded["analysis_sha256"]


def _verify_pilot_log(rows: list[dict], arms: dict) -> None:
    if not rows:
        raise ValueError("bound pilot log is empty")
    summary = {arm: {"conversations": set(), "requests": 0, "failures": 0} for arm in ARMS}
    seen_requests = set()
    for i, row in enumerate(rows):
        arm, conversation = row.get("arm"), row.get("conversation")
        if arm not in summary or not isinstance(conversation, str) or not conversation:
            raise ValueError(f"pilot log row {i} has invalid arm/conversation")
        for key in ("cached", "written", "uncached"):
            value = row.get(key)
            if type(value) is not int or value < 0:
                raise ValueError(f"pilot log row {i} {key} must be a nonnegative integer")
        latency = row.get("latency_s")
        if isinstance(latency, bool) or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0:
            raise ValueError(f"pilot log row {i} latency_s must be finite and nonnegative")
        if type(row.get("success")) is not bool:
            raise ValueError(f"pilot log row {i} success must be boolean")
        for key in ("correct", "blind_acceptable"):
            if row.get(key) is not None and type(row[key]) is not bool:
                raise ValueError(f"pilot log row {i} {key} must be boolean or null")
        request_id = row.get("request_id")
        if request_id is not None:
            identity = (row.get("cache_scope"), request_id)
            if identity in seen_requests:
                raise ValueError("bound pilot log contains duplicate request_id")
            seen_requests.add(identity)
        summary[arm]["conversations"].add(conversation)
        summary[arm]["requests"] += 1
        summary[arm]["failures"] += row["success"] is False
    for arm in ARMS:
        actual = {"conversations": len(summary[arm]["conversations"]), "requests": summary[arm]["requests"],
                  "failures": summary[arm]["failures"]}
        if actual != arms[arm]:
            raise ValueError(f"bound pilot log does not match arm summary for {arm}")


def _recompute_gates(rows: list[dict], prices: dict) -> dict[str, float]:
    grouped = {arm: {} for arm in ARMS}
    for row in rows:
        grouped[row["arm"]].setdefault(row["conversation"], []).append(row)
    write = prices["cache_write"] / prices["uncached_input"]
    read = prices["cache_read"] / prices["uncached_input"]

    def mean_input_cost(arm):
        costs = [math.fsum(r["uncached"] + write * r["written"] + read * r["cached"] for r in turns)
                 for turns in grouped[arm].values()]
        return math.fsum(costs) / len(costs)

    def error_rate(arm, key):
        values = [row[key] for row in rows if row["arm"] == arm and row.get(key) is not None]
        if not values:
            raise ValueError(f"bound pilot log has no {key} observations for {arm}")
        return sum(not value for value in values) / len(values)

    def failure_rate(arm):
        values = [row["success"] for row in rows if row["arm"] == arm]
        return sum(not value for value in values) / len(values)

    def p90(arm):
        values = sorted(row["latency_s"] for row in rows if row["arm"] == arm)
        return values[min(len(values) - 1, int(.9 * len(values)))]

    front, placed, fixed = "front-tuned", "placed", "fixed-tail"
    front_cost, placed_cost, fixed_cost = mean_input_cost(front), mean_input_cost(placed), mean_input_cost(fixed)
    if front_cost <= 0 or fixed_cost <= 0:
        raise ValueError("bound pilot log has a zero input-cost baseline")
    return {
        "placed_vs_fixed_tail_input_cost": round(placed_cost / fixed_cost, 4),
        "placed_vs_front_tuned_input_cost": round(placed_cost / front_cost, 4),
        "quality_noninferiority": round(error_rate(placed, "correct") - error_rate(front, "correct"), 4),
        "blind_quality_noninferiority": round(
            error_rate(placed, "blind_acceptable") - error_rate(front, "blind_acceptable"), 4),
        "latency": round(p90(placed) - p90(front), 4),
        "failure_rate": round(failure_rate(placed) - failure_rate(front), 4),
    }


def validate_evidence(doc: dict, artifact_dir: str | Path | None = None) -> dict:
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
        if gate["estimate"] > gate["limit"]:
            raise ValueError(f"release gate {name} estimate exceeds its preregistered limit")
    _keys(doc["artifacts"], {"pilot_log_sha256", "blind_review_log_sha256", "analysis_sha256"}, "artifacts")
    if any(not isinstance(value, str) or not HASH.fullmatch(value) for value in doc["artifacts"].values()):
        raise ValueError("artifact identities must be sha256:<64 lowercase hex>")
    if artifact_dir is None:
        raise ValueError("bound release artifacts are required")
    pilot_rows, blind_artifact, analysis_artifact = _load_bound_artifacts(doc, artifact_dir)
    _verify_pilot_log(pilot_rows, doc["arms"])
    recomputed = _recompute_gates(pilot_rows, doc["actual_prices_per_million_tokens"])
    for name, estimate in recomputed.items():
        if doc["gates"][name]["estimate"] != estimate:
            raise ValueError(f"release gate {name} estimate does not match bound pilot log")
    if not isinstance(blind_artifact, dict) or blind_artifact != doc["blind_review"]:
        raise ValueError("bound blind-review artifact does not match release evidence")
    if not isinstance(analysis_artifact, dict) or analysis_artifact.get("gates") != doc["gates"]:
        raise ValueError("bound analysis artifact does not match release gates")
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
    parser.add_argument("--artifact-dir", required=True,
                        help="directory containing pilot.jsonl, blind-review.json and analysis.json")
    args = parser.parse_args()
    evidence = Path(args.evidence)
    if not evidence.is_file():
        raise SystemExit(f"release blocked: missing {evidence}")
    try:
        summary = validate_evidence(json.loads(evidence.read_text()), args.artifact_dir)
        version = project_version(args.pyproject)
        if args.tag is not None and args.tag != f"v{version}":
            raise ValueError(f"tag {args.tag!r} does not match project version v{version}")
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        raise SystemExit(f"release blocked: {exc}") from exc
    print(json.dumps({"release_gate": "pass", "project_version": version, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()

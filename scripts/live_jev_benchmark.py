#!/usr/bin/env python3
"""Run the labeled Jev task-shape benchmark using OPENROUTER_API_KEY.

The script prints only case labels and aggregate telemetry. It never prints the
credential or benchmark state payloads.
"""
from __future__ import annotations

import json

from pcf.jit.jev_benchmark import default_jev_benchmark_cases, run_jev_benchmark
from pcf.jit.jev_ood_benchmark import hard_jev_benchmark_cases
from pcf.jit.jev_openrouter import OpenRouterJevClient
from pcf.jit.jev_selective import evaluate_jev_selective_evidence


def main() -> int:
    client = OpenRouterJevClient.from_env()
    cases = default_jev_benchmark_cases() + hard_jev_benchmark_cases()
    summary = run_jev_benchmark(client, cases)
    selective = evaluate_jev_selective_evidence(summary)

    for item in summary.observations:
        print(
            json.dumps(
                {
                    "case": item.case_name,
                    "expected": item.expected.value,
                    "observed": item.observed.value if item.observed is not None else None,
                    "matched": item.matched,
                    "confidence": item.confidence,
                    "cost_usd": item.cost_usd,
                    "latency_ms": round(item.latency_ms, 3),
                    "input_tokens": item.input_tokens,
                    "output_tokens": item.output_tokens,
                    "error_type": item.error_type,
                },
                sort_keys=True,
            )
        )

    print(
        json.dumps(
            {
                "summary": {
                    "total_cases": summary.total_cases,
                    "successes": summary.successes,
                    "errors": summary.errors,
                    "matches": summary.matches,
                    "accuracy": summary.accuracy,
                    "successful_accuracy": summary.successful_accuracy,
                    "low_confidence_rate": summary.low_confidence_rate,
                    "mean_confidence": summary.mean_confidence,
                    "mean_latency_ms": round(summary.mean_latency_ms, 3),
                    "total_cost_usd": summary.total_cost_usd,
                    "accuracy_by_expected": {
                        primitive.value: value for primitive, value in summary.accuracy_by_expected().items()
                    },
                    "confusion_matrix": {
                        expected.value: row for expected, row in summary.confusion_matrix().items()
                    },
                    "selective_evidence": {
                        "selected_cases": selective.selected_cases,
                        "selected_matches": selective.selected_matches,
                        "abstained_cases": selective.abstained_cases,
                        "coverage": selective.coverage,
                        "selected_accuracy": selective.selected_accuracy,
                        "accuracy_lcb": selective.accuracy_lcb,
                        "error_rate": selective.error_rate,
                        "eligible": selective.eligible,
                        "reasons": list(selective.reasons),
                    },
                }
            },
            sort_keys=True,
        )
    )
    return 1 if summary.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

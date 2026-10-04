#!/usr/bin/env python3
"""Minimal live Jev/OpenRouter smoke test.

Requires OPENROUTER_API_KEY. Prints decision telemetry only; it never prints the key
or the input state.
"""
from __future__ import annotations

import json

from pcf.jit import OpenRouterJevClient


def main() -> None:
    client = OpenRouterJevClient.from_env()
    advice = client.route(
        {
            "operation": "normalize_phone_number",
            "description": "Normalize a phone number into E.164 using deterministic parsing rules.",
            "output_schema": "string",
            "side_effects": "none",
        }
    )
    print(
        json.dumps(
            {
                "requested_model": advice.requested_model,
                "served_model": advice.served_model,
                "provider": advice.provider,
                "primitive": advice.primitive.value,
                "confidence": advice.confidence,
                "probabilities": {key.value: value for key, value in advice.probabilities.items()},
                "input_tokens": advice.usage.input_tokens,
                "output_tokens": advice.usage.output_tokens,
                "cost_usd": advice.usage.cost_usd,
                "latency_ms": round(advice.latency_ms, 3),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

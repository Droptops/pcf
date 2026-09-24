#!/usr/bin/env python3
"""A six-turn session across two model families, with the router deciding each turn on measured cache
warmth plus a calibrated confidence. Every number printed is computed by the simulated engines, not typed in.

Run:  python examples/demo.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pcf import Context, Segment  # noqa: E402
from pcf.families.sim import family_a, family_b  # noqa: E402
from pcf.router import Candidate, PlattScaledSource, Router  # noqa: E402

SYSTEM = ("You are a support agent for a telecom carrier. Be brief and cite the order id. " * 12).strip()
TOOLS = [{"name": "lookup_order", "description": "Find an order by id",
          "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}]

# Prices: frontier at $1.00/MTok input and $0.10/MTok cache read; cheap at $0.20 / $0.02 (illustrative).
frontier, cheap = family_a(min_cacheable=32), family_b(min_cacheable=32)
engines = {frontier.descriptor.model_id: frontier, cheap.descriptor.model_id: cheap}


def raw_scorer(ctx: Context, cand: Candidate) -> float:
    """Stand-in for Jev: long or multi-part questions are 'hard'."""
    user = next(s for s in ctx.segments if s.kind == "user").content
    return 0.35 if (len(user) > 60 or " and " in user) else 0.9


router = Router([
    Candidate(frontier.compiler, frontier.cache, 1.0, 0.10, is_fallback=True),
    Candidate(cheap.compiler, cheap.cache, 0.2, 0.02),
], PlattScaledSource(raw_scorer, a=2.0, b=0.0), threshold=0.8)

turns = [
    "Where is order 1001?",
    "And 1002?",
    "Compare the delivery estimates for 1001 and 1002 and tell me which to cancel if only one can arrive by Friday.",
    "Thanks. Status of 1003?",
    "Explain why 1002 was delayed and draft an apology that references our SLA and the credit policy.",
    "Ok. 1004?",
]

ctx = Context([Segment("tools", "tools", TOOLS), Segment("sys", "system", SYSTEM)], session_id="demo")
prev_assistant = None
print(f"{'turn':>4}  {'chosen':<12} {'esc':<4} {'conf':>5}  {'warm':>5} {'cold':>5}  {'cost$':>10}  why")
for n, q in enumerate(turns, start=1):
    segs = [s for s in ctx.segments if s.kind != "user"]
    prev_user = next((s for s in ctx.segments if s.kind == "user"), None)
    if prev_user is not None:
        segs.append(Segment(f"h{n-1}", "history", [{"role": "user", "content": prev_user.content},
                                                    {"role": "assistant", "content": prev_assistant}]))
    segs.append(Segment(f"u{n}", "user", q, stable=False))
    ctx = Context(segs, session_id="demo")

    decision = router.route(ctx, now=float(n), request_id=f"demo-{n}")
    usage, _ = engines[decision.chosen].run(ctx, now=float(n))
    rep = next(c for c in decision.candidates if c.model_id == decision.chosen)
    conf = "-" if decision.confidence is None else f"{decision.confidence:.2f}"
    print(f"{n:>4}  {decision.chosen:<12} {'yes' if decision.escalate else 'no':<4} {conf:>5}  "
          f"{usage.cache_read_input_tokens:>5} {usage.cold_tokens:>5}  {rep.est_input_cost_usd:>10.7f}  {decision.reason}")
    prev_assistant = f"Order {1000 + n} update sent."

print("\nLast decision as JSON (validates against spec/schemas/route-decision.schema.json):")
print(json.dumps(decision.to_json(), indent=1)[:900] + "\n...")

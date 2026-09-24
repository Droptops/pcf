import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402

from pcf import Context, Segment  # noqa: E402

SYSTEM = ("You are a support agent for a telecom carrier. Be brief and cite the order id. " * 12).strip()
TOOLS = [
    {"name": "lookup_order", "description": "Find an order by id",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "issue_credit", "description": "Issue an account credit",
     "parameters": {"type": "object", "properties": {"amount": {"type": "number"}}, "required": ["amount"]}},
]
MEMORY = {"customer_tier": "gold", "preferred_language": "es", "open_tickets": 2}


@pytest.fixture
def base_ctx() -> Context:
    return Context([
        Segment("tools", "tools", TOOLS),
        Segment("sys", "system", SYSTEM),
        Segment("mem", "memory", MEMORY),
    ], session_id="test-session")


def turn(ctx: Context, n: int, user_text: str, assistant_text: str | None) -> Context:
    """Append the current user turn; if an assistant reply is given, fold the pair into history first."""
    segs = [s for s in ctx.segments if s.kind != "user"]
    prev_user = next((s for s in ctx.segments if s.kind == "user"), None)
    if prev_user is not None and assistant_text is not None:
        segs.append(Segment(f"h{n-1}", "history", [
            {"role": "user", "content": prev_user.content},
            {"role": "assistant", "content": assistant_text},
        ]))
    segs.append(Segment(f"u{n}", "user", user_text, stable=False))
    return Context(segs, session_id=ctx.session_id)

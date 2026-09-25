"""Incremental prefix compilation must match rendering and encoding every prefix from scratch."""
from __future__ import annotations

import random
from time import perf_counter

import pytest

from pcf import Context, PrefixCache, Segment
from pcf.compiler import ContextCompiler
from pcf.families.anthropic_adapter import AnthropicCompiler
from pcf.families.openai_adapter import OpenAICompiler
from pcf.families.sim import SimEngine, family_a, family_b

THINKING = {"provider": "anthropic", "block": {"type": "thinking", "thinking": "", "signature": "s"}}
REASONING = {"provider": "openai", "block": {"type": "reasoning", "id": "rs", "summary": []}}


def _history(rng, k):
    """One history segment: a plain exchange, a tool round, or a tool result for earlier calls."""
    kind = rng.choice(["chat", "chat", "call"])
    turns = [{"role": "user", "content": f"question {k} " + "é  \"quoted\" " * rng.randint(0, 3)}]
    if kind == "chat":
        reply = {"role": "assistant", "content": f"answer {k} " * rng.randint(1, 30)}
        if rng.random() < .3:
            reply["provider_blocks"] = [rng.choice([THINKING, REASONING])]
        return [Segment(f"h{k}", "history", [*turns, reply])]
    calls = [{"id": f"c{k}_{j}", "name": "lookup", "arguments": {"q": k, "j": j}} for j in range(rng.randint(1, 3))]
    turns.append({"role": "assistant", "content": "checking" if rng.random() < .5 else "", "tool_calls": calls})
    results = [{"role": "tool", "call_id": c["id"], "content": {"rows": [k, j]}, "is_error": rng.random() < .2}
               for j, c in enumerate(calls)]
    if rng.random() < .5:  # results in the same segment, or a separate one
        return [Segment(f"h{k}", "history", turns + results)]
    return [Segment(f"h{k}", "history", turns), Segment(f"h{k}r", "history", results)]


def _context(seed):
    rng = random.Random(seed)
    segs = []
    if rng.random() < .6:
        segs.append(Segment("t", "tools", [{"name": "lookup", "description": "find", "parameters": {"type": "object"}}]))
    segs.append(Segment("s", "system", "policy " * rng.randint(1, 200)))
    authorities = sorted(rng.choice(["data", "data", "instruction"]) for _ in range(rng.randint(0, 3)))[::-1]
    for m, authority in enumerate(authorities):  # instruction memory precedes data
        segs.append(Segment(f"m{m}", "memory", {"module": m, "rows": list(range(rng.randint(0, 20)))},
                            authority=authority, provenance=f"mod{m}"))
    if rng.random() < .3:
        segs.append(Segment("d", "document", "doc " * 50))
    for k in range(rng.randint(0, 12)):
        segs.extend(_history(rng, k))
    for m in range(rng.randint(0, 2)):
        segs.append(Segment(f"tail{m}", "memory", {"tail": m}, False, provenance=f"tail{m}"))
    segs.append(Segment("u", "user", "next " * rng.randint(1, 5), stable=False))
    return Context(segs, cache_namespace=f"ns{seed % 3}")


def _slow(compiler):
    """The same compiler, forced onto the render-every-prefix path."""
    return type("Slow", (type(compiler),), {"render_prefixes": ContextCompiler.render_prefixes})


COMPILERS = [lambda: OpenAICompiler("gpt-5.6"), lambda: OpenAICompiler("gpt-5.6", reasoning_items="drop"),
             lambda: AnthropicCompiler("claude-sonnet-5"),
             lambda: AnthropicCompiler("claude-sonnet-5", thinking_blocks="drop"),
             lambda: family_a().compiler, lambda: family_b().compiler]


@pytest.mark.parametrize("make", COMPILERS)
def test_incremental_compile_matches_rendering_every_prefix(make):
    fast = make()
    slow = fast.__class__.__new__(_slow(fast))
    slow.__dict__.update(fast.__dict__)
    for seed in range(60):
        ctx = _context(seed)
        a, b = fast.compile(ctx), slow.compile(ctx)
        assert a == b, seed


def test_openai_marker_prefix_is_what_the_simulator_writes_and_reads():
    compiler = OpenAICompiler("gpt-5.6")
    ctx = Context([Segment("s", "system", "policy " * 1000),
                   Segment("h", "history", [{"role": "user", "content": "Q"},
                                            {"role": "assistant", "content": "answer " * 1000}]),
                   Segment("u", "user", "Next")])
    engine = SimEngine(compiler, PrefixCache(1800))
    usage, compiled = engine.run(ctx, 0)
    covered = compiler.covered_tokens(ctx, 1, compiled.cum_tokens)
    assert covered < compiled.cum_tokens[1]  # the marker stops before the assistant reply
    assert usage.cache_creation_input_tokens == covered and usage.cache_read_input_tokens == 0
    # The next request carries the same marker: it reads exactly the covered prefix, not the whole segment.
    nxt = Context([*ctx.segments[:2], Segment("h2", "history", [{"role": "user", "content": "Next"},
                                                                 {"role": "assistant", "content": "ok"}]),
                   Segment("u", "user", "Again")])
    usage2, _ = engine.run(nxt, 1)
    assert usage2.cache_read_input_tokens == covered
    warmth = compiler.warmth(nxt, PrefixCache(1800), 1)
    assert warmth.cache_creation_tokens <= compiler.compile(nxt).marker_prefixes[-1][2]


def test_long_conversations_compile_in_near_linear_time():
    compiler = OpenAICompiler("gpt-5.6")

    def ctx(n):
        return Context([Segment("s", "system", "Rule " * 100),
                        *[Segment(f"h{i}", "history", [{"role": "user", "content": f"{i} q " * 20},
                                                       {"role": "assistant", "content": "a " * 200}])
                          for i in range(n)], Segment("u", "user", "Next")])
    timings = []
    for n in (200, 800):
        c = ctx(n)
        started = perf_counter()
        compiler.compile(c)
        timings.append(perf_counter() - started)
    # Rebuilding every prefix grows about 16x from 200 to 800 segments; allow generous machine noise.
    assert timings[1] < 8 * timings[0] + 0.5

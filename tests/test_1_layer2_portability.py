"""Falsification 1 (SPEC.md "Cache and routing"): the Layer 2 portability guarantee on an A/B/A session.

Three assertions, each catching a different bug class:
  (a) cold tokens billed == oracle on EVERY call     -> cross-family leakage, TTL bugs, accounting drift
  (b) each stable segment billed cold <= 1x per family within a TTL window -> bad breakpoint placement
  (c) both compiled requests carry every tool and every history turn -> state dropped in translation

The oracle re-derives the SPEC cache rule with its own dict; it does not import PrefixCache. Mutation guards at the
bottom inject one bug each and assert the corresponding assertion FAILS, so the test has teeth.
"""
from __future__ import annotations

import pytest
from conftest import turn

from pcf import Context, PrefixCache, Segment
from pcf.compiler import choose_breakpoints
from pcf.families.sim import SimEngine, family_a, family_b

TTL = 300


class Oracle:
    """SPEC.md "Cache and routing", written independently: longest live cached prefix; writes at breakpoints >= min."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], tuple[int, float]] = {}  # (family, prefix) -> (cum, expires)

    def expected_cold_then_record(self, family: str, chain: list[str], cum: list[int], breakpoints: list[int],
                                  min_cacheable: int, ttl: float, now: float) -> int:
        longest = -1
        for i in range(len(chain) - 1, -1, -1):
            e = self.entries.get((family, chain[i]))
            if e is not None and e[1] > now:  # live while now < expires_at
                longest = i
                self.entries[(family, chain[i])] = (e[0], now + ttl)  # read refreshes TTL
                break
        warm = cum[longest] if longest >= 0 else 0
        for b in breakpoints:
            if b > longest and cum[b] >= min_cacheable:
                self.entries[(family, chain[b])] = (cum[b], now + ttl)
        return cum[-1] - warm


def run_session(engines: dict[str, SimEngine], base: Context, schedule: list[tuple[str, float]]):
    """schedule: (family, now) per turn. Returns per-call records for assertions."""
    oracle = Oracle()
    ctx = base
    records = []
    prev_assistant = None
    for n, (fam, now) in enumerate(schedule, start=1):
        ctx = turn(ctx, n, f"Question {n}: where is order {1000 + n}?", prev_assistant)
        eng = engines[fam]
        usage, compiled = eng.run(ctx, now)
        d = eng.descriptor
        expected = oracle.expected_cold_then_record(
            fam, compiled.chain, compiled.cum_tokens, compiled.breakpoints, d.min_cacheable_tokens, d.ttl_seconds, now)
        records.append((fam, now, ctx, compiled, usage, expected))
        prev_assistant = f"Order {1000 + n} ships Tuesday."
    return records


def cold_stable_segments(ctx: Context, compiled, usage):
    """Which stable segments were billed cold on this call (derived from warm tokens, not from the cache)."""
    cum = compiled.cum_tokens
    warm = usage.cache_read_input_tokens
    hit = cum.index(warm) if warm else -1
    return [s for i, s in enumerate(ctx.segments) if i > hit and s.stable]


SCHEDULE = [("A", 0.0), ("A", 10.0), ("B", 20.0), ("B", 30.0), ("A", 40.0), ("B", 50.0), ("A", 60.0)]


def _engines(store_a=None, store_b=None):
    return {"A": family_a(store_a, min_cacheable=32, ttl_seconds=TTL), "B": family_b(store_b, min_cacheable=32, ttl_seconds=TTL)}


def test_cold_tokens_equal_oracle_on_every_call(base_ctx):
    for fam, now, ctx, compiled, usage, expected in run_session(_engines(), base_ctx, SCHEDULE):
        assert usage.cold_tokens == expected, f"{fam}@{now}: billed {usage.cold_tokens} cold, spec says {expected}"
        assert usage.total_input_tokens == compiled.total_tokens


def test_first_call_to_cold_family_is_full_prefill_and_nothing_after_is(base_ctx):
    """Physics: the first call to B is fully cold. The guarantee is on every call after."""
    recs = run_session(_engines(), base_ctx, SCHEDULE)
    first_b = next(r for r in recs if r[0] == "B")
    assert first_b[4].cache_read_input_tokens == 0 and first_b[4].cold_tokens == first_b[3].total_tokens
    for fam, now, ctx, compiled, usage, _ in recs[1:]:
        if (fam, now) == (first_b[0], first_b[1]):
            continue
        assert usage.cache_read_input_tokens > 0, f"{fam}@{now} should have read something from cache"


def test_each_stable_segment_billed_cold_at_most_once_per_family(base_ctx):
    seen: dict[tuple[str, str], int] = {}
    for fam, now, ctx, compiled, usage, _ in run_session(_engines(), base_ctx, SCHEDULE):
        for seg in cold_stable_segments(ctx, compiled, usage):
            seen[(fam, seg.hash)] = seen.get((fam, seg.hash), 0) + 1
    repeats = {k: v for k, v in seen.items() if v > 1}
    assert not repeats, f"stable segments billed cold more than once within TTL: {repeats}"


def test_ttl_expiry_rebills_and_oracle_agrees(base_ctx):
    schedule = [("A", 0.0), ("A", 10.0), ("A", 10.0 + TTL)]  # third call lands exactly on expiry
    recs = run_session(_engines(), base_ctx, schedule)
    assert recs[2][4].cache_read_input_tokens == 0, "expired entries must not be read"
    for _, _, _, _, usage, expected in recs:
        assert usage.cold_tokens == expected


def test_compiled_requests_preserve_tools_and_history(base_ctx):
    recs = run_session(_engines(), base_ctx, SCHEDULE)
    _, _, ctx, _, _, _ = recs[-1]
    tool_names = {t["name"] for s in ctx.segments if s.kind == "tools" for t in s.content}
    history_turns = [t["content"] for s in ctx.segments if s.kind == "history" for t in s.content]
    assert len(history_turns) == 2 * (len(SCHEDULE) - 1)
    for eng in _engines().values():
        req = eng.compiler.compile(ctx).request
        assert {t["name"] for t in req["tools"]} == tool_names
        rendered = [m["content"] for m in req["messages"]]
        kept = [c for c in rendered if c in history_turns]
        assert kept == history_turns, f"{eng.descriptor.family} dropped or reordered history"
        assert req == eng.compiler.compile(ctx).request, "compile must be deterministic"


def test_empty_segments_bill_nothing(base_ctx):
    ctx = turn(base_ctx, 1, "hi", None)
    padded = Context((*ctx.segments[:3], Segment("h0", "history", []), ctx.segments[3]), session_id=ctx.session_id)
    for eng in _engines().values():
        a, b = eng.compiler.compile(ctx), eng.compiler.compile(padded)
        assert b.segment_tokens[3] == 0 and b.total_tokens == a.total_tokens, "an empty segment renders nothing"


# ---------------------------------------------------------------- mutation guards: the tests must be able to fail


class _StoreIgnoringCompatKey(PrefixCache):
    def write(self, compat_key, prefix_hash, cum_tokens, now, **kwargs):
        from pcf import sha256_tag
        super().write(sha256_tag("mutant-shared"), prefix_hash, cum_tokens, now, **kwargs)

    def peek(self, compat_key, chain, now, **kwargs):
        from pcf import sha256_tag
        return super().peek(sha256_tag("mutant-shared"), chain, now, **kwargs)


def test_mutation_cross_family_leak_is_caught(base_ctx):
    shared = _StoreIgnoringCompatKey(ttl_seconds=TTL)
    with pytest.raises(AssertionError, match="spec says"):
        for fam, now, ctx, compiled, usage, expected in run_session(_engines(shared, shared), base_ctx, SCHEDULE):
            assert usage.cold_tokens == expected, f"{fam}@{now}: billed {usage.cold_tokens} cold, spec says {expected}"


class _StoreNeverExpiring(PrefixCache):
    def write(self, compat_key, prefix_hash, cum_tokens, now, **kwargs):
        kwargs["ttl_seconds"] = 1e12
        super().write(compat_key, prefix_hash, cum_tokens, now, **kwargs)


def test_mutation_ignored_ttl_is_caught(base_ctx):
    schedule = [("A", 0.0), ("A", 10.0), ("A", 10.0 + TTL + 1)]
    engines = {"A": family_a(_StoreNeverExpiring(TTL), min_cacheable=32, ttl_seconds=TTL), "B": family_b()}
    with pytest.raises(AssertionError):
        for _, _, _, _, usage, expected in run_session(engines, base_ctx, schedule):
            assert usage.cold_tokens == expected


def test_mutation_breakpoint_on_varying_block_is_caught(base_ctx, monkeypatch):
    """The documented common mistake: breakpoint on the block that changes every request."""
    import pcf.compiler as compiler_mod

    global _original_compile
    _original_compile = compiler_mod.ContextCompiler.compile
    monkeypatch.setattr(compiler_mod.ContextCompiler, "compile", _compile_with_breakpoint_on_last_segment)
    seen: dict[tuple[str, str], int] = {}
    for fam, now, ctx, compiled, usage, _ in run_session(_engines(), base_ctx, SCHEDULE):
        for seg in cold_stable_segments(ctx, compiled, usage):
            seen[(fam, seg.hash)] = seen.get((fam, seg.hash), 0) + 1
    assert any(v > 1 for v in seen.values()), "mutation not detected: stable segments should be re-billed"


def _compile_with_breakpoint_on_last_segment(self, ctx):
    from dataclasses import replace
    from pcf.segments import canonical_bytes
    compiled = _original_compile(self, ctx)
    bp = [len(ctx.segments) - 1]
    return replace(compiled, request_json=canonical_bytes(self.render(ctx, bp)), breakpoints=tuple(bp))


def test_breakpoints_never_land_on_unstable_segments(base_ctx):
    ctx = turn(base_ctx, 1, "hello", None)
    for bp in choose_breakpoints(ctx, 4):
        assert ctx.segments[bp].stable
    assert (len(ctx.segments) - 1) not in choose_breakpoints(ctx, 4)

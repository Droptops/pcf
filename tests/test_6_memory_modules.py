"""Memory modules: memory segments sharing a provenance get their own breakpoint anchor.

Editing a later module must keep earlier modules warm; editing an earlier module must not (the prefix chain
still invalidates everything after a change). Memory without provenance keeps the single shared anchor.
"""
from __future__ import annotations

import pytest

from pcf import Context, Segment
from pcf.compiler import choose_breakpoints
from pcf.families.sim import family_a

SYSTEM = " ".join(f"rule{i}" for i in range(80))


def _module(name: str, version: int = 0) -> str:
    return " ".join(f"{name}{i}v{version}" for i in range(100))


def _ctx(modules: bool, edit: str | None = None) -> Context:
    mem = [Segment(f"m{name}", "memory", _module(name, int(name == edit)),
                   provenance=name if modules else None) for name in "abc"]
    return Context([Segment("s", "system", SYSTEM), *mem, Segment("u", "user", "hi", stable=False)])


def _cold_after_edit(modules: bool, edit: str) -> tuple[int, list[int]]:
    engine = family_a()
    engine.run(_ctx(modules), 0)
    usage, compiled = engine.run(_ctx(modules, edit), 10)
    return usage.cold_tokens, list(compiled.segment_tokens)


def test_modules_get_their_own_anchors_and_plain_memory_does_not():
    assert choose_breakpoints(_ctx(modules=True), 4) == [0, 1, 2, 3]
    assert choose_breakpoints(_ctx(modules=False), 4) == [0, 3]


def test_editing_the_last_module_keeps_earlier_modules_warm():
    cold, tokens = _cold_after_edit(modules=True, edit="c")
    assert cold == tokens[3] + tokens[4]  # module c and the user turn only
    baseline, _ = _cold_after_edit(modules=False, edit="c")
    assert baseline == sum(tokens[1:])  # a, b, c and the user turn


def test_editing_the_first_module_still_invalidates_everything_after_it():
    cold, tokens = _cold_after_edit(modules=True, edit="a")
    assert cold == sum(tokens[1:])
    assert cold == _cold_after_edit(modules=False, edit="a")[0]


def _history(n: int) -> Segment:
    return Segment(f"h{n}", "history", [{"role": "user", "content": f"question {n} " * 30},
                                        {"role": "assistant", "content": f"answer {n} " * 30}])


def _tail_ctx(version: int) -> Context:
    return Context([Segment("s", "system", SYSTEM), _history(0), _history(1),
                    Segment("live", "memory", _module("c", version), stable=False),
                    Segment("u", "user", "hi", stable=False)])


def test_tail_memory_is_allowed_only_before_user_turns():
    _tail_ctx(0)
    for bad in ([_history(0), Segment("m", "memory", "x"), _history(1)],
                [_history(0), Segment("u", "user", "hi"), Segment("m", "memory", "x")],
                [_history(0), Segment("m", "memory", "x"), Segment("d", "document", "y")]):
        with pytest.raises(ValueError, match="order"):
            Context([Segment("s", "system", SYSTEM), *bad])


def test_editing_tail_memory_keeps_history_warm():
    engine = family_a()
    engine.run(_tail_ctx(0), 0)
    usage, compiled = engine.run(_tail_ctx(1), 10)
    assert usage.cold_tokens == compiled.segment_tokens[3] + compiled.segment_tokens[4]


def test_tail_memory_renders_as_one_user_message_with_the_turn():
    from pcf.families.anthropic_adapter import AnthropicCompiler
    from pcf.families.openai_adapter import OpenAICompiler
    messages = AnthropicCompiler("claude-sonnet-5").compile(_tail_ctx(0)).request["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert [b["text"] for b in messages[-1]["content"]][-1] == "hi"
    inputs = OpenAICompiler("gpt-5.6").compile(_tail_ctx(0)).request["input"]
    assert '"source":null' in inputs[-2]["content"][0]["text"] and inputs[-1]["content"][0]["text"] == "hi"


def _placed_session(edits: list[int], place: bool, cold_at: tuple[int, ...] = (),
                    report_cold: bool = True, **options) -> tuple[int, list[str]]:
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    engine, placer, history, total, tail, now = family_a(), MemoryPlacer(WordTokenizer(), **options), [], 0, [], 0
    for n, version in enumerate(edits):
        now += 1000 if n in cold_at else 10  # 1000s outlives the 300s TTL
        memory = [Segment("ma", "memory", _module("a")), Segment("mc", "memory", _module("c", version))]
        front, tail = placer.split(memory, history, cold=report_cold and n in cold_at) if place else (memory, [])
        ctx = Context([Segment("s", "system", SYSTEM), *front, *history, *tail, Segment("u", "user", "hi", stable=False)])
        total += engine.run(ctx, now)[0].cold_tokens
        history.append(_history(n))
    return total, [s.id for s in tail]


def test_placer_moves_only_the_changing_module_and_saves_tokens():
    busy = list(range(10))
    cost, tail = _placed_session(busy, place=True)
    assert tail == ["mc"]
    assert cost < _placed_session(busy, place=False)[0]


def test_placer_keeps_stable_memory_in_front_at_no_extra_cost():
    quiet = [0] * 10
    assert _placed_session(quiet, place=True) == (_placed_session(quiet, place=False)[0], [])


def test_placer_keeps_changing_memory_in_front_without_history_and_rejects_other_kinds():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    placer = MemoryPlacer(WordTokenizer())
    for version in range(3):
        front, tail = placer.split([Segment("mc", "memory", _module("c", version))], [])
        assert tail == []  # with no history after it, the front never costs more than the tail
    with pytest.raises(ValueError, match="memory"):
        placer.split([Segment("d", "document", "x")], [])


BUSY_THEN_QUIET = [*range(8), *[7] * 16]


PLAIN = {"write_multiplier": 1.0, "read_multiplier": 0.0}  # the plain token rule


def test_quiet_module_returns_to_front_when_the_cache_is_cold():
    warm_cost, warm_tail = _placed_session(BUSY_THEN_QUIET, place=True, **PLAIN)
    assert warm_tail == ["mc"]  # 16 quiet turns do not repay re-billing the history
    cold_cost, cold_tail = _placed_session(BUSY_THEN_QUIET, place=True, cold_at=(8, 16), **PLAIN)
    assert cold_tail == []
    unreported, unreported_tail = _placed_session(BUSY_THEN_QUIET, place=True, cold_at=(8, 16), report_cold=False,
                                                  **PLAIN)
    assert unreported_tail == ["mc"] and cold_cost < unreported
    assert cold_cost < _placed_session(BUSY_THEN_QUIET, place=False, cold_at=(8, 16))[0]


LONG_QUIET = [*range(8), *[7] * 60]


def test_quiet_module_returns_while_warm_when_the_conversation_will_last():
    sticky, sticky_tail = _placed_session(LONG_QUIET, place=True)
    returned, returned_tail = _placed_session(LONG_QUIET, place=True, expected_turns=100)
    assert sticky_tail == ["mc"] and returned_tail == []
    assert returned < sticky
    periodic = [n // 6 for n in range(60)]  # a steady period is never quiet long enough to bounce
    assert _placed_session(periodic, place=True, expected_turns=60) == _placed_session(periodic, place=True)


def test_decay_must_be_a_fraction():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    for bad in (1, -0.1, float("nan"), True):
        with pytest.raises(ValueError):
            MemoryPlacer(WordTokenizer(), decay=bad)


def test_tail_is_sticky_while_the_cache_is_warm():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    placer, history = MemoryPlacer(WordTokenizer()), [_history(n) for n in range(5)]
    versions = [0, 1] + [1] * 10  # one change, then quiet: the lifetime rate falls below the threshold
    tails = [placer.split([Segment("mc", "memory", _module("c", v))], history)[1] for v in versions]
    assert tails[1] and all(tails[1:])
    assert placer.split([Segment("mc", "memory", _module("c", 1))], history, cold=True) == (
        [Segment("mc", "memory", _module("c", 1))], [])


def test_cache_prices_shift_the_placement_threshold():
    from pcf.placement import MemoryPlacer

    class XTokenizer:  # one token per "x", so memory and history sizes are exact
        tokenizer_hash, is_estimate = "sha256:" + "0" * 64, True

        def count(self, text):
            return text.count("x")

    history = [Segment("h", "history", [{"role": "user", "content": "x" * 90}])]

    def tail_after_one_change(**prices):  # p = 1/2 after two looks; m = 100, H = 90
        placer = MemoryPlacer(XTokenizer(), **prices)
        return [placer.split([Segment("m", "memory", "x" * 100 + v)], history)[1] != [] for v in "ab"][-1]

    assert not tail_after_one_change(**PLAIN)  # 0.5 * 190 = 95 < 100: plain token rule keeps it in front
    assert tail_after_one_change(write_multiplier=1.25, read_multiplier=0.1)  # 0.5 * 1.15 * 190 > 0.9 * 100
    with pytest.raises(ValueError):
        MemoryPlacer(XTokenizer(), write_multiplier=0.1, read_multiplier=0.1)


def test_placer_takes_prices_from_a_router_candidate():
    from pcf.cache import PrefixCache
    from pcf.families.openai_adapter import OpenAICompiler
    from pcf.placement import MemoryPlacer
    from pcf.router import Candidate
    compiler = OpenAICompiler("gpt-5.6")
    placer = MemoryPlacer.for_candidate(Candidate(compiler, PrefixCache(1800), 2.0, 0.2, is_fallback=True),
                                        expected_turns=40)
    assert placer.tokenizer is compiler.tokenizer
    assert (placer.write_multiplier, placer.read_multiplier) == (1.25, 0.1)
    assert placer.min_cacheable_tokens == compiler.descriptor.min_cacheable_tokens
    assert placer.expected_turns == 40
    with pytest.raises(ValueError):
        MemoryPlacer.for_candidate(Candidate(compiler, PrefixCache(1800), 0.0, 0.0, is_fallback=True))



def test_placer_keeps_instruction_memory_in_front_and_tail_copies_keep_authority():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    placer, history = MemoryPlacer(WordTokenizer()), [_history(n) for n in range(5)]
    for version in range(4):
        rules = Segment("rules", "memory", _module("r", version), authority="instruction")
        data = Segment("mc", "memory", _module("c", version))
        front, tail = placer.split([rules, data], history)
    assert [s.id for s in front] == ["rules"] and [s.id for s in tail] == ["mc"]
    assert front[0].authority == "instruction" and tail[0].authority == "data" and not tail[0].stable
    Context([Segment("s", "system", "sys"), *front, *history, *tail, Segment("u", "user", "hi", stable=False)])


def test_placer_counts_later_front_modules_in_what_a_change_rebills():
    from pcf.placement import MemoryPlacer

    class XTokenizer:
        tokenizer_hash, is_estimate = "sha256:" + "0" * 64, True

        def count(self, text):
            return text.count("x")

    placer = MemoryPlacer(XTokenizer())
    for v in "ab":  # cart changes once (p = 1/2); a large stable catalog follows it; no history yet
        cart = Segment("cart", "memory", "x" * 300 + v)
        catalog = Segment("catalog", "memory", "x" * 3000)
        front, tail = placer.split([cart, catalog], [])
    # 0.5 * (300 + 3000) > 300: moving the cart saves re-billing the catalog behind it.
    assert [s.id for s in front] == ["catalog"] and [s.id for s in tail] == ["cart"]


def test_placer_anchors_the_first_front_module_on_the_turn_a_module_moves():
    from pcf.families.openai_adapter import OpenAICompiler
    from pcf.placement import MemoryPlacer
    compiler = OpenAICompiler("gpt-5.6")
    placer = MemoryPlacer(compiler.tokenizer, write_multiplier=1.25, read_multiplier=.1)
    history = [Segment("h", "history", [{"role": "user", "content": "q " * 3000},
                                        {"role": "assistant", "content": "a"}])]

    def memory(turn):
        return [Segment("ref", "memory", "reference " * 800, provenance="ref"),
                Segment("plan", "memory", "plan basic", provenance="plan"),
                Segment("bal", "memory", f"balance {turn}", provenance="bal")]
    turns = [placer.split(memory(turn), history) for turn in range(6)]
    move = next(i for i, (_, tail) in enumerate(turns) if tail)
    front, tail = turns[move]
    assert [s.id for s in tail] == ["bal"]
    assert [(s.id, s.stable) for s in front] == [("ref", True), ("plan", False)]
    assert all(s.stable for front, _ in turns[:move] + turns[move + 1:] for s in front)
    ctx = Context([Segment("s", "system", "policy"), *front, *history, *tail, Segment("u", "user", "q", stable=False)])
    assert "ref" in [ctx.segments[i].id for i in compiler.compile(ctx).breakpoints]


def test_placer_keeps_the_last_front_anchor_when_modules_are_only_appended():
    from pcf.families.openai_adapter import OpenAICompiler
    from pcf.placement import MemoryPlacer
    placer = MemoryPlacer(OpenAICompiler("gpt-5.6").tokenizer, write_multiplier=1.25, read_multiplier=.1)
    base = [Segment("ref", "memory", "reference " * 800, provenance="ref"),
            Segment("plan", "memory", "plan basic", provenance="plan")]
    placer.split(base, [])
    front, _ = placer.split([*base, Segment("new", "memory", "new module", provenance="new")], [])
    assert [(s.id, s.stable) for s in front] == [("ref", True), ("plan", True), ("new", False)]
    front, _ = placer.split([*base, Segment("new", "memory", "new module", provenance="new")], [])
    assert all(s.stable for s in front)


def test_placer_takes_prices_and_minimum_from_a_compiler_and_does_not_move_uncacheable_memory():
    from pcf.families.anthropic_adapter import AnthropicCompiler
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    placer = MemoryPlacer.for_compiler(AnthropicCompiler("claude-sonnet-5"), expected_turns=30)
    assert (placer.write_multiplier, placer.read_multiplier, placer.expected_turns) == (1.25, 0.1, 30)
    assert placer.min_cacheable_tokens > 0
    small = MemoryPlacer(WordTokenizer(), min_cacheable_tokens=10_000)
    history = [_history(n) for n in range(5)]
    tails = [small.split([Segment("mc", "memory", _module("c", v))], history)[1] for v in range(6)]
    assert not any(tails)  # below the minimum nothing is cached, so moving saves nothing


def test_placement_turn_is_idempotent_and_rejects_changed_retry_inputs():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    placer = MemoryPlacer(WordTokenizer())
    history = [_history(0)]
    memory = [Segment("mc", "memory", _module("c", 0))]
    first = placer.split(memory, history, turn_id="request-0", expected_revision=0)
    state = placer.export_state()
    assert placer.split(memory, history, turn_id="request-0", expected_revision=0) == first
    assert placer.export_state() == state and placer.revision == 1
    with pytest.raises(ValueError, match="different placement inputs"):
        placer.split([Segment("mc", "memory", _module("c", 1))], history, turn_id="request-0")


def test_placement_state_survives_json_round_trip_and_delayed_retry():
    import json
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    tokenizer, history = WordTokenizer(), [_history(n) for n in range(4)]
    original = MemoryPlacer(tokenizer, expected_turns=30)
    inputs = [[Segment("mc", "memory", _module("c", v))] for v in (0, 1, 2)]
    decisions = [original.split(mem, history, turn_id=f"request-{n}", expected_revision=n)
                 for n, mem in enumerate(inputs)]
    saved = json.loads(json.dumps(original.export_state()))
    restarted = MemoryPlacer.from_state(tokenizer, saved)
    assert restarted.export_state() == saved and restarted.revision == 3
    before = restarted.export_state()
    assert restarted.split(inputs[0], history, turn_id="request-0") == decisions[0]
    assert restarted.export_state() == before  # a delayed retry is not a new observation
    next_memory = [Segment("mc", "memory", _module("c", 3))]
    assert restarted.split(next_memory, history, turn_id="request-3", expected_revision=3) == \
        original.split(next_memory, history, turn_id="request-3", expected_revision=3)
    assert restarted.export_state() == original.export_state()


def test_expected_revision_rejects_one_of_two_concurrent_turns():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pcf.families.sim import WordTokenizer
    from pcf.placement import ConcurrentPlacementUpdate, MemoryPlacer
    placer, barrier = MemoryPlacer(WordTokenizer()), threading.Barrier(2)
    history = [_history(0)]

    def attempt(n):
        barrier.wait()
        try:
            placer.split([Segment("mc", "memory", _module("c", n))], history,
                         turn_id=f"request-{n}", expected_revision=0)
            return "committed"
        except ConcurrentPlacementUpdate:
            return "conflict"

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(attempt, (0, 1)))
    assert sorted(results) == ["committed", "conflict"]
    assert placer.revision == 1 and placer.export_state()["turns"] == 1


def test_placement_state_rejects_configuration_drift():
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    state = MemoryPlacer(WordTokenizer()).export_state()
    with pytest.raises(ValueError, match="configuration"):
        MemoryPlacer(WordTokenizer(), decay=0.5).restore_state(state)

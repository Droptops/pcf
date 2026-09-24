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


def _placed_session(edits: list[int], place: bool) -> tuple[int, list[str]]:
    from pcf.families.sim import WordTokenizer
    from pcf.placement import MemoryPlacer
    engine, placer, history, total, tail = family_a(), MemoryPlacer(WordTokenizer()), [], 0, []
    for n, version in enumerate(edits):
        memory = [Segment("ma", "memory", _module("a")), Segment("mc", "memory", _module("c", version))]
        front, tail = placer.split(memory, history) if place else (memory, [])
        ctx = Context([Segment("s", "system", SYSTEM), *front, *history, *tail, Segment("u", "user", "hi", stable=False)])
        total += engine.run(ctx, n * 10)[0].cold_tokens
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

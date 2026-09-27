"""Regression tests for the September red-team findings."""
from __future__ import annotations

import json

import pytest

from pcf import Context, Segment
from pcf.families.anthropic_adapter import AnthropicCompiler
from pcf.families.openai_adapter import OpenAICompiler
from pcf.placement import ConcurrentPlacementUpdate, MemoryPlacer


class XTokenizer:
    tokenizer_hash = "sha256:" + "0" * 64
    is_estimate = True

    def count(self, text):
        return text.count("x")


def test_cache_minimum_counts_the_lead_before_memory():
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 90}])]
    memory0 = [Segment("m", "memory", "x" * 100 + "a")]
    memory1 = [Segment("m", "memory", "x" * 100 + "b")]

    eligible = MemoryPlacer(XTokenizer(), min_cacheable_tokens=1000)
    eligible.split(memory0, history, prefix_tokens=950)
    assert [s.id for s in eligible.split(memory1, history, prefix_tokens=950)[1]] == ["m"]

    ineligible = MemoryPlacer(XTokenizer(), min_cacheable_tokens=1000)
    ineligible.split(memory0, history, prefix_tokens=0)
    assert ineligible.split(memory1, history, prefix_tokens=0)[1] == []


def test_cache_minimum_counts_the_history_a_change_rebills():
    # 300 + 50k history is far above the minimum: a module that changes every turn must not pin the history
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 50_000}])]
    placer = MemoryPlacer(XTokenizer(), min_cacheable_tokens=1024)
    tails = [placer.split([Segment("m", "memory", "x" * 300 + str(turn))], history)[1] for turn in range(4)]
    assert [[s.id for s in tail] for tail in tails] == [[], ["m"], ["m"], ["m"]]


def test_cache_minimum_band_keeps_a_module_in_front_when_the_tail_leaves_the_prompt_uncached():
    # 2500 + 1200 < 4096 <= 2500 + 1200 + 600: in the tail nothing is cached and the lead, history and module are
    # billed uncached every turn; in front the prompt is cached, and a change every 3rd turn costs less
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 1200}])]
    placer = MemoryPlacer(XTokenizer(), min_cacheable_tokens=4096)
    tails = [placer.split([Segment("m", "memory", "x" * 600 + str(turn // 3))], history, prefix_tokens=2500)[1]
             for turn in range(30)]
    assert tails == [[]] * 30


def test_cache_minimum_band_lets_a_module_that_changes_every_turn_go_to_the_tail():
    # once its decayed rate p has p * (w - r) > 1 - r, cache writes cost more than the uncached prompt
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 1200}])]
    placer = MemoryPlacer(XTokenizer(), min_cacheable_tokens=4096)
    tails, rates = [], []
    for turn in range(10):
        memory = [Segment("m", "memory", "x" * 600 + str(turn))]
        tails.append([s.id for s in placer.split(memory, history, prefix_tokens=2500)[1]])
        rates.append(placer.export_state()["seen"]["m"]["rate"])
    assert tails == [[]] * 5 + [["m"]] * 5
    assert rates[4] * (1.25 - 0.1) <= 1 - 0.1 < rates[5] * (1.25 - 0.1)


def test_cache_minimum_keeps_a_module_outside_the_band_in_the_tail_in_either_order():
    # i (1500 tokens) changes every turn, j (1700) every other turn; history 1400, minimum 4096. In one pass with i
    # first, i went back to the front because 1400 + 1500 stays below the minimum, then j lifted the prompt to 4600
    # and i's changes rewrote it every turn; with j first, i stayed in the tail. Now j goes back first, i would lift
    # 1400 + 1700 to the minimum, and i stays in the tail in either order once its rate leaves the band.
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 1400}])]
    tails = {}
    for order in ("ij", "ji"):
        placer = MemoryPlacer(XTokenizer(), min_cacheable_tokens=4096)
        tails[order] = []
        for turn in range(20):
            modules = {"i": Segment("i", "memory", "x" * 1500 + str(turn)),
                       "j": Segment("j", "memory", "x" * 1700 + str(turn // 2))}
            tails[order].append([s.id for s in placer.split([modules[k] for k in order], history)[1]])
    assert tails["ij"] == tails["ji"] == [[]] * 5 + [["i"]] * 15


def test_atomic_snapshot_cannot_be_advanced_by_a_later_turn():
    placer = MemoryPlacer(XTokenizer())
    memory = [Segment("m", "memory", "x" * 10)]
    _, _, snapshot = placer.split_and_snapshot(memory, [], turn_id="t0", expected_revision=0)
    placer.split(memory, [], turn_id="t1", expected_revision=1)
    assert snapshot["revision"] == 1
    assert snapshot["turns"] == 1
    assert set(snapshot["decisions"]) == {"t0"}
    assert placer.export_state()["revision"] == 2


def test_idempotency_history_is_bounded():
    placer = MemoryPlacer(XTokenizer(), max_idempotency_entries=2)
    memory = [Segment("m", "memory", "x" * 10)]
    for i in range(3):
        placer.split(memory, [], turn_id=f"t{i}", expected_revision=i)
    assert list(placer.export_state()["decisions"]) == ["t1", "t2"]
    with pytest.raises(ValueError, match="requires expected_revision"):
        placer.split(memory, [], turn_id="t0")


def test_retry_eviction_after_restore_removes_the_oldest_decision():
    memory = [Segment("m", "memory", "x" * 10)]
    placer = MemoryPlacer(XTokenizer(), max_idempotency_entries=3)
    first = {turn_id: placer.split(memory, [], turn_id=turn_id, expected_revision=n)
             for n, turn_id in enumerate(("r9", "r10", "r11"))}  # age order differs from id order
    restored = MemoryPlacer.from_state(XTokenizer(), json.loads(json.dumps(placer.export_state())))
    restored.split(memory, [], turn_id="r12", expected_revision=3)
    assert set(restored.export_state()["decisions"]) == {"r10", "r11", "r12"}
    before = restored.export_state()
    assert restored.split(memory, [], turn_id="r10") == first["r10"]
    assert restored.export_state() == before
    with pytest.raises(ConcurrentPlacementUpdate):
        restored.split(memory, [], turn_id="r9", expected_revision=0)


def test_restore_rejects_missing_or_duplicate_decision_revisions():
    memory = [Segment("m", "memory", "x" * 10)]
    placer = MemoryPlacer(XTokenizer())
    for n in range(2):
        placer.split(memory, [], turn_id=f"t{n}", expected_revision=n)
    missing, duplicate = (json.loads(json.dumps(placer.export_state())) for _ in range(2))
    del missing["decisions"]["t0"]["revision"]
    duplicate["decisions"]["t0"]["revision"] = duplicate["decisions"]["t1"]["revision"]
    with pytest.raises(ValueError, match="invalid stored placement decision"):
        MemoryPlacer(XTokenizer()).restore_state(missing)
    with pytest.raises(ValueError, match="unique"):
        MemoryPlacer(XTokenizer()).restore_state(duplicate)


def test_restore_rejects_more_decisions_than_the_cap():
    memory = [Segment("m", "memory", "x" * 10)]
    placer = MemoryPlacer(XTokenizer(), max_idempotency_entries=2)
    for n in range(3):
        placer.split(memory, [], turn_id=f"t{n}", expected_revision=n)
    state = json.loads(json.dumps(placer.export_state()))
    MemoryPlacer.from_state(XTokenizer(), state)
    state["decisions"]["t0"] = {**state["decisions"]["t1"], "revision": 1}  # the decision eviction removed
    with pytest.raises(ValueError, match="exceed max_idempotency_entries"):
        MemoryPlacer.from_state(XTokenizer(), state)


def test_openai_cache_affecting_settings_are_compiled_and_fingerprinted():
    ctx = Context([Segment("s", "system", "x" * 1200), Segment("u", "user", "hi", stable=False)])
    low = OpenAICompiler("gpt-5.6", reasoning={"effort": "low"}, text={"verbosity": "low"})
    high = OpenAICompiler("gpt-5.6", reasoning={"effort": "high"}, text={"verbosity": "low"})
    request = low.compile(ctx).request
    assert request["reasoning"] == {"effort": "low"}
    assert request["text"] == {"verbosity": "low"}
    assert low.candidate_fingerprint != high.candidate_fingerprint


def test_default_openai_context_does_not_force_a_shared_cache_partition():
    default = Context([Segment("s", "system", "x" * 1200), Segment("u", "user", "hi", stable=False)])
    tenant = Context(default.segments, cache_namespace="tenant-123")
    compiler = OpenAICompiler("gpt-5.6")
    assert "prompt_cache_key" not in compiler.compile(default).request
    assert compiler.compile(tenant).request["prompt_cache_key"].startswith("pcf-")


def test_model_specific_cache_read_prices_feed_placement():
    assert MemoryPlacer.for_compiler(AnthropicCompiler("claude-fable-5-1")).read_multiplier == .025
    assert MemoryPlacer.for_compiler(AnthropicCompiler("claude-sonnet-5")).read_multiplier == .1
    with pytest.raises(ValueError, match="cache-read multiplier"):
        MemoryPlacer.for_compiler(OpenAICompiler("gpt-5.5"))

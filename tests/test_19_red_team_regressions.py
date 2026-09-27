"""Regression tests for the September red-team findings."""
from __future__ import annotations

import pytest

from pcf import Context, Segment
from pcf.families.anthropic_adapter import AnthropicCompiler
from pcf.families.openai_adapter import OpenAICompiler
from pcf.placement import MemoryPlacer


class XTokenizer:
    tokenizer_hash = "sha256:" + "0" * 64
    is_estimate = True

    def count(self, text):
        return text.count("x")


def test_cache_minimum_uses_prefix_before_module_not_history_suffix():
    history = [Segment("h", "history", [{"role": "user", "content": "x" * 90}])]
    memory0 = [Segment("m", "memory", "x" * 100 + "a")]
    memory1 = [Segment("m", "memory", "x" * 100 + "b")]

    eligible = MemoryPlacer(XTokenizer(), min_cacheable_tokens=1000)
    eligible.split(memory0, history, prefix_tokens=950)
    assert [s.id for s in eligible.split(memory1, history, prefix_tokens=950)[1]] == ["m"]

    ineligible = MemoryPlacer(XTokenizer(), min_cacheable_tokens=1000)
    ineligible.split(memory0, history, prefix_tokens=0)
    assert ineligible.split(memory1, history, prefix_tokens=0)[1] == []


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

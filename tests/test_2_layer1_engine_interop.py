"""Falsification 2 (SPEC.md "Cache and routing"): same weights on two engines sharing a store must bill exactly like
one engine alone. Measured by tokens billed, never by an engine's own hit counter.

Mutation guard: a descriptor that folds the engine name into compat_key makes the alternating session
bill more than the baseline, and the equality assertion fails.
"""
from __future__ import annotations

import hashlib

import pytest
from conftest import turn

from pcf import CacheDescriptor, PrefixCache
from pcf.families.sim import SimCompiler, SimEngine, WordTokenizer, sim_descriptor

TURNS = 10


def _session(engines: list[SimEngine], base):
    ctx, prev, cold_total, calls = base, None, 0, []
    for n in range(1, TURNS + 1):
        ctx = turn(ctx, n, f"Turn {n}: status of order {n}?", prev)
        usage, _ = engines[n % len(engines)].run(ctx, now=float(n))
        cold_total += usage.cold_tokens
        calls.append(usage)
        prev = f"Order {n} is on the truck."
    return cold_total, calls


def _same_model_two_engines(store: PrefixCache):
    tok = WordTokenizer()
    d = sim_descriptor("sim-a", "sim-a-large", tok.tokenizer_hash, min_cacheable=32)
    comp = SimCompiler(d, tok)
    e1 = SimEngine(comp, store, engine_name="vllm", engine_version="0.11", block_tokens=16)
    e2 = SimEngine(comp, store, engine_name="sglang", engine_version="0.5", block_tokens=32)
    return e1, e2


def test_engine_identity_is_outside_compat_key():
    e1, e2 = _same_model_two_engines(PrefixCache(300))
    d1, d2 = e1.descriptor, e2.descriptor
    assert d1.engine != d2.engine and d1.layout.block_tokens != d2.layout.block_tokens
    assert d1.compat_key == d2.compat_key


def test_two_engines_sharing_store_bill_exactly_like_one(base_ctx):
    baseline, _ = _session([SimEngine(_same_model_two_engines(PrefixCache(300))[0].compiler, PrefixCache(300))], base_ctx)
    e1, e2 = _same_model_two_engines(PrefixCache(300))
    alternating, calls = _session([e1, e2], base_ctx)
    assert alternating == baseline, f"alternating engines billed {alternating} cold vs single-engine {baseline}"
    assert all(u.cache_read_input_tokens > 0 for u in calls[1:]), "every call after the first should read cache"


def test_two_engines_with_separate_stores_bill_more(base_ctx):
    """Control: without a shared store (or a transfer layer), each engine pays its own prefill."""
    baseline, _ = _session([SimEngine(_same_model_two_engines(PrefixCache(300))[0].compiler, PrefixCache(300))], base_ctx)
    e1, _ = _same_model_two_engines(PrefixCache(300))
    e2, _ = _same_model_two_engines(PrefixCache(300))
    e2 = SimEngine(e2.compiler, PrefixCache(300), engine_name="sglang", engine_version="0.5", block_tokens=32)
    separate, _ = _session([e1, e2], base_ctx)
    assert separate > baseline


# ---------------------------------------------------------------- mutation guard


def test_mutation_engine_in_compat_key_is_caught(base_ctx, monkeypatch):
    baseline, _ = _session([SimEngine(_same_model_two_engines(PrefixCache(300))[0].compiler, PrefixCache(300))], base_ctx)

    real = CacheDescriptor.compat_key.fget

    def buggy(self):  # folds engine identity into the key: the failure mode Layer 1 exists to remove
        eng = f"{self.engine.name}:{self.engine.version}" if self.engine else "none"
        return "sha256:" + hashlib.sha256((real(self) + eng).encode()).hexdigest()

    monkeypatch.setattr(CacheDescriptor, "compat_key", property(buggy))
    e1, e2 = _same_model_two_engines(PrefixCache(300))
    assert e1.descriptor.compat_key != e2.descriptor.compat_key  # the bug is in place
    alternating, _ = _session([e1, e2], base_ctx)
    with pytest.raises(AssertionError):
        assert alternating == baseline

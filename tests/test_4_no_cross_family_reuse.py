"""Falsification 4 (SPEC.md "Cache and routing"): a cache hit across two different compat_keys is a keying bug,
never a discovery. The prefix chain is content-addressed and therefore IDENTICAL across families; the compiler
cache_key (compat_key plus compiler_id and tokenizer_hash) is what separates them, so this is the assertion
that has to hold.
"""
from __future__ import annotations

import pytest
from conftest import turn

from pcf import Layout, PrefixCache, first_divergence
from pcf.descriptor import CacheDescriptor, sha256_tag
from pcf.families.sim import family_a, family_b


def test_same_pcf_same_chain_different_keys_no_hit(base_ctx):
    ctx = turn(base_ctx, 1, "Where is order 1?", None)
    A, B = family_a(min_cacheable=32), family_b(min_cacheable=32)
    ua, ca = A.run(ctx, now=0.0)
    assert ua.cache_creation_input_tokens > 0  # A wrote entries
    cb = B.compiler.compile(ctx)
    assert ca.chain == cb.chain, "chain is content-addressed and must not depend on the family"
    assert first_divergence(ca.chain, cb.chain) == len(ca.chain)
    assert A.descriptor.compat_key != B.descriptor.compat_key
    # B's lookup against A's populated store must miss even though the chain is identical.
    assert A.cache.lookup(B.compiler.cache_key, cb.native_chain, now=1.0) == -1 < A.cache.lookup(
        A.compiler.cache_key, ca.native_chain, now=1.0)
    assert B.compiler.warmth(ctx, A.cache, now=1.0).warm_tokens == 0


def test_compat_key_sensitivity():
    tok = sha256_tag("tok")
    base = dict(family="x", model_id="m", weights_hash=sha256_tag("w1"), tokenizer_hash=tok,
                layout=Layout("rope", "bf16", "gqa"), min_cacheable_tokens=1, max_breakpoints=4, ttl_seconds=300)
    d = CacheDescriptor(**base)
    assert CacheDescriptor(**base).compat_key == d.compat_key  # identical => same
    assert CacheDescriptor(**{**base, "weights_hash": sha256_tag("w2")}).compat_key != d.compat_key
    assert CacheDescriptor(**{**base, "tokenizer_hash": sha256_tag("tok2")}).compat_key != d.compat_key
    assert CacheDescriptor(**{**base, "layout": Layout("rope", "fp8", "gqa")}).compat_key != d.compat_key
    assert CacheDescriptor(**{**base, "layout": Layout("rope", "bf16", "gqa", kv_quant="int8")}).compat_key != d.compat_key
    # informational fields never change the key
    assert CacheDescriptor(**{**base, "family": "y", "model_id": "renamed"}).compat_key == d.compat_key
    assert d.with_engine("vllm", "0.11", block_tokens=16).compat_key == d.compat_key


# ---------------------------------------------------------------- mutation guard


class _StoreIgnoringCompatKey(PrefixCache):
    def write(self, compat_key, prefix_hash, cum_tokens, now, **kwargs):
        from pcf import sha256_tag
        super().write(sha256_tag("mutant-shared"), prefix_hash, cum_tokens, now, **kwargs)

    def peek(self, compat_key, chain, now, **kwargs):
        from pcf import sha256_tag
        return super().peek(sha256_tag("mutant-shared"), chain, now, **kwargs)


def test_mutation_shared_keyspace_produces_false_cross_family_hit(base_ctx):
    ctx = turn(base_ctx, 1, "Where is order 1?", None)
    shared = _StoreIgnoringCompatKey(300)
    A, B = family_a(shared, min_cacheable=32), family_b(shared, min_cacheable=32)
    A.run(ctx, now=0.0)
    with pytest.raises(AssertionError):
        assert B.compiler.warmth(ctx, B.cache, now=1.0).warm_tokens == 0

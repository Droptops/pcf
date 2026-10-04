import pytest

from pcf import Context, Segment
from pcf.families.sim import family_a
from pcf.jit import (
    PCFInputCostEstimate,
    ShadowRequest,
    ShadowRun,
    apply_pcf_input_cost,
    estimate_pcf_input_cost,
    make_pcf_metered_shadow_runner,
)
from pcf.router import Candidate


def context():
    return Context([
        Segment("s", "system", "stable system prompt " * 20),
        Segment("m", "memory", "stable account facts " * 10),
        Segment("u", "user", "what is the answer?", stable=False),
    ])


def candidate(engine=None, **kwargs):
    engine = engine or family_a(min_cacheable=1)
    return Candidate(
        engine.compiler,
        engine.cache,
        input_price_per_mtok=kwargs.pop("input_price_per_mtok", 2.0),
        cache_read_price_per_mtok=kwargs.pop("cache_read_price_per_mtok", 0.2),
        **kwargs,
    )


def populate_compiler_cache(cand, ctx, now):
    compiled = cand.compiler.compile(ctx)
    for _, digest, tokens in compiled.marker_prefixes:
        if tokens >= cand.compiler.descriptor.min_cacheable_tokens:
            cand.cache.write(
                compiled.cache_key,
                digest,
                tokens,
                now,
                namespace=ctx.cache_namespace,
                ttl_seconds=cand.compiler.descriptor.ttl_seconds,
            )


def test_cost_split_matches_existing_pcf_router_formula():
    cand = candidate()
    ctx = context()
    estimate = estimate_pcf_input_cost(ctx, cand, now=0)
    warmth = cand.compiler.warmth(ctx, cand.cache, 0)

    assert estimate.token_cost_usd == pytest.approx(warmth.uncached_tokens * cand.input_price_per_mtok / 1e6)
    assert estimate.memory_cost_usd == pytest.approx(
        (
            warmth.cache_creation_tokens * cand.write_price
            + warmth.warm_tokens * cand.cache_read_price_per_mtok
        ) / 1e6
    )
    assert estimate.total_input_cost_usd == pytest.approx(estimate.token_cost_usd + estimate.memory_cost_usd)
    assert estimate.cold_tokens == warmth.cold_tokens
    assert estimate.cache_state == "simulated"
    assert not estimate.token_count_is_estimate


def test_explicit_cache_write_price_is_respected():
    cand = candidate(cache_write_price_per_mtok=7.0)
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    assert estimate.memory_cost_usd == pytest.approx(estimate.cache_creation_tokens * 7.0 / 1e6)


def test_warm_cache_moves_cost_from_creation_or_uncached_to_cache_read():
    cand = candidate()
    ctx = context()
    cold = estimate_pcf_input_cost(ctx, cand, now=0)
    populate_compiler_cache(cand, ctx, 0)
    warm = estimate_pcf_input_cost(ctx, cand, now=1)

    assert cold.warm_tokens == 0
    assert warm.warm_tokens > 0
    assert warm.cache_creation_tokens <= cold.cache_creation_tokens
    assert warm.total_input_cost_usd < cold.total_input_cost_usd


def test_apply_cost_preserves_execution_fields_and_adds_metering_metadata():
    estimate = estimate_pcf_input_cost(context(), candidate(), now=0)
    run = ShadowRun(
        output={"answer": 42},
        latency_ms=12,
        authority_ok=False,
        freshness_ok=True,
        metadata={"trace": "abc"},
    )
    metered = apply_pcf_input_cost(run, estimate)

    assert metered.output == {"answer": 42}
    assert metered.latency_ms == 12
    assert not metered.authority_ok
    assert metered.token_cost_usd == estimate.token_cost_usd
    assert metered.memory_cost_usd == estimate.memory_cost_usd
    assert metered.metadata["trace"] == "abc"
    assert metered.metadata["pcf_warm_tokens"] == estimate.warm_tokens
    assert metered.metadata["pcf_cache_state"] == estimate.cache_state


def test_apply_cost_rejects_double_metering_and_metadata_collision():
    estimate = estimate_pcf_input_cost(context(), candidate(), now=0)
    with pytest.raises(ValueError, match="already contains"):
        apply_pcf_input_cost(ShadowRun(output="x", token_cost_usd=0.01), estimate)
    with pytest.raises(ValueError, match="collides"):
        apply_pcf_input_cost(ShadowRun(output="x", metadata={"pcf_warm_tokens": 9}), estimate)


def test_metered_runner_snapshots_cost_before_execution_populates_cache():
    cand = candidate()
    ctx = context()

    def execute(request):
        populate_compiler_cache(cand, request.payload, request.metadata["now"])
        return ShadowRun(output="ok", latency_ms=3)

    runner = make_pcf_metered_shadow_runner(
        cand,
        execute,
        context_of=lambda request: request.payload,
        now_of=lambda request: request.metadata["now"],
    )
    first = runner(ShadowRequest("r1", "fp", ctx, {"now": 0.0}))
    second = runner(ShadowRequest("r2", "fp", ctx, {"now": 1.0}))

    assert first.metadata["pcf_warm_tokens"] == 0
    assert second.metadata["pcf_warm_tokens"] > 0
    assert second.token_cost_usd + second.memory_cost_usd < first.token_cost_usd + first.memory_cost_usd


def test_unknown_provider_cache_state_is_preserved_not_invented():
    from pcf.cache import PrefixCache
    from pcf.families.openai_adapter import OpenAICompiler

    compiler = OpenAICompiler("gpt-5.6")
    cand = Candidate(compiler, PrefixCache(compiler.descriptor.ttl_seconds), 2.0, 0.2)
    estimate = estimate_pcf_input_cost(context(), cand, now=0)
    assert estimate.cache_state == "unknown"
    assert estimate.warm_tokens == 0
    assert estimate.token_count_is_estimate


def test_invalid_bridge_inputs_fail_closed():
    cand = candidate()
    with pytest.raises(ValueError, match="Context"):
        estimate_pcf_input_cost("not-context", cand, now=0)
    with pytest.raises(ValueError, match="Candidate"):
        estimate_pcf_input_cost(context(), "not-candidate", now=0)
    with pytest.raises(ValueError, match="ShadowRun"):
        apply_pcf_input_cost("not-run", PCFInputCostEstimate(
            "model", 0, 0, 0, 0, 0, "simulated", 0, 0, 0, 0, False
        ))

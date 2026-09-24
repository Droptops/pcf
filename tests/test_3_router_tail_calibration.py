"""Falsification 3 (SPEC.md "Cache and routing"): the router's confidence must be calibrated on the TAIL.

Synthetic population with a known ground truth so calibration can be measured exactly:
  difficulty d ~ head U(0, 0.5) with prob 0.8, tail U(0.6, 1.0) with prob 0.2
  P(cheap model suffices | d) = sigmoid(4 - 8d)        # head ~0.5-0.98, tail ~0.02-0.31

Two raw scorers stand in for Jev:
  good: sees d, but at half the true scale and with noise   -> Platt scaling recovers calibration
  bad:  sees d on the head but treats every tail prompt as an easy one (the "jagged" failure)
        -> head ECE looks fine after fitting; tail ECE is terrible. This is the mutation guard, and it
           is also the whole reason SPEC says head calibration is not evidence of tail calibration.
"""
from __future__ import annotations

import math
import random

import pytest

from pcf import Context, Segment
from pcf.families.sim import family_a, family_b
from pcf.router import (Candidate, PlattScaledSource, Router, UncalibratedSource, ValidationSample,
                        expected_calibration_error, fit_platt)

TAIL_ECE_BOUND = 0.10
THRESHOLD = 0.80


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def truth(d: float) -> float:
    return sigmoid(4 - 8 * d)


def difficulty_of(ctx: Context) -> float:
    user = next(s for s in ctx.segments if s.kind == "user")
    return float(user.content.split("difficulty=")[1].split(";")[0])


def make_population(n: int, seed: int):
    rng = random.Random(seed)
    rows = []  # (ctx, is_tail, label)
    for i in range(n):
        is_tail = rng.random() < 0.2
        d = rng.uniform(0.6, 1.0) if is_tail else rng.uniform(0.0, 0.5)
        ctx = Context([
            Segment("sys", "system", "You are a support agent."),
            Segment(f"u{i}", "user", f"difficulty={d:.4f}; question {i}", stable=False),
        ])
        rows.append((ctx, is_tail, 1 if rng.random() < truth(d) else 0))
    return rows


def good_scorer(rng: random.Random):
    def raw(ctx: Context, cand: Candidate) -> float:
        d = difficulty_of(ctx)
        return sigmoid(0.5 * (4 - 8 * d) + rng.gauss(0, 0.4))
    return raw


def bad_scorer(rng: random.Random):
    def raw(ctx: Context, cand: Candidate) -> float:
        d = difficulty_of(ctx)
        d_seen = d if d <= 0.5 else 0.3  # blind to the tail
        return sigmoid(0.5 * (4 - 8 * d_seen) + rng.gauss(0, 0.4))
    return raw


def _candidates():
    A, B = family_a(), family_b()
    return [
        Candidate(A.compiler, A.cache, input_price_per_mtok=1.0, cache_read_price_per_mtok=0.1, is_fallback=True),
        Candidate(B.compiler, B.cache, input_price_per_mtok=0.2, cache_read_price_per_mtok=0.02),
    ]


def _fit_and_eval(scorer_factory, seed=7, fit_on="all"):
    """fit_on="head" models the realistic case: the labeled calibration set is dominated by common traffic."""
    cands = _candidates()
    cheap = cands[1]
    train = make_population(3000, seed)
    test = make_population(3000, seed + 1)
    src = PlattScaledSource(scorer_factory(random.Random(seed)), scorer_version=scorer_factory.__name__ + ":1")
    src.fit([(ctx, cheap, y) for ctx, t, y in train if fit_on == "all" or not t])
    validation = make_population(3000, seed + 101)
    src.validate([ValidationSample(ctx, cheap, y, tail) for ctx, tail, y in validation], dataset_id=f"synthetic-validation-{seed}")
    probs = [src.p_sufficient(ctx, cheap) for ctx, _, _ in test]
    head = [(p, y) for p, (_, t, y) in zip(probs, test) if not t]
    tail = [(p, y) for p, (_, t, y) in zip(probs, test) if t]
    return src, cands, test, probs, head, tail


def test_good_scorer_is_calibrated_on_head_and_tail():
    _, _, _, _, head, tail = _fit_and_eval(good_scorer, fit_on="all")
    head_ece = expected_calibration_error([p for p, _ in head], [y for _, y in head])
    tail_ece = expected_calibration_error([p for p, _ in tail], [y for _, y in tail])
    assert head_ece < TAIL_ECE_BOUND, f"head ECE {head_ece:.3f}"
    assert tail_ece < TAIL_ECE_BOUND, f"tail ECE {tail_ece:.3f}"


def test_calibration_set_must_include_tail_samples():
    """Finding, not assumption: even a scorer whose SIGNAL is right on the tail drifts there when the
    calibration set is head-only (noise is amplified differently across the two difficulty ranges, and
    a head-fit compensation does not transfer). So the SPEC tail requirement is stronger than "evaluate on a
    tail holdout": the labeled set used to fit the calibrator has to contain tail examples too."""
    _, _, _, _, _, tail_all = _fit_and_eval(good_scorer, fit_on="all")
    src, cands, _, _, head_only, tail_head = _fit_and_eval(good_scorer, fit_on="head")
    ece_all = expected_calibration_error([p for p, _ in tail_all], [y for _, y in tail_all])
    ece_head_fit = expected_calibration_error([p for p, _ in tail_head], [y for _, y in tail_head])
    ece_head_on_head = expected_calibration_error([p for p, _ in head_only], [y for _, y in head_only])
    assert ece_head_on_head < TAIL_ECE_BOUND, "head-only fit is fine on the head"
    assert ece_all < TAIL_ECE_BOUND < ece_head_fit, f"tail ECE: fit-on-all {ece_all:.3f}, fit-on-head {ece_head_fit:.3f}"
    assert src.validation_for(cands[1], THRESHOLD) is None, "tail ECE alone must block routing"


def test_routing_on_calibrated_confidence_holds_the_quality_floor():
    src, cands, test, probs, _, _ = _fit_and_eval(good_scorer)
    router = Router(cands, src, threshold=THRESHOLD)
    routed_cheap_labels, escalations = [], 0
    for (ctx, is_tail, y), _ in zip(test, probs):
        d = router.route(ctx, now=0.0)
        if d.chosen == "sim-b-small":
            routed_cheap_labels.append(y)
        else:
            escalations += 1
    realized = sum(routed_cheap_labels) / len(routed_cheap_labels)
    # If we only route cheap at p >= 0.8, the realized sufficiency rate must be near or above 0.8.
    assert realized >= THRESHOLD - 0.05, f"realized sufficiency {realized:.3f} below floor"
    assert escalations > 0, "some tail prompts must escalate"


def test_uncalibrated_source_forces_fallback():
    cands = _candidates()
    router = Router(cands, UncalibratedSource(lambda ctx, c: 0.99), threshold=0.5)
    ctx = make_population(1, 0)[0][0]
    d = router.route(ctx, now=0.0)
    assert d.escalate and d.chosen == "sim-a-large" and d.confidence_source == "uncalibrated"


def test_validation_needs_samples_at_the_threshold():
    """A well-calibrated source that never reaches the threshold has measured nothing about routed quality."""
    cheap = _candidates()[1]
    ctxs = [Context([Segment("s", "system", "sys"), Segment(f"u{i}", "user", f"q{i}", stable=False)])
            for i in range(260)]
    src = PlattScaledSource(lambda ctx, c: 0.5, scorer_version="const:1")
    src.fit([(c, cheap, i % 2) for i, c in enumerate(ctxs[:100])])  # calibrated p = 0.5 everywhere
    rec = src.validate([ValidationSample(c, cheap, i % 2, i % 4 < 2) for i, c in enumerate(ctxs[100:])],
                       dataset_id="nothing-selected")
    assert rec.n_selected == 0 and rec.ece < 0.1 and rec.tail_ece < 0.1 and not rec.passed


def test_platt_fit_converges_on_ordinary_data():
    """Seed 2 head-only data used to stall at float resolution and raise 'did not converge'."""
    rows, raw, cheap = make_population(3000, 2), good_scorer(random.Random(2)), _candidates()[1]
    a, b = fit_platt([raw(ctx, cheap) for ctx, t, _ in rows if not t], [y for _, t, y in rows if not t])
    assert math.isfinite(a) and math.isfinite(b)


# ---------------------------------------------------------------- mutation guard


def test_mutation_tail_blind_scorer_passes_head_and_fails_tail():
    """Calibrated on head-dominated traffic, a tail-blind scorer looks fine on the head and is
    catastrophic on the tail. This is why SPEC.md demands a tail holdout, not a global ECE."""
    _, _, _, _, head, tail = _fit_and_eval(bad_scorer, fit_on="head")
    head_ece = expected_calibration_error([p for p, _ in head], [y for _, y in head])
    tail_ece = expected_calibration_error([p for p, _ in tail], [y for _, y in tail])
    assert head_ece < TAIL_ECE_BOUND, f"head ECE {head_ece:.3f} — the bad scorer should look fine on the head"
    with pytest.raises(AssertionError):
        assert tail_ece < TAIL_ECE_BOUND, f"tail ECE {tail_ece:.3f}"
    assert tail_ece > 0.4, f"tail ECE {tail_ece:.3f} should be catastrophic, not marginal"


def test_mutation_global_fit_cannot_rescue_a_tail_blind_scorer():
    """Even fit on everything, Platt scaling cannot fix a scorer whose ordering is wrong on the tail."""
    _, _, _, _, _, tail = _fit_and_eval(bad_scorer, fit_on="all")
    tail_ece = expected_calibration_error([p for p, _ in tail], [y for _, y in tail])
    assert tail_ece > TAIL_ECE_BOUND


def test_mutation_tail_blind_scorer_breaks_the_quality_floor():
    src, cands, test, probs, _, _ = _fit_and_eval(bad_scorer)
    router = Router(cands, src, threshold=THRESHOLD)
    # The underlying scorer still fails on the tail; the new validation gate
    # prevents that scorer from qualifying the cheap candidate.
    tail_cheap = [y for (_, is_tail, y), p in zip(test, probs) if is_tail and p >= THRESHOLD]
    assert tail_cheap
    realized = sum(tail_cheap) / len(tail_cheap)
    assert realized < THRESHOLD - 0.05
    for ctx, is_tail, _ in test[:50]:
        assert router.route(ctx, now=0).escalate

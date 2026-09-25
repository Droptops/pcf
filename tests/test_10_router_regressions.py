"""Regression tests for confidence transport redirects, validation evidence minimums, duplicate samples and token accounting."""
from __future__ import annotations

import io
import urllib.request
from email.message import Message
from urllib.response import addinfourl

import pytest

from pcf import Context, PrefixCache, Segment, canonical_bytes
from pcf.families.sim import CharChunkTokenizer, SimCompiler, SimEngine, family_a, family_b, sim_descriptor
from pcf.router import Candidate, ConfidenceSource, JevConfidenceSource, Router, ValidationSample, http_transport


class _KnownSource(ConfidenceSource):
    name = "review-known-probability"

    def __init__(self):
        super().__init__(source_version="1", rubric_id="synthetic-sufficiency")

    def p_sufficient(self, ctx, candidate):
        return ctx.segments[-1].content["p"]


def _context(key, p=1):
    return Context([Segment("u", "user", {"question": key, "p": p})])


def _candidates():
    a, b = family_a(), family_b()
    return [Candidate(a.compiler, a.cache, 1, .1, is_fallback=True),
            Candidate(b.compiler, b.cache, .2, .02)]


def _samples(candidate, tail_selected):
    rows = [ValidationSample(_context(f"head-{i}"), candidate, 1, False) for i in range(70)]
    rows += [ValidationSample(_context(f"tail-{i}", int(i < tail_selected)), candidate,
                              int(i < tail_selected), True) for i in range(30)]
    return rows


@pytest.mark.parametrize("tail_selected", [0, 1, 19, 20])
def test_tail_selection_needs_the_full_evidence_minimum(tail_selected):
    candidates, source = _candidates(), _KnownSource()
    record = source.validate(_samples(candidates[1], tail_selected), dataset_id="tail-evidence")
    assert record.ece == record.tail_ece == 0  # calibration alone is not a selection-quality measurement
    assert record.n_tail_selected == tail_selected
    assert record.passed is (tail_selected >= 20)
    decision = Router(candidates, source).route(_context("unseen-tail"), now=0)
    if tail_selected < 20:
        assert "tail-selected quality or sample count below requirement" in record.failures
        assert decision.escalate and decision.chosen == candidates[0].model_id
        assert decision.candidates[1].p_sufficient is None
    else:
        assert not decision.escalate and decision.chosen == candidates[1].model_id


@pytest.mark.parametrize("renamed", [False, True])
def test_repeated_contexts_cannot_inflate_validation_counts(renamed):
    candidate, source = _candidates()[1], _KnownSource()
    rows = []
    for i in range(100):
        ctx = Context([Segment(f"u{i}" if renamed else "u", "user", {"question": "same", "p": 1},
                               stable=bool(i % 2))], session_id=f"session-{i}")
        rows.append(ValidationSample(ctx, candidate, 1, True))
    with pytest.raises(ValueError, match="unique contexts"):
        source.validate(rows, dataset_id="repeated-context")
    assert source.validation_for(candidate, .8) is None


def test_validation_id_records_its_sample_requirements():
    candidate, source = _candidates()[1], _KnownSource()
    rows = _samples(candidate, 30)
    first = source.validate(rows, dataset_id="sample-policy", min_selected_samples=20)
    second = source.validate(rows, dataset_id="sample-policy", min_selected_samples=21)
    assert first.passed and second.passed
    assert first.validation_id != second.validation_id
    assert second.to_json()["min_samples"] == 100
    assert second.to_json()["min_tail_samples"] == 30
    assert second.to_json()["min_selected_samples"] == 21


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("target", ["https://redirect-target.invalid/collect",
                                   "http://api.typesafe.ai/collect",
                                   "https://api.typesafe.ai/elsewhere"])
def test_confidence_transport_never_follows_redirects(monkeypatch, status, target):
    requests = []

    class OfflineServer(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, req):
            return self.respond(req)

        def http_open(self, req):
            return self.respond(req)

        def respond(self, req):
            requests.append(req)
            headers = Message()
            if len(requests) == 1:
                headers["Location"] = target
                reply = addinfourl(io.BytesIO(b""), headers, req.full_url, status)
                reply.msg = "Redirect"
            else:
                reply = addinfourl(io.BytesIO(b"{}"), headers, req.full_url, 200)
                reply.msg = "OK"
            return reply

    build_opener = urllib.request.build_opener
    # Exercise the production handler chain. Both old urlopen and the fixed local
    # opener get an offline server; the test must never reach the network.
    monkeypatch.setattr(urllib.request, "build_opener",
                        lambda *handlers: build_opener(OfflineServer(), *handlers))
    monkeypatch.setattr(urllib.request, "urlopen", build_opener(OfflineServer()).open)
    source = JevConfidenceSource(http_transport(api_key="DUMMY-REVIEW-KEY"), model="jev-1.13.0")
    decision = Router([_candidates()[0]], source, score_unvalidated=True).route(_context("redirect"), now=0)
    assert decision.escalate and decision.candidates[0].confidence_error == "confidence transport unavailable"
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") == "Bearer DUMMY-REVIEW-KEY"
    assert requests[0].get_method() == "POST"


@pytest.mark.parametrize("make_engine", [family_a, family_b])
@pytest.mark.parametrize("write_split", [False, True])
def test_regrouping_history_preserves_cache_counts(make_engine, write_split):
    engine = make_engine(min_cacheable=1)
    turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    whole = Context([Segment("h", "history", turns)])
    split = Context([Segment("h1", "history", turns[:1]), Segment("h2", "history", turns[1:])])
    first, second = (split, whole) if write_split else (whole, split)
    written, compiled = engine.run(first, now=0)
    estimate = engine.compiler.warmth(second, engine.cache, now=1)
    read, regrouped = engine.run(second, now=1)
    assert compiled.native_chain[-1] == regrouped.native_chain[-1]
    # Independent count of the actual rendered message list, with no PCF segment wrappers.
    expected = engine.compiler.tokenizer.count(canonical_bytes({"messages": turns}).decode())
    assert compiled.total_tokens == regrouped.total_tokens == expected
    assert read.cache_read_input_tokens == estimate.warm_tokens == written.cache_creation_input_tokens == expected
    assert read.cold_tokens == estimate.cold_tokens == 0
    assert engine.cache.cum_tokens(compiled.cache_key, compiled.native_chain[-1]) == expected


def test_chunk_tokenizer_configuration_changes_cache_identity():
    store = PrefixCache(300)
    engines = []
    for size in (1, 4):
        tok = CharChunkTokenizer(chars_per_token=size)
        descriptor = sim_descriptor("sim", "same-model", tok.tokenizer_hash, min_cacheable=1)
        engines.append(SimEngine(SimCompiler(descriptor, tok), store))
    assert engines[0].compiler.cache_key != engines[1].compiler.cache_key
    ctx = Context([Segment("s", "system", "stable prefix")])
    engines[0].run(ctx, now=0)
    assert engines[1].run(ctx, now=1)[0].cache_read_input_tokens == 0


@pytest.mark.parametrize("size", [0, -1, True, float("nan"), float("inf")])
def test_chunk_tokenizer_rejects_invalid_configuration(size):
    with pytest.raises(ValueError):
        CharChunkTokenizer(chars_per_token=size)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1.5, -0.1, "0.9", None])
def test_invalid_confidence_from_a_validated_scorer_falls_back(bad):
    class _Glitching(_KnownSource):
        def p_sufficient(self, ctx, candidate):
            return bad if ctx.segments[-1].content["question"] == "unseen" else super().p_sufficient(ctx, candidate)

    candidates, source = _candidates(), _Glitching()
    assert source.validate(_samples(candidates[1], 20), dataset_id="invalid-score").passed
    decision = Router(candidates, source).route(_context("unseen"), now=0)
    assert decision.escalate and decision.chosen == candidates[0].model_id
    report = decision.candidates[1]
    assert report.p_sufficient is None and report.confidence_error.startswith("invalid confidence score")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), "0.9", None, True, -0.1, 1.5])
def test_fitted_platt_raw_failure_uses_fallback(bad):
    from pcf.router import PlattScaledSource
    candidates = _candidates()
    def raw(ctx, cand):
        value = ctx.segments[-1].content
        return bad if value["question"] == "glitch" else value["p"]
    def population(tag):
        for i in range(300):
            high = i < 150
            yield (_context(f"{tag}-{i}", .9 if high else .1), (i // 10) % 2 == 0,
                   int(i % 10 != 9) if high else int(i % 10 == 9))
    source = PlattScaledSource(raw, scorer_version="raw-failure:1")
    source.fit([(ctx, candidates[1], y) for ctx, tail, y in population("fit")])
    record = source.validate([ValidationSample(ctx, candidates[1], y, tail)
                              for ctx, tail, y in population("validation")], dataset_id="raw-failure")
    assert record.passed and record.n_tail_selected == 80
    decision = Router(candidates, source).route(_context("glitch"), 0)
    assert decision.escalate and decision.chosen == candidates[0].model_id
    assert decision.candidates[1].confidence_error.startswith("invalid raw confidence score")


def test_platt_preserves_configuration_and_programming_errors():
    from pcf.router import PlattScaledSource
    candidate = _candidates()[1]
    source = PlattScaledSource(lambda ctx, cand: .9, a=float("nan"))
    with pytest.raises(ValueError, match="coefficients"):
        source.p_sufficient(_context("test"), candidate)
    def broken(ctx, cand):
        raise ValueError("scorer programming error")
    with pytest.raises(ValueError, match="scorer programming error"):
        PlattScaledSource(broken).p_sufficient(_context("test"), candidate)


def test_calibration_metrics_are_correctly_rounded_on_every_python_version():
    import math
    from fractions import Fraction

    from pcf.router.calibration import brier_score, expected_calibration_error
    probs = [0.0, 0.54, 0.79, 0.33, 0.6, 0.8, 0.64, 0.55, 0.18, 0.09, 0.55, 0.85]
    labels = [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
    terms = [(p - y) ** 2 for p, y in zip(probs, labels)]
    naive = 0.0
    for t in terms:
        naive += t
    exact = float(sum(Fraction(t) for t in terms))
    assert naive != exact  # left-to-right addition (Python < 3.12 sum) rounds differently here
    assert brier_score(probs, labels) == exact / len(probs) == math.fsum(terms) / len(probs)
    groups = {}
    for p, y in zip(probs, labels):
        groups.setdefault(min(int(p * 10), 9), []).append(y - p)
    ece = float(sum(abs(sum(Fraction(v) for v in g)) for g in groups.values())) / len(probs)
    assert expected_calibration_error(probs, labels) == ece

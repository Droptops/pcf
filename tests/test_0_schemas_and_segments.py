"""Schema conformance for every artifact the library emits, plus the hashing/ordering rules of SPEC.md §2."""
from __future__ import annotations

import json
import pathlib

import jsonschema
import pytest
from conftest import turn

from pcf import Context, Segment, canonical_bytes
from pcf.families.sim import family_a, family_b
from pcf.router import Candidate, PlattScaledSource, Router

SCHEMAS = pathlib.Path(__file__).parent.parent / "spec" / "schemas"


def _schema(name: str):
    return json.loads((SCHEMAS / name).read_text())


def test_pcf_document_validates_and_roundtrips(base_ctx):
    ctx = turn(base_ctx, 1, "hi", None)
    doc = ctx.to_json()
    jsonschema.validate(doc, _schema("pcf.schema.json"))
    back = Context.from_json(json.loads(json.dumps(doc)))
    assert back.prefix_chain() == ctx.prefix_chain()


def test_descriptor_validates():
    for eng in (family_a(), family_b()):
        jsonschema.validate(eng.descriptor.to_json(), _schema("cache-descriptor.schema.json"))
        jsonschema.validate(eng.compiler.descriptor.to_json(), _schema("cache-descriptor.schema.json"))


def test_route_decision_validates(base_ctx):
    A, B = family_a(), family_b()
    router = Router([
        Candidate(A.compiler, A.cache, 1.0, 0.1, is_fallback=True),
        Candidate(B.compiler, B.cache, 0.2, 0.02),
    ], PlattScaledSource(lambda ctx, c: 0.9), threshold=0.8)
    doc = router.route(turn(base_ctx, 1, "hi", None), now=0.0, request_id="r1").to_json()
    jsonschema.validate(doc, _schema("route-decision.schema.json"))


def test_ordering_rule_is_enforced():
    with pytest.raises(ValueError, match="SPEC.md"):
        Context([Segment("u", "user", "x", stable=False), Segment("s", "system", "y")])
    Context([Segment("t", "tools", []), Segment("s", "system", "y"), Segment("m", "memory", {}),
             Segment("d", "document", "doc"), Segment("h", "history", []), Segment("u", "user", "x", stable=False)])


def test_segment_hash_ignores_id_and_stable_flag():
    a = Segment("one", "system", "same", stable=True)
    b = Segment("two", "system", "same", stable=False)
    assert a.hash == b.hash


def test_segment_hash_depends_on_kind_and_content():
    assert Segment("x", "system", "same").hash != Segment("x", "memory", "same").hash
    assert Segment("x", "system", "a").hash != Segment("x", "system", "b").hash


def test_canonicalization_prevents_type_collisions():
    assert canonical_bytes("a") != canonical_bytes({"a": 1})
    assert canonical_bytes({"b": 1, "a": 2}) == canonical_bytes({"a": 2, "b": 1})
    assert Segment("x", "memory", {"b": 1, "a": 2}).hash == Segment("x", "memory", {"a": 2, "b": 1}).hash


def test_declared_hash_mismatch_is_hard_error(base_ctx):
    doc = base_ctx.to_json()
    doc["segments"][0]["hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="declared hash"):
        Context.from_json(doc)


def test_prefix_chain_diverges_at_the_changed_segment(base_ctx):
    ctx1 = turn(base_ctx, 1, "hi", None)
    ctx2 = ctx1.with_replaced("mem", Segment("mem", "memory", {"customer_tier": "silver"}))
    chain1, chain2 = ctx1.prefix_chain(), ctx2.prefix_chain()
    assert chain1[:2] == chain2[:2] and chain1[2:] != chain2[2:]

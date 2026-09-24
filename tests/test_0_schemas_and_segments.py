"""Schema conformance for every artifact the library emits, plus the hashing/ordering rules of SPEC.md "Document"."""
from __future__ import annotations

import copy
import json
import pathlib
import re

import jsonschema
import pytest
from conftest import turn

from pcf import Context, Segment, canonical_bytes
from pcf.families.sim import family_a, family_b
from pcf.router import Candidate, PlattScaledSource, Router
from pcf.schemas import validate

SCHEMAS = pathlib.Path(__file__).parent.parent / "spec" / "schemas"


def _schema(name: str):
    return json.loads((SCHEMAS / name).read_text())


def test_pcf_document_validates_and_roundtrips(base_ctx):
    ctx = Context(turn(base_ctx, 1, "hi", None).segments, "s", "tenant-a")
    doc = ctx.to_json()
    jsonschema.validate(doc, _schema("pcf.schema.json"))
    back = Context.from_json(json.loads(json.dumps(doc)))
    assert back == ctx and back.prefix_chain() == ctx.prefix_chain()


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
    cand = doc["candidates"][0]
    for bad in ({**doc, "chosen": " "}, {**doc, "request_id": "\x85"},
                {**doc, "source_fingerprint": doc["source_fingerprint"] + "\n"},
                {**doc, "candidates": [{**cand, "family": " "}]}, {**doc, "candidates": [{**cand, "model_id": "\t"}]},
                {**doc, "candidates": [{**cand, "compat_key": cand["compat_key"] + "\n"}]}):
        with pytest.raises(ValueError, match="validation failed"):
            validate("route-decision", bad)


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
    assert chain1[:2] == chain2[:2] and all(a != b for a, b in zip(chain1[2:], chain2[2:]))


def test_core_invariants_are_enforced():
    call = {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "t", "arguments": {}}]}
    res = {"role": "tool", "call_id": "c", "content": "r"}
    for bad, why in [([Segment("m", "memory", "x"), Segment("d", "document", "y", authority="instruction")],
                      "precede data"),
                     ([Segment("h", "history", [res])], "orphan"),
                     ([Segment("h", "history", [call, res, call, res])], "call ids must be unique"),
                     ([Segment("h", "history", [call]), Segment("u", "user", "x")], "resolve tool calls"),
                     ([Segment("h", "history", [call, {"role": "user", "content": "x"}, res])], "must follow"),
                     ([Segment("t", "tools", [{"name": "t", "parameters": {}}] * 2)], "tool names must be unique")]:
        with pytest.raises(ValueError, match=why):
            Context(bad)
    with pytest.raises(ValueError, match="promoted"):
        Segment("h", "history", [], authority="instruction")
    s = Segment("m", "memory", {"a": [1]})
    s.content["a"].append(2)
    assert s.content == {"a": [1]}
    variants = ({}, {"authority": "instruction"}, {"provenance": "p"})
    assert len({Segment("d", "document", "x", **kw).hash for kw in variants}) == 3
    assert Segment("x", "memory", '{"a":1}').hash != Segment("x", "memory", {"a": 1}).hash
    legacy = {"role": "tool", "content": {"tool_use_id": "c", "content": "r", "is_error": True}}
    assert Segment("h", "history", [call, legacy]).content[1] == {**res, "is_error": True}
    mixed = {"role": "tool", "is_error": True, "content": {"tool_use_id": "c", "content": "r"}}
    with pytest.raises(ValueError):  # mixed 0.1/0.2 form must not silently flip an outer is_error
        Segment("h", "history", [call, mixed])
    doc = Context([Segment("h", "history", [call, res])]).to_json()
    del doc["segments"][0]["content"][1]["is_error"]  # the hash covers is_error, so the schema must require it
    with pytest.raises(ValueError, match="validation failed"):
        Context.from_json(doc)


def test_spec_schemas_mirror_packaged_schemas():
    from importlib.resources import files
    for n in ("pcf", "cache-descriptor", "route-decision"):
        packaged = files("pcf.schemas").joinpath(f"{n}.schema.json").read_bytes()
        assert (SCHEMAS / f"{n}.schema.json").read_bytes() == packaged


def _set(path, value):
    def mutate(doc):
        *parents, last = path
        node = doc
        for key in parents:
            node = node[key]
        node[last] = value
    return mutate


def test_schema_rejects_what_the_runtime_rejects():
    """Every probe breaks exactly one runtime rule; the published schema must reject it too."""
    call = {"id": "c", "name": "t", "arguments": {}}
    turns = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [call]},
             {"role": "tool", "call_id": "c", "content": "r"}]
    doc = Context([Segment("t", "tools", [{"name": "t", "parameters": {}}]), Segment("s", "system", "sys"),
                   Segment("h", "history", turns), Segment("u", "user", "hi", stable=False)], "s1", "ns").to_json()
    for seg in doc["segments"]:
        del seg["hash"]  # probes change content; hashes are checked separately
    tool_use = {"type": "tool_use", "id": "x", "name": "t", "input": {}}
    H = ("segments", 2, "content")
    probes = [_set(("segments", 1, "content"), ""), _set(("segments", 3, "content"), ""),
              _set((*H, 0, "content"), ""), _set((*H, 0, "content"), {}), _set((*H, 0, "content"), tool_use),
              _set(H, [turns[0], {"role": "assistant", "content": ""}]),
              _set(("segments", 1, "provenance"), " "), _set(("session_id",), " "), _set(("cache_namespace",), "\x1c"),
              _set(("segments", 0, "content", 0, "name"), " "), _set((*H, 1, "tool_calls", 0, "name"), " "),
              _set((*H, 1, "tool_calls", 0, "id"), " "), _set((*H, 2, "call_id"), " "),
              _set(("segments", 1, "id"), "s\n")]
    Context.from_json(doc)  # control: the unmodified document is valid
    for probe in probes:
        bad = copy.deepcopy(doc)
        probe(bad)
        with pytest.raises(ValueError):  # runtime, bypassing the schema
            Context(tuple(Segment(r["id"], r["kind"], r["content"], r.get("stable"), authority=r.get("authority"),
                                  provenance=r.get("provenance")) for r in bad["segments"]),
                    bad.get("session_id"), bad.get("cache_namespace", "default"))
        with pytest.raises(ValueError, match="validation failed"):
            validate("pcf", bad)
    hashed = Context([Segment("s", "system", "sys")]).to_json()
    hashed["segments"][0]["hash"] += "\n"
    with pytest.raises(ValueError, match="validation failed"):
        validate("pcf", hashed)
    # Constructors still migrate the 0.1 assistant tool_use form, but serialized documents must use the 0.2 form.
    legacy = [turns[0], {"role": "assistant", "content": tool_use},
              {"role": "tool", "call_id": "x", "content": "r", "is_error": False}]
    Segment("h", "history", legacy)
    bad = copy.deepcopy(doc)
    _set(H, legacy)(bad)
    with pytest.raises(ValueError, match="validation failed"):
        validate("pcf", bad)


def test_nonblank_pattern_matches_the_runtime_whitespace_rule():
    """Python re and ECMA-262 disagree on \\s, so the schemas spell out str.strip()'s whitespace set."""
    for name in ("pcf", "cache-descriptor", "route-decision"):
        pattern = _schema(f"{name}.schema.json")["$defs"]["nonblank"]["pattern"]
        assert "\\s" not in pattern.lower()
        wrong = [hex(c) for c in [*range(0x3001), 0xFEFF]
                 if (re.fullmatch(pattern, chr(c)) is None) != chr(c).isspace()]
        assert not wrong, f"{name}: {wrong}"

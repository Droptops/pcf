"""PCF 0.2: immutable JSON snapshots, explicit authority and typed tool history."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

import rfc8785

from .validation import ID_PATTERN, nonempty

PCF_VERSION = "0.2"
KIND_RANK = {"tools": 0, "system": 1, "memory": 2, "document": 2, "history": 3, "user": 4}


def canonical_bytes(content: Any) -> bytes:
    """RFC 8785 JCS; rejects non-finite numbers, unsafe integers and invalid Unicode."""
    try:
        return rfc8785.dumps(content)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"content is not canonical JSON: {exc}") from exc


def _sha256(*parts: bytes) -> str:
    return "sha256:" + hashlib.sha256(b"".join(parts)).hexdigest()


def text_content(content: Any) -> str:
    return content if isinstance(content, str) else canonical_bytes(content).decode("utf-8")


def _fields(record: dict, allowed: set[str], required: set[str]) -> None:
    if not isinstance(record, dict) or set(record) - allowed or required - set(record):
        raise ValueError(f"invalid fields: expected {sorted(required)}, allowed {sorted(allowed)}")


def normalize_history(turns: Any) -> list[dict]:
    if not isinstance(turns, list):
        raise ValueError("history content must be an array")
    result = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("history entries must be objects")
        role = turn.get("role")
        if role not in {"user", "assistant", "tool"}:
            raise ValueError("history role must be user, assistant or tool")
        if role == "tool":
            _fields(turn, {"role", "call_id", "content", "is_error"}, {"role", "content"})
            content = turn["content"]
            call_id = turn.get("call_id")
            # Lossless migration of the 0.1 embedded tool-result representation.
            if call_id is None and isinstance(content, dict) and "tool_use_id" in content:
                _fields(turn, {"role", "content"}, {"role", "content"})  # 0.1 kept is_error inside content
                _fields(content, {"tool_use_id", "content", "is_error"}, {"tool_use_id", "content"})
                call_id = content["tool_use_id"]
                turn = {**turn, "is_error": content.get("is_error", False)}
                content = content["content"]
            nonempty(call_id, "tool result call_id")
            if type(turn.get("is_error", False)) is not bool:
                raise ValueError("is_error must be boolean")
            result.append({"role": "tool", "call_id": call_id, "content": content,
                           "is_error": turn.get("is_error", False)})
            continue
        _fields(turn, {"role", "content", "tool_calls"}, {"role"})
        content = turn.get("content", "")
        calls = turn.get("tool_calls", [])
        if isinstance(content, dict) and content.get("type") == "tool_use":
            if calls or role != "assistant":
                raise ValueError("tool_use must be a standalone assistant call")
            _fields(content, {"type", "id", "name", "input"}, {"type", "id", "name", "input"})
            calls = [{"id": content["id"], "name": content["name"], "arguments": content["input"]}]
            content = ""
        if calls and role != "assistant":
            raise ValueError("only assistant entries can contain tool calls")
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be an array")
        clean_calls = []
        for call in calls:
            _fields(call, {"id", "name", "arguments"}, {"id", "name", "arguments"})
            nonempty(call["id"], "call id")
            nonempty(call["name"], "tool name")
            if not isinstance(call["arguments"], dict):
                raise ValueError("tool arguments must be an object")
            clean_calls.append(dict(call))
        if not isinstance(content, (str, dict)):
            raise ValueError("history text content must be a string or JSON object")
        if not content and not clean_calls:
            raise ValueError("empty history entry")
        item = {"role": role, "content": content}
        if clean_calls:
            item["tool_calls"] = clean_calls
        result.append(item)
    return result


@dataclass(frozen=True, init=False)
class Segment:
    id: str
    kind: str
    stable: bool
    authority: str
    provenance: str | None
    _json: bytes

    def __init__(self, id: str, kind: str, content: Any, stable: bool | None = None,
                 *, authority: str | None = None, provenance: str | None = None) -> None:
        if not isinstance(id, str) or not ID_PATTERN.fullmatch(id):
            raise ValueError("invalid segment id")
        if kind not in KIND_RANK:
            raise ValueError(f"unknown segment kind {kind!r}")
        stable = kind != "user" if stable is None else stable
        if type(stable) is not bool:
            raise ValueError("stable must be boolean")
        if authority is None:
            authority = "instruction" if kind in {"tools", "system"} else "data"
        if authority not in {"instruction", "data"}:
            raise ValueError("authority must be instruction or data")
        if kind in {"tools", "system"} and authority != "instruction":
            raise ValueError("tools/system require explicit application instruction authority")
        if kind in {"history", "user"} and authority != "data":
            raise ValueError("history/user cannot be promoted to instruction authority")
        if provenance is not None:
            nonempty(provenance, "provenance")
        content = json.loads(canonical_bytes(content))  # snapshot caller-owned data first
        if kind == "history":
            content = normalize_history(content)
        elif kind == "tools":
            if not isinstance(content, list):
                raise ValueError("tools content must be an array")
            for tool in content:
                _fields(tool, {"name", "description", "parameters"}, {"name", "parameters"})
                nonempty(tool["name"], "tool name")
                if not isinstance(tool.get("description", ""), str) or not isinstance(tool["parameters"], dict):
                    raise ValueError("tool description must be text and parameters must be an object")
        elif not isinstance(content, (str, dict)) or content == "":
            raise ValueError(f"{kind} content must be nonempty text or a JSON object")
        for name, value in {"id": id, "kind": kind, "stable": stable, "authority": authority,
                            "provenance": provenance, "_json": canonical_bytes(content)}.items():
            object.__setattr__(self, name, value)

    @property
    def content(self) -> Any:
        """Return an isolated copy so callers cannot mutate a hashed snapshot."""
        return json.loads(self._json)

    @property
    def hash(self) -> str:
        return _sha256(b"pcf:segment:0.2\x00", canonical_bytes({
            "kind": self.kind, "authority": self.authority, "provenance": self.provenance,
            "content": self.content}))

    def to_json(self) -> dict:
        result = {"id": self.id, "kind": self.kind, "content": self.content,
                  "stable": self.stable, "authority": self.authority, "hash": self.hash}
        if self.provenance is not None:
            result["provenance"] = self.provenance
        return result


@dataclass(frozen=True)
class Context:
    segments: tuple[Segment, ...]
    session_id: str | None = None
    cache_namespace: str = "default"

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(self.segments))
        if not self.segments or any(not isinstance(s, Segment) for s in self.segments):
            raise ValueError("context requires at least one Segment")
        nonempty(self.cache_namespace, "cache_namespace")
        if self.session_id is not None:
            nonempty(self.session_id, "session_id")
        if len({s.id for s in self.segments}) != len(self.segments):
            raise ValueError("segment ids must be unique")
        self.validate_order()
        names = [t["name"] for s in self.segments if s.kind == "tools" for t in s.content]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        self.validate_tool_history()

    def validate_order(self) -> None:
        # Memory may also follow history (tail memory), so volatile state does not invalidate cached history.
        ranks, seen_history = [], False
        for s in self.segments:
            seen_history = seen_history or s.kind == "history"
            ranks.append(3.5 if s.kind == "memory" and seen_history else KIND_RANK[s.kind])
        if any(b < a for a, b in zip(ranks, ranks[1:])):
            raise ValueError('invalid segment order; see SPEC.md "Document"')
        seen_data = False
        for seg in self.segments:
            if seg.authority == "data":
                seen_data = True
            elif seen_data:
                raise ValueError("instruction segments must precede data; compiler will not reorder authority")

    def validate_tool_history(self, *, require_resolved: bool = False) -> None:
        seen, pending = set(), set()
        for seg in self.segments:
            for turn in seg.content if seg.kind == "history" else []:
                if turn["role"] == "tool":
                    call_id = turn["call_id"]
                    if call_id not in pending:
                        raise ValueError("orphan or duplicate tool result")
                    pending.remove(call_id)
                else:
                    if pending:
                        raise ValueError("tool results must follow the corresponding calls before another turn")
                    for call in turn.get("tool_calls", []):
                        if call["id"] in seen:
                            raise ValueError("tool call ids must be unique")
                        seen.add(call["id"])
                        pending.add(call["id"])
            if seg.kind == "user" and pending:
                raise ValueError("resolve tool calls before adding a user turn")
        if require_resolved and pending:
            raise ValueError("resolve pending tool calls before compiling an inference request")

    def prefix_chain(self) -> list[str]:
        prev = _sha256(b"pcf:0.2")
        out = []
        for seg in self.segments:
            prev = _sha256(bytes.fromhex(prev[7:]), bytes.fromhex(seg.hash[7:]))
            out.append(prev)
        return out

    def with_appended(self, *segs: Segment) -> Context:
        return Context((*self.segments, *segs), self.session_id, self.cache_namespace)

    def with_replaced(self, seg_id: str, new: Segment) -> Context:
        if not any(s.id == seg_id for s in self.segments):
            raise KeyError(seg_id)
        return Context(tuple(new if s.id == seg_id else s for s in self.segments), self.session_id, self.cache_namespace)

    def to_json(self) -> dict:
        result = {"pcf_version": PCF_VERSION, "segments": [s.to_json() for s in self.segments],
                  "cache_namespace": self.cache_namespace}
        if self.session_id is not None:
            result["session_id"] = self.session_id
        return result

    @classmethod
    def from_json(cls, doc: dict) -> Context:
        from .schemas import validate
        validate("pcf", doc)
        segments = []
        for raw in doc["segments"]:
            seg = Segment(raw["id"], raw["kind"], raw["content"], raw.get("stable"),
                          authority=raw.get("authority"), provenance=raw.get("provenance"))
            if "hash" in raw and raw["hash"] != seg.hash:
                raise ValueError(f"segment {seg.id!r}: declared hash does not match content")
            segments.append(seg)
        return cls(tuple(segments), doc.get("session_id"), doc.get("cache_namespace", "default"))


def first_divergence(a: Iterable[str], b: Iterable[str]) -> int:
    i = 0
    for x, y in zip(a, b):
        if x != y:
            return i
        i += 1
    return i

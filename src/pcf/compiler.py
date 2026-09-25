"""Compilation and conservative, explicitly labelled cache-cost estimates."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

from .cache import PrefixCache
from .descriptor import CacheDescriptor, hash_object
from .segments import Context, Segment, canonical_bytes, text_content
from .validation import integer, number


class Tokenizer(Protocol):
    tokenizer_hash: str
    is_estimate: bool
    def count(self, text: str) -> int: ...


@dataclass(frozen=True)
class Usage:
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    input_tokens: int

    def __post_init__(self):
        for name in ("cache_read_input_tokens", "cache_creation_input_tokens", "input_tokens"):
            integer(getattr(self, name), name)

    @property
    def cold_tokens(self):
        return self.cache_creation_input_tokens + self.input_tokens

    @property
    def total_input_tokens(self):
        return self.cache_read_input_tokens + self.cold_tokens


class UnsupportedRequest(ValueError):
    """The target API rejects this request shape; another candidate may still serve the context."""


@dataclass(frozen=True)
class Warmth:
    warm_prefix_segments: int
    warm_tokens: int
    cold_tokens: int
    cache_creation_tokens: int
    uncached_tokens: int
    cache_state: str
    observed_at: float


@dataclass(frozen=True)
class CompiledPrompt:
    descriptor: CacheDescriptor
    request_json: bytes
    segment_tokens: tuple[int, ...]
    breakpoints: tuple[int, ...]
    chain: tuple[str, ...]  # portable identity, independent of provider
    native_chain: tuple[str, ...]  # compiled input prefix, excludes output/cache controls
    cache_key: str
    cache_namespace: str
    lookup_indices: tuple[int, ...]
    token_count_is_estimate: bool

    @property
    def request(self) -> dict[str, Any]:
        return json.loads(self.request_json)

    @property
    def cum_tokens(self) -> list[int]:
        acc, result = 0, []
        for n in self.segment_tokens:
            acc += n
            result.append(acc)
        return result

    @property
    def total_tokens(self):
        return sum(self.segment_tokens)


def segment_text(seg: Segment) -> str:
    return text_content(seg.content)


def data_text(seg: Segment) -> str:
    """Preserve data kind/provenance without moving it into an instruction channel."""
    return canonical_bytes({"kind": seg.kind, "source": seg.provenance, "data": seg.content}).decode()


def choose_breakpoints(ctx: Context, max_breakpoints: int, supported=None, *, history_slots: int = 2) -> list[int]:
    """Keep stable group anchors and recent history endpoints, up to the explicit budget.

    History endpoints are retained individually so an append-only explicit-mode
    request can still name a previously written endpoint. Empty segments, and
    indices rejected by ``supported``, have no native marker location and do not
    consume budget. Memory segments sharing a ``provenance`` form one module with its
    own anchor, so a change to a later module keeps earlier modules warm; memory without
    provenance shares the document anchor. Tail memory (after history) is rebuilt every turn
    in a conversation, so an entry there is reused only by a request that repeats the same
    history (a retry); it never takes a breakpoint. A history boundary that leaves a tool call
    waiting for its result can never end a request, so it takes no slot either. Stability is
    a hint, not a cache guarantee.

    Over budget (and above a budget of one, which marks the newest candidate), the last anchor
    is kept (with no anchor, the oldest candidate stands in); then the last anchor of the
    leading tools/system run when ``history_slots`` endpoints (or every history candidate, if
    fewer) still fit beside both anchors, so one mislabelled
    volatile module cannot take the system prompt out of cache; then the newest history
    endpoints, up to ``history_slots``; then the newest remaining candidates. ``history_slots`` is how
    many history endpoints a compiler needs for the next request to name the endpoint this
    request writes: providers that read only at markers present in the request need one
    per markable segment a request can append.
    """
    integer(max_breakpoints, "max_breakpoints")
    integer(history_slots, "history_slots")
    first_history = next((i for i, seg in enumerate(ctx.segments) if seg.kind == "history"), len(ctx.segments))
    pending, settled = set(), set()  # history indices whose end leaves no tool call waiting for its result
    for i, seg in enumerate(ctx.segments):
        if seg.kind == "history":
            for turn in seg.content:
                if turn["role"] == "tool":
                    pending.discard(turn["call_id"])
                else:
                    pending.update(call["id"] for call in turn.get("tool_calls", []))
            if not pending:
                settled.add(i)
    groups, history = {}, []
    for i, seg in enumerate(ctx.segments):
        if not seg.stable or not seg.content or (supported is not None and not supported(i)):
            continue
        if seg.kind == "history":
            if i in settled:
                history.append(i)
        elif i > first_history and seg.kind in {"memory", "document"}:
            continue  # tail memory: rebuilt every turn, so an entry here is rewritten, not read
        elif seg.kind == "memory" and seg.provenance is not None:
            groups[("memory", seg.provenance)] = i
        else:
            groups["context" if seg.kind in {"memory", "document"} else seg.kind] = i
    candidates = sorted([*groups.values(), *history])
    if len(candidates) <= max_breakpoints:
        return candidates
    if max_breakpoints == 0:
        return []
    if max_breakpoints == 1:
        return [candidates[-1]]
    anchors = [i for i in candidates if ctx.segments[i].kind not in {"history", "user"}]
    anchor = anchors[-1] if anchors else candidates[0]
    keep = {anchor}
    need = min(history_slots, len(history))
    lead = [i for i in anchors if ctx.segments[i].kind in {"tools", "system"}]
    if lead and lead[-1] != anchor and max_breakpoints - 2 >= need:
        keep.add(lead[-1])
    reserve = min(need, max_breakpoints - len(keep))  # history endpoints before any newer non-history candidate
    if reserve:
        keep |= set([i for i in history if i not in keep][-reserve:])
    room = max_breakpoints - len(keep)
    rest = [i for i in candidates if i not in keep]
    return sorted({*keep, *(rest[-room:] if room else [])})


class ContextCompiler(ABC):
    descriptor: CacheDescriptor
    tokenizer: Tokenizer
    compiler_id = "pcf.base:0.2"
    cache_mode = "simulated"
    implicit_breakpoint = False  # the provider places its own breakpoint when a request carries none
    history_slots = 2  # history endpoints kept over budget; see choose_breakpoints

    @abstractmethod
    def render(self, ctx: Context, breakpoints: list[int]) -> dict[str, Any]: ...

    def supports_boundary(self, ctx: Context, index: int) -> bool:
        return bool(ctx.segments[index].content)

    @property
    def generation_identity(self) -> dict:
        """Non-default settings that change what the model does or sees (e.g. thinking); part of quality identity."""
        return {}

    def covered_tokens(self, ctx: Context, index: int, cum_tokens: list[int]) -> int:
        """Prefix tokens a marker on segment ``index`` caches; adapters whose marker sits inside a segment trim it."""
        return cum_tokens[index]

    def native_input(self, request: dict) -> dict:
        """Subclasses must include every input-affecting field, excluding generation controls."""
        return {k: v for k, v in request.items() if k not in {
            "model", "max_tokens", "max_output_tokens", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention"}}

    def lookup_boundaries(self, breakpoints: list[int], positions: list[int]) -> list[int]:
        return list(range(len(positions)))

    def native_positions(self, request: dict) -> int:
        return len(request.get("messages", request.get("input", [])))

    def native_token_count(self, native: dict) -> int:
        """Count the same rendered input whose digest names the cached prefix.

        Empty containers carry no input. Provider counts remain estimates; the
        simulator defines its billing units as tokens of this canonical JSON.
        A supplied counter must be deterministic and monotonic on native prefixes.
        """
        payload = {key: value for key, value in native.items() if value not in (None, [], {}, "")}
        return integer(self.tokenizer.count(canonical_bytes(payload).decode("utf-8")), "token count") if payload else 0

    @property
    def cache_key(self):
        return self.cache_key_for(self.descriptor)

    def cache_key_for(self, descriptor):
        return hash_object("pcf:cache:0.2", {"execution": descriptor.compat_key,
                           "compiler": self.compiler_id, "counter": self.tokenizer.tokenizer_hash,
                           "accounting": "native-input:1"})

    @property
    def candidate_fingerprint(self):
        return hash_object("pcf:candidate:0.2", {"model": self.descriptor.model_id, "cache": self.cache_key})

    def compile(self, ctx: Context) -> CompiledPrompt:
        ctx.validate_tool_history(require_resolved=True)
        breakpoints = choose_breakpoints(ctx, self.descriptor.max_breakpoints, lambda i: self.supports_boundary(ctx, i),
                                         history_slots=self.history_slots)
        native, positions, counts = [], [], []
        previous_count = 0
        for end in range(1, len(ctx.segments) + 1):
            prefix = Context(ctx.segments[:end], ctx.session_id, ctx.cache_namespace)
            rendered = self.render(prefix, [])
            native_input = self.native_input(rendered)
            native.append(hash_object("pcf:native:0.2", native_input))
            positions.append(self.native_positions(rendered))
            cumulative = self.native_token_count(native_input)
            if cumulative < previous_count:
                raise ValueError("token counter must be monotonic on native input prefixes")
            counts.append(cumulative - previous_count)
            previous_count = cumulative
        return CompiledPrompt(self.descriptor, canonical_bytes(self.render(ctx, breakpoints)), tuple(counts),
                              tuple(breakpoints), tuple(ctx.prefix_chain()), tuple(native), self.cache_key,
                              ctx.cache_namespace, tuple(self.lookup_boundaries(breakpoints, positions)),
                              self.tokenizer.is_estimate or self.descriptor.identity_kind != "simulated")

    def warmth(self, ctx: Context, cache: PrefixCache, now: float) -> Warmth:
        number(now, "now")
        compiled = self.compile(ctx)
        # Closed APIs expose aggregate usage after execution, not a live preflight
        # cache inventory. Never promote a caller's local dictionary into knowledge
        # of a provider's cache. Native reuse is unknown until the provider responds.
        simulated = self.descriptor.identity_kind == "simulated"
        hit = cache.peek(compiled.cache_key, compiled.native_chain, now, namespace=ctx.cache_namespace,
                         eligible_indices=compiled.lookup_indices) if simulated else -1
        warm = compiled.cum_tokens[hit] if hit >= 0 else 0
        # An implicit breakpoint's position is unknown; assume the most it could write (the whole prompt).
        implicit = (len(ctx.segments) - 1,) if self.implicit_breakpoint and not compiled.breakpoints else ()
        cum, covered = compiled.cum_tokens, warm
        for index in compiled.breakpoints or implicit:
            reach = self.covered_tokens(ctx, index, cum) if index in compiled.breakpoints else cum[index]
            if index > hit and reach >= self.descriptor.min_cacheable_tokens:
                covered = max(covered, reach)
        written = covered - warm
        cold = compiled.total_tokens - warm
        return Warmth(hit + 1, warm, cold, written, cold - written,
                      "simulated" if simulated else "unknown", now)

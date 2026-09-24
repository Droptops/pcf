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


def choose_breakpoints(ctx: Context, max_breakpoints: int) -> list[int]:
    """Keep stable group anchors and recent history endpoints, up to the explicit budget.

    History endpoints are retained individually so an append-only explicit-mode
    request can still name a previously written endpoint. Empty segments have no
    native marker location. Stability is a hint, not a cache guarantee.
    """
    integer(max_breakpoints, "max_breakpoints")
    groups, history = {}, []
    for i, seg in enumerate(ctx.segments):
        if not seg.stable or not seg.content:
            continue
        if seg.kind == "history":
            history.append(i)
        else:
            groups["context" if seg.kind in {"memory", "document"} else seg.kind] = i
    candidates = sorted([*groups.values(), *history])
    if len(candidates) <= max_breakpoints:
        return candidates
    if max_breakpoints == 0:
        return []
    anchors = [i for i in candidates if ctx.segments[i].kind not in {"history", "user"}]
    anchor = anchors[-1] if anchors else candidates[0]
    if max_breakpoints == 1:
        return [candidates[-1]]
    return sorted({anchor, *[i for i in candidates if i != anchor][-(max_breakpoints - 1):]})


class ContextCompiler(ABC):
    descriptor: CacheDescriptor
    tokenizer: Tokenizer
    compiler_id = "pcf.base:0.2"
    cache_mode = "simulated"

    @abstractmethod
    def render(self, ctx: Context, breakpoints: list[int]) -> dict[str, Any]: ...

    def supports_boundary(self, ctx: Context, index: int) -> bool:
        return bool(ctx.segments[index].content)

    def native_input(self, request: dict) -> dict:
        """Subclasses must include every input-affecting field, excluding generation controls."""
        return {k: v for k, v in request.items() if k not in {
            "model", "max_tokens", "max_output_tokens", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention"}}

    def lookup_boundaries(self, breakpoints: list[int], positions: list[int]) -> list[int]:
        return list(range(len(positions)))

    def native_positions(self, request: dict) -> int:
        return len(request.get("messages", request.get("input", [])))

    @property
    def cache_key(self):
        return self.cache_key_for(self.descriptor)

    def cache_key_for(self, descriptor):
        return hash_object("pcf:cache:0.2", {"execution": descriptor.compat_key,
                           "compiler": self.compiler_id, "counter": self.tokenizer.tokenizer_hash})

    @property
    def candidate_fingerprint(self):
        return hash_object("pcf:candidate:0.2", {"model": self.descriptor.model_id, "cache": self.cache_key})

    def compile(self, ctx: Context) -> CompiledPrompt:
        ctx.validate_tool_history(require_resolved=True)
        breakpoints = [i for i in choose_breakpoints(ctx, self.descriptor.max_breakpoints)
                       if self.supports_boundary(ctx, i)]
        native, positions = [], []
        for end in range(1, len(ctx.segments) + 1):
            prefix = Context(ctx.segments[:end], ctx.session_id, ctx.cache_namespace)
            rendered = self.render(prefix, [])
            native.append(hash_object("pcf:native:0.2", self.native_input(rendered)))
            positions.append(self.native_positions(rendered))
        counts = tuple(integer(self.tokenizer.count(segment_text(s)), "token count") for s in ctx.segments)
        return CompiledPrompt(self.descriptor, canonical_bytes(self.render(ctx, breakpoints)), counts,
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
        last_write = hit
        for index in compiled.breakpoints:
            if index > hit and compiled.cum_tokens[index] >= self.descriptor.min_cacheable_tokens:
                last_write = index
        written = (compiled.cum_tokens[last_write] if last_write >= 0 else 0) - warm
        cold = compiled.total_tokens - warm
        return Warmth(hit + 1, warm, cold, written, cold - written,
                      "simulated" if simulated else "unknown", now)

"""Two simulated model families with different tokenizers, plus a serving engine that bills like a vendor.

Nothing here is a language model. The engine computes exactly what a prompt cache would bill for a
request given the store's state, using Anthropic's three-way usage split. That is the quantity the
falsification tests assert on.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from ..cache import PrefixCache
from ..compiler import CompiledPrompt, ContextCompiler, Usage
from ..descriptor import CacheDescriptor, Layout, sha256_tag
from ..segments import Context

_WORDISH = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class WordTokenizer:
    """Family A: one token per word or punctuation mark."""

    tokenizer_hash: str = sha256_tag("sim-tokenizer:wordish-v1")
    is_estimate: bool = False

    def count(self, text: str) -> int:
        return len(_WORDISH.findall(text))


@dataclass(frozen=True)
class CharChunkTokenizer:
    """Family B: ceil(chars / chars_per_token). Deliberately different counts from A for the same text."""

    chars_per_token: float = 3.7
    tokenizer_hash: str = sha256_tag("sim-tokenizer:charchunk-3.7-v1")
    is_estimate: bool = False

    def count(self, text: str) -> int:
        return math.ceil(len(text) / self.chars_per_token) if text else 0


def sim_descriptor(family: str, model_id: str, tokenizer_hash: str, *, min_cacheable: int = 64,
                   max_breakpoints: int = 4, ttl_seconds: int = 300) -> CacheDescriptor:
    return CacheDescriptor(
        family=family,
        model_id=model_id,
        weights_hash=sha256_tag(f"weights:{family}:{model_id}"),
        tokenizer_hash=tokenizer_hash,
        layout=Layout(positional="rope", dtype="bf16", attention="gqa"),
        min_cacheable_tokens=min_cacheable,
        identity_kind="simulated",
        max_breakpoints=max_breakpoints,
        ttl_seconds=ttl_seconds,
    )


class SimCompiler(ContextCompiler):
    compiler_id = "pcf.sim:0.2"
    """Renders a PCF into a neutral request. Every segment lands somewhere; nothing is dropped."""

    def __init__(self, descriptor: CacheDescriptor, tokenizer: Any) -> None:
        self.descriptor = descriptor
        self.tokenizer = tokenizer

    def render(self, ctx: Context, breakpoints: list[int]) -> dict[str, Any]:
        marks = set(breakpoints)
        tools: list[Any] = []
        system: list[dict[str, Any]] = []
        messages: list[dict[str, Any]] = []
        for i, seg in enumerate(ctx.segments):
            if seg.kind == "tools":
                tools.extend(seg.content)
                if i in marks and tools:
                    tools[-1] = {**tools[-1], "cache_mark": True}
            elif seg.authority == "instruction":
                system.append({"kind": seg.kind, "text": seg.content, **({"cache_mark": True} if i in marks else {})})
            elif seg.kind == "history":
                for j, turn in enumerate(seg.content):
                    msg = dict(turn)
                    if i in marks and j == len(seg.content) - 1:
                        msg["cache_mark"] = True
                    messages.append(msg)
            else:  # data and user
                messages.append({"role": "user", "content": seg.content, **({"cache_mark": True} if i in marks else {})})
        return {"model": self.descriptor.model_id, "tools": tools, "system": system, "messages": messages}


@dataclass
class SimEngine:
    """A serving engine for one family. Several engines may share one PrefixCache (Layer 1 interop)."""

    compiler: ContextCompiler
    cache: PrefixCache
    engine_name: str = "sim-engine"
    engine_version: str = "1"
    block_tokens: int = 16

    @property
    def descriptor(self) -> CacheDescriptor:
        return self.compiler.descriptor.with_engine(self.engine_name, self.engine_version, self.block_tokens)

    def run(self, ctx: Context, now: float) -> tuple[Usage, CompiledPrompt]:
        compiled = self.compiler.compile(ctx)
        key = self.compiler.cache_key_for(self.descriptor)
        cum = compiled.cum_tokens
        hit = self.cache.peek(key, compiled.native_chain, now, namespace=ctx.cache_namespace, eligible_indices=compiled.lookup_indices)
        if hit >= 0:
            self.cache.touch(key, compiled.native_chain[hit], now, namespace=ctx.cache_namespace)
        warm = cum[hit] if hit >= 0 else 0

        # Write an entry at every breakpoint past the hit whose prefix meets the vendor minimum.
        last_written = hit
        for b in compiled.breakpoints:
            if b > hit and cum[b] >= self.descriptor.min_cacheable_tokens:
                self.cache.write(key, compiled.native_chain[b], cum[b], now, namespace=ctx.cache_namespace, ttl_seconds=self.descriptor.ttl_seconds)
                last_written = b
        created = (cum[last_written] if last_written >= 0 else 0) - warm
        after = compiled.total_tokens - warm - created
        return Usage(cache_read_input_tokens=warm, cache_creation_input_tokens=created, input_tokens=after), compiled


def family_a(store: PrefixCache | None = None, **kw: Any) -> SimEngine:
    tok = WordTokenizer()
    # `store if ... is not None` — an empty PrefixCache is falsy (__len__ == 0), so `store or ...` would drop it.
    return SimEngine(SimCompiler(sim_descriptor("sim-a", "sim-a-large", tok.tokenizer_hash, **kw), tok),
                     store if store is not None else PrefixCache(ttl_seconds=kw.get("ttl_seconds", 300)))


def family_b(store: PrefixCache | None = None, **kw: Any) -> SimEngine:
    tok = CharChunkTokenizer()
    return SimEngine(SimCompiler(sim_descriptor("sim-b", "sim-b-small", tok.tokenizer_hash, **kw), tok),
                     store if store is not None else PrefixCache(ttl_seconds=kw.get("ttl_seconds", 300)))

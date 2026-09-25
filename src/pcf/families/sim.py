"""Two simulated model families with different tokenizers, plus a serving engine that bills like a vendor.

Nothing here is a language model. The engine computes exactly what a prompt cache would bill for a
request given the store's state, using Anthropic's three-way usage split. That is the quantity the
falsification tests assert on.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from ..cache import PrefixCache
from ..compiler import CompiledPrompt, ContextCompiler, Usage
from ..descriptor import CacheDescriptor, Layout, hash_object, sha256_tag
from ..segments import Context
from ..validation import number

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
    tokenizer_hash: str = field(init=False)
    is_estimate: bool = False

    def __post_init__(self):
        size = number(self.chars_per_token, "chars_per_token", minimum=1e-12)
        object.__setattr__(self, "chars_per_token", size)
        object.__setattr__(self, "tokenizer_hash", hash_object("sim-tokenizer:charchunk-v2", {"chars_per_token": size}))

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
    """Renders a PCF into a neutral request. Every segment lands somewhere; nothing is dropped."""
    compiler_id = "pcf.sim:0.2"

    def __init__(self, descriptor: CacheDescriptor, tokenizer: Any) -> None:
        self.descriptor = descriptor
        self.tokenizer = tokenizer

    def render(self, ctx: Context, breakpoints: list[int]) -> dict[str, Any]:
        result = self._result([], [], [])
        for result in self._steps(ctx, breakpoints):
            pass
        return result

    def render_prefixes(self, ctx):
        return self._steps(ctx, [])

    def _result(self, tools: list, system: list, messages: list) -> dict[str, Any]:
        return {"model": self.descriptor.model_id, "tools": tools, "system": system, "messages": messages}

    def _steps(self, ctx: Context, breakpoints: list[int]):
        """The request after each segment; lists only grow."""
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
                    msg = {k: v for k, v in turn.items() if k != "provider_blocks"}  # opaque to the simulator
                    if i in marks and j == len(seg.content) - 1:
                        msg["cache_mark"] = True
                    messages.append(msg)
            else:  # data and user
                messages.append({"role": "user", "content": seg.content, **({"cache_mark": True} if i in marks else {})})
            yield self._result(tools, system, messages)


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
        # Reads and writes use the prefix each marker actually covers (its native hash and token count), which
        # for an OpenAI history marker ends before the segment's trailing assistant items.
        candidates = compiled.read_candidates()
        found = self.cache.peek(key, [digest for _, digest, _ in candidates], now, namespace=ctx.cache_namespace)
        warm = candidates[found][0] if found >= 0 else 0
        if found >= 0:
            self.cache.touch(key, candidates[found][1], now, namespace=ctx.cache_namespace)

        # Write an entry at every marker past the hit whose prefix meets the vendor minimum.
        written = warm
        for _, digest, tokens in compiled.marker_prefixes:
            if tokens > warm and tokens >= self.descriptor.min_cacheable_tokens:
                self.cache.write(key, digest, tokens, now, namespace=ctx.cache_namespace,
                                 ttl_seconds=self.descriptor.ttl_seconds)
                written = max(written, tokens)
        created = written - warm
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

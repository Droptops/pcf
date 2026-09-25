"""Anthropic native text/tool compiler; external documents remain data messages."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..compiler import ContextCompiler, UnsupportedRequest, Usage, data_text
from ..descriptor import CacheDescriptor, Layout, sha256_tag
from ..segments import Context, text_content
from ..validation import integer

# Minimum cacheable prefix per platform.claude.com/docs/en/build-with-claude/prompt-caching, checked 2026-09-25.
MIN_CACHEABLE = {"claude-fable-5-1": 512, "claude-fable-5": 512, "claude-mythos-5-1": 512, "claude-mythos-5": 512,
                 "claude-opus-5-5": 512, "claude-opus-5": 512, "claude-mythos-preview": 2048,
                 "claude-sonnet-5": 1024, "claude-haiku-4-5": 4096,
                 "claude-sonnet-4-6": 1024, "claude-opus-4-8": 1024,
                 "claude-opus-4-7": 2048, "claude-opus-4-6": 4096}
# "Claude 4.6 and later models and Claude Mythos Preview" reject a final assistant turn (prefill) with a 400, per
# platform.claude.com/docs/en/api/errors ("Prefill not supported"), checked 2026-09-24. IDs per Anthropic's models
# overview; unlisted models are not checked.
PREFILL_REJECTED = {"claude-fable-5-1", "claude-fable-5", "claude-mythos-5-1", "claude-mythos-5", "claude-mythos-preview",
                    "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5", "claude-sonnet-4-6",
                    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6"}


THINKING_BLOCKS = {"thinking", "redacted_thinking"}


def history_from_response(response: Any) -> list[dict]:
    """Normalize supported Anthropic assistant content blocks without data loss.

    Thinking and redacted_thinking blocks are kept verbatim as ``provider_blocks`` so they can be replayed to the
    model that produced them. A turn replays its blocks before its text and calls, so thinking that follows text
    starts a new assistant turn; consecutive assistant turns render as one message in the original order.
    """
    blocks = response.get("content", []) if isinstance(response, dict) else getattr(response, "content", [])
    turns: list[dict] = []
    text, calls, kept = [], [], []

    def close() -> None:
        if "".join(text) or calls:  # empty text does not end a turn, so thinking blocks around it stay together
            turns.append({"role": "assistant", "content": "".join(text), **({"tool_calls": list(calls)} if calls else {}),
                          **({"provider_blocks": list(kept)} if kept else {})})
            text.clear(), calls.clear(), kept.clear()

    for block in blocks:
        block = block if isinstance(block, dict) else block.model_dump(exclude_none=True)
        typ = block.get("type")
        if typ in THINKING_BLOCKS:
            if calls:  # a call's result must follow it; thinking cannot sit between them in neutral history
                raise ValueError("Anthropic thinking after tool_use is not representable")
            close()  # thinking after text opens the next turn
            kept.append({"provider": "anthropic", "block": block})
        elif typ == "text":
            if (calls and block.get("text")) or block.get("citations"):  # would reorder text or drop citations
                raise ValueError("Anthropic text after tool_use or with citations is not representable")
            text.append(block.get("text", ""))
        elif typ == "tool_use":
            calls.append({"id": block["id"], "name": block["name"], "arguments": block["input"]})
        else:
            raise ValueError(f"unsupported Anthropic response block: {typ!r}")
    if kept and not ("".join(text) or calls):
        raise ValueError("Anthropic thinking without text or tool_use is not representable")
    close()
    return turns


@dataclass(frozen=True)
class HeuristicTokenizer:
    tokenizer_hash: str = sha256_tag("heuristic:chars/4")
    is_estimate: bool = True
    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class ScaledTokenizer:
    """A deterministic correction of another counter by a fixed factor, with its own identity.

    Measured 2026-09-25 against billed input on the placement harness (results/2026-09-25/): chars/4 undercounts
    claude-sonnet-5 by 1.21-1.42x and overcounts gpt-5.6 at 0.86-0.98x (billed over estimated input per run). A fixed factor
    narrows cross-family cost comparisons; it does not make counts exact.
    """

    is_estimate = True

    def __init__(self, base, factor: float) -> None:
        if not (isinstance(factor, (int, float)) and not isinstance(factor, bool) and math.isfinite(factor)
                and factor > 0):
            raise ValueError("factor must be a positive finite number")
        self.base, self.factor = base, float(factor)
        self.tokenizer_hash = sha256_tag(f"scaled:{self.factor!r}:{base.tokenizer_hash}")  # normalized: 2 == 2.0

    def count(self, text: str) -> int:
        return math.ceil(self.base.count(text) * self.factor)


def anthropic_descriptor(model_id, *, ttl="5m", min_cacheable_tokens=None):
    if ttl not in {"5m", "1h"}:
        raise ValueError("ttl must be 5m or 1h")
    minimum = MIN_CACHEABLE.get(model_id) if min_cacheable_tokens is None else min_cacheable_tokens
    if minimum is None:
        raise ValueError(f"unknown cache profile for {model_id!r}; supply min_cacheable_tokens")
    integer(minimum, "min_cacheable_tokens", minimum=1)
    return CacheDescriptor("anthropic", model_id, sha256_tag(f"opaque:anthropic:{model_id}"),
                           sha256_tag(f"opaque:anthropic-tokenizer:{model_id}"), Layout("rope", "bf16", "gqa"),
                           minimum, 4, 3600 if ttl == "1h" else 300, identity_kind="opaque",
                           cache_write_multiplier=2 if ttl == "1h" else 1.25)


class AnthropicCompiler(ContextCompiler):
    compiler_id = "pcf.anthropic.messages:0.2"
    cache_mode = "explicit"

    def __init__(self, model_id: str, *, ttl="5m", tokenizer=None, max_tokens=1024, min_cacheable_tokens=None,
                 thinking: dict | None = None, thinking_blocks: str = "replay"):
        """``thinking`` is sent as the request's thinking configuration (omitted when None). ``thinking_blocks``
        chooses what happens to thinking blocks kept in history: "replay" sends them back unchanged (required
        within a tool round on thinking models), "drop" strips every one. A replayed block is bound to the exact
        prefix that produced it; see SPEC for tail memory."""
        self.descriptor = anthropic_descriptor(model_id, ttl=ttl, min_cacheable_tokens=min_cacheable_tokens)
        self.tokenizer = tokenizer if tokenizer is not None else HeuristicTokenizer()
        self.ttl, self.max_tokens = ttl, integer(max_tokens, "max_tokens", minimum=1)
        if thinking is not None and (not isinstance(thinking, dict) or not isinstance(thinking.get("type"), str)):
            raise ValueError("thinking must be an object with a string type")
        if thinking_blocks not in {"replay", "drop"}:
            raise ValueError("thinking_blocks must be replay or drop")
        self.thinking, self.thinking_blocks = (dict(thinking) if thinking else None), thinking_blocks

    @property
    def generation_identity(self) -> dict:
        return {**({"thinking": self.thinking} if self.thinking is not None else {}),
                **({"thinking_blocks": self.thinking_blocks} if self.thinking_blocks != "replay" else {})}

    def _mark(self):
        return {"type": "ephemeral", **({"ttl": "1h"} if self.ttl == "1h" else {})}

    def render(self, ctx: Context, breakpoints: list[int]) -> dict:
        marks, tools, system, messages = set(breakpoints), [], [], []
        def message(role, blocks):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].extend(blocks)
            else:
                messages.append({"role": role, "content": blocks})
        for i, seg in enumerate(ctx.segments):
            last = None
            if seg.kind == "tools":
                for tool in seg.content:
                    last = {"name": tool["name"], "description": tool.get("description", ""),
                            "input_schema": tool["parameters"]}
                    tools.append(last)
            elif seg.authority == "instruction":
                last = {"type": "text", "text": text_content(seg.content)}
                system.append(last)
            elif seg.kind == "history":
                for turn in seg.content:
                    if turn["role"] == "tool":
                        last = {"type": "tool_result", "tool_use_id": turn["call_id"],
                                "content": text_content(turn["content"]), "is_error": turn["is_error"]}
                        message("user", [last])
                    else:
                        blocks = [dict(b["block"]) for b in turn.get("provider_blocks", [])
                                  if b["provider"] == "anthropic" and self.thinking_blocks == "replay"]
                        if turn["content"]:
                            blocks.append({"type": "text", "text": text_content(turn["content"])})
                        for call in turn.get("tool_calls", []):
                            blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["arguments"]})
                        message(turn["role"], blocks)
                        last = blocks[-1]
            else:
                last = {"type": "text", "text": data_text(seg) if seg.kind in {"memory", "document"} else text_content(seg.content)}
                message("user", [last])
            if i in marks and last is not None:
                last["cache_control"] = self._mark()
        result = {"model": self.descriptor.model_id, "max_tokens": self.max_tokens, "messages": messages}
        if self.thinking is not None:
            result["thinking"] = dict(self.thinking)
        if tools:
            result["tools"] = tools
        if system:
            result["system"] = system
        return result

    def compile(self, ctx: Context):
        # Checked on the full request only: render() also builds prefix hashes, where these shapes are normal.
        compiled = super().compile(ctx)
        messages = compiled.request["messages"]
        if not messages:
            raise UnsupportedRequest("Anthropic Messages requires at least one message")
        if messages[-1]["role"] == "assistant" and self.descriptor.model_id in PREFILL_REJECTED:
            raise UnsupportedRequest(f"{self.descriptor.model_id} rejects a final assistant turn (prefill)")
        return compiled

    def native_positions(self, request):
        n = len(request.get("tools", [])) + len(request.get("system", []))
        previous = None
        for message in request.get("messages", []):
            for block in message["content"]:
                kind = block["type"]
                if kind not in {"tool_use", "tool_result"} or kind != previous:
                    n += 1
                previous = kind
        return n

    def lookup_boundaries(self, breakpoints, positions):
        return [i for i, position in enumerate(positions)
                if any(0 <= positions[b] - position < 20 for b in breakpoints if i <= b)]

    @staticmethod
    def usage_from_response(resp: Any) -> Usage:
        u = resp["usage"] if isinstance(resp, dict) else resp.usage
        def get(key):
            return (u.get(key, 0) if isinstance(u, dict) else getattr(u, key, 0)) or 0
        return Usage(get("cache_read_input_tokens"), get("cache_creation_input_tokens"), get("input_tokens"))

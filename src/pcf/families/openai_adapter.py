"""Responses API compiler with native content markers and stable cache partitions."""
from __future__ import annotations

from typing import Any

from ..compiler import ContextCompiler, Usage, data_text
from ..descriptor import CacheDescriptor, Layout, hash_object, sha256_tag
from ..segments import Context, Segment, text_content
from ..validation import integer
from .anthropic_adapter import HeuristicTokenizer
from .capabilities import OpenAICapabilities, openai_profile


def history_from_response(response: Any) -> list[dict]:
    """Losslessly normalize supported Responses output items into PCF history.

    Reasoning items are kept verbatim as ``provider_blocks`` on the assistant turn they precede, so they can be
    replayed (replaying needs the item stored server-side, or requested with encrypted content).
    """
    raw = response.get("output", []) if isinstance(response, dict) else getattr(response, "output", [])
    turns: list[dict] = []
    pending: list[dict] = []  # reasoning items waiting for the assistant turn they precede

    def take_pending(turn: dict) -> dict:
        if pending:
            turn.setdefault("provider_blocks", []).extend(pending)
            pending.clear()
        return turn

    for item in raw:
        item = item if isinstance(item, dict) else item.model_dump(exclude_none=True)
        typ = item.get("type")
        if typ == "reasoning":
            pending.append({"provider": "openai", "block": item})
        elif typ == "message":
            blocks = item.get("content", [])
            unsupported = [b.get("type") for b in blocks
                           if b.get("type") not in {"output_text", "text"} or b.get("annotations")]
            if unsupported:
                raise ValueError(f"unsupported Responses content blocks: {unsupported}")
            text = "".join(b.get("text", "") for b in blocks)
            if text:
                turns.append(take_pending({"role": "assistant", "content": text}))
        elif typ == "function_call":
            import json
            try:
                args = json.loads(item["arguments"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid Responses function_call arguments") from exc
            if not isinstance(args, dict):
                raise ValueError("function_call arguments must be an object")
            call = {"id": item["call_id"], "name": item["name"], "arguments": args}
            if turns and turns[-1]["role"] == "assistant":  # parallel calls (or calls after text) form one turn
                if pending:  # rendering replays provider blocks before the turn's text and calls
                    raise ValueError("reasoning between output items of one turn is not representable")
                turns[-1].setdefault("tool_calls", []).append(call)
            else:
                turns.append(take_pending({"role": "assistant", "content": "", "tool_calls": [call]}))
        elif typ == "function_call_output":
            if pending:
                raise ValueError("reasoning before a tool output is not representable")
            output = item.get("output", "")
            if isinstance(output, list):
                unsupported = [x.get("type") for x in output if x.get("type") != "input_text"]
                if unsupported:
                    raise ValueError(f"unsupported function_call_output content parts: {unsupported}")
                output = "".join(x.get("text", "") for x in output)
            if not isinstance(output, (str, dict)):
                raise ValueError("function_call_output output must be text or JSON")
            turns.append({"role": "tool", "call_id": item["call_id"], "content": output, "is_error": False})
        else:
            raise ValueError(f"unsupported Responses output item: {typ!r}")
    if pending:
        raise ValueError("reasoning without a following message or call is not representable")
    return turns


def openai_descriptor(model_id, *, capabilities=None):
    p = openai_profile(model_id, capabilities)
    return CacheDescriptor("openai", model_id, sha256_tag(f"opaque:openai:{model_id}"),
                           sha256_tag(f"opaque:openai-tokenizer:{model_id}"), Layout("rope", "bf16", "gqa"),
                           p.min_tokens, p.max_breakpoints, p.ttl_seconds,
                           identity_kind="opaque", cache_write_multiplier=p.write_multiplier)


class OpenAICompiler(ContextCompiler):
    compiler_id = "pcf.openai.responses:0.2"
    implicit_breakpoint = True  # unmarked requests use implicit mode, where OpenAI picks the breakpoint
    # Reads only hit markers present in the request, so a request that appends two markable segments
    # (e.g. [user, call] then [tool]) must still name the endpoint the previous request wrote.
    history_slots = 3

    def __init__(self, model_id: str, *, tokenizer=None, capabilities: OpenAICapabilities | None = None,
                 reasoning_items: str = "replay"):
        """``reasoning_items`` chooses what happens to reasoning items kept in history: "replay" sends them before
        the assistant turn they preceded, "drop" strips them."""
        if reasoning_items not in {"replay", "drop"}:
            raise ValueError("reasoning_items must be replay or drop")
        self.reasoning_items = reasoning_items
        self.profile = openai_profile(model_id, capabilities)
        self.descriptor = openai_descriptor(model_id, capabilities=self.profile)
        self.tokenizer = tokenizer if tokenizer is not None else HeuristicTokenizer()
        self.explicit = self.profile.explicit
        self.cache_mode = "explicit" if self.explicit else "implicit"

    def supports_boundary(self, ctx, index):
        seg = ctx.segments[index]
        if seg.kind == "tools" or not seg.content:
            return False  # definitions are covered by a later native message boundary
        if seg.kind == "history":
            # Assistant text and call items cannot carry a marker (the API accepts one on assistant
            # output_text but writes nothing), so the marker goes on the segment's last user/tool item.
            return any(t["role"] == "tool" or (t["role"] == "user" and t["content"]) for t in seg.content)
        return True

    def covered_tokens(self, ctx, index, cum_tokens):
        """A history marker sits on the last user/tool item, so trailing assistant turns are not written yet.

        The covered prefix is counted in the same native units as ``cum_tokens``: the request rendered with
        segment ``index`` cut after that item.
        """
        seg = ctx.segments[index]
        if seg.kind != "history":
            return cum_tokens[index]
        cut = max((k for k, t in enumerate(seg.content)
                   if t["role"] == "tool" or (t["role"] == "user" and t["content"])), default=None)
        if cut is None or cut == len(seg.content) - 1:
            return cum_tokens[index]
        head = Segment(seg.id, "history", seg.content[:cut + 1], seg.stable, provenance=seg.provenance)
        prefix = Context([*ctx.segments[:index], head], ctx.session_id, ctx.cache_namespace)
        return min(cum_tokens[index], self.native_token_count(self.native_input(self.render(prefix, []))))

    def render(self, ctx: Context, breakpoints: list[int]) -> dict:
        marks, tools, inputs, marked = set(breakpoints), [], [], False
        for i, seg in enumerate(ctx.segments):
            last = None
            if seg.kind == "tools":
                tools.extend({"type": "function", "name": tool["name"], "description": tool.get("description", ""),
                              "parameters": tool["parameters"], "strict": False} for tool in seg.content)
            elif seg.kind == "history":
                for turn in seg.content:
                    if turn["role"] == "tool":
                        # Responses does not expose Anthropic's is_error field; preserve it
                        # explicitly in the result data without changing the native item type.
                        output = turn["content"] if not turn["is_error"] else {"is_error": True, "content": turn["content"]}
                        last = {"type": "input_text", "text": text_content(output)}
                        inputs.append({"type": "function_call_output", "call_id": turn["call_id"], "output": [last]})
                    else:
                        if self.reasoning_items == "replay":
                            inputs.extend(dict(b["block"]) for b in turn.get("provider_blocks", [])
                                          if b["provider"] == "openai")
                        if turn["content"] and turn["role"] == "user":
                            last = {"type": "input_text", "text": text_content(turn["content"])}
                            inputs.append({"role": "user", "content": [last]})
                        elif turn["content"]:  # assistant input rejects input_text; plain text has no marker
                            inputs.append({"role": "assistant", "content": text_content(turn["content"])})
                        for call in turn.get("tool_calls", []):
                            inputs.append({"type": "function_call", "call_id": call["id"], "name": call["name"],
                                           "arguments": text_content(call["arguments"])})
            else:
                last = {"type": "input_text", "text": data_text(seg) if seg.kind in {"memory", "document"} and seg.authority == "data" else text_content(seg.content)}
                inputs.append({"role": "developer" if seg.authority == "instruction" else "user", "content": [last]})
            if i in marks and last is not None:
                last["prompt_cache_breakpoint"], marked = {"mode": "explicit"}, True
        result = {"model": self.descriptor.model_id, "input": inputs,
                  "prompt_cache_key": "pcf-" + hash_object("pcf:partition:0.2", {"namespace": ctx.cache_namespace})[7:39]}
        if tools:
            result["tools"] = tools
        if self.explicit:  # explicit mode with no marker disables caching, so fall back to implicit
            result["prompt_cache_options"] = {"mode": "explicit" if marked else "implicit",
                                              "ttl": self.profile.retention}
        else:
            result["prompt_cache_retention"] = self.profile.retention
        return result

    def lookup_boundaries(self, breakpoints, positions):
        return breakpoints if self.explicit else []

    @staticmethod
    def usage_from_response(resp: Any) -> Usage:
        u = resp["usage"] if isinstance(resp, dict) else resp.usage
        get = (lambda obj, key, default=0: obj.get(key, default)) if isinstance(u, dict) else (lambda obj, key, default=0: getattr(obj, key, default))
        total = integer(get(u, "input_tokens") or 0, "input_tokens")
        detail = get(u, "input_tokens_details", None)
        read = integer(get(detail, "cached_tokens") or 0, "cached_tokens") if detail is not None else 0
        written = integer(get(detail, "cache_write_tokens") or 0, "cache_write_tokens") if detail is not None else 0
        if read + written > total:
            raise ValueError("provider cache usage exceeds total input tokens")
        return Usage(read, written, total - read - written)

"""Responses API compiler with native content markers and stable cache partitions."""
from __future__ import annotations

from typing import Any

from ..compiler import ContextCompiler, Usage, data_text
from ..descriptor import CacheDescriptor, Layout, hash_object, sha256_tag
from ..segments import Context, text_content
from ..validation import integer
from .anthropic_adapter import HeuristicTokenizer
from .capabilities import OpenAICapabilities, openai_profile


def history_from_response(response: Any) -> list[dict]:
    """Losslessly normalize supported Responses output items into PCF history."""
    raw = response.get("output", []) if isinstance(response, dict) else getattr(response, "output", [])
    turns: list[dict] = []
    for item in raw:
        item = item if isinstance(item, dict) else item.model_dump()
        typ = item.get("type")
        if typ == "message":
            blocks = item.get("content", [])
            unsupported = [b.get("type") for b in blocks if b.get("type") not in {"output_text", "text"}]
            if unsupported:
                raise ValueError(f"unsupported Responses content blocks: {unsupported}")
            text = "".join(b.get("text", "") for b in blocks)
            if text:
                turns.append({"role": "assistant", "content": text})
        elif typ == "function_call":
            import json
            try:
                args = json.loads(item["arguments"])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError("invalid Responses function_call arguments") from exc
            if not isinstance(args, dict):
                raise ValueError("function_call arguments must be an object")
            turns.append({"role": "assistant", "content": "", "tool_calls":
                          [{"id": item["call_id"], "name": item["name"], "arguments": args}]})
        elif typ == "function_call_output":
            output = item.get("output", "")
            if isinstance(output, list):
                output = "".join(x.get("text", "") for x in output if x.get("type") == "input_text")
            if not isinstance(output, (str, dict)):
                raise ValueError("function_call_output output must be text or JSON")
            turns.append({"role": "tool", "call_id": item["call_id"], "content": output, "is_error": False})
        else:
            raise ValueError(f"unsupported Responses output item: {typ!r}")
    return turns


def openai_descriptor(model_id, *, capabilities=None):
    p = openai_profile(model_id, capabilities)
    return CacheDescriptor("openai", model_id, sha256_tag(f"opaque:openai:{model_id}"),
                           sha256_tag(f"opaque:openai-tokenizer:{model_id}"), Layout("rope", "bf16", "gqa"),
                           p.min_tokens, p.max_breakpoints, p.ttl_seconds,
                           identity_kind="opaque", cache_write_multiplier=p.write_multiplier)


class OpenAICompiler(ContextCompiler):
    compiler_id = "pcf.openai.responses:0.2"

    def __init__(self, model_id: str, *, tokenizer=None, capabilities: OpenAICapabilities | None = None):
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
            last = seg.content[-1]
            return last["role"] == "tool" or not last.get("tool_calls")
        return True

    def render(self, ctx: Context, breakpoints: list[int]) -> dict:
        marks, tools, inputs = set(breakpoints), [], []
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
                        if turn["content"]:
                            last = {"type": "input_text", "text": text_content(turn["content"])}
                            inputs.append({"role": turn["role"], "content": [last]})
                        for call in turn.get("tool_calls", []):
                            inputs.append({"type": "function_call", "call_id": call["id"], "name": call["name"],
                                           "arguments": text_content(call["arguments"])})
                            last = None  # a call item is not a supported content-block endpoint
            else:
                last = {"type": "input_text", "text": data_text(seg) if seg.kind in {"memory", "document"} and seg.authority == "data" else text_content(seg.content)}
                inputs.append({"role": "developer" if seg.authority == "instruction" else "user", "content": [last]})
            if i in marks and last is not None:
                last["prompt_cache_breakpoint"] = {"mode": "explicit"}
        result = {"model": self.descriptor.model_id, "input": inputs,
                  "prompt_cache_key": "pcf-" + hash_object("pcf:partition:0.2", {"namespace": ctx.cache_namespace})[7:39]}
        if tools:
            result["tools"] = tools
        if self.explicit:
            result["prompt_cache_options"] = {"mode": "explicit", "ttl": self.profile.retention}
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

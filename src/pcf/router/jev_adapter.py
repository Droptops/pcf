"""Jev (TypeSafe System One) as the router's confidence source (SPEC.md "Cache and routing").

Request/response shape per https://docs.typesafe.ai/api (fetched 2026-09-24):
  POST https://api.typesafe.ai/v1/systemone
  Authorization: Bearer <API_KEY>
  body: {"model": ..., "state": str|object|array, "questions": {<key>: {"type": "noul"|"choice"|"score", ...}}}
  resp: {"model": ..., "answers": {<key>: {"type": "noul", "noul": 0.95}}, "usage": {"input_tokens", "output_tokens"}}

OpenRouter serves the same body and answer shape at POST https://openrouter.ai/api/alpha/decisions with
a Bearer OpenRouter key (per @openrouter/sdk 1.3.26, alphaDecisionsCreate). Replies name the dated snapshot
(e.g. "typesafe/jev-1.13-20260917", observed 2026-09-24), so pin that ID: the alias fails the pin check.

Network is injected via `transport` so nothing here dials out unless a caller passes a real one.
Limits per docs: Choice <= 255 options, Score 2-10 levels. State size limits are not restated here;
TypeSafe publishes them and the API returns an error when exceeded.
"""
from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from http.client import HTTPException
from urllib.error import HTTPError
from typing import Any

from ..segments import Context, Segment
from .base import Candidate, ConfidenceSource, ConfidenceUnavailable
from ..descriptor import sha256_tag, hash_object
from ..validation import number
from urllib.parse import urlparse

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
OPENROUTER_DECISIONS_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
DEFAULT_MODEL = "jev-latest"  # never eligible for validation; pass a dated id such as "typesafe/jev-1.13-20260917"

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _prompt_only(ctx: Context) -> Context:
    """The context without opaque provider reasoning state, which is not part of the prompt being judged."""
    if not any(s.kind == "history" and any("provider_blocks" in t for t in s.content) for s in ctx.segments):
        return ctx
    segments = [Segment(s.id, s.kind, [{k: v for k, v in t.items() if k != "provider_blocks"} for t in s.content],
                        s.stable, authority=s.authority, provenance=s.provenance) if s.kind == "history" else s
                for s in ctx.segments]
    return Context(segments, ctx.session_id, ctx.cache_namespace)


def build_request(ctx: Context, candidate_model_id: str, *, model: str = DEFAULT_MODEL, rubric: str = "An acceptable answer is correct, follows the application instructions, preserves tool semantics, and satisfies the user request.") -> dict[str, Any]:
    """The exact wire body. State is the PCF document itself; Jev reads text/JSON only."""
    return {
        "model": model,
        "state": {
            "candidate_model": candidate_model_id,
            "context": _prompt_only(ctx).to_json(),
        },
        "questions": {
            "sufficient": {
                "type": "noul",
                "instructions": (
                    "Given `context` (the full prompt a model will receive), will `candidate_model` "
                    "produce an answer that a careful reviewer would accept without escalating to a "
                    "more capable model?"
                ),
                "criteria": {"true": rubric, "false": "The candidate's answer fails that acceptance rubric."},
            }
        },
    }


def parse_noul(resp: dict[str, Any], key: str = "sufficient") -> float:
    ans = resp["answers"][key]
    if ans.get("type") != "noul":
        raise ValueError(f"expected noul answer for {key!r}, got {ans.get('type')!r}")
    return number(ans["noul"], "noul", maximum=1)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A confidence endpoint must answer directly; credentials never follow redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "confidence endpoint redirects are not permitted", headers, fp)


def http_transport(api_key: str | None = None, endpoint: str = JEV_ENDPOINT, timeout: float = 10.0) -> Transport:
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("confidence endpoint must use HTTPS without URL credentials")
    key = api_key or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY not set")
    timeout = number(timeout, "timeout", minimum=1e-12)
    opener = urllib.request.build_opener(_NoRedirect())

    def send(body: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    return send


def openrouter_transport(api_key: str | None = None, timeout: float = 10.0) -> Transport:
    """Jev through OpenRouter's Decisions API; pin a dated model such as "typesafe/jev-1.13-20260917"."""
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    return http_transport(key, OPENROUTER_DECISIONS_ENDPOINT, timeout)


class JevConfidenceSource(ConfidenceSource):
    name = "jev-noul"

    def __init__(self, transport: Transport, model: str = DEFAULT_MODEL, *, rubric: str = "An acceptable answer is correct, follows the application instructions, preserves tool semantics, and satisfies the user request.") -> None:
        super().__init__(source_version=model, rubric_id=sha256_tag(rubric))
        self.rubric = rubric
        self.transport = transport
        self.model = model
        self.last_response: dict[str, Any] | None = None

    def context_key(self, ctx: Context) -> str:
        return _prompt_only(ctx).prefix_chain()[-1]  # Jev never sees provider_blocks

    @property
    def fingerprint(self):
        return hash_object("pcf:jev-source:0.2", {"base": super().fingerprint, "model": self.model, "rubric": self.rubric})

    def can_validate(self, candidate):
        return super().can_validate(candidate) and "latest" not in self.model

    def p_sufficient(self, ctx: Context, candidate: Candidate) -> float:
        body = build_request(ctx, candidate.model_id, model=self.model, rubric=self.rubric)
        try:
            response = self.transport(body)
        except (OSError, ValueError, HTTPException) as exc:  # URLError/timeouts are OSError; bad JSON is ValueError
            raise ConfidenceUnavailable("confidence transport unavailable") from exc
        self.last_response = response
        try:
            if "latest" not in self.model and response.get("model") != self.model:
                raise ConfidenceUnavailable("confidence model revision differs from the pinned version")
            return parse_noul(response)
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ConfidenceUnavailable("malformed confidence response") from exc

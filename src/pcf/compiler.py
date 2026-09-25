"""Compilation and conservative, explicitly labelled cache-cost estimates."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Protocol

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
    # (segment index, native prefix hash, prefix tokens) for each marker: the prefix the marker actually caches,
    # which can end inside its segment (an OpenAI history marker sits before trailing assistant items)
    marker_prefixes: tuple[tuple[int, str, int], ...] = ()

    def read_candidates(self) -> list[tuple[int, str, int]]:
        """(prefix tokens, native hash, last whole segment) a cache read can match, shortest first.

        At a marker the readable prefix is the one the marker writes; elsewhere it ends at the segment."""
        cum = self.cum_tokens
        marks = {i: (tokens, digest, i if tokens >= cum[i] else i - 1) for i, digest, tokens in self.marker_prefixes}
        return sorted({marks.get(i, (cum[i], self.native_chain[i], i)) for i in self.lookup_indices})

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
    volatile module cannot take the system prompt out of cache (a memory or document anchor directly after that
    run takes the place: it covers the system prompt too, and it is the context other conversations share; every
    request makes the same choice, so a later request reads the entry the first one wrote); then the newest history
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
        after = lead[-1] + 1  # shared context right after the system run covers the system too
        shared = after in anchors and after != anchor and ctx.segments[after].kind in {"memory", "document"}
        keep.add(after if shared else lead[-1])
    reserve = min(need, max_breakpoints - len(keep))  # history endpoints before any newer non-history candidate
    if reserve:
        keep |= set([i for i in history if i not in keep][-reserve:])
    room = max_breakpoints - len(keep)
    rest = [i for i in candidates if i not in keep]
    return sorted({*keep, *(rest[-room:] if room else [])})


def _jcs(value: Any) -> str:
    return canonical_bytes(value).decode("utf-8")


class _PrefixEncoder:
    """RFC 8785 JSON of successive native prefixes, reusing each list element's encoding and hashing state.

    Valid when each prefix's lists extend the previous prefix's lists and only their last element may have
    changed; ``ContextCompiler.render_prefixes`` overrides promise that. ``text`` is byte-identical to
    ``canonical_bytes`` of the same object and ``digest`` to ``hash_object(domain, ...)``.
    """

    def __init__(self, domain: str | None = None) -> None:
        self.encoded: dict[str, list[str]] = {}
        self.chunks: list[str] = []
        self.hasher = hashlib.sha256(domain.encode() + b"\x00") if domain is not None else None
        self.fed = 0  # chunks[:fed] are already in the hasher
        self.domain = domain

    def update(self, value: dict) -> None:
        chunks, settled = ["{"], None
        for n, key in enumerate(sorted(value, key=lambda k: k.encode("utf-16-be"))):  # JCS: UTF-16 key order
            head, item = ("," if n else "") + _jcs(key) + ":", value[key]
            if isinstance(item, list):
                done = self.encoded.setdefault(key, [])
                keep = max(0, min(len(done), len(item)) - 1)  # re-encode the last element: it may have grown
                del done[keep:]
                done.extend(("," if i else "") + _jcs(x) for i, x in enumerate(item[keep:], keep))
                chunks.append(head + "[")
                if settled is None:  # the first list's earlier elements cannot change in a later prefix
                    settled = len(chunks) + max(0, len(done) - 1)
                chunks.extend(done)
                chunks.append("]")
            else:
                chunks.append(head + _jcs(item))
        chunks.append("}")
        if self.hasher is not None:
            same = _common_prefix(self.chunks, chunks)
            if same < self.fed:  # an assumption failed: rebuild the hashing state
                self.hasher, self.fed = hashlib.sha256(self.domain.encode() + b"\x00"), 0
            target = min(same if settled is None else settled, len(chunks))
            if target > self.fed:
                self.hasher.update("".join(chunks[self.fed:target]).encode())
                self.fed = target
        self.chunks = chunks

    @property
    def text(self) -> str:
        return "".join(self.chunks)

    @property
    def digest(self) -> str:
        h = self.hasher.copy()
        h.update("".join(self.chunks[self.fed:]).encode())
        return "sha256:" + h.hexdigest()


def _common_prefix(a: list[str], b: list[str]) -> int:
    """Length of the longest common leading run of two chunk lists (C-speed slice comparisons)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


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
        return self.marker_prefix(ctx, index, None, cum_tokens)[1]

    def marker_prefix(self, ctx: Context, index: int, native_hash: str | None,
                      cum_tokens: list[int]) -> tuple[str | None, int]:
        """(native hash, tokens) of the prefix a marker on segment ``index`` caches; by default the whole prefix
        through that segment. Adapters whose marker sits inside a segment return the shorter prefix."""
        return native_hash, cum_tokens[index]

    def render_prefixes(self, ctx: Context) -> Iterator[dict[str, Any]]:
        """The unmarked request for each prefix ``ctx.segments[:end]``, in order.

        Each item is read before the next is produced. An override may yield live accumulators instead of fresh
        copies, provided each prefix's lists extend the previous prefix's lists with only their last element
        changed; compilation then reuses earlier encodings (see ``_PrefixEncoder``)."""
        for end in range(1, len(ctx.segments) + 1):
            yield self.render(Context(ctx.segments[:end], ctx.session_id, ctx.cache_namespace), [])

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
        incremental = type(self).render_prefixes is not ContextCompiler.render_prefixes
        hashes, counts = _PrefixEncoder("pcf:native:0.2"), _PrefixEncoder()
        native, positions, sizes = [], [], []
        previous_count = 0
        for rendered in self.render_prefixes(ctx):
            native_input = self.native_input(rendered)
            if incremental:  # same bytes as canonical_bytes, without re-encoding or re-hashing the shared prefix
                hashes.update(native_input)
                digest = hashes.digest
                payload = {k: v for k, v in native_input.items() if v not in (None, [], {}, "")}
                if payload:
                    counts.update(payload)
                cumulative = integer(self.tokenizer.count(counts.text), "token count") if payload else 0
            else:
                digest, cumulative = hash_object("pcf:native:0.2", native_input), self.native_token_count(native_input)
            native.append(digest)
            positions.append(self.native_positions(rendered))
            if cumulative < previous_count:
                raise ValueError("token counter must be monotonic on native input prefixes")
            sizes.append(cumulative - previous_count)
            previous_count = cumulative
        cum = list(_accumulate(sizes))
        markers = tuple((b, *self.marker_prefix(ctx, b, native[b], cum)) for b in breakpoints)
        return CompiledPrompt(self.descriptor, canonical_bytes(self.render(ctx, breakpoints)), tuple(sizes),
                              tuple(breakpoints), tuple(ctx.prefix_chain()), tuple(native), self.cache_key,
                              ctx.cache_namespace, tuple(self.lookup_boundaries(breakpoints, positions)),
                              self.tokenizer.is_estimate or self.descriptor.identity_kind != "simulated", markers)

    def warmth(self, ctx: Context, cache: PrefixCache, now: float) -> Warmth:
        number(now, "now")
        compiled = self.compile(ctx)
        # Closed APIs expose aggregate usage after execution, not a live preflight
        # cache inventory. Never promote a caller's local dictionary into knowledge
        # of a provider's cache. Native reuse is unknown until the provider responds.
        simulated = self.descriptor.identity_kind == "simulated"
        warm, hit = read_hit(compiled, cache, now) if simulated else (0, -1)
        # An implicit breakpoint's position is unknown; assume the most it could write (the whole prompt).
        cum = compiled.cum_tokens
        reaches = [tokens for _, _, tokens in compiled.marker_prefixes]
        if self.implicit_breakpoint and not compiled.breakpoints:
            reaches = [cum[-1]] if cum else []
        covered = max([warm] + [r for r in reaches if r > warm and r >= self.descriptor.min_cacheable_tokens])
        written = covered - warm
        cold = compiled.total_tokens - warm
        return Warmth(hit + 1, warm, cold, written, cold - written,
                      "simulated" if simulated else "unknown", now)


def _accumulate(sizes):
    total = 0
    for n in sizes:
        total += n
        yield total


def read_hit(compiled: CompiledPrompt, cache: PrefixCache, now: float) -> tuple[int, int]:
    """(warm tokens, last warm segment index) for the longest live prefix a read can match, or (0, -1)."""
    candidates = compiled.read_candidates()
    found = cache.peek(compiled.cache_key, [digest for _, digest, _ in candidates], now,
                       namespace=compiled.cache_namespace)
    return (candidates[found][0], candidates[found][2]) if found >= 0 else (0, -1)

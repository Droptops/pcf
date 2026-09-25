"""Self-placing memory: each module picks front or tail from its own observed change rate.

A front module is re-billed with everything after it when it changes (H: the front modules that follow it plus
history): expected p * (m + H) per turn.
A tail module is re-billed every turn: m. A module goes to the tail when p * (m + H) > m, where p is its
observed change rate. With cache prices (write w, read r, relative to uncached input) the comparison is
p * (w - r) * (m + H) > (1 - r) * m: a changed front prefix is written instead of read, while a tail module is
paid uncached instead of read. The defaults w=1, r=0 reduce to the plain token rule.
Modules are assumed stable until a change is seen, so stable memory never moves.

Moving back to the front re-bills m + H once. While the cache is warm a tail module returns once it has gone quiet: unchanged for more than twice its average
gap between changes, it is judged by its decayed change rate, and it returns when that rate makes the front
cheaper per turn and the saving over the turns still to come repays the rewrite. Pass `expected_turns` (the typical
conversation length) when the application knows it; otherwise the placer assumes as many more turns as the
conversation has had,
under which returning rarely pays once history grows faster per turn than the module, since the rewrite grows too.
A module that changes on a steady period is never quiet that long, so it does not bounce. When the caller reports a cold cache (everything is re-billed anyway), modules are
re-placed from a decayed change rate. A module never moves while it and everything after it are below the
provider's minimum cacheable length: nothing there is cached, so moving it saves nothing. The defaults are the
common cache prices (write 1.25, read 0.1); `for_compiler` takes the write price and minimum from a compiler.
State is per module id; there is no global policy. This is a cost heuristic, not a quality guarantee.
"""
from __future__ import annotations

from .compiler import Tokenizer, segment_text
from .segments import Segment
from .validation import integer, number


class MemoryPlacer:
    def __init__(self, tokenizer: Tokenizer, *, decay: float = 0.7, write_multiplier: float = 1.25,
                 read_multiplier: float = 0.1, min_cacheable_tokens: int = 0,
                 expected_turns: int | None = None) -> None:
        number(decay, "decay")
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        number(read_multiplier, "read_multiplier")
        number(write_multiplier, "write_multiplier")
        if not 0 <= read_multiplier < 1 or write_multiplier <= read_multiplier:
            raise ValueError("need 0 <= read_multiplier < 1 and write_multiplier > read_multiplier")
        integer(min_cacheable_tokens, "min_cacheable_tokens")
        if expected_turns is not None:
            integer(expected_turns, "expected_turns", minimum=1)
        self.expected_turns = expected_turns
        self.tokenizer = tokenizer
        self.min_cacheable_tokens = min_cacheable_tokens
        self._turns = 0
        self._quiet: dict[str, int] = {}  # id -> turns since the module last changed
        self.decay = decay
        self.write_multiplier = write_multiplier
        self.read_multiplier = read_multiplier
        # id -> (observations, changes, decayed rate, last hash, in tail)
        self._seen: dict[str, tuple[int, int, float, str, bool]] = {}
        self._front: tuple[str, ...] | None = None  # front module ids of the previous call

    @classmethod
    def for_compiler(cls, compiler, *, read_multiplier: float = 0.1, decay: float = 0.7,
                     expected_turns: int | None = None) -> "MemoryPlacer":
        """A placer with the compiler's tokenizer, cache write price and minimum cacheable length."""
        d = compiler.descriptor
        return cls(compiler.tokenizer, decay=decay, write_multiplier=d.cache_write_multiplier,
                   read_multiplier=read_multiplier, min_cacheable_tokens=d.min_cacheable_tokens,
                   expected_turns=expected_turns)

    @classmethod
    def for_candidate(cls, candidate, *, decay: float = 0.7) -> "MemoryPlacer":
        """A placer priced from a router ``Candidate``: its tokenizer, write price and cache-read price."""
        base = number(candidate.input_price_per_mtok, "input_price_per_mtok", minimum=1e-12)
        return cls(candidate.compiler.tokenizer, decay=decay, write_multiplier=candidate.write_price / base,
                   read_multiplier=candidate.cache_read_price_per_mtok / base)

    def split(self, memory: list[Segment], history: list[Segment], *,
              cold: bool = False) -> tuple[list[Segment], list[Segment]]:
        """Return (front, tail); call once per turn. Pass cold=True when the provider cache has expired.

        Predict cold from time (``now - last_request >= descriptor.ttl_seconds``), not from a zero cache read in
        usage: by then that request has already rewritten the cache in the old layout, so re-placing re-bills it.

        A change to a front module re-bills everything after it: history and the front modules that follow
        it, so H for each module counts both. List modules stable-first; order is kept. Modules with
        instruction authority never move (instructions precede data). Tail copies are unstable so they
        get no breakpoint.

        On a warm turn where the front changes, providers that read only at markers present in the request need a
        marker at an entry written earlier. When modules were only appended, the previous last front module's entry
        still holds, and the appended modules are returned unstable so it keeps the anchor. Otherwise the prefix
        changes after the first front module, whose entry the conversation's first request wrote, and the front
        modules after the first are returned unstable.
        """
        if any(seg.kind != "memory" for seg in memory):
            raise ValueError("only memory segments can be placed")
        behind = sum(self.tokenizer.count(segment_text(h)) for h in history)
        self._turns += 1
        placed = [False] * len(memory)
        for i in range(len(memory) - 1, -1, -1):  # later front modules add to what an earlier change re-bills
            seg = memory[i]
            obs, changes, rate, last, in_tail = self._seen.get(seg.id, (-1, 0, 0.0, seg.hash, False))
            changed = seg.hash != last
            quiet = 0 if changed else self._quiet.get(seg.id, 0) + 1
            self._quiet[seg.id] = quiet
            obs, changes = obs + 1, changes + changed
            rate = self.decay * rate + (1 - self.decay) * changed
            m = self.tokenizer.count(segment_text(seg))
            if seg.authority == "instruction":
                in_tail = False
            elif cold:
                in_tail = self._tail_pays(rate, m, behind)
                if not in_tail:
                    obs, changes = 0, 0  # back in front: evidence restarts
            elif m + behind < self.min_cacheable_tokens:
                pass  # nothing here is cached: moving it saves nothing
            elif not in_tail:
                in_tail = self._tail_pays(changes / (obs + 1), m, behind)
            elif quiet > 2 * (obs + 1) / max(changes, 1) and self._return_pays(rate, m, behind):
                in_tail, obs, changes = False, 0, 0  # back in front: evidence restarts
            self._seen[seg.id] = (obs, changes, rate, seg.hash, in_tail)
            placed[i] = in_tail
            if not in_tail:
                behind += m
        front = [seg for seg, in_tail in zip(memory, placed) if not in_tail]
        ids = tuple(seg.id for seg in front)
        self._previous, self._front = self._front or (), ids
        moved = bool(self._previous) and ids != self._previous and not cold
        if moved:
            previous = len(self._previous) if ids[:len(self._previous)] == self._previous else 1
            front = front[:previous] + [Segment(seg.id, "memory", seg.content, False, authority=seg.authority,
                                                provenance=seg.provenance) for seg in front[previous:]]
        tail = [Segment(seg.id, "memory", seg.content, False, authority=seg.authority, provenance=seg.provenance)
                for seg, in_tail in zip(memory, placed) if in_tail]
        return front, tail

    def _return_pays(self, p: float, m: int, behind: int) -> bool:
        """A warm move back: the per-turn saving over the turns seen so far repays one rewrite of m + H."""
        w, r = self.write_multiplier, self.read_multiplier
        saving = (1 - r) * m - p * (w - r) * (m + behind)
        turns = self._turns if self.expected_turns is None else self.expected_turns - self._turns
        return saving > 0 and saving * turns > (w - r) * (m + behind)

    def _tail_pays(self, p: float, m: int, behind: int) -> bool:
        w, r = self.write_multiplier, self.read_multiplier
        return p * (w - r) * (m + behind) > (1 - r) * m

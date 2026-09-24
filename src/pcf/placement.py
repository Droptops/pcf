"""Self-placing memory: each module picks front or tail from its own observed change rate.

A front module is re-billed with all history after it when it changes: expected p * (m + H) per turn.
A tail module is re-billed every turn: m. A module goes to the tail when p * (m + H) > m, where p is its
observed change rate. With cache prices (write w, read r, relative to uncached input) the comparison is
p * (w - r) * (m + H) > (1 - r) * m: a changed front prefix is written instead of read, while a tail module is
paid uncached instead of read. The defaults w=1, r=0 reduce to the plain token rule.
Modules are assumed stable until a change is seen, so stable memory never moves.

Moving back to the front re-bills m + H at once, so while the cache is warm the tail is sticky. When the
caller reports a cold cache (everything is re-billed anyway), modules are re-placed from a decayed change
rate, so a module that went quiet returns to the front. State is per module id; there is no global policy.
This is a cost heuristic, not a quality guarantee.
"""
from __future__ import annotations

from .compiler import Tokenizer, segment_text
from .segments import Segment
from .validation import number


class MemoryPlacer:
    def __init__(self, tokenizer: Tokenizer, *, decay: float = 0.7, write_multiplier: float = 1.0,
                 read_multiplier: float = 0.0) -> None:
        number(decay, "decay")
        if not 0 <= decay < 1:
            raise ValueError("decay must be in [0, 1)")
        number(read_multiplier, "read_multiplier")
        number(write_multiplier, "write_multiplier")
        if not 0 <= read_multiplier < 1 or write_multiplier <= read_multiplier:
            raise ValueError("need 0 <= read_multiplier < 1 and write_multiplier > read_multiplier")
        self.tokenizer = tokenizer
        self.decay = decay
        self.write_multiplier = write_multiplier
        self.read_multiplier = read_multiplier
        # id -> (observations, changes, decayed rate, last hash, in tail)
        self._seen: dict[str, tuple[int, int, float, str, bool]] = {}

    @classmethod
    def for_candidate(cls, candidate, *, decay: float = 0.7) -> "MemoryPlacer":
        """A placer priced from a router ``Candidate``: its tokenizer, write price and cache-read price."""
        base = number(candidate.input_price_per_mtok, "input_price_per_mtok", minimum=1e-12)
        return cls(candidate.compiler.tokenizer, decay=decay, write_multiplier=candidate.write_price / base,
                   read_multiplier=candidate.cache_read_price_per_mtok / base)

    def split(self, memory: list[Segment], history: list[Segment], *,
              cold: bool = False) -> tuple[list[Segment], list[Segment]]:
        """Return (front, tail); call once per turn. Pass cold=True when the provider cache has expired.

        Tail copies are unstable so they get no breakpoint.
        """
        if any(seg.kind != "memory" for seg in memory):
            raise ValueError("only memory segments can be placed")
        history_tokens = sum(self.tokenizer.count(segment_text(h)) for h in history)
        front, tail = [], []
        for seg in memory:
            obs, changes, rate, last, in_tail = self._seen.get(seg.id, (-1, 0, 0.0, seg.hash, False))
            changed = seg.hash != last
            obs, changes = obs + 1, changes + changed
            rate = self.decay * rate + (1 - self.decay) * changed
            m = self.tokenizer.count(segment_text(seg))
            if cold:
                in_tail = self._tail_pays(rate, m, history_tokens)
                if not in_tail:
                    obs, changes = 0, 0  # back in front: evidence restarts
            elif not in_tail:
                in_tail = self._tail_pays(changes / (obs + 1), m, history_tokens)
            self._seen[seg.id] = (obs, changes, rate, seg.hash, in_tail)
            if in_tail:
                tail.append(Segment(seg.id, "memory", seg.content, False, provenance=seg.provenance))
            else:
                front.append(seg)
        return front, tail

    def _tail_pays(self, p: float, m: int, history_tokens: int) -> bool:
        w, r = self.write_multiplier, self.read_multiplier
        return p * (w - r) * (m + history_tokens) > (1 - r) * m

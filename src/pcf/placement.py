"""Self-placing memory: each module picks front or tail from its own observed change rate.

A front module is re-billed with all history after it when it changes: expected p * (m + H) per turn.
A tail module is re-billed every turn: m. A module goes to the tail when p * (m + H) > m, where p is its
observed change rate. Modules are assumed stable until a change is seen, so stable memory never moves.
State is per module id; there is no global policy. This is a cost heuristic, not a quality guarantee.
"""
from __future__ import annotations

from .compiler import Tokenizer, segment_text
from .segments import Segment


class MemoryPlacer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._seen: dict[str, tuple[int, int, str]] = {}  # id -> (observations, changes, last hash)

    def split(self, memory: list[Segment], history: list[Segment]) -> tuple[list[Segment], list[Segment]]:
        """Return (front, tail); call once per turn. Tail copies are unstable so they get no breakpoint."""
        if any(seg.kind != "memory" for seg in memory):
            raise ValueError("only memory segments can be placed")
        history_tokens = sum(self.tokenizer.count(segment_text(h)) for h in history)
        front, tail = [], []
        for seg in memory:
            obs, changes, last = self._seen.get(seg.id, (-1, 0, seg.hash))
            obs, changes = obs + 1, changes + (seg.hash != last)
            self._seen[seg.id] = (obs, changes, seg.hash)
            m = self.tokenizer.count(segment_text(seg))
            if changes / (obs + 1) * (m + history_tokens) > m:
                tail.append(Segment(seg.id, "memory", seg.content, False, provenance=seg.provenance))
            else:
                front.append(seg)
        return front, tail

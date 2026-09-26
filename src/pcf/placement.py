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

import hashlib
from threading import RLock

from .compiler import Tokenizer, segment_text
from .segments import Segment, canonical_bytes
from .validation import digest, integer, nonempty, number


PLACER_STATE_VERSION = 1


class ConcurrentPlacementUpdate(RuntimeError):
    """The caller planned from a stale placement-state revision."""


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
        self._revision = 0
        # turn id -> (input fingerprint, ((front module id, returned stability), ...))
        self._decisions: dict[str, tuple[str, tuple[tuple[str, bool], ...]]] = {}
        self._lock = RLock()

    @classmethod
    def for_compiler(cls, compiler, *, read_multiplier: float = 0.1, decay: float = 0.7,
                     expected_turns: int | None = None) -> "MemoryPlacer":
        """A placer with the compiler's tokenizer, cache write price and minimum cacheable length."""
        d = compiler.descriptor
        return cls(compiler.tokenizer, decay=decay, write_multiplier=d.cache_write_multiplier,
                   read_multiplier=read_multiplier, min_cacheable_tokens=d.min_cacheable_tokens,
                   expected_turns=expected_turns)

    @classmethod
    def for_candidate(cls, candidate, *, decay: float = 0.7,
                      expected_turns: int | None = None) -> "MemoryPlacer":
        """A placer configured from a router ``Candidate``, including the provider cache minimum."""
        base = number(candidate.input_price_per_mtok, "input_price_per_mtok", minimum=1e-12)
        return cls(candidate.compiler.tokenizer, decay=decay, write_multiplier=candidate.write_price / base,
                   read_multiplier=candidate.cache_read_price_per_mtok / base,
                   min_cacheable_tokens=candidate.compiler.descriptor.min_cacheable_tokens,
                   expected_turns=expected_turns)

    @property
    def revision(self) -> int:
        """Monotonic revision for optimistic single-writer persistence."""
        with self._lock:
            return self._revision

    def _configuration(self) -> dict:
        return {"tokenizer_hash": self.tokenizer.tokenizer_hash, "decay": self.decay,
                "write_multiplier": self.write_multiplier, "read_multiplier": self.read_multiplier,
                "min_cacheable_tokens": self.min_cacheable_tokens, "expected_turns": self.expected_turns}

    def export_state(self) -> dict:
        """Return a versioned JSON-compatible snapshot of all placement and idempotency state.

        Persist this document with a compare-and-set on ``revision``. It contains hashes and placement decisions,
        not module contents. One state document belongs to one conversation.
        """
        with self._lock:
            return {"memory_placer_state_version": PLACER_STATE_VERSION,
                    "configuration": self._configuration(), "revision": self._revision, "turns": self._turns,
                    "quiet": dict(sorted(self._quiet.items())),
                    "seen": {key: {"observations": value[0], "changes": value[1], "rate": value[2],
                                    "last_hash": value[3], "in_tail": value[4]}
                             for key, value in sorted(self._seen.items())},
                    "front": list(self._front) if self._front is not None else None,
                    "decisions": {key: {"input_hash": value[0],
                                         "front": [{"id": seg_id, "stable": stable}
                                                   for seg_id, stable in value[1]]}
                                  for key, value in sorted(self._decisions.items())}}

    def restore_state(self, state: dict) -> None:
        """Replace state from :meth:`export_state`; reject incompatible or malformed snapshots."""
        if not isinstance(state, dict) or set(state) != {
                "memory_placer_state_version", "configuration", "revision", "turns", "quiet", "seen", "front",
                "decisions"}:
            raise ValueError("invalid MemoryPlacer state fields")
        if state["memory_placer_state_version"] != PLACER_STATE_VERSION:
            raise ValueError("unsupported MemoryPlacer state version")
        if state["configuration"] != self._configuration():
            raise ValueError("MemoryPlacer state configuration does not match this placer")
        revision = integer(state["revision"], "revision")
        turns = integer(state["turns"], "turns")
        if revision < 0 or turns < 0 or revision != turns:
            raise ValueError("MemoryPlacer state needs equal nonnegative revision and turns")
        quiet_raw, seen_raw, front_raw, decisions_raw = (state[k] for k in ("quiet", "seen", "front", "decisions"))
        if not isinstance(quiet_raw, dict) or not isinstance(seen_raw, dict) or not isinstance(decisions_raw, dict):
            raise ValueError("MemoryPlacer state maps are invalid")
        if front_raw is not None and (not isinstance(front_raw, list) or
                                      any(not isinstance(x, str) or not x for x in front_raw) or
                                      len(front_raw) != len(set(front_raw))):
            raise ValueError("MemoryPlacer state front is invalid")
        quiet = {}
        for key, value in quiet_raw.items():
            nonempty(key, "module id")
            value = integer(value, "quiet turns")
            if value < 0:
                raise ValueError("quiet turns must be nonnegative")
            quiet[key] = value
        seen = {}
        fields = {"observations", "changes", "rate", "last_hash", "in_tail"}
        for key, value in seen_raw.items():
            nonempty(key, "module id")
            if not isinstance(value, dict) or set(value) != fields:
                raise ValueError("invalid observed module state")
            observations = integer(value["observations"], "observations")
            changes = integer(value["changes"], "changes")
            rate = number(value["rate"], "rate")
            if observations < 0 or changes < 0 or changes > observations or not 0 <= rate <= 1:
                raise ValueError("invalid observed module counters")
            digest(value["last_hash"], "last_hash")
            if type(value["in_tail"]) is not bool:
                raise ValueError("in_tail must be boolean")
            seen[key] = (observations, changes, rate, value["last_hash"], value["in_tail"])
        decisions = {}
        for turn_id, value in decisions_raw.items():
            nonempty(turn_id, "turn_id")
            if not isinstance(value, dict) or set(value) != {"input_hash", "front"}:
                raise ValueError("invalid stored placement decision")
            digest(value["input_hash"], "input_hash")
            if not isinstance(value["front"], list):
                raise ValueError("stored decision front must be a list")
            front = []
            for item in value["front"]:
                if not isinstance(item, dict) or set(item) != {"id", "stable"} or type(item["stable"]) is not bool:
                    raise ValueError("invalid stored front module")
                front.append((nonempty(item["id"], "module id"), item["stable"]))
            if len(front) != len({item[0] for item in front}):
                raise ValueError("stored front module ids must be unique")
            decisions[turn_id] = (value["input_hash"], tuple(front))
        with self._lock:
            self._revision, self._turns, self._quiet, self._seen = revision, turns, quiet, seen
            self._front = tuple(front_raw) if front_raw is not None else None
            self._decisions = decisions

    @classmethod
    def from_state(cls, tokenizer: Tokenizer, state: dict) -> "MemoryPlacer":
        """Construct a placer from a durable snapshot, checking tokenizer and configuration identity."""
        if not isinstance(state, dict) or not isinstance(state.get("configuration"), dict):
            raise ValueError("invalid MemoryPlacer state")
        config = state["configuration"]
        required = {"tokenizer_hash", "decay", "write_multiplier", "read_multiplier", "min_cacheable_tokens",
                    "expected_turns"}
        if set(config) != required or config["tokenizer_hash"] != tokenizer.tokenizer_hash:
            raise ValueError("MemoryPlacer state tokenizer/configuration mismatch")
        placer = cls(tokenizer, decay=config["decay"], write_multiplier=config["write_multiplier"],
                     read_multiplier=config["read_multiplier"], min_cacheable_tokens=config["min_cacheable_tokens"],
                     expected_turns=config["expected_turns"])
        placer.restore_state(state)
        return placer

    @staticmethod
    def _input_hash(memory: list[Segment], history: list[Segment], cold: bool) -> str:
        value = {"memory": [{"id": s.id, "hash": s.hash, "stable": s.stable} for s in memory],
                 "history": [{"id": s.id, "hash": s.hash, "stable": s.stable} for s in history], "cold": cold}
        return "sha256:" + hashlib.sha256(canonical_bytes(value)).hexdigest()

    @staticmethod
    def _with_stability(seg: Segment, stable: bool) -> Segment:
        return seg if seg.stable == stable else Segment(seg.id, "memory", seg.content, stable,
                                                        authority=seg.authority, provenance=seg.provenance)

    def _replay(self, memory: list[Segment], decision: tuple[str, tuple[tuple[str, bool], ...]]):
        front_spec = decision[1]
        front_ids = {seg_id for seg_id, _ in front_spec}
        by_id = {seg.id: seg for seg in memory}
        if len(by_id) != len(memory) or any(seg_id not in by_id for seg_id, _ in front_spec):
            raise ValueError("stored placement decision does not match memory modules")
        front = [self._with_stability(by_id[seg_id], stable) for seg_id, stable in front_spec]
        tail = [self._with_stability(seg, False) for seg in memory if seg.id not in front_ids]
        return front, tail

    def split(self, memory: list[Segment], history: list[Segment], *, cold: bool = False,
              turn_id: str | None = None, expected_revision: int | None = None) -> tuple[list[Segment], list[Segment]]:
        """Return (front, tail); call once per turn. Pass cold=True when the provider cache has expired.

        Supply a durable unique ``turn_id`` in production. Retrying that id with byte-identical inputs returns the
        recorded decision without observing the modules again; different inputs for the same id are rejected.
        ``expected_revision`` provides optimistic single-writer concurrency: a new turn raises
        :class:`ConcurrentPlacementUpdate` when another worker committed first. Persist :meth:`export_state` with
        the same compare-and-set. Calls without these arguments retain the pre-0.3 behavior.

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
        if turn_id is not None:
            nonempty(turn_id, "turn_id")
        if expected_revision is not None:
            integer(expected_revision, "expected_revision")
            if expected_revision < 0:
                raise ValueError("expected_revision must be nonnegative")
        input_hash = self._input_hash(memory, history, cold)
        with self._lock:
            if turn_id is not None and turn_id in self._decisions:
                decision = self._decisions[turn_id]
                if decision[0] != input_hash:
                    raise ValueError("turn_id was already used with different placement inputs")
                return self._replay(memory, decision)
            if expected_revision is not None and expected_revision != self._revision:
                raise ConcurrentPlacementUpdate(
                    f"placement revision changed: expected {expected_revision}, found {self._revision}")
            result = self._split_new(memory, history, cold)
            self._revision += 1
            if turn_id is not None:
                self._decisions[turn_id] = (input_hash, tuple((seg.id, seg.stable) for seg in result[0]))
            return result

    def _split_new(self, memory: list[Segment], history: list[Segment], cold: bool):
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

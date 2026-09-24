"""Bounded prefix metadata store. Inspection is read-only; execution touches TTL."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import RLock

from .validation import digest, integer, nonempty, number


@dataclass(frozen=True)
class Entry:
    cum_tokens: int
    expires_at: float
    ttl_seconds: float
    observed_at: float


@dataclass
class PrefixCache:
    ttl_seconds: float
    max_entries: int = 4096
    _entries: dict[tuple[str, str, str], Entry] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    def __post_init__(self):
        number(self.ttl_seconds, "ttl_seconds", minimum=1e-12)
        integer(self.max_entries, "max_entries", minimum=1)

    @staticmethod
    def _key(compat_key, prefix_hash, namespace):
        return (nonempty(namespace, "namespace"), digest(compat_key, "compat_key"), digest(prefix_hash, "prefix_hash"))

    def write(self, compat_key: str, prefix_hash: str, cum_tokens: int, now: float,
              *, namespace: str = "default", ttl_seconds: float | None = None) -> None:
        key = self._key(compat_key, prefix_hash, namespace)
        integer(cum_tokens, "cum_tokens")
        number(now, "now")
        ttl = self.ttl_seconds if ttl_seconds is None else number(ttl_seconds, "ttl_seconds", minimum=1e-12)
        with self._lock:
            self.prune(now)
            previous = self._entries.get(key)
            if previous is not None and now < previous.observed_at:
                raise ValueError("cache event time moved backwards")
            if key not in self._entries and len(self._entries) >= self.max_entries:
                del self._entries[min(self._entries, key=lambda k: self._entries[k].observed_at)]
            self._entries[key] = Entry(cum_tokens, now + ttl, ttl, now)

    def peek(self, compat_key: str, chain: list[str], now: float, *, namespace: str = "default",
             eligible_indices: list[int] | None = None) -> int:
        """Longest eligible live prefix. Does not mutate entries, expiry, or eviction state."""
        number(now, "now")
        digest(compat_key, "compat_key")
        nonempty(namespace, "namespace")
        indices = range(len(chain) - 1, -1, -1) if eligible_indices is None else sorted(set(eligible_indices), reverse=True)
        with self._lock:
            for i in indices:
                if i < 0 or i >= len(chain):
                    raise ValueError("cache lookup index outside prefix chain")
                entry = self._entries.get(self._key(compat_key, chain[i], namespace))
                if entry is not None and now < entry.expires_at:
                    return i
        return -1

    def lookup(self, compat_key: str, chain: list[str], now: float, **kwargs) -> int:
        """Compatibility alias for read-only peek; call touch after actual reuse."""
        return self.peek(compat_key, chain, now, **kwargs)

    def touch(self, compat_key: str, prefix_hash: str, now: float, *, namespace: str = "default") -> bool:
        number(now, "now")
        key = self._key(compat_key, prefix_hash, namespace)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or now >= entry.expires_at:
                return False
            if now < entry.observed_at:
                raise ValueError("cache event time moved backwards")
            self._entries[key] = replace(entry, expires_at=now + entry.ttl_seconds, observed_at=now)
            return True

    def cum_tokens(self, compat_key: str, prefix_hash: str, *, namespace: str = "default") -> int:
        with self._lock:
            return self._entries[self._key(compat_key, prefix_hash, namespace)].cum_tokens

    def evict(self, compat_key: str, prefix_hash: str, *, namespace: str = "default") -> None:
        with self._lock:
            self._entries.pop(self._key(compat_key, prefix_hash, namespace), None)

    def prune(self, now: float) -> int:
        number(now, "now")
        with self._lock:
            stale = [k for k, e in self._entries.items() if now >= e.expires_at]
            for key in stale:
                del self._entries[key]
            return len(stale)

    def __len__(self):
        with self._lock:
            return len(self._entries)

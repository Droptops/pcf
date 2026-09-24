"""Explicit API profiles. Unknown model IDs require a caller-supplied profile.

These profiles record documented behavior as of 2026-09-24, not live API validation. Sources: openai-python
3.19.2 docstrings (PromptCacheOptions, prompt_cache_retention) and the guide they cite,
https://developers.openai.com/api/docs/guides/prompt-caching (write premium and minimum length: checked via
secondary summaries, not fetched directly). in_memory 600s and 24h 86400s are nominal: the guide states a range
and a maximum, not fixed TTLs.
Old-model minimums vary with request settings; no exact old-model boundary is asserted.
"""
from dataclasses import dataclass

from ..validation import integer, number


@dataclass(frozen=True)
class OpenAICapabilities:
    explicit: bool
    retention: str
    ttl_seconds: int
    min_tokens: int = 1024  # estimate/profile threshold; never exact native billing
    write_multiplier: float | None = None  # default: 1.25 with explicit caching (gpt-5.6+), else free writes
    max_breakpoints: int = 4

    def __post_init__(self):
        if type(self.explicit) is not bool:
            raise ValueError("explicit must be boolean")
        if self.retention not in ({"30m"} if self.explicit else {"in_memory", "24h"}):
            raise ValueError("retention incompatible with cache profile")
        integer(self.ttl_seconds, "ttl_seconds", minimum=1)
        integer(self.min_tokens, "min_tokens")
        integer(self.max_breakpoints, "max_breakpoints")
        if self.max_breakpoints > 4 or (not self.explicit and self.max_breakpoints != 0):
            raise ValueError("invalid explicit breakpoint budget")
        if self.write_multiplier is None:
            object.__setattr__(self, "write_multiplier", 1.25 if self.explicit else 1)
        number(self.write_multiplier, "write_multiplier")


_MODERN = OpenAICapabilities(True, "30m", 1800)
_LEGACY = OpenAICapabilities(False, "in_memory", 600, write_multiplier=1, max_breakpoints=0)
_LONG = OpenAICapabilities(False, "24h", 86400, write_multiplier=1, max_breakpoints=0)
OPENAI_PROFILES = {
    "gpt-5.6": _MODERN, "gpt-5.6-sol": _MODERN, "gpt-5.6-terra": _MODERN,
    "gpt-6-astra": _MODERN, "gpt-6-sol": _MODERN, "gpt-6-luna": _MODERN,
    "gpt-5.5": _LONG, "gpt-5.5-pro": _LONG,
    "gpt-5.4": _LEGACY, "gpt-5.2": _LEGACY, "gpt-5.1": _LEGACY, "gpt-4.1": _LEGACY,
}


def openai_profile(model_id, override=None):
    if override is not None:
        if not isinstance(override, OpenAICapabilities):
            raise ValueError("capabilities must be an OpenAICapabilities profile")
        return override
    try:
        return OPENAI_PROFILES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown cache profile for {model_id!r}; supply explicit capabilities") from exc

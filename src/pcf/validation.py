"""Small, shared runtime validators. Serialized schemas live in pcf.schemas."""
from __future__ import annotations

import math
import re
from numbers import Real

HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def number(value, name: str, *, minimum: float = 0, maximum: float | None = None) -> float:
    try:
        finite = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)
    except OverflowError:  # int too large for float
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its allowed range")
    return float(value)


def integer(value, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def nonempty(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def digest(value, name: str) -> str:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a tagged SHA-256 digest")
    return value

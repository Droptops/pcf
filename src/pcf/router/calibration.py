"""Validated calibration metrics and damped, objective-checked Platt fitting."""
from __future__ import annotations

import math
from collections.abc import Sequence

from ..validation import integer, number


def checked_outcomes(probs: Sequence[float], labels: Sequence[int]):
    probs, labels = list(probs), list(labels)
    if not probs or len(probs) != len(labels):
        raise ValueError("probabilities and labels must have equal, nonzero length")
    for p in probs:
        number(p, "probability", maximum=1)
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("labels must be integer zero or one")
    return probs, labels


def expected_calibration_error(probs, labels, n_bins=10):
    probs, labels = checked_outcomes(probs, labels)
    integer(n_bins, "n_bins", minimum=1)
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    return sum(abs(sum(y - p for p, y in group)) for group in bins) / len(probs)


def brier_score(probs, labels):
    probs, labels = checked_outcomes(probs, labels)
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)


def _sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def _logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def fit_platt(raw_probs, labels, iters=100, *, l2=1e-6):
    """Minimize mean logistic NLL + l2*a²/2 with backtracking Newton steps.

    Constant features get an intercept-only fit. Single-class/empty evidence is
    rejected. Failure to converge raises instead of publishing extreme scores.
    Fitting does not grant permission to route: separate held-out validation does.
    """
    raw_probs, labels = checked_outcomes(raw_probs, labels)
    integer(iters, "iters", minimum=1)
    number(l2, "l2", minimum=1e-12)
    if len(set(labels)) < 2:
        raise ValueError("calibration fit needs both outcome classes")
    z = [_logit(p) for p in raw_probs]
    mean = sum(labels) / len(labels)
    if max(z) - min(z) < 1e-12:
        return 0.0, _logit(mean)
    a, b = 0.0, _logit(mean)
    n = len(z)
    def loss(slope, intercept):
        total = 0.0
        for zi, yi in zip(z, labels):
            t = slope * zi + intercept
            total += max(t, 0) + math.log1p(math.exp(-abs(t))) - yi * t
        return total / n + .5 * l2 * slope * slope
    for _ in range(iters):
        ga, gb, haa, hab, hbb = l2 * a, 0.0, l2, 0.0, 0.0
        for zi, yi in zip(z, labels):
            q = _sigmoid(a * zi + b)
            r, w = (q - yi) / n, q * (1 - q) / n
            ga += r * zi
            gb += r
            haa += w * zi * zi
            hab += w * zi
            hbb += w
        if max(abs(ga), abs(gb)) < 1e-9:
            return a, b
        det = haa * hbb - hab * hab
        if det <= 1e-18 or not math.isfinite(det):
            raise ValueError("ill-conditioned calibration fit")
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        old = loss(a, b)
        step = 1.0
        for _ in range(60):
            aa, bb = a - step * da, b - step * db
            if loss(aa, bb) <= old - 1e-4 * step * (ga * da + gb * db):
                a, b = aa, bb
                break
            step *= .5
        else:
            raise ValueError("calibration line search failed")
    raise ValueError("calibration did not converge within iteration budget")


def apply_platt(raw_prob, a, b):
    number(raw_prob, "raw probability", maximum=1)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, b)):
        raise ValueError("calibration coefficients must be finite")
    return _sigmoid(a * _logit(raw_prob) + b)

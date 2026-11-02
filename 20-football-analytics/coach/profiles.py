"""Player profiles for the AI coach: one provider-independent format, filled from open data now and from a live
sports-data API later. Design: docs/superpowers/specs/2026-11-02-player-profiles-design.md.

Every skill is an offset on the logit scale of an existing model (pass_skill: the pass model; finishing: the shot
xG), stored with its sample size n and the unshrunk estimate raw. The coach uses logit(p) += a * offset; a missing
field leaves the coach exactly as without profiles. Offsets with sample sizes only: no ratings, no judgement labels.

Shrinkage (empirical Bayes): one Newton step from 0 for a per-player logit offset with a Gaussian prior of precision k,

    raw   = sum w (y - p) / sum w p (1 - p)
    logit = sum w (y - p) / (sum w p (1 - p) + k)

so a player with a handful of passes stays near 0 (the model's expectation). k and the scale a are chosen by
held-out log-loss (choose_k): each match's rows get offsets built from the other matches only.
"""

import numpy as np

VERSION = 1
SKILLS = ("pass_skill", "finishing")
K_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _arrays(keys, y, p, w):
    keys = np.asarray(keys, dtype=str)
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return keys, y, p, w


def residual_sums(keys, y, p, w=None) -> dict:
    """key -> (rows, sum w (y - p), sum w p (1 - p)); rows whose key is "" (unknown player) are ignored."""
    keys, y, p, w = _arrays(keys, y, p, w)
    ok = keys != ""
    if not ok.any():
        return {}
    u, inv = np.unique(keys[ok], return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    g = np.bincount(inv, weights=(w * (y - p))[ok], minlength=len(u))
    h = np.bincount(inv, weights=(w * p * (1 - p))[ok], minlength=len(u))
    return {str(k): (int(a), float(b), float(c)) for k, a, b, c in zip(u, n, g, h)}


def shrink(sums, k: float) -> dict:
    """(rows, sum w (y - p), sum w p (1 - p)) -> {"logit": shrunk offset, "n": rows, "raw": unshrunk offset}."""
    n, g, h = sums
    return {"logit": g / (h + k) if h + k > 0 else 0.0, "n": int(n), "raw": g / h if h > 0 else None}


def offsets(keys, y, p, w=None, k: float = 30.0) -> dict:
    """Shrunk offset of every player key against the expectations p."""
    return {key: shrink(s, k) for key, s in residual_sums(keys, y, p, w).items()}


def offset_vector(keys, table: dict) -> np.ndarray:
    """Per row: the player's offset, 0 when the player has no profile."""
    return np.array([table[k]["logit"] if k in table else 0.0 for k in np.asarray(keys, dtype=str)], float)


def apply(p, off, a: float = 1.0) -> np.ndarray:
    return sigmoid(logit(p) + a * np.asarray(off, float))


def row_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float(np.sum(w * row_loss(y, p)) / np.sum(w))


def held_out_sums(groups, keys, y, p, w=None):
    """Per row: (rows, sum w (y - p), sum w p (1 - p)) of its player over the OTHER groups (zeros when none)."""
    keys, y, p, w = _arrays(keys, y, p, w)
    groups = np.asarray(groups)
    n, g, h = (np.zeros(len(y)) for _ in range(3))
    for grp in np.unique(groups):
        test = groups == grp
        sums = residual_sums(keys[~test], y[~test], p[~test], w[~test])
        for i in np.flatnonzero(test):
            s = sums.get(keys[i])
            if s is not None:
                n[i], g[i], h[i] = s
    return n, g, h


def choose_k(groups, keys, y, p, w=None, ks=K_GRID, scales=A_GRID) -> dict:
    """(k, a) with the lowest held-out log-loss when each group (match) gets offsets built from the other groups only.
    `p` must be out-of-fold expectations. Ties keep the first in grid order (k, then a). curve_a1: log-loss per k at
    a = 1 (when 1 is in the grid)."""
    n, g, h = held_out_sums(groups, keys, y, p, w)
    lp = logit(p)
    grid = {}
    for k in ks:
        off = np.where(h + k > 0, g / np.maximum(h + k, 1e-12), 0.0)
        for a in scales:
            grid[(float(k), float(a))] = log_loss(y, sigmoid(lp + a * off), w)
    best = min(grid, key=grid.get)
    out = {
        "k": best[0],
        "a": best[1],
        "log_loss": grid[best],
        "base_log_loss": log_loss(y, p, w),
        "coverage": float(np.mean(n > 0)),
    }
    if 1.0 in scales:
        out["curve_a1"] = {str(float(k)): grid[(float(k), 1.0)] for k in ks}
    return out

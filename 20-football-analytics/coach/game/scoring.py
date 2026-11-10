"""Playmaker points: 100 for options 'among the best', 0-90 by rank for the others (spec section 4)."""

import numpy as np

AMONG_BEST = 0.2
# Partial credit falls quadratically with rank: with a linear rule the second-best-looking pass already earned ~80 and
# trivial strategies (most advanced pass, shoot) beat the pros (docs/playmaker_content_eval.json).
POWER = 2.0


def share_top(refit_scores: np.ndarray) -> np.ndarray:
    """[R, K] scores under R refits -> [K] share of refits in which each option has the highest score (ties: all)."""
    s = np.asarray(refit_scores, float)
    top = s >= s.max(axis=1, keepdims=True) - 1e-15
    return top.mean(axis=0)


def points(main: np.ndarray, share: np.ndarray, among_best: float = AMONG_BEST, power: float = POWER) -> np.ndarray:
    main, share = np.asarray(main, float), np.asarray(share, float)
    full = share >= among_best
    full[int(np.argmax(main))] = True
    out = np.zeros(len(main), int)
    out[full] = 100
    rest = [i for i in np.argsort(-main, kind="stable") if not full[i]]
    if len(rest) == 1:
        out[rest[0]] = 45
    elif rest:
        r = len(rest)
        for rank, i in enumerate(rest, start=1):
            out[i] = int(round(90 * ((r - rank) / (r - 1)) ** power))
    return out

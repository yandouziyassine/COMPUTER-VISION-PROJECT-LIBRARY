"""Bootstrap refits: the four coach models refitted on whole-match resamples of the training matches."""

import zlib

import numpy as np

from .. import advice as A
from .. import datasets as D
from .. import export as X

N_REFITS = 20


def refit_models(target: str, tables: dict, n: int = N_REFITS, seed: int = 0) -> list:
    """[(advice functions, resampled match ids)] for match `target`, never trained on it."""
    others = sorted(mid for mid in tables if mid != target)
    rng = np.random.default_rng([seed, zlib.crc32(target.encode())])
    out = []
    for _ in range(n):
        pick = [others[i] for i in rng.integers(0, len(others), len(others))]
        keys = [k for k in ("pass", "value", "shot", "arrival") if k in tables[others[0]]]
        stacked = {k: D.stack([tables[mid][k] for mid in pick]) for k in keys}
        stacked["shot"] = D.drop_penalties(stacked["shot"])
        out.append((A.from_fitted(X.fit_models(stacked)), pick))
    return out

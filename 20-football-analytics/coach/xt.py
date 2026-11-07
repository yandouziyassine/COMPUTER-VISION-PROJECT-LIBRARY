"""Expected threat (xT, Singh 2019) re-implemented on the SPADL-like actions of coach.vaep (coach insight, part C).

Grid of NX x NY = 16 x 12 cells over the 105 x 68 m pitch, in the frame of the acting team (it attacks +x). Per cell,
from the actions starting there: s = shots / (shots + moves), m = moves / (shots + moves), g = scoring probability of
a shot there, T[c, c'] = completed moves from c that end in c' / moves from c (failed moves count in the denominator:
the ball is lost). xT = s * g + m * T @ xT, iterated from 0 until the largest change is below TOL (value iteration).
Moves = passes of every kind and carries; shots = open-play and free-kick shots (penalties out).

Deviation (documented in docs/coach_benchmark.md): Singh fitted a season of events; 6 matches have ~130 non-penalty
shots, so g from goals alone is 0 or 1 in most cells. By default g is the provider's xG of the shots in the cell,
shrunk towards the overall mean with SHRINK pseudo-shots (goal_source="xg"); goal_source="goals" uses goals.
Value of a move: xT(end) - xT(start) when it succeeds, -xT(start) when it fails (the threat held is lost).
"""

import json

import numpy as np

from . import vaep as V

NX, NY, LENGTH, WIDTH = 16, 12, 105.0, 68.0
TOL, MAX_ITER, SHRINK = 1e-6, 1000, 2.0
XT_SHOTS = ("shot", "shot_freekick")


def cell(x, y):
    """Cell index cx * NY + cy of attack-normalised centred points (clipped to the pitch)."""
    cx = np.clip(((np.asarray(x, float) + LENGTH / 2) / LENGTH * NX).astype(int), 0, NX - 1)
    cy = np.clip(((np.asarray(y, float) + WIDTH / 2) / WIDTH * NY).astype(int), 0, NY - 1)
    return cx * NY + cy


class ExpectedThreat:
    def __init__(self, goal_source: str = "xg", shrink: float = SHRINK):
        self.goal_source, self.shrink = goal_source, shrink

    def fit(self, a):
        n = NX * NY
        c0, c1 = cell(a["x0"], a["y0"]), cell(a["x1"], a["y1"])
        move = np.isin(a["type"], V.MOVES)
        shot = np.isin(a["type"], XT_SHOTS)
        n_move = np.bincount(c0[move], minlength=n).astype(float)
        n_shot = np.bincount(c0[shot], minlength=n).astype(float)
        total = n_move + n_shot
        self.s_ = np.divide(n_shot, total, out=np.zeros(n), where=total > 0)
        self.m_ = np.divide(n_move, total, out=np.zeros(n), where=total > 0)
        goals = a["goal"][shot].astype(float)
        xg = a["xg"][shot]
        target = goals if self.goal_source == "goals" else np.where(np.isfinite(xg), xg, goals)
        prior = float(target.mean()) if len(target) else 0.0
        num = np.bincount(c0[shot], weights=target, minlength=n) + self.shrink * prior
        den = n_shot + self.shrink
        self.g_ = np.divide(num, den, out=np.full(n, prior), where=den > 0)
        ok = move & (a["result"] == 1)
        T = np.zeros((n, n))
        np.add.at(T, (c0[ok], c1[ok]), 1.0)
        self.T_ = T / np.maximum(n_move, 1.0)[:, None]
        xt = np.zeros(n)
        it = 0
        while it < MAX_ITER:
            it += 1
            new = self.s_ * self.g_ + self.m_ * (self.T_ @ xt)
            done = float(np.max(np.abs(new - xt))) < TOL
            xt = new
            if done:
                break
        self.xt_, self.iterations_ = xt, it
        return self

    def grid(self):
        """xT as an [NX, NY] array (x towards the attacked goal, then y)."""
        return self.xt_.reshape(NX, NY)

    def action_values(self, a):
        """xT(end) - xT(start) for completed moves, -xT(start) for failed ones, NaN for shots."""
        start = self.xt_[cell(a["x0"], a["y0"])]
        end = self.xt_[cell(a["x1"], a["y1"])]
        v = np.where(a["result"] == 1, end - start, -start)
        return np.where(np.isin(a["type"], V.MOVES), v, np.nan)

    def to_json(self) -> str:
        """What a browser needs: the grid, 6 decimals."""
        return json.dumps({"nx": NX, "ny": NY, "xt": [round(float(v), 6) for v in self.xt_]})

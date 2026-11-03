"""Ghost runs and ghost defenders (coach insight, part B): counterfactual positions scored with the same models.

Pure functions of one normalised frame state (features.state; the team in possession attacks +x) and the advice
models (advice.from_fitted, or models.linear_predictor on the exported weights). Deterministic: ties keep the lower
player row and fan index (strict comparisons).

Fan: FAN_N points at distance REACH = SPEED * HORIZON_S (5 m/s for 1.5 s = 7.5 m) from the player's position, at
FAN_DEG degrees around his heading: the direction of his velocity when he moves at >= MIN_SPEED m/s, else towards the
opponent goal (+x) for an attacker and towards his own goal (+x as well, he defends it) for a defender. Points are
clipped to the pitch. The moved player stands still at his ghost point (velocity 0); everyone else keeps position and
velocity (nobody reacts: a limit of the counterfactual).

Ghost runs (attack): for each teammate j of the carrier, the score of the pass to j (advice.py's formula) with j at
each fan point; his ghost is the best point when it beats the pass to his real position by more than RUN_MIN_GAIN.
At most MAX_GHOSTS, largest gain first. Ghost defence: greedy, at most MAX_GHOSTS defenders one at a time; at each step
every remaining defender at every fan point, keep the move that lowers the attacking team's best option score
(advice.options, all options) the most, when by more than DEF_MIN_GAIN; the move stays for the next steps.
"""

import math

import numpy as np

from . import advice as A
from . import features as F

SPEED, HORIZON_S = 5.0, 1.5
REACH = SPEED * HORIZON_S
FAN_N, FAN_DEG = 7, (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
MIN_SPEED = 1.0
RUN_MIN_GAIN = 0.005  # xG
DEF_MIN_GAIN = 0.002  # xG
MAX_GHOSTS = 3
TIE = 1e-9  # two candidates closer than this could swap between languages: flagged "fragile"


def params() -> dict:
    """The constants the website's port needs (written into the parity fixture)."""
    return {
        "speed": SPEED,
        "horizon_s": HORIZON_S,
        "reach_m": REACH,
        "fan_deg": list(FAN_DEG),
        "min_speed": MIN_SPEED,
        "run_min_gain": RUN_MIN_GAIN,
        "def_min_gain": DEF_MIN_GAIN,
        "max_ghosts": MAX_GHOSTS,
        "tie": TIE,
    }


def fan(pos, vel) -> np.ndarray:
    """[FAN_N, 2] points REACH metres from pos around the heading (velocity direction, or +x when slower than
    MIN_SPEED), clipped to the pitch."""
    pos, vel = np.asarray(pos, float), np.asarray(vel, float)
    speed = math.hypot(vel[0], vel[1])
    heading = math.atan2(vel[1], vel[0]) if speed >= MIN_SPEED else 0.0
    out = []
    for d in FAN_DEG:
        a = heading + d * (math.pi / 180)
        out.append([pos[0] + REACH * math.cos(a), pos[1] + REACH * math.sin(a)])
    return np.clip(np.array(out), [-52.5, -34.0], [52.5, 34.0])


def moved(s: dict, side: str, row: int, point) -> dict:
    """Copy of s with player `row` of side "att" / "def" standing still at `point`."""
    pos, vel = s[side].copy(), s[side + "_v"].copy()
    pos[row], vel[row] = point, 0.0
    return {**s, side: pos, side + "_v": vel}


def pass_scores(states: list, carrier: int, target: int, models: dict, v_now: float) -> np.ndarray:
    """advice.py's score of the pass carrier -> target in each state (one batch per model)."""
    here = states[0]["att"][carrier]
    X = np.array([[F.pass_features(s, carrier, target)[k] for k in F.PASS_FEATURES] for s in states])
    p = np.asarray(models["pass"](X), float)
    to = [s["att"][target] for s in states]
    loss = [F.lane_loss_point(s, here, s["att"][target]) for s in states]
    v = A.value_at(models, to)
    v_opp = A.value_at(models, [-pt for pt in loss])
    return p * v - (1 - p) * v_opp - v_now


def ghost_runs(s: dict, carrier: int, models: dict) -> list:
    """Ghosts of the carrier's teammates: [{row, player, from, to, score_now, score, gain, fragile}], gain > RUN_MIN_GAIN,
    largest first, at most MAX_GHOSTS."""
    v_now = float(A.value_at(models, [s["att"][carrier]])[0])
    out = []
    for j in range(len(s["att"])):
        if j == carrier:
            continue
        pts = fan(s["att"][j], s["att_v"][j])
        scores = pass_scores([s] + [moved(s, "att", j, pt) for pt in pts], carrier, j, models, v_now)
        now, cand = float(scores[0]), scores[1:]
        k = int(np.argmax(cand))  # first maximum: the lower fan index wins a tie
        gain = float(cand[k]) - now
        if gain > RUN_MIN_GAIN:
            others = np.delete(cand, k)
            margin = min(abs(gain - RUN_MIN_GAIN), float(cand[k] - others.max()) if len(others) else math.inf)
            out.append(
                {
                    "row": j,
                    "player": int(s["att_idx"][j]),
                    "from": s["att"][j].tolist(),
                    "to": pts[k].tolist(),
                    "fan_index": k,
                    "score_now": now,
                    "score": float(cand[k]),
                    "gain": gain,
                    "fragile": bool(margin < TIE),
                }
            )
    out.sort(key=lambda g: (-g["gain"], g["row"]))
    for a, b in zip(out[:MAX_GHOSTS], out[1 : MAX_GHOSTS + 1]):
        if a["gain"] - b["gain"] < TIE:
            a["fragile"] = b["fragile"] = True
    return out[:MAX_GHOSTS]


def best_score(s: dict, carrier: int, models: dict) -> float:
    return float(A.options(s, carrier, models)[0]["score"])


def ghost_defence(s: dict, carrier: int, models: dict) -> list:
    """Greedy ghost defenders: [{row, player, from, to, best_before, best_after, gain, fragile}] in the order chosen,
    gain = best_before - best_after > DEF_MIN_GAIN, at most MAX_GHOSTS."""
    out, cur = [], s
    before = best_score(cur, carrier, models)
    left = list(range(len(s["def"])))
    for _ in range(MAX_GHOSTS):
        best, second = None, -math.inf
        for d in left:
            pts = fan(cur["def"][d], cur["def_v"][d])
            for k, pt in enumerate(pts):
                after = best_score(moved(cur, "def", d, pt), carrier, models)
                gain = before - after
                if best is None or gain > best[0]:
                    if best is not None:
                        second = max(second, best[0])
                    best = (gain, d, k, pt, after)
                else:
                    second = max(second, gain)
        if best is None or best[0] <= DEF_MIN_GAIN:
            break
        gain, d, k, pt, after = best
        margin = min(gain - second, abs(gain - DEF_MIN_GAIN))
        out.append(
            {
                "row": d,
                "player": int(s["def_idx"][d]),
                "from": cur["def"][d].tolist(),
                "to": pt.tolist(),
                "fan_index": k,
                "best_before": before,
                "best_after": after,
                "gain": gain,
                "fragile": bool(margin < TIE),
            }
        )
        cur = moved(cur, "def", d, pt)
        before = after
        left.remove(d)
    return out

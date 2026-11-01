"""Rank the ball carrier's options on one frame state, in expected-goal units.

`models` maps "pass" / "value" / "shot" to a function X[n, k] -> number[n] over the feature order of
features.PASS_FEATURES / VALUE_FEATURES / SHOT_FEATURES (see from_fitted for sklearn models): the pass-completion
probability, the value V (expected goals for minus against within the horizon, for the team in possession with the ball
at that point) and the shot's goal probability (xG).

Score of an option = p * V(target) + (1 - p) * (-V_opp(loss point)) - V(now): the expected change of the net expected
goals. V_opp is the value for the opponent in possession (the point mirrored), and the loss point is where a failed
pass is lost (features.lane_loss_point) or, for a carry, the carry point itself. A shot scores xG - V(now).
"""

import numpy as np

from . import features as F
from . import pitch_control as PC

CARRY_DIST, CARRY_ANGLES, SHOT_RANGE = 6.0, (-60, -30, 0, 30, 60), 35.0


def from_fitted(fitted: dict) -> dict:
    """Wrap fitted sklearn objects: classifiers give P(y = 1), regressors their prediction."""

    def fn(m):
        if hasattr(m, "predict_proba"):
            return lambda X: m.predict_proba(np.atleast_2d(X))[:, 1]
        return lambda X: m.predict(np.atleast_2d(X))

    return {k: fn(m) for k, m in fitted.items()}


def value_at(models, pts):
    """V at points [N, 2] (normalised frame: the team in possession attacks +x)."""
    return models["value"](np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in pts]))


def options(s, carrier: int, models: dict) -> list:
    out, loss = [], []
    here = s["att"][carrier]
    for j in range(len(s["att"])):
        if j == carrier:
            continue
        f = F.pass_features(s, carrier, j)
        p = float(models["pass"](np.array([[f[k] for k in F.PASS_FEATURES]]))[0])
        out.append({"kind": "pass", "target": int(s["att_idx"][j]), "to": s["att"][j].tolist(), "p": p})
        loss.append(F.lane_loss_point(s, here, s["att"][j]))
    for a in CARRY_ANGLES:
        pt = here + CARRY_DIST * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        pt = np.clip(pt, [-52.5, -34], [52.5, 34])
        p = float(PC.control(pt[None], s["att"], s["att_v"], s["def"], s["def_v"])[0])
        out.append({"kind": "carry", "angle": a, "to": pt.tolist(), "p": p})
        loss.append(pt)
    v_now = float(value_at(models, [here])[0])
    v_keep = value_at(models, [np.array(o["to"]) for o in out])
    v_opp = value_at(models, [-l for l in loss])  # the opponent attacks the other way: mirror the loss point
    for o, vk, vo in zip(out, v_keep, v_opp):
        o["value"], o["v_lost"] = float(vk), float(vo)
        o["score"] = float(o["p"] * vk - (1 - o["p"]) * vo - v_now)
    sf = F.shot_features(here, s["def"])
    if sf["goal_dist"] <= SHOT_RANGE:
        p = float(models["shot"](np.array([[sf[k] for k in F.SHOT_FEATURES]]))[0])
        out.append({"kind": "shot", "to": [52.5, 0.0], "p": p, "value": 1.0, "score": p - v_now})
    return sorted(out, key=lambda o: -o["score"])

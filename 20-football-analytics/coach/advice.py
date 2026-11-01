"""Rank the ball carrier's options on one frame state.

`models` maps "pass" / "value" / "shot" to a function X[n, k] -> probability[n] over the feature order of
features.PASS_FEATURES / VALUE_FEATURES / SHOT_FEATURES (see from_fitted for sklearn models).
"""

import numpy as np

from . import features as F
from . import pitch_control as PC

CARRY_DIST, CARRY_ANGLES, SHOT_RANGE = 6.0, (-60, -30, 0, 30, 60), 35.0


def from_fitted(fitted: dict) -> dict:
    return {k: (lambda X, m=m: m.predict_proba(np.atleast_2d(X))[:, 1]) for k, m in fitted.items()}


def _value(models, pts):
    return models["value"](np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in pts]))


def options(s, carrier: int, models: dict) -> list:
    out = []
    here = s["att"][carrier]
    for j in range(len(s["att"])):
        if j == carrier:
            continue
        f = F.pass_features(s, carrier, j)
        p = float(models["pass"](np.array([[f[k] for k in F.PASS_FEATURES]]))[0])
        out.append({"kind": "pass", "target": int(s["att_idx"][j]), "to": s["att"][j].tolist(), "p": p})
    for a in CARRY_ANGLES:
        pt = here + CARRY_DIST * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        pt = np.clip(pt, [-52.5, -34], [52.5, 34])
        p = float(PC.control(pt[None], s["att"], s["att_v"], s["def"], s["def_v"])[0])
        out.append({"kind": "carry", "angle": a, "to": pt.tolist(), "p": p})
    to = [np.array(o["to"]) for o in out]
    v_keep = _value(models, to)
    v_lose = _value(models, [-t for t in to])  # opponent attacks the other way
    for o, vk, vl in zip(out, v_keep, v_lose):
        o["value"], o["score"] = float(vk), float(o["p"] * vk - (1 - o["p"]) * vl)
    sf = F.shot_features(here, s["def"])
    if sf["goal_dist"] <= SHOT_RANGE:
        p = float(models["shot"](np.array([[sf[k] for k in F.SHOT_FEATURES]]))[0])
        out.append({"kind": "shot", "to": [52.5, 0.0], "p": p, "value": 1.0, "score": p})
    return sorted(out, key=lambda o: -o["score"])

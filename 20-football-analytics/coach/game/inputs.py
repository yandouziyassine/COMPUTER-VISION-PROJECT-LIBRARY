"""Model-independent inputs of one decision and fast scores for any model set (same formula as advice.options).

Option order (k = index): passes by target player index, then carries by angle (advice.CARRY_ANGLES order), then the
shot when the carrier is within advice.SHOT_RANGE of the goal."""

import numpy as np

from .. import advice as A
from .. import features as F
from .. import pitch_control as PC


def _vf(points) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, float))
    return np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in pts]).reshape(
        len(pts), len(F.VALUE_FEATURES)
    )


def inputs(s: dict, carrier: int) -> dict:
    here = s["att"][carrier]
    rows = sorted((j for j in range(len(s["att"])) if j != carrier), key=lambda j: int(s["att_idx"][j]))
    pass_x = np.array([[F.pass_features(s, carrier, j)[k] for k in F.PASS_FEATURES] for j in rows])
    pass_to = np.array([s["att"][j] for j in rows], float)
    pass_loss = np.array([F.lane_loss_point(s, here, s["att"][j]) for j in rows], float)
    carry_to, carry_p = [], []
    for a in A.CARRY_ANGLES:
        pt = here + A.CARRY_DIST * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        pt = np.clip(pt, [-52.5, -34], [52.5, 34])
        carry_to.append(pt)
        carry_p.append(float(PC.control(pt[None], s["att"], s["att_v"], s["def"], s["def_v"])[0]))
    carry_to = np.array(carry_to)
    sf = F.shot_features(here, s["def"])
    shot_x = np.array([[sf[k] for k in F.SHOT_FEATURES]]) if sf["goal_dist"] <= A.SHOT_RANGE else None
    return {
        "targets": [int(s["att_idx"][j]) for j in rows],
        "angles": list(A.CARRY_ANGLES),
        "pass_x": pass_x,
        "pass_to": pass_to,
        "carry_to": carry_to,
        "carry_p": np.array(carry_p),
        "vf_here": _vf([here]),
        "vf_pass_to": _vf(pass_to),
        "vf_carry_to": _vf(carry_to),
        "vf_pass_loss": _vf(-pass_loss),
        "vf_carry_loss": _vf(-carry_to),
        "shot_x": shot_x,
    }


def key(o: dict) -> tuple:
    """Identity of an option across model sets: ("pass", player) / ("carry", angle) / ("shot",)."""
    if o["kind"] == "pass":
        return ("pass", int(o["target"]))
    if o["kind"] == "carry":
        return ("carry", int(o["angle"]))
    return ("shot",)


def order(k: tuple) -> tuple:
    return ({"pass": 0, "carry": 1, "shot": 2}[k[0]], *k[1:])


def options(inp: dict) -> list:
    """The options in k order: {"k", "kind", "target" | "angle", "to"} (to: normalised metres)."""
    out = [{"kind": "pass", "target": t, "to": inp["pass_to"][i].tolist()} for i, t in enumerate(inp["targets"])]
    out += [{"kind": "carry", "angle": a, "to": inp["carry_to"][i].tolist()} for i, a in enumerate(inp["angles"])]
    if inp["shot_x"] is not None:
        out.append({"kind": "shot", "to": [52.5, 0.0]})
    for k, o in enumerate(out):
        o["k"] = k
    return out


def parts(inp: dict, models: dict) -> dict:
    """p, value, v_lost, score per option (k order) and v_now under one model set."""
    v_now = float(models["value"](inp["vf_here"])[0])
    p_pass = models["pass"](inp["pass_x"])
    v_pass = models["arrival" if "arrival" in models else "value"](inp["vf_pass_to"])
    p = np.concatenate([p_pass, inp["carry_p"]])
    vk = np.concatenate([v_pass, models["value"](inp["vf_carry_to"])])
    vo = np.concatenate([models["value"](inp["vf_pass_loss"]), models["value"](inp["vf_carry_loss"])])
    score = p * vk - (1 - p) * vo - v_now
    if inp["shot_x"] is not None:
        ps = float(models["shot"](inp["shot_x"])[0])
        p, vk, vo = np.append(p, ps), np.append(vk, 1.0), np.append(vo, 0.0)
        score = np.append(score, ps - v_now)
    return {"p": p, "value": vk, "v_lost": vo, "score": score, "v_now": v_now}


def scores(inp: dict, models: dict) -> np.ndarray:
    return parts(inp, models)["score"]

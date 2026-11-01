"""Labelled tables for the three models. Every row carries its match id, for leave-one-match-out evaluation."""

import numpy as np

from . import features as F
from .sync import event_xy

HORIZON = 10.0  # seconds: "does this possession lead to a shot soon?"


def _table(rows, ys, mid, names):
    X = np.array([[r[k] for k in names] for r in rows], float).reshape(len(rows), len(names))
    return {"X": X, "y": np.array(ys, int), "match": np.array([mid] * len(rows))}


def intended_target(m, e, s, where):
    """Teammate (row of s["att"]) an unsuccessful pass was meant for.

    In the DFL events the Recipient of an unsuccessful pass is the opponent who got the ball, so the intended receiver
    is not recorded. It is inferred from the ball path: the ball is followed from the pass until it is closest to that
    opponent (the interception point E); the intended target is the teammate nearest to the ray passer -> E among
    those at least half-way along it. Returns None when the path cannot be read (untracked opponent, E < 3 m away).
    """
    r = e["recipient"]
    if r is None or r == e["player"]:
        return None
    h = m.halves[e["half"]]
    lo, hi = e["frame"], min(len(h.t), e["frame"] + 4 * 25 + 1)
    d = np.linalg.norm(h.xy[lo:hi, r] - h.ball[lo:hi, :2], axis=1)
    if not np.isfinite(d).any() or np.nanmin(d) > 3.0:
        return None
    end = h.ball[lo + int(np.nanargmin(d)), :2] * s["direction"]  # normalised like the state
    a = s["att"][where[e["player"]]]
    v = end - a
    length = float(np.linalg.norm(v))
    if length < 3.0:
        return None
    u = v / length
    best, best_cost = None, np.inf
    for pid, j in where.items():
        if pid == e["player"]:
            continue
        w = s["att"][j] - a
        if float(w @ u) < 0.5 * length:
            continue
        cost = abs(float(w[0] * u[1] - w[1] * u[0]))  # perpendicular distance to the ray
        if cost < best_cost:
            best, best_cost = j, cost
    return best


def pass_table(m) -> dict:
    rows, ys = [], []
    for e in m.events:
        if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
            continue
        if e["team"] < 0 or e["player"] is None or e["recipient"] == e["player"]:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where:
            continue
        if e["success"]:
            if e["recipient"] not in where:
                continue
            target = where[e["recipient"]]
        else:  # the recorded recipient is an opponent: infer the intended teammate from the ball path
            target = intended_target(m, e, s, where)
            if target is None:
                continue
        rows.append(F.pass_features(s, where[e["player"]], target))
        ys.append(int(e["success"]))
    return _table(rows, ys, m.match_id, F.PASS_FEATURES)


def value_table(m, every: float = 1.0) -> dict:
    rows, ys = [], []
    for hi, h in enumerate(m.halves):
        shots = {0: [], 1: []}
        for e in m.events:
            if e["type"] == "shot" and e.get("half") == hi and e["team"] in shots:
                shots[e["team"]].append(e["t"])
        step = max(1, int(round(every * 25)))
        for i in range(0, len(h.t), step):
            team = int(h.owner[i])
            if not h.alive[i] or team < 0 or np.isnan(h.ball[i, 0]):
                continue
            s = F.state(m, hi, i, team)
            rows.append(F.value_features(s["ball"]))
            ys.append(int(any(0 < t - h.t[i] <= HORIZON for t in shots[team])))
    return _table(rows, ys, m.match_id, F.VALUE_FEATURES)


def shot_table(m) -> dict:
    rows, ys = [], []
    for e in m.events:
        if e["type"] != "shot" or e.get("frame") is None or e["team"] < 0:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        loc = event_xy(m, e) * s["direction"]  # event location, normalised like the state
        rows.append(F.shot_features(loc, s["def"]))
        ys.append(int(e["goal"]))
    return _table(rows, ys, m.match_id, F.SHOT_FEATURES)


def stack(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in ("X", "y", "match")}

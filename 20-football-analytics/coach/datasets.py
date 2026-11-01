"""Labelled tables for the three models. Every row carries its match id, for leave-one-match-out evaluation."""

import numpy as np

from . import features as F
from .sync import event_xy

HORIZON = 10.0  # seconds: "does this possession lead to a shot soon?"


def _table(rows, ys, mid, names):
    X = np.array([[r[k] for k in names] for r in rows], float).reshape(len(rows), len(names))
    return {"X": X, "y": np.array(ys, int), "match": np.array([mid] * len(rows))}


def pass_table(m) -> dict:
    rows, ys = [], []
    for e in m.events:
        if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
            continue
        if e["player"] is None or e["recipient"] is None or e["player"] == e["recipient"]:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where or e["recipient"] not in where:
            continue
        rows.append(F.pass_features(s, where[e["player"]], where[e["recipient"]]))
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

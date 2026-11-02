"""Labelled tables for the three models. Every row carries its match id, for leave-one-match-out evaluation.

Pass table: open-play passes, y = completed. Value table: one row per second of live play, y = expected goals of the
team in possession minus expected goals against it within the next H seconds (provider xG per shot, all shots
including penalties and direct free kicks). Shot table: y = goal, penalties flagged (and kept out of the fit).
"""

import numpy as np

from . import features as F
from .sync import event_xy

HORIZON = 15.0  # seconds: value = net expected goals within this horizon (10 / 15 / 20 s are compared)
HORIZONS = (10.0, 15.0, 20.0)
XG_DEFAULT = 0.1  # shots without a provider xG (none in the 7 matches, kept for robustness)
MAX_FLIGHT_S, LEFT_PASSER_M, TOUCH_M, RAY_TOL_M, FALLBACK_M = 6.0, 3.0, 1.5, 5.0, 10.0


def _table(rows, ys, mid, names, **extra):
    X = np.array([[r[k] for k in names] for r in rows], float).reshape(len(rows), len(names))
    return {"X": X, "y": np.array(ys, int), "match": np.array([mid] * len(rows)), **extra}


def when(m, e) -> float:
    """Event time on the tracking clock: the time of its snapped frame, else the shifted nominal time."""
    if e.get("frame") is not None:
        return float(m.halves[e["half"]].t[e["frame"]])
    return float(e.get("t_sync", e["t"]))


def shots_by_half(m) -> list:
    """Per half: arrays (time, team, xG) of every shot with a team, sorted by time."""
    out = []
    for hi in range(len(m.halves)):
        rows = sorted(
            (when(m, e), e["team"], e.get("xg") if e.get("xg") is not None else XG_DEFAULT)
            for e in m.events
            if e["type"] == "shot" and e.get("half") == hi and e["team"] in (0, 1)
        )
        out.append(np.array(rows, float).reshape(-1, 3))
    return out


def net_xg(shots, t: float, team: int, horizon: float) -> float:
    """xG of `team` minus xG against it, for shots in (t, t + horizon]."""
    if not len(shots):
        return 0.0
    sel = (shots[:, 0] > t) & (shots[:, 0] <= t + horizon)
    mine = shots[sel & (shots[:, 1] == team), 2].sum()
    return float(mine - shots[sel & (shots[:, 1] != team), 2].sum())


def _ball_path_end(m, e, s):
    """Where an unsuccessful pass ended: (end point in the state's frame, interceptor player index or None).

    The ball is followed from the release; once it is more than LEFT_PASSER_M from the passer, the first frame it is
    within TOUCH_M of an opponent is the interception. Without one, the first dead-ball frame is the end point.
    """
    h = m.halves[e["half"]]
    f = e["frame"]
    up = min(len(h.t), f + int(MAX_FLIGHT_S * F.match_fps(m)) + 1)
    ball = h.ball[f:up, :2]
    left = np.flatnonzero(np.linalg.norm(h.xy[f:up, e["player"]] - ball, axis=1) > LEFT_PASSER_M)
    if not len(left):
        return None
    k0 = int(left[0])
    opp = np.flatnonzero(m.team == 1 - e["team"])
    d = np.linalg.norm(h.xy[f + k0 : up][:, opp, :] - ball[k0:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    near = np.flatnonzero(d.min(axis=1) <= TOUCH_M)
    if len(near):
        k = int(near[0])
        return ball[k0 + k] * s["direction"], int(opp[int(np.argmin(d[k]))])
    dead = np.flatnonzero(~h.alive[f + k0 : up])
    if len(dead) and np.isfinite(ball[k0 + int(dead[0])]).all():
        return ball[k0 + int(dead[0])] * s["direction"], None
    return None


def intended_target(m, e, s, where):
    """(row of s["att"] of the teammate an unsuccessful pass was meant for, interceptor player index) or None.

    In the DFL events the Recipient of an unsuccessful pass is the opponent who got the ball, so the intended receiver
    is not recorded. It is inferred from the ball path (_ball_path_end): the teammate nearest to the ray passer ->
    end point among those at least half-way along it and within RAY_TOL_M of it; without such a teammate (or without
    an interception, e.g. a ball out of play) the teammate nearest to the end point, if within FALLBACK_M.
    """
    end = _ball_path_end(m, e, s)
    if end is None:
        return None
    point, interceptor = end
    a = s["att"][where[e["player"]]]
    rows = [j for pid, j in where.items() if pid != e["player"]]
    if not rows:
        return None
    v = point - a
    length = float(np.linalg.norm(v))
    if interceptor is not None and length >= 3.0:
        u = v / length
        best, best_cost = None, np.inf
        for j in rows:
            w = s["att"][j] - a
            cost = abs(float(w[0] * u[1] - w[1] * u[0]))  # perpendicular distance to the ray
            if float(w @ u) >= 0.5 * length and cost <= RAY_TOL_M and cost < best_cost:
                best, best_cost = j, cost
        if best is not None:
            return best, interceptor
    dist = [float(np.linalg.norm(s["att"][j] - point)) for j in rows]
    k = int(np.argmin(dist))
    return (rows[k], interceptor) if dist[k] <= FALLBACK_M else None


def pass_samples(m):
    """Usable open-play passes: dicts with the event (and its index in m.events), the state at the release, carrier /
    target rows of s["att"], and for unsuccessful passes the interceptor's row in s["def"] (None when unknown).

    When the provider records the intended receiver of failed passes too (e["target_known"], SkillCorner), it is used
    as is and the ball path only names the interceptor; otherwise (DFL) the target is inferred from the ball path."""
    for idx, e in enumerate(m.events):
        if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
            continue
        if e["team"] < 0 or e["player"] is None or e["recipient"] == e["player"]:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where:
            continue
        interceptor = None
        if e["success"] or e.get("target_known"):
            if e["recipient"] not in where:
                continue
            target = where[e["recipient"]]
            if not e["success"]:
                end = _ball_path_end(m, e, s)
                who = end[1] if end is not None else None
                if who is not None and who in s["def_idx"]:
                    interceptor = int(np.flatnonzero(s["def_idx"] == who)[0])
        else:  # the recorded recipient is an opponent: infer the intended teammate from the ball path
            found = intended_target(m, e, s, where)
            if found is None:
                continue
            target, who = found
            if who is not None and who in s["def_idx"]:
                interceptor = int(np.flatnonzero(s["def_idx"] == who)[0])
        yield {
            "index": idx,
            "event": e,
            "state": s,
            "carrier": where[e["player"]],
            "target": target,
            "interceptor": interceptor,
        }


def completion_rate(m) -> float:
    """True open-play completion rate of the match, from all its pass events (placed on the tracking or not)."""
    ev = [
        e for e in m.events if e["type"] == "pass" and not e.get("set_piece") and not e.get("cross") and e["team"] >= 0
    ]
    return float(np.mean([e["success"] for e in ev])) if ev else float("nan")


def pass_table(m) -> dict:
    """X: features at the release; weight: reweighting to the true completion rate. X_noint: the same with the interceptor removed from the defenders of unsuccessful
    passes (he stands on the lane by construction); X_geom: geometry only (no defenders), both leak-free variants."""
    rows, rows_noint, ys, idx, target = [], [], [], [], []
    for p in pass_samples(m):
        idx.append(p["index"])
        target.append(int(p["state"]["att_idx"][p["target"]]))
        s, c, t = p["state"], p["carrier"], p["target"]
        rows.append(F.pass_features(s, c, t))
        s2 = s if p["interceptor"] is None else F.without_defender(s, p["interceptor"])
        rows_noint.append(F.pass_features(s2, c, t))
        ys.append(int(p["event"]["success"]))
    table = _table(rows, ys, m.match_id, F.PASS_FEATURES)
    table["X_noint"] = _table(rows_noint, ys, m.match_id, F.PASS_FEATURES)["X"]
    table["event_idx"] = np.array(idx, int)  # row -> index in m.events
    table["target_idx"] = np.array(target, int)  # row -> player index of the (given or inferred) target
    # failed passes are harder to place (intended receiver, release frame) than completed ones, so the table over-represents
    # completions: weight the failed rows so that the weighted completion rate equals the match's true rate
    y = table["y"]
    n_ok, n_fail, rate = int(y.sum()), int((1 - y).sum()), completion_rate(m)
    w_fail = n_ok * (1 - rate) / (rate * n_fail) if n_ok and n_fail and 0 < rate < 1 else 1.0
    table["weight"] = np.where(y == 1, 1.0, w_fail)
    return table


def value_table(m, horizons=(HORIZON,), every: float = 1.0) -> dict:
    """y: net xG within the first horizon; y_h maps every horizon to its labels; half / t / team locate each row
    (labels can be recomputed with another shot value, see cross_league.value_labels)."""
    rows, ys, where = [], {h: [] for h in horizons}, []
    shots = shots_by_half(m)
    step = max(1, int(round(every * F.match_fps(m))))
    for hi, h in enumerate(m.halves):
        for i in range(0, len(h.t), step):
            team = int(h.owner[i])
            if not h.alive[i] or team < 0 or np.isnan(h.ball[i, 0]):
                continue
            s = F.state(m, hi, i, team)
            rows.append(F.value_features(s["ball"]))
            where.append((hi, float(h.t[i]), team))
            for hz in horizons:
                ys[hz].append(net_xg(shots[hi], float(h.t[i]), team, hz))
    X = np.array([[r[k] for k in F.VALUE_FEATURES] for r in rows], float).reshape(len(rows), len(F.VALUE_FEATURES))
    y_h = {hz: np.array(v) for hz, v in ys.items()}
    w = np.array(where, float).reshape(len(where), 3)
    return {
        "X": X,
        "y": y_h[horizons[0]],
        "match": np.array([m.match_id] * len(rows)),
        "y_h": y_h,
        "half": w[:, 0].astype(int),
        "t": w[:, 1],
        "team": w[:, 2].astype(int),
    }


def shot_table(m) -> dict:
    """Every located shot; `penalty` flags penalties (kept out of the fit by drop_penalties); `xg` is the provider's."""
    rows, ys, pen, xg = [], [], [], []
    for e in m.events:
        if e["type"] != "shot" or e.get("frame") is None or e["team"] < 0:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        loc = event_xy(m, e) * s["direction"]  # event location, normalised like the state
        rows.append(F.shot_features(loc, s["def"]))
        ys.append(int(e["goal"]))
        pen.append(e.get("set_piece") == "Penalty")
        xg.append(e["xg"] if e.get("xg") is not None else np.nan)
    return _table(rows, ys, m.match_id, F.SHOT_FEATURES, penalty=np.array(pen, bool), xg=np.array(xg, float))


def drop_penalties(table) -> dict:
    keep = ~table["penalty"]
    return {k: v[keep] for k, v in table.items()}


def stack(tables):
    out = {}
    for k, v in tables[0].items():
        if isinstance(v, dict):
            out[k] = {h: np.concatenate([t[k][h] for t in tables]) for h in v}
        else:
            out[k] = np.concatenate([t[k] for t in tables])
    return out

"""VAEP (Decroos et al., KDD 2019) re-implemented on the IDSSE events, and the SPADL-like action table xT uses too.

    python -m coach.benchmark     # runs it (coach insight, part C); this module has no main

Action table (actions(m)): one row per on-ball action, a dict of NumPy arrays ordered by half and time: match, half,
t (s since the half's first frame), team, player, type (TYPES), result (1 success / 0 fail; a shot succeeds when it
scores), x0, y0, x1, y1 (metres, centred, normalised so that the ACTING team attacks +x), event (index in m.events,
-1 for carries), xg (provider xG of a shot, else NaN), goal (1 for a shot that scored).

The IDSSE events hold passes and shots only (no duels, tackles, interceptions, clearances or dribbles), so the table
has passes (by set piece: throw_in, freekick, corner, goalkick, kickoff; else cross or pass), shots (shot,
shot_freekick, shot_penalty) and carries taken from the tracking:
- a completed pass ends where the recipient stands when the ball first comes within ARRIVE_M of him after the
  release (within MAX_FLIGHT_S); otherwise, and for a failed pass, it ends at the start of the next action of the
  half (for a failed pass: where the other team next played the ball, or the restart);
- a shot ends at the goal centre;
- carry: after a completed pass, when the recipient plays the next action of the half CARRY_MIN_M..CARRY_MAX_M away
  from where he received the ball and within CARRY_MAX_S, a carry from the reception point to that action's start
  (the way socceraction adds dribbles). Carries succeed by construction.

VAEP: the game state of an action is the action and the two before it (features()); labels: the acting team scores /
concedes within the next K actions of the half, the action included (labels()); two classifiers give P_score and
P_concede; value = (P_score(a_i) - P_score(a_i-1)) - (P_concede(a_i) - P_concede(a_i-1)), where the previous pair is
swapped when a_i-1 was by the other team and set to 0 at the start of a half or after a goal (values()). lomo() fits
the classifiers without the match it scores; with select_inner each classifier is chosen among CANDIDATES (gradient
boosting with more or less regularisation, or the base rate) by leave-one-match-out on the training matches only.
"""

import numpy as np

from . import datasets as D
from . import features as F
from . import geometry as G
from . import models as ML
from .sync import event_xy

TYPES = (
    "pass",
    "cross",
    "throw_in",
    "freekick",
    "corner",
    "goalkick",
    "kickoff",
    "carry",
    "shot",
    "shot_freekick",
    "shot_penalty",
)
SHOTS = ("shot", "shot_freekick", "shot_penalty")
MOVES = tuple(t for t in TYPES if t not in SHOTS)
SET_PIECES = {
    "ThrowIn": "throw_in",
    "FreeKick": "freekick",
    "CornerKick": "corner",
    "GoalKick": "goalkick",
    "KickOff": "kickoff",
}
ARRIVE_M, MAX_FLIGHT_S = 2.0, 6.0
CARRY_MIN_M, CARRY_MAX_M, CARRY_MAX_S = 3.0, 60.0, 10.0
GOAL = np.array([G.HALF_LENGTH, 0.0])


def action_type(e) -> str:
    sp = e.get("set_piece")
    if e["type"] == "shot":
        return {"Penalty": "shot_penalty", "FreeKick": "shot_freekick"}.get(sp, "shot")
    if sp in SET_PIECES:
        return SET_PIECES[sp]
    return "cross" if e.get("cross") else "pass"


def _player(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _arrival(m, e):
    """Tracking position of the recipient when the ball first comes within ARRIVE_M of him after the release."""
    r = e.get("recipient")
    if e.get("frame") is None or not _player(r) or not 0 <= r < len(m.player_ids):
        return None
    h = m.halves[e["half"]]
    f = int(e["frame"])
    up = min(len(h.t), f + int(MAX_FLIGHT_S * F.match_fps(m)) + 1)
    if up <= f + 1:
        return None
    d = np.linalg.norm(h.xy[f + 1 : up, r] - h.ball[f + 1 : up, :2], axis=1)
    k = np.flatnonzero(d <= ARRIVE_M)
    return h.xy[f + 1 + int(k[0]), r].astype(float) if len(k) else None


def actions(m) -> dict:
    """The SPADL-like action table of match m (see the module docstring)."""
    dirs = [G.attack_dirs(h, m.team) for h in m.halves]
    ev = [
        (i, e)
        for i, e in enumerate(m.events)
        if e["type"] in ("pass", "shot")
        and e.get("half") is not None
        and e.get("team") in (0, 1)
        and np.isfinite(e.get("x", np.nan))
    ]
    ev.sort(key=lambda ie: (ie[1]["half"], D.when(m, ie[1])))
    base = [
        {
            "half": e["half"],
            "t": D.when(m, e) - float(m.halves[e["half"]].t[0]),
            "team": int(e["team"]),
            "player": int(e["player"]) if _player(e.get("player")) else -1,
            "type": action_type(e),
            "start": event_xy(m, e).astype(float),
            "event": i,
            "e": e,
        }
        for i, e in ev
    ]
    rows = []
    for k, a in enumerate(base):
        e = a["e"]
        nxt = base[k + 1] if k + 1 < len(base) and base[k + 1]["half"] == a["half"] else None
        if a["type"] in SHOTS:
            end, result = GOAL * dirs[a["half"]][a["team"]], int(bool(e.get("goal")))
        else:
            result = int(bool(e.get("success")))
            got = _arrival(m, e) if result else None
            end = got if got is not None else (nxt["start"] if nxt is not None else a["start"])
        rows.append({**a, "end": end, "result": result})
        if a["type"] in MOVES and result and nxt is not None and nxt["team"] == a["team"]:
            if nxt["player"] == e.get("recipient"):
                gap = float(np.linalg.norm(nxt["start"] - end))
                if CARRY_MIN_M <= gap <= CARRY_MAX_M and nxt["t"] - a["t"] <= CARRY_MAX_S:
                    rows.append(
                        {
                            **a,
                            "t": (a["t"] + nxt["t"]) / 2,
                            "player": nxt["player"],
                            "type": "carry",
                            "start": end,
                            "end": nxt["start"],
                            "event": -1,
                            "e": None,
                            "result": 1,
                        }
                    )
    n = len(rows)
    d = np.array([dirs[r["half"]][r["team"]] for r in rows], float).reshape(n, 1)
    s = np.array([r["start"] for r in rows], float).reshape(n, 2) * d
    en = np.array([r["end"] for r in rows], float).reshape(n, 2) * d
    xg = [r["e"]["xg"] if r["type"] in SHOTS and r["e"].get("xg") is not None else np.nan for r in rows]
    return {
        "match": np.array([str(m.match_id)] * n),
        "half": np.array([r["half"] for r in rows], int),
        "t": np.array([r["t"] for r in rows], float),
        "team": np.array([r["team"] for r in rows], int),
        "player": np.array([r["player"] for r in rows], int),
        "type": np.array([r["type"] for r in rows], dtype="U16"),
        "result": np.array([r["result"] for r in rows], int),
        "x0": s[:, 0],
        "y0": s[:, 1],
        "x1": en[:, 0],
        "y1": en[:, 1],
        "event": np.array([r["event"] for r in rows], int),
        "xg": np.array(xg, float),
        "goal": np.array([int(r["type"] in SHOTS and r["result"] == 1) for r in rows], int),
    }


K = 10  # labels: goal within the next K actions of the half (the action included)
BLOCK = tuple(f"type_{t}" for t in TYPES) + (
    "result",
    "x0",
    "y0",
    "x1",
    "y1",
    "dx",
    "dy",
    "length",
    "start_goal_dist",
    "start_goal_angle",
    "end_goal_dist",
    "end_goal_angle",
)
FEATURES = tuple(f"a{lag}_{name}" for lag in range(3) for name in BLOCK) + (
    "team_1",
    "team_2",
    "dt_1",
    "dt_2",
    "gap_x",
    "gap_y",
    "goals_for",
    "goals_against",
    "goal_diff",
    "half",
    "t",
)


def labels(a, k: int = K) -> tuple:
    """(scores, concedes): 1 when the acting team scores / concedes within actions i .. i + k - 1 of the same half."""
    n = len(a["team"])
    scores, concedes = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        for j in range(i, min(n, i + k)):
            if a["half"][j] != a["half"][i] or a["match"][j] != a["match"][i]:
                break
            if a["goal"][j]:
                if a["team"][j] == a["team"][i]:
                    scores[i] = 1
                else:
                    concedes[i] = 1
    return scores, concedes


def score_before(a) -> tuple:
    """(goals for, goals against) of the acting team before each action, per match."""
    n = len(a["team"])
    gf, ga, tally = np.zeros(n), np.zeros(n), {}
    for i in range(n):
        g = tally.setdefault(a["match"][i], [0, 0])
        gf[i], ga[i] = g[a["team"][i]], g[1 - a["team"][i]]
        if a["goal"][i]:
            g[a["team"][i]] += 1
    return gf, ga


def _previous(a, idx, fallback):
    """idx where it is an action of the same half and match as each row, else `fallback`."""
    j = np.maximum(idx, 0)
    ok = (idx >= 0) & (a["half"][j] == a["half"]) & (a["match"][j] == a["match"])
    return np.where(ok, j, fallback)


def _block(a, idx, ref_team):
    """BLOCK features of the actions idx, seen from ref_team (an action by the other team is rotated by 180 deg)."""
    flip = np.where(a["team"][idx] == ref_team, 1.0, -1.0)
    x0, y0, x1, y1 = (a[k][idx] * flip for k in ("x0", "y0", "x1", "y1"))
    dx, dy = x1 - x0, y1 - y0
    d0, g0 = G.goal_geometry(np.c_[x0, y0])
    d1, g1 = G.goal_geometry(np.c_[x1, y1])
    onehot = (a["type"][idx][:, None] == np.array(TYPES)[None]).astype(float)
    cols = [a["result"][idx], x0, y0, x1, y1, dx, dy, np.hypot(dx, dy), d0, g0, d1, g1]
    return np.column_stack([onehot] + cols)


def features(a) -> np.ndarray:
    """Game state of each action: BLOCK of the action and of the two before it (from the acting team's side; at the
    start of a half the missing ones repeat the earliest available), plus team / time / gap / score features."""
    n = len(a["team"])
    i = np.arange(n)
    p1 = _previous(a, i - 1, i)
    p2 = _previous(a, i - 2, p1)
    ref = a["team"]
    flip1 = np.where(a["team"][p1] == ref, 1.0, -1.0)
    gf, ga = score_before(a)
    extra = [
        (a["team"][p1] == ref).astype(float),
        (a["team"][p2] == ref).astype(float),
        a["t"] - a["t"][p1],
        a["t"] - a["t"][p2],
        a["x0"] - a["x1"][p1] * flip1,
        a["y0"] - a["y1"][p1] * flip1,
        gf,
        ga,
        gf - ga,
        a["half"].astype(float),
        a["t"],
    ]
    return np.column_stack([_block(a, idx, ref) for idx in (i, p1, p2)] + extra).reshape(n, len(FEATURES))


def values(a, p_score, p_concede) -> dict:
    """offensive = P_score(i) - P_score(i-1), defensive = -(P_concede(i) - P_concede(i-1)), value = their sum; the
    previous pair is swapped when action i-1 was by the other team and is 0 at the start of a half or after a goal."""
    n = len(p_score)
    prev = np.arange(n) - 1
    j = np.maximum(prev, 0)
    start = (prev < 0) | (a["half"][j] != a["half"]) | (a["match"][j] != a["match"]) | (a["goal"][j] == 1)
    same = a["team"][j] == a["team"]
    ps = np.where(start, 0.0, np.where(same, p_score[j], p_concede[j]))
    pc = np.where(start, 0.0, np.where(same, p_concede[j], p_score[j]))
    off, dfn = p_score - ps, -(p_concede - pc)
    return {"offensive": off, "defensive": dfn, "value": off + dfn}


# Classifier candidates. "boost_300" is the coach's ML.fit_boosting; with 7 matches (19 goals) it overfits, so lomo()
# picks per held-out match, by inner leave-one-match-out log-loss on the training matches only, among these.
CANDIDATES = {
    "boost_300": {"max_iter": 300, "learning_rate": 0.05, "max_depth": 3},
    "boost_50_leaf200": {
        "max_iter": 50,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 200,
        "l2_regularization": 1.0,
    },
    "boost_30_leaf400": {
        "max_iter": 30,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_samples_leaf": 400,
        "l2_regularization": 1.0,
    },
    "base_rate": None,
}
TARGETS = ("score", "concede")


def fit_classifier(name: str, X, y):
    if CANDIDATES[name] is None:
        return ML.fit_base_rate(X, y)
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(random_state=0, early_stopping=False, **CANDIDATES[name]).fit(X, y)


def fit(a, config=None) -> dict:
    """The two classifiers on an action table (several matches stacked); config: target -> CANDIDATES name."""
    config = config or {t: "boost_300" for t in TARGETS}
    X = features(a)
    y = dict(zip(TARGETS, labels(a)))
    return {t: fit_classifier(config[t], X, y[t]) for t in TARGETS}


def predict(models, a) -> dict:
    X = features(a)
    ps = models["score"].predict_proba(X)[:, 1]
    pc = models["concede"].predict_proba(X)[:, 1]
    return {"p_score": ps, "p_concede": pc, **values(a, ps, pc)}


def _xy(tables: dict) -> dict:
    """match id -> (features, {target: labels}); features of one match equal its rows in a stacked table."""
    return {mid: (features(a), dict(zip(TARGETS, labels(a)))) for mid, a in tables.items()}


def select(tables: dict, xy=None, cache=None) -> tuple:
    """(target -> chosen CANDIDATES name, target -> name -> pooled inner log-loss) by leave-one-match-out over
    `tables` (the training matches only). `cache` shares the inner fits between outer folds (keyed by the matches
    trained on)."""
    xy = _xy(tables) if xy is None else xy
    cache = {} if cache is None else cache
    ids = list(tables)
    scores = {}
    for t in TARGETS:
        scores[t] = {}
        for name in CANDIDATES:
            ys, ps = [], []
            for mid in ids:
                train = tuple(sorted(o for o in ids if o != mid))
                key = (train, t, name)
                if key not in cache:
                    X = np.vstack([xy[o][0] for o in train])
                    y = np.concatenate([xy[o][1][t] for o in train])
                    cache[key] = fit_classifier(name, X, y) if 0 < y.sum() < len(y) else ML.fit_base_rate(X, y)
                ys.append(xy[mid][1][t])
                ps.append(cache[key].predict_proba(xy[mid][0])[:, 1])
            scores[t][name] = ML.metrics(np.concatenate(ys), np.concatenate(ps))["log_loss"]
    return {t: min(scores[t], key=scores[t].get) for t in TARGETS}, scores


def lomo(tables: dict, select_inner: bool = False) -> dict:
    """match id -> predict() on that match with the classifiers fitted on the other matches, plus "trained_on" and
    "config" (with select_inner, chosen by select() on the other matches only; else boost_300)."""
    out, xy, cache = {}, (_xy(tables) if select_inner else None), {}
    for mid in tables:
        others = [o for o in tables if o != mid]
        config, inner = None, None
        if select_inner:
            config, inner = select({o: tables[o] for o in others}, {o: xy[o] for o in others}, cache)
        config = config or {t: "boost_300" for t in TARGETS}
        models = fit(D.stack([tables[o] for o in others]), config)
        out[mid] = {**predict(models, tables[mid]), "trained_on": others, "config": config, "inner_log_loss": inner}
    return out

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
"""

import numpy as np

from . import datasets as D
from . import features as F
from . import geometry as G
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

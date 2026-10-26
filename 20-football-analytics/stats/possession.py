"""Ball control, possession with hysteresis, pass detection, pass matching and pass networks."""

import numpy as np

from .tracks import runs

DEAD = -2  # ball not visible / out of play
FREE = -1  # ball in flight or loose: nobody within the control radius


def ball_owner(tracks, r_control=1.5):
    """Per frame: index of the player controlling the ball (nearest one within `r_control` m), FREE or DEAD."""
    d = np.linalg.norm(tracks.xy - tracks.ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    idx = np.argmin(d, axis=1)
    dmin = d[np.arange(len(d)), idx]
    owner = np.where(dmin <= r_control, idx, FREE)
    owner[np.isnan(tracks.ball[:, 0])] = DEAD
    return owner


def possession(tracks, owner, min_frames=5):
    """Team in possession per frame (0/1); -1 before anyone has had the ball and while the ball is dead.

    Hysteresis: possession changes only after the other team has controlled the ball for `min_frames` control
    frames in a row (frames in flight neither count nor reset), so a ball passing near an opponent does not
    flip possession."""
    out = np.full(len(owner), -1)
    state, cand, count, per = -1, -1, 0, None
    for t, o in enumerate(owner):
        if tracks.period[t] != per:
            per, state, cand, count = tracks.period[t], -1, -1, 0
        if o == DEAD:
            continue
        if o >= 0:
            tm = tracks.team[o]
            count = count + 1 if tm == cand else 1
            cand = tm
            if tm != state and count >= min_frames:
                state = tm
        out[t] = state
    return out


def possession_share(poss):
    """Share of team 0 among the frames where some team has the ball."""
    p = poss[poss >= 0]
    return float(np.mean(p == 0)) if len(p) else float("nan")


def control_segments(owner, min_control=1):
    """Runs of the same controlling player lasting >= min_control frames, plus dead-ball runs (as breaks)."""
    segs = []
    for v, s, e in runs(owner):
        if v == DEAD:
            segs.append((DEAD, s, e))
        elif v >= 0 and e - s >= min_control:
            segs.append((int(v), s, e))
    return segs


def detect_passes(tracks, owner, min_control=1, max_flight_s=4.0, min_dist=3.0):
    """Passes = control by player A, then (after the ball is in flight, or directly) control by a team-mate B != A.

    Control runs shorter than `min_control` frames are ignored, so a ball flying past a defender does not cut a pass
    in two. Returns dicts with frame indices (start = last frame of A's control, end = first frame of B's control),
    player indices, team and ball positions."""
    out = []
    segs = control_segments(owner, min_control)
    for (a, _sa, ea), (b, sb, _eb) in zip(segs[:-1], segs[1:]):
        if a == DEAD or b == DEAD or a == b or tracks.team[a] != tracks.team[b]:
            continue
        start, end = ea - 1, sb
        if (end - start) / tracks.fps > max_flight_s or tracks.period[start] != tracks.period[end]:
            continue
        p0, p1 = tracks.ball[start], tracks.ball[end]
        if np.linalg.norm(p1 - p0) < min_dist:
            continue
        out.append(
            {
                "start": int(start),
                "end": int(end),
                "from": int(a),
                "to": int(b),
                "team": int(tracks.team[a]),
                "x0": float(p0[0]),
                "y0": float(p0[1]),
                "x1": float(p1[0]),
                "y1": float(p1[1]),
            }
        )
    return out


def match_passes(detected, truth, tol_frames=25, use_ids=True):
    """Greedy one-to-one matching on start frame (|dt| <= tol_frames), same team and, if `use_ids`, same passer and
    receiver. Both lists hold dicts with start, team, from, to. Returns precision, recall, F1 and counts."""
    used = np.zeros(len(detected), bool)
    starts = np.array([d["start"] for d in detected]) if detected else np.zeros(0)
    tp = 0
    for g in truth:
        best, best_dt = -1, tol_frames + 1
        for j in np.flatnonzero(np.abs(starts - g["start"]) <= tol_frames):
            d = detected[j]
            if used[j] or d["team"] != g["team"]:
                continue
            if use_ids and (d["from"] != g["from"] or d["to"] != g["to"]):
                continue
            dt = abs(d["start"] - g["start"])
            if dt < best_dt:
                best, best_dt = j, dt
        if best >= 0:
            used[best] = True
            tp += 1
    fp, fn = len(detected) - tp, len(truth) - tp
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "tp": tp, "fp": fp, "fn": fn}


def pass_network(tracks, passes, team, poss=None, min_minutes=1.0):
    """networkx DiGraph of one team's passes. Node attrs: `pos` (average position, while the team has the ball if
    `poss` is given), `name`, `n_passes`; edge attr `weight` = number of passes A -> B."""
    import networkx as nx

    g = nx.DiGraph()
    mask = (poss == team) if poss is not None else np.ones(tracks.n_frames, bool)
    for i in np.flatnonzero(tracks.team == team):
        pts = tracks.xy[mask, i]
        pts = pts[~np.isnan(pts[:, 0])]
        if len(pts) >= tracks.fps * 60 * min_minutes:
            g.add_node(int(i), pos=tuple(pts.mean(0)), name=tracks.names[i], n_passes=0)
    for p in passes:
        if p["team"] != team or p["from"] not in g or p["to"] not in g:
            continue
        g.nodes[p["from"]]["n_passes"] += 1
        w = g.edges[p["from"], p["to"]]["weight"] + 1 if g.has_edge(p["from"], p["to"]) else 1
        g.add_edge(p["from"], p["to"], weight=w)
    return g

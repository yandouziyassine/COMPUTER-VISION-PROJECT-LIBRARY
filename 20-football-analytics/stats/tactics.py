"""Team shape: playing direction, formation (Hungarian match to templates), compactness, defensive line, pressing."""

import numpy as np

# Template outfield positions as (depth, width): depth 0 = own goal line, 1 = opponent's; width 0..1 across.
_ROW = {
    1: [0.5],
    2: [0.35, 0.65],
    3: [0.25, 0.5, 0.75],
    4: [0.12, 0.37, 0.63, 0.88],
    5: [0.08, 0.29, 0.5, 0.71, 0.92],
}


def _lines(*lines):
    return np.array([(d, w) for d, n in lines for w in _ROW[n]], float)


FORMATIONS = {
    "4-4-2": _lines((0.2, 4), (0.5, 4), (0.78, 2)),
    "4-3-3": _lines((0.2, 4), (0.48, 3), (0.78, 3)),
    "4-2-3-1": _lines((0.2, 4), (0.4, 2), (0.6, 3), (0.8, 1)),
    "3-5-2": _lines((0.2, 3), (0.48, 5), (0.78, 2)),
    "5-3-2": _lines((0.2, 5), (0.45, 3), (0.75, 2)),
}


def _z(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-9)


def match_formation(avg, templates=FORMATIONS):
    """avg: (10, 2) average (depth, width) of the outfield players. Both shapes are standardised per axis (so the
    team's length and width don't matter), players are assigned to template slots with the Hungarian algorithm, and
    the template with the lowest mean squared distance wins. Returns (name, {name: cost})."""
    from scipy.optimize import linear_sum_assignment

    a = _z(np.asarray(avg, float))
    costs = {}
    for name, tpl in templates.items():
        c = ((a[:, None, :] - _z(tpl)[None, :, :]) ** 2).sum(-1)
        r, k = linear_sum_assignment(c)
        costs[name] = float(c[r, k].mean())
    return min(costs, key=costs.get), costs


def attack_sign(tracks, team):
    """+1 if `team` attacks towards x = L in this window (its players are, on average, left of the opponents)."""
    mx = [np.nanmean(tracks.xy[:, tracks.team == t, 0]) for t in (0, 1)]
    return 1 if mx[team] < mx[1 - team] else -1


def to_attack_frame(xy, sign, pitch):
    """Positions -> (depth from own goal line, width) in metres, for a team attacking with `sign`."""
    L, W = pitch
    out = np.array(xy, float)
    if sign < 0:
        out[..., 0] = L - out[..., 0]
        out[..., 1] = W - out[..., 1]
    return out


def outfield(tracks, team, sign=None, min_presence=0.5):
    """(outfield player indices, goalkeeper index) for one team. The goalkeeper is the player deepest on average;
    outfield = the 10 other players most present (each present in >= min_presence of the frames)."""
    sign = attack_sign(tracks, team) if sign is None else sign
    idx = np.flatnonzero(tracks.team == team)
    pres = (~np.isnan(tracks.xy[:, idx, 0])).mean(0)
    idx, pres = idx[pres >= min_presence], pres[pres >= min_presence]
    if len(idx) == 0:
        return idx, -1
    depth = np.nanmean(to_attack_frame(tracks.xy[:, idx], sign, tracks.pitch)[..., 0], axis=0)
    gk = idx[np.argmin(depth)]
    keep = idx != gk
    of, pres = idx[keep], pres[keep]
    return of[np.argsort(-pres, kind="stable")][:10], int(gk)


def formation(tracks, team):
    """Formation of `team` over the window: (name, costs), or (None, {}) if < 10 outfield players were tracked."""
    sign = attack_sign(tracks, team)
    of, _ = outfield(tracks, team, sign)
    if len(of) < 10:
        return None, {}
    depth_width = np.nanmean(to_attack_frame(tracks.xy[:, of], sign, tracks.pitch), axis=0)
    return match_formation(depth_width)


def hull_area(pts):
    """Area of the convex hull of 2-D points (monotone chain + shoelace formula). Fewer than 3 points -> 0."""
    pts = np.unique(np.asarray(pts, float), axis=0)
    if len(pts) < 3:
        return 0.0

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(points):
        h = []
        for p in points:
            while len(h) >= 2 and cross(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h

    hull = np.array(half(pts)[:-1] + half(pts[::-1])[:-1])
    x, y = hull[:, 0], hull[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def shape_stats(tracks, team, step=5):
    """Mean compactness (convex-hull area of the outfield players, m^2) and defensive-line height (distance of the
    deepest outfield player from the own goal line, m), sampled every `step` frames."""
    sign = attack_sign(tracks, team)
    of, _ = outfield(tracks, team, sign)
    area, line = [], []
    for t in range(0, tracks.n_frames, step):
        p = tracks.xy[t, of]
        p = p[~np.isnan(p[:, 0])]
        if len(p) < 8:  # need most of the team to describe its shape
            continue
        area.append(hull_area(p))
        line.append(to_attack_frame(p, sign, tracks.pitch)[:, 0].min())
    return {
        "compactness_m2": float(np.mean(area)) if area else float("nan"),
        "defensive_line_m": float(np.mean(line)) if line else float("nan"),
    }


def in_build_up_zone(x, pressing_sign, pitch, share=0.6):
    """True where x lies in the opponent's own `share` of the pitch, seen from the pressing team (which attacks
    with `pressing_sign`)."""
    depth = np.asarray(x) if pressing_sign > 0 else pitch[0] - np.asarray(x)
    return depth >= (1 - share) * pitch[0]


def ppda_counts(tracks, passes, poss, team):
    """PPDA-style pressing counts with `team` pressing: opponent passes started in the opponent's own 60 % and
    possession regains by `team` in that zone (from the hysteresis possession). PPDA = passes / regains."""
    sign = attack_sign(tracks, team)
    opp_passes = sum(1 for p in passes if p["team"] != team and in_build_up_zone(p["x0"], sign, tracks.pitch))
    regains = 0
    for t in np.flatnonzero((poss[1:] == team) & (poss[:-1] == 1 - team)) + 1:
        if not np.isnan(tracks.ball[t, 0]) and in_build_up_zone(tracks.ball[t, 0], sign, tracks.pitch):
            regains += 1
    return opp_passes, regains

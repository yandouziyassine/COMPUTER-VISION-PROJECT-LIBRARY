"""Attach each event to a half and a frame index, and measure how well event locations match the ball.

Event coordinates are 0..105 x 0..68; tracking is centred. The two Y conventions (as is / flipped) are compared on
the unsnapped event-to-ball distances and the better one is kept (stored in m.meta["event_axis"]).

Time mapping (checked on all 7 matches): the timezone handling is right (UTC vs +02:00, offset-aware parsing), but the
event timestamps are hand-made: each half has its own clock offset against the tracking (median -1.6 s .. +1.4 s)
plus ~1 s of jitter. So per half the median shift is estimated (best frame within +-SEARCH_S of each event), then each
event is snapped within +-SNAP_S around its shifted time to the frame where the ball is closest to the event location,
among frames where the acting player (passer / shooter) is within RELEASE_M of the ball: the release, never the later
interception. The provider's CalculatedFrame is used when it exists and agrees with the event location. Events that
cannot be placed get frame None and a reason in e["drop"]: no_half, no_position, no_release, far (more than MAX_DIST_M
from the ball), window_edge (best frame is at the edge of the window, i.e. the true release is probably outside).
"""

import numpy as np

from .match import FPS, frame_at

AXES = {"same_y": lambda x, y: (x - 52.5, y - 34.0), "flip_y": lambda x, y: (x - 52.5, 34.0 - y)}
SEARCH_S, SNAP_S, RELEASE_M, MAX_DIST_M = 4.0, 1.5, 2.0, 3.0


def _half_of(m, t):
    for i, h in enumerate(m.halves):
        if h.t[0] - 1 <= t <= h.t[-1] + 1:
            return i
    return None


def _is_index(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _ball_dist(h, lo, hi, loc):
    return np.hypot(h.ball[lo:hi, 0] - loc[0], h.ball[lo:hi, 1] - loc[1])


def _pct(a, q):
    return float(np.nanpercentile(a, q)) if len(a) else float("nan")


def attach_frames(m) -> dict:
    n_players = len(m.player_ids)
    search, radius = int(round(SEARCH_S * FPS)), int(round(SNAP_S * FPS))
    # 1. half and nominal frame
    located = []
    for e in m.events:
        e["half"], e["drop"], e["sync"] = _half_of(m, e["t"]), None, None
        if e["half"] is None:
            e["frame"], e["drop"] = None, "no_half"
            continue
        e["frame"] = frame_at(m.halves[e["half"]], e["t"])
        e["nominal"] = e["frame"]
        if not np.isfinite(e["x"]):
            e["frame"], e["drop"] = None, "no_position"
        elif np.all(np.isfinite(m.halves[e["half"]].ball[e["nominal"], :2])):
            located.append(e)
    # 2. axis convention on the unsnapped distances
    nominal = {}
    for name, f in AXES.items():
        nominal[name] = [
            float(np.hypot(*(np.array(f(e["x"], e["y"])) - m.halves[e["half"]].ball[e["nominal"], :2])))
            for e in located
        ]
    axis = min(nominal, key=lambda k: np.median(nominal[k]) if nominal[k] else float("inf"))
    m.meta["event_axis"] = axis
    loc = lambda e: AXES[axis](e["x"], e["y"])
    # 3. clock offset per half: median of the best-frame shift of the events that find the ball nearby
    shift = []
    for hi, h in enumerate(m.halves):
        found = []
        for e in located:
            if e["half"] != hi:
                continue
            lo, up = max(0, e["nominal"] - search), min(len(h.t), e["nominal"] + search + 1)
            d = _ball_dist(h, lo, up, loc(e))
            if np.isfinite(d).any() and np.nanmin(d) <= MAX_DIST_M:
                found.append(lo + int(np.nanargmin(d)) - e["nominal"])
        shift.append(int(np.median(found)) if found else 0)
    # 4. snap
    for e in m.events:
        if e["half"] is None or e["drop"] == "no_position":
            continue
        hi = e["half"]
        h = m.halves[hi]
        e["t_sync"] = e["t"] + shift[hi] / FPS  # used when the event cannot be placed
        where = loc(e)
        calc = e.get("calc_frame")
        if calc is not None:
            i = int(np.searchsorted(h.n, calc))
            if i < len(h.n) and h.n[i] == calc:
                d = _ball_dist(h, i, i + 1, where)[0]
                if np.isfinite(d) and d <= MAX_DIST_M:
                    e["frame"], e["sync"], e["dist"] = i, "calc", float(d)
                    continue
        centre = e["nominal"] + shift[hi]
        lo_u, hi_u = centre - radius, centre + radius
        lo, up = max(0, lo_u), min(len(h.t), hi_u + 1)
        if lo >= up:
            e["frame"], e["drop"] = None, "no_release"
            continue
        d = _ball_dist(h, lo, up, where)
        ok = np.isfinite(d)
        actor = e.get("player")
        if _is_index(actor) and 0 <= actor < n_players:
            ok &= np.linalg.norm(h.xy[lo:up, actor] - h.ball[lo:up, :2], axis=1) <= RELEASE_M
        if not ok.any():
            e["frame"], e["drop"] = None, "no_release"
            continue
        k = int(np.argmin(np.where(ok, d, np.inf)))
        f = lo + k
        if d[k] > MAX_DIST_M:
            e["frame"], e["drop"] = None, "far"
        elif f - lo_u <= 1 or hi_u - f <= 1:
            e["frame"], e["drop"] = None, "window_edge"
        else:
            e["frame"], e["sync"], e["dist"] = f, "snap", float(d[k])
    # 5. report
    kept = [e for e in m.events if e.get("frame") is not None]
    for e in kept:
        e["sync_shift_s"] = (e["frame"] - e["nominal"]) / FPS
        e["t_sync"] = float(m.halves[e["half"]].t[e["frame"]])
    reasons = {}
    for e in m.events:
        if e.get("frame") is None:
            reasons[e["drop"]] = reasons.get(e["drop"], 0) + 1
    d = [e["dist"] for e in kept]
    passes = [e for e in m.events if e["type"] == "pass"]
    shots = [e for e in m.events if e["type"] == "shot"]
    return {
        "axis": axis,
        "nominal_median_dist_m": _pct(nominal[axis], 50),
        "nominal_p90_dist_m": _pct(nominal[axis], 90),
        "median_dist_m": _pct(d, 50),
        "p90_dist_m": _pct(d, 90),
        "shift_s": [s / FPS for s in shift],
        "events": len(m.events),
        "kept": len(kept),
        "dropped": reasons,
        "passes": len(passes),
        "passes_kept": sum(e.get("frame") is not None for e in passes),
        "shots": len(shots),
        "shots_kept": sum(e.get("frame") is not None for e in shots),
        "calc_frames_used": sum(e.get("sync") == "calc" for e in kept),
    }


def event_xy(m, e):
    """Event location in centred tracking coordinates, using the convention chosen by attach_frames."""
    return np.array(AXES[m.meta.get("event_axis", "same_y")](e["x"], e["y"]))

"""Attach each event to a half and a frame index, and measure how well event locations match the ball.

Event coordinates are 0..105 x 0..68; tracking is centred. Two conventions are tried (Y as is, or flipped) and the
one with the smaller median event-to-ball distance is kept; the residuals are the sync-quality numbers in the README.
"""

import numpy as np

from .match import frame_at

AXES = {"same_y": lambda x, y: (x - 52.5, y - 34.0), "flip_y": lambda x, y: (x - 52.5, 34.0 - y)}


def _half_of(m, t):
    for i, h in enumerate(m.halves):
        if h.t[0] - 1 <= t <= h.t[-1] + 1:
            return i
    return None


def attach_frames(m) -> dict:
    located = []
    for e in m.events:
        e["half"] = _half_of(m, e["t"])
        if e["half"] is None:
            e["frame"] = None
            continue
        h = m.halves[e["half"]]
        e["frame"] = frame_at(h, e["t"])
        if np.isfinite(e["x"]) and np.all(np.isfinite(h.ball[e["frame"], :2])):
            located.append(e)
    best = None
    for name, f in AXES.items():
        d = [np.hypot(*(np.array(f(e["x"], e["y"])) - m.halves[e["half"]].ball[e["frame"], :2])) for e in located]
        med = float(np.median(d)) if d else float("nan")
        if best is None or med < best[1]:
            best = (name, med, d)
    name, med, d = best
    m.meta["event_axis"] = name
    return {
        "axis": name,
        "median_dist_m": med,
        "p90_dist_m": float(np.percentile(d, 90)) if d else float("nan"),
        "events": len(m.events),
        "located": len(located),
    }


def event_xy(m, e):
    """Event location in centred tracking coordinates, using the convention chosen by attach_frames."""
    return np.array(AXES[m.meta.get("event_axis", "same_y")](e["x"], e["y"]))

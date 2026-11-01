"""Attach each event to a half and a frame index, and measure how well event locations match the ball.

Event coordinates are 0..105 x 0..68; tracking is centred. Two conventions are tried (Y as is, or flipped) and the
one with the smaller median event-to-ball distance is kept; the residuals are the sync-quality numbers in the README.

Time mapping (checked on J03WOH): the timezone handling is right (UTC vs +02:00 offset-aware), but the event
timestamps are hand-made and lead the tracking by about 1.2 s on average with ~1 s of jitter, so the ball is not yet
(or no longer) at the event location at the nominal frame (median distance 7.7 m). Each event is therefore snapped to
the frame, within +-SNAP_S seconds of its nominal time, where the ball is closest to the event location. After the
snap the passer is within 2 m of the ball for ~3/4 of the passes (39 % at the nominal time). Both the nominal and the
snapped residuals are reported.
"""

import numpy as np

from .match import frame_at

AXES = {"same_y": lambda x, y: (x - 52.5, y - 34.0), "flip_y": lambda x, y: (x - 52.5, 34.0 - y)}
SNAP_S = 3.0


def _half_of(m, t):
    for i, h in enumerate(m.halves):
        if h.t[0] - 1 <= t <= h.t[-1] + 1:
            return i
    return None


def _snap(h, nominal, loc, radius):
    """Frame within +-radius frames of nominal whose ball position is closest to loc; returns (frame, distance)."""
    lo, hi = max(0, nominal - radius), min(len(h.t), nominal + radius + 1)
    d = np.hypot(h.ball[lo:hi, 0] - loc[0], h.ball[lo:hi, 1] - loc[1])
    if not np.isfinite(d).any():
        return nominal, float("nan")
    k = int(np.nanargmin(d))
    return lo + k, float(d[k])


def attach_frames(m) -> dict:
    radius = int(round(SNAP_S * 25))
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
        snapped = []
        for e in located:
            h = m.halves[e["half"]]
            snapped.append(_snap(h, e["frame"], f(e["x"], e["y"]), radius))
        d = [s[1] for s in snapped]
        med = float(np.nanmedian(d)) if d else float("nan")
        if best is None or med < best[1]:
            best = (name, med, snapped)
    name, med, snapped = best
    nominal = [
        float(np.hypot(*(np.array(AXES[name](e["x"], e["y"])) - m.halves[e["half"]].ball[e["frame"], :2])))
        for e in located
    ]
    shift = []
    for e, (frame, _) in zip(located, snapped):
        shift.append((frame - e["frame"]) / 25.0)
        e["sync_shift_s"] = shift[-1]
        e["frame"] = frame
    m.meta["event_axis"] = name
    d = [s[1] for s in snapped]
    pct = lambda a, q: float(np.nanpercentile(a, q)) if len(a) else float("nan")
    return {
        "axis": name,
        "median_dist_m": med,
        "p90_dist_m": pct(d, 90),
        "nominal_median_dist_m": pct(nominal, 50),
        "nominal_p90_dist_m": pct(nominal, 90),
        "median_shift_s": pct(shift, 50),
        "p90_abs_shift_s": pct(np.abs(shift), 90),
        "snap_s": SNAP_S,
        "events": len(m.events),
        "located": len(located),
    }


def event_xy(m, e):
    """Event location in centred tracking coordinates, using the convention chosen by attach_frames."""
    return np.array(AXES[m.meta.get("event_axis", "same_y")](e["x"], e["y"]))

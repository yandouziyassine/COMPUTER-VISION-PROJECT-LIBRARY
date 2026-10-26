"""Physical statistics: distance, speed (Savitzky-Golay), sprints."""

import math

import numpy as np

from .tracks import runs

MAX_SPEED = 12.0  # m/s; faster than any footballer, so higher values are tracking errors
SPRINT_SPEED = 7.0  # m/s (25.2 km/h)
SPRINT_MIN_S = 1.0


def savgol_coeffs(window, poly, deriv=0):
    """Savitzky-Golay filter coefficients (same as scipy.signal.savgol_coeffs with use='dot'), NumPy only."""
    h = window // 2
    t = np.arange(-h, h + 1, dtype=float)
    a = np.vander(t, poly + 1, increasing=True)
    return np.linalg.pinv(a)[deriv] * math.factorial(deriv)


def _odd_pad(x, h):
    """Point-symmetric extension around the end samples: keeps the slope at the edges (no fake stop)."""
    return np.concatenate([2 * x[0] - x[h:0:-1], x, 2 * x[-1] - x[-2 : -h - 2 : -1]])


def fill_gaps(a, max_gap):
    """Linearly interpolate NaN runs of at most `max_gap` frames that have valid data on both sides. a: (T,) or (T, D)."""
    a = np.array(a, float)
    two_d = a.ndim == 2
    a2 = a if two_d else a[:, None]
    missing = np.isnan(a2[:, 0])
    for v, s, e in runs(missing):
        if v and s > 0 and e < len(a2) and e - s <= max_gap:
            w = np.arange(1, e - s + 1)[:, None] / (e - s + 1)
            a2[s:e] = a2[s - 1] + w * (a2[e] - a2[s - 1])
    return a2 if two_d else a2[:, 0]


def _filter_segments(pos, w, coeffs):
    """Apply a Savitzky-Golay kernel to every NaN-free run of `pos` (T, 2); runs shorter than half the window are
    returned as None so the caller can fall back to something simpler."""
    h = w // 2
    k = coeffs[::-1]
    out = np.full_like(pos, np.nan)
    short = []
    for valid, s, e in runs(~np.isnan(pos[:, 0])):
        if not valid:
            continue
        if e - s <= h:
            short.append((s, e))
            continue
        for d in range(2):
            out[s:e, d] = np.convolve(_odd_pad(pos[s:e, d], h), k, "valid")
    return out, short


def _window(window_s, fps):
    return max(3, int(round(window_s * fps)) | 1)


def velocity(xy, fps, window_s=1.0, poly=2, max_gap_s=1.0, vmax=MAX_SPEED):
    """Smoothed velocity of one track. xy: (T, 2) metres. Returns (velocity (T, 2), speed (T,)), NaN where unknown.

    Gaps up to `max_gap_s` are interpolated; each continuous piece is differentiated with a Savitzky-Golay filter
    (window `window_s`, polynomial order `poly`); speeds are capped at `vmax`. Call it once per half (a track jumps
    when the teams change ends)."""
    pos = fill_gaps(xy, int(round(max_gap_s * fps)))
    w = _window(window_s, fps)
    vel, short = _filter_segments(pos, w, savgol_coeffs(w, poly, 1))
    vel *= fps
    for s, e in short:
        if e - s >= 2:
            vel[s:e] = np.gradient(pos[s:e], axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    over = speed > vmax
    vel[over] *= (vmax / speed[over])[:, None]
    speed = np.minimum(speed, vmax)
    return vel, speed


def smooth_positions(xy, fps, window_s=1.0, poly=2):
    """Savitzky-Golay smoothed positions of one track (short pieces are left as they are)."""
    w = _window(window_s, fps)
    out, short = _filter_segments(np.asarray(xy, float), w, savgol_coeffs(w, poly, 0))
    for s, e in short:
        out[s:e] = xy[s:e]
    return out


def smooth_tracks(tracks, window_s=1.0):
    """Copy of `tracks` with every player and the ball smoothed, half by half."""
    t = tracks.copy()
    for per in np.unique(t.period):
        m = t.period == per
        for i in range(t.xy.shape[1]):
            t.xy[m, i] = smooth_positions(t.xy[m, i], t.fps, window_s)
        t.ball[m] = smooth_positions(t.ball[m], t.fps, window_s)
    return t


def sprints(speed, fps, threshold=SPRINT_SPEED, min_s=SPRINT_MIN_S):
    """Sprints = runs of at least `min_s` seconds above `threshold` m/s. Returns (count, distance in metres)."""
    fast = np.nan_to_num(speed, nan=0.0) > threshold
    n, dist = 0, 0.0
    for v, s, e in runs(fast):
        if v and e - s >= math.ceil(min_s * fps - 1e-9):
            n += 1
            dist += float(np.sum(speed[s:e])) / fps
    return n, dist


def physical_stats(tracks, window_s=1.0):
    """Per-player table: minutes tracked, distance (m), top speed (m/s), sprints, sprint distance."""
    rows = []
    halves = [tracks.period == p for p in np.unique(tracks.period)]
    for i in range(tracks.xy.shape[1]):
        sp = np.full(tracks.n_frames, np.nan)
        for m in halves:  # positions jump at half-time (teams change ends): differentiate each half on its own
            sp[m] = velocity(tracks.xy[m, i], tracks.fps, window_s=window_s)[1]
        valid = ~np.isnan(sp)
        if valid.sum() < tracks.fps:  # less than a second on the pitch
            continue
        n, sd = sprints(sp, tracks.fps)
        rows.append(
            {
                "player": tracks.names[i],
                "team": int(tracks.team[i]),
                "minutes": valid.sum() / tracks.fps / 60,
                "distance_m": float(np.nansum(sp)) / tracks.fps,
                "max_speed": float(np.nanmax(sp)),
                "sprints": n,
                "sprint_m": sd,
            }
        )
    return rows


def team_physical(rows, team):
    """Team summary from `physical_stats` rows: distance per 90 min of one player, mean top speed, sprints."""
    r = [x for x in rows if x["team"] == team]
    minutes = sum(x["minutes"] for x in r)
    return {
        "distance_per90_m": sum(x["distance_m"] for x in r) / max(minutes, 1e-9) * 90,
        "mean_top_speed": float(np.mean([x["max_speed"] for x in r])) if r else float("nan"),
        "sprints": sum(x["sprints"] for x in r),
        "player_minutes": minutes,
    }

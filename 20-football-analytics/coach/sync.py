"""Attach each event to a half and a frame index, and measure how well the events match the tracking.

Event coordinates are 0..105 x 0..68; tracking is centred. The two Y conventions (as is / flipped) are compared on
the unsnapped event-to-ball distances and the better one is kept (stored in m.meta["event_axis"]).

The event timestamps are hand-made: each half has its own clock offset against the tracking (-1.6 s .. +2.1 s) plus
about a second of jitter per event, and the logged spot is a few metres off (median 2.5 m from the passer). Two
methods place an event (pass or shot) on a frame:

* "release" (default, coach v2). The tracking says when a player releases the ball: a *release* of player p is the
  last frame of a run where p is within TOUCH_M of the ball, after which the ball gets more than LEFT_M away from p
  within LEFT_S (the kick). Per half, the clock offset is the peak of the distribution of (release time - event time)
  over all events' actors; per event a local offset (median of the NEIGHBOURS nearest events' matched offsets) absorbs
  the slow drift. Each event then takes the release of its actor within +-WINDOW_S of its locally shifted time that
  minimises (dt / SIGMA_T)^2 + (min(spot error, 15) / SIGMA_D)^2 minus a bonus when what follows agrees with the event:
  the recorded recipient is the next player to touch the ball (completed pass), an opponent touches it or the ball
  goes dead (failed pass). A completed pass whose recipient never comes within RECEIVE_M of the ball within FLIGHT_S
  is dropped (no_reception). On the 7 Bundesliga matches it keeps 91 % of the passes (69 % with "legacy"), the chosen
  frame is the kick (ball speed +7 m/s across it, +2 m/s for "legacy") and on shots it agrees with the provider's
  CalculatedFrame within 0.2 s for 94 % of them (sync_eval.py).
* "legacy" (coach v1). Per half the median shift that brings the ball nearest to the logged spot (within +-SEARCH_S),
  then a snap within +-SNAP_S to the frame where the ball is nearest the spot while the actor is within RELEASE_M of
  it. It trusts the hand-logged spot: a pass logged 3 m off or 2 s late is dropped (far / window_edge / no_release).

Both use the provider's CalculatedFrame for shots when it exists and agrees with the event location. Events that
cannot be placed get frame None and a reason in e["drop"]: no_half, no_position, no_release, far, window_edge
(legacy), no_reception, duplicate (release: two events on one release, the worse one loses it).
"""

import numpy as np

from .match import FPS, frame_at

AXES = {"same_y": lambda x, y: (x - 52.5, y - 34.0), "flip_y": lambda x, y: (x - 52.5, 34.0 - y)}
# legacy
SEARCH_S, SNAP_S, RELEASE_M, MAX_DIST_M = 4.0, 1.5, 2.0, 3.0
# release
TOUCH_M, LEFT_M, LEFT_S, GAP_FRAMES = 1.5, 3.0, 0.8, 2
OFFSET_RANGE_S, OFFSET_BW_S, WINDOW_S, NEIGHBOURS = 5.0, 0.25, 3.0, 10
SIGMA_T, SIGMA_D, SPOT_CAP_M, FAR_M = 0.8, 5.0, 15.0, 15.0
FLIGHT_S, RECEIVE_M = 6.0, 2.0
BONUS = {"recipient": 4.0, "opponent": 2.0, "dead": 1.0}
METHODS = ("release", "legacy")
SYNC_FIELDS = ("frame", "drop", "sync", "nominal", "t_sync", "dist", "sync_shift_s", "reception", "follow")


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
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _fps(m) -> float:
    return float(m.meta.get("fps", FPS)) if getattr(m, "meta", None) else FPS


def _actor(m, e):
    a = e.get("player")
    return int(a) if _is_index(a) and 0 <= a < len(m.player_ids) else None


def attach_frames(m, method: str = "release", use_calc: bool = True) -> dict:
    """Place every event of m on a frame (in place: e["half"], e["frame"], e["drop"], ...) and return the audit.
    method: "release" (default) or "legacy"; use_calc=False ignores the provider's CalculatedFrame (used by
    sync_eval to check the release detector against it)."""
    assert method in METHODS, method
    for e in m.events:  # re-syncing a saved match: forget the previous placement
        for k in SYNC_FIELDS:
            e.pop(k, None)
    located = _locate(m)
    nominal, axis = _choose_axis(m, located)
    if method == "legacy":
        shift = _attach_legacy(m, axis, located, use_calc)
    else:
        shift = _attach_release(m, axis, use_calc)
    kept = [e for e in m.events if e.get("frame") is not None]
    for e in kept:
        e["sync_shift_s"] = (e["frame"] - e["nominal"]) / _fps(m)
        e["t_sync"] = float(m.halves[e["half"]].t[e["frame"]])
    reasons = {}
    for e in m.events:
        if e.get("frame") is None:
            reasons[e["drop"]] = reasons.get(e["drop"], 0) + 1
    d = [e["dist"] for e in kept]
    passes = [e for e in m.events if e["type"] == "pass"]
    shots = [e for e in m.events if e["type"] == "shot"]
    return {
        "method": method,
        "axis": axis,
        "nominal_median_dist_m": _pct(nominal[axis], 50),
        "nominal_p90_dist_m": _pct(nominal[axis], 90),
        "median_dist_m": _pct(d, 50),
        "p90_dist_m": _pct(d, 90),
        "shift_s": [round(s, 3) for s in shift],
        "events": len(m.events),
        "kept": len(kept),
        "dropped": reasons,
        "passes": len(passes),
        "passes_kept": sum(e.get("frame") is not None for e in passes),
        "shots": len(shots),
        "shots_kept": sum(e.get("frame") is not None for e in shots),
        "calc_frames_used": sum(e.get("sync") == "calc" for e in kept),
        "quality": quality(m),
    }


def _locate(m) -> list:
    """Half and nominal frame of every event; returns the events whose nominal frame has a ball position."""
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
    return located


def _choose_axis(m, located) -> tuple:
    nominal = {}
    for name, f in AXES.items():
        nominal[name] = [
            float(np.hypot(*(np.array(f(e["x"], e["y"])) - m.halves[e["half"]].ball[e["nominal"], :2])))
            for e in located
        ]
    axis = min(nominal, key=lambda k: np.median(nominal[k]) if nominal[k] else float("inf"))
    m.meta["event_axis"] = axis
    return nominal, axis


def _try_calc(h, e, where) -> bool:
    """Use the provider's CalculatedFrame when it exists and the ball is within MAX_DIST_M of the event there."""
    calc = e.get("calc_frame")
    if calc is None:
        return False
    i = int(np.searchsorted(h.n, calc))
    if i < len(h.n) and h.n[i] == calc:
        d = _ball_dist(h, i, i + 1, where)[0]
        if np.isfinite(d) and d <= MAX_DIST_M:
            e["frame"], e["sync"], e["dist"] = i, "calc", float(d)
            return True
    return False


# ------------------------------------------------------------------------------------------------------- legacy


def _attach_legacy(m, axis, located, use_calc) -> list:
    n_players = len(m.player_ids)
    fps = _fps(m)
    search, radius = int(round(SEARCH_S * fps)), int(round(SNAP_S * fps))
    loc = lambda e: AXES[axis](e["x"], e["y"])
    # clock offset per half: median of the best-frame shift of the events that find the ball nearby
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
    for e in m.events:
        if e["half"] is None or e["drop"] == "no_position":
            continue
        hi = e["half"]
        h = m.halves[hi]
        e["t_sync"] = e["t"] + shift[hi] / fps  # used when the event cannot be placed
        where = loc(e)
        if use_calc and _try_calc(h, e, where):
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
    return [s / fps for s in shift]


# ------------------------------------------------------------------------------------------------------ release


def ball_distances(h) -> np.ndarray:
    """[F, P] distance of every player to the ball (inf when either is missing)."""
    d = np.linalg.norm(h.xy - h.ball[:, None, :2], axis=2)
    return np.where(np.isfinite(d), d, np.inf)


def releases(d: np.ndarray, p: int, fps: float = FPS) -> np.ndarray:
    """Release frames of player p: the last frame of each run within TOUCH_M of the ball (runs separated by at most
    GAP_FRAMES frames are merged) after which the ball gets more than LEFT_M away from p within LEFT_S."""
    idx = np.flatnonzero(d[:, p] <= TOUCH_M)
    if not len(idx):
        return np.array([], int)
    ends = idx[np.append(np.diff(idx) > GAP_FRAMES, True)]
    k = max(1, int(round(LEFT_S * fps)))
    out = []
    for f in ends:
        seg = d[f + 1 : f + 1 + k, p]
        seg = seg[np.isfinite(seg)]
        if len(seg) and seg.max() > LEFT_M:
            out.append(int(f))
    return np.array(out, int)


def next_touch(d: np.ndarray, f: int, p: int, horizon: int) -> tuple:
    """(player, frame) of the first touch (within TOUCH_M) after the ball has left p at frame f, within `horizon`
    frames of f; p himself counts (a dribble or a one-two). (None, None) when nobody touches it."""
    n = len(d)
    k = f + 1
    while k < n and d[k, p] <= TOUCH_M:
        k += 1
    seg = d[k : min(n, f + horizon)]
    if not len(seg):
        return None, None
    hit = np.flatnonzero((seg <= TOUCH_M).any(axis=1))
    if not len(hit):
        return None, None
    j = int(hit[0])
    return int(np.argmin(seg[j])), k + j


def _follow(m, h, d, e, f, horizon) -> tuple:
    """What follows a pass released at f: (tag, reception frame). tag "recipient" (completed: the recorded recipient
    touches next), "opponent" (failed: an opponent touches next), "dead" (failed: nobody touches it or the ball goes
    out of play), None otherwise."""
    if e["type"] != "pass":
        return None, None
    who, at = next_touch(d, f, e["player"], horizon)
    if e.get("success"):
        return ("recipient", at) if who is not None and who == e.get("recipient") else (None, None)
    if who is not None and m.team[who] == 1 - e["team"]:
        return "opponent", at
    if who is None or not h.alive[min(len(h.t) - 1, f + horizon // 2)]:
        return "dead", None
    return None, None


def _clock_offset(dts) -> float:
    """Peak of a Gaussian kernel density of the candidate offsets (s)."""
    if not len(dts):
        return 0.0
    dts = np.asarray(dts, float)
    grid = np.arange(-OFFSET_RANGE_S, OFFSET_RANGE_S + 1e-9, 0.04)
    dens = np.exp(-0.5 * ((dts[None, :] - grid[:, None]) / OFFSET_BW_S) ** 2).sum(axis=1)
    return float(grid[int(np.argmax(dens))])


def _attach_release(m, axis, use_calc) -> list:
    fps = _fps(m)
    loc = lambda e: np.array(AXES[axis](e["x"], e["y"]))
    horizon, window = int(round(FLIGHT_S * fps)), WINDOW_S * fps
    shifts = []
    for hi, h in enumerate(m.halves):
        evs = [e for e in m.events if e.get("half") == hi and e["drop"] != "no_position"]
        d = ball_distances(h)
        rel = {}
        for e in evs:
            a = _actor(m, e)
            if a is not None and a not in rel:
                rel[a] = releases(d, a, fps)
        cand = {id(e): rel[_actor(m, e)] if _actor(m, e) is not None else np.array([], int) for e in evs}
        # 1. clock offset of the half
        dts = []
        for e in evs:
            c = cand[id(e)]
            dts.extend(((c[np.abs(c - e["nominal"]) <= OFFSET_RANGE_S * fps] - e["nominal"]) / fps).tolist())
        g0 = _clock_offset(dts)
        shifts.append(g0)
        # 2. local offset: matched offsets of the neighbouring events (nearest release to the half's offset)
        first_t, first_dt = [], []
        for e in evs:
            c = cand[id(e)]
            w = c[np.abs(c - e["nominal"] - g0 * fps) <= window]
            if len(w):
                first_t.append(e["t"])
                first_dt.append((w[np.argmin(np.abs(w - e["nominal"] - g0 * fps))] - e["nominal"]) / fps)
        order = np.argsort(first_t)
        first_t, first_dt = np.asarray(first_t)[order], np.asarray(first_dt)[order]
        # 3. choose a release per event
        chosen = {}  # frame -> (cost, event)
        ranked = {}
        for e in evs:
            j = int(np.searchsorted(first_t, e["t"]))
            lo, up = max(0, j - NEIGHBOURS), min(len(first_t), j + NEIGHBOURS)
            local = float(np.median(first_dt[lo:up])) if up > lo else g0
            e["t_sync"] = e["t"] + local  # used when the event cannot be placed
            where = loc(e)
            if use_calc and _try_calc(h, e, where):
                continue
            a = _actor(m, e)
            if a is None:  # no tracked actor: nothing ties the event to a release
                e["frame"], e["drop"] = None, "no_release"
                continue
            centre = e["nominal"] + local * fps
            c = cand[id(e)]
            c = c[np.abs(c - centre) <= window]
            if not len(c):
                e["frame"], e["drop"] = None, "no_release"
                continue
            options, reasons = [], set()
            for f in c:
                f = int(f)
                spot = float(np.hypot(*(h.ball[f, :2] - where)))
                if spot > FAR_M:
                    reasons.add("far")
                    continue
                if not _received(e, d, f, horizon):
                    reasons.add("no_reception")
                    continue
                tag, at = _follow(m, h, d, e, f, horizon)
                cost = ((f - centre) / fps / SIGMA_T) ** 2 + (min(spot, SPOT_CAP_M) / SIGMA_D) ** 2
                options.append((cost - BONUS.get(tag, 0.0), f, spot, tag, at))
            if not options:
                e["frame"] = None
                e["drop"] = "no_reception" if "no_reception" in reasons else "far"
                continue
            options.sort()
            ranked[id(e)] = options
            _place(e, chosen, ranked)
    return shifts


def _received(e, d, f, horizon) -> bool:
    """A completed pass released at f must reach its recipient: within RECEIVE_M of the ball within FLIGHT_S."""
    r = e.get("recipient")
    if e["type"] != "pass" or not e.get("success") or not _is_index(r) or not 0 <= r < d.shape[1]:
        return True
    return bool(np.any(d[f : f + horizon, r] <= RECEIVE_M))


def _place(e, chosen, ranked):
    """Give e its best remaining release; a release already taken goes to the cheaper event (the other retries)."""
    options = ranked[id(e)]
    while options:
        cost, f, spot, tag, at = options.pop(0)
        if f in chosen and chosen[f][0] <= cost:
            continue
        other = chosen[f][1] if f in chosen else None
        chosen[f] = (cost, e)
        e["frame"], e["drop"], e["sync"], e["dist"], e["follow"] = f, None, "release", spot, tag
        if at is not None:
            e["reception"] = int(at)
        if other is not None:
            other["frame"], other["drop"], other["sync"] = None, "duplicate", None
            other.pop("reception", None)
            _place(other, chosen, ranked)
        return
    e["frame"], e["drop"] = None, "duplicate"


# ------------------------------------------------------------------------------------------------------ quality


def quality(m) -> dict:
    """Independent checks of the placed passes (none of them is the quantity a method optimises, except the spot):
    ball speed jump across the frame (the kick), passer-ball distance there, the logged spot's distance, completed
    passes whose recipient touches the ball next / comes within RECEIVE_M within FLIGHT_S, failed passes followed by an
    opponent's touch or a dead ball."""
    fps = _fps(m)
    horizon = int(round(FLIGHT_S * fps))
    dmat = [ball_distances(h) for h in m.halves]
    passer, jump, spot, rec_next, rec_near, fail_ok = [], [], [], [], [], []
    for e in m.events:
        f, a = e.get("frame"), _actor(m, e)
        if e["type"] != "pass" or f is None or a is None:
            continue
        h, d = m.halves[e["half"]], dmat[e["half"]]
        passer.append(d[f, a])
        spot.append(e.get("dist", np.nan))
        k = max(1, int(round(0.2 * fps)))
        if f - k >= 0 and f + k + 1 <= len(h.t):
            v = np.linalg.norm(np.diff(h.ball[f - k : f + k + 1, :2], axis=0), axis=1) * fps
            jump.append(float(v[k:].mean() - v[:k].mean()))
        who, _ = next_touch(d, f, a, horizon)
        r = e.get("recipient")
        if e.get("success") and _is_index(r) and 0 <= r < d.shape[1]:
            rec_next.append(who == r)
            rec_near.append(bool(np.any(d[f : f + horizon, r] <= RECEIVE_M)))
        elif not e.get("success"):
            fail_ok.append(who is None or m.team[who] == 1 - e["team"])
    mean = lambda a: float(np.mean(a)) if len(a) else float("nan")
    return {
        "passes_placed": len(passer),
        "passer_ball_median_m": _pct(passer, 50),
        "passer_ball_p90_m": _pct(passer, 90),
        "ball_speed_jump_median_ms": _pct(jump, 50),
        "spot_median_m": _pct(spot, 50),
        "completed_recipient_next": mean(rec_next),
        "completed_recipient_within_2m": mean(rec_near),
        "failed_opponent_or_dead_next": mean(fail_ok),
    }


def event_xy(m, e):
    """Event location in centred tracking coordinates, using the convention chosen by attach_frames."""
    return np.array(AXES[m.meta.get("event_axis", "same_y")](e["x"], e["y"]))

"""Inject the errors of our broadcast pipeline into clean tracks: position noise, ID swaps, missing frames.

Position noise follows the error measured in project 20 v1 (pitch-position error of the landmark homography:
median 2.5 m, 90th percentile 6.2 m). Its direction is isotropic and its magnitude has a heavier tail than a
Gaussian (p90 / median = 2.48 instead of 1.82), so it is modelled as a 2-D Gaussian times a log-normal scale.
`tau_s` sets how fast the error changes: 0 = independent every frame (worst case), ~1 s = smooth drift, which is
closer to a homography error that changes with the camera.
"""

import numpy as np

POS_MEDIAN, POS_P90 = 2.5, 6.2


def smooth_normal(shape, tau_frames, rng):
    """Standard-normal noise along axis 0 (time), Gaussian-smoothed with std `tau_frames`, rescaled to unit std."""
    z = rng.standard_normal(shape)
    if tau_frames <= 0.5:
        return z
    r = int(3 * tau_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / tau_frames) ** 2)
    k /= np.sqrt((k**2).sum())  # unit variance after convolution of white noise
    flat = z.reshape(len(z), -1)
    pad = np.pad(flat, ((r, r), (0, 0)), mode="wrap")
    out = np.stack([np.convolve(pad[:, j], k, "valid") for j in range(flat.shape[1])], axis=1)
    return out.reshape(shape)


def _calibrate(median=POS_MEDIAN, p90=POS_P90, n=400_000):
    """(log-normal sigma, scale) so that |scale * exp(sigma*z) * g| has the target median and 90th percentile."""
    rng = np.random.default_rng(0)
    g = np.linalg.norm(rng.standard_normal((n, 2)), axis=1)
    z = rng.standard_normal(n)
    lo, hi = 0.0, 2.0
    for _ in range(40):  # bisection on the p90/median ratio, which grows with sigma
        s = (lo + hi) / 2
        m = g * np.exp(s * z)
        if np.quantile(m, 0.9) / np.median(m) < p90 / median:
            lo = s
        else:
            hi = s
    m = g * np.exp(s * z)
    return s, median / np.median(m)


_CAL = {}


def position_noise(n_frames, n_tracks, fps, tau_s, rng, median=POS_MEDIAN, p90=POS_P90):
    """(n_frames, n_tracks, 2) error vectors in metres with the target median / p90 magnitude."""
    key = (median, p90)
    if key not in _CAL:
        _CAL[key] = _calibrate(median, p90)
    sigma, scale = _CAL[key]
    tau = tau_s * fps
    g = smooth_normal((n_frames, n_tracks, 2), tau, rng)
    z = smooth_normal((n_frames, n_tracks), tau, rng)
    return scale * np.exp(sigma * z)[..., None] * g


def drop_frames(present, frac, gap_frames, rng):
    """Mask (True = drop) removing about `frac` of the present frames in gaps of geometric length (mean gap_frames)."""
    T = len(present)
    drop = np.zeros(T, bool)
    target = frac * present.sum()
    if target <= 0:
        return drop
    for _ in range(100_000):
        if (drop & present).sum() >= target:
            break
        s = rng.integers(0, T)
        drop[s : s + rng.geometric(1 / max(gap_frames, 1))] = True
    return drop


def inject(
    tracks, rng, tau_s=None, pos_scale=1.0, shared=False, local_scale=0.2, missing_frac=0.0, gap_s=1.0,
    swaps_per_min=0.0, ball=True,
):  # fmt: skip
    """Return a corrupted copy of `tracks`.

    tau_s: None = no position noise; otherwise its time correlation in seconds (0 = independent per frame).
    pos_scale: multiplies the position error (0.5 = a pipeline twice as accurate as v1's homography).
    shared: False = every object gets its own error (pessimistic); True = a homography error shared by all
        objects in the frame (they are seen through the same camera model) plus an independent part scaled by
        `local_scale` (0.2 -> median 0.5 m).
    missing_frac: share of each track's frames dropped, in gaps of mean length gap_s.
    swaps_per_min: identity swaps; at each one, a player's track and that of the nearest other player exchange
        identities from that frame on (the team label stays with the identity, as in a tracker).
    ball: apply position noise and missing frames to the ball too."""
    t = tracks.copy()
    T, N, _ = t.xy.shape
    n_swaps = rng.poisson(swaps_per_min * T / t.fps / 60) if swaps_per_min > 0 else 0
    for f in np.sort(rng.integers(0, T, n_swaps)):
        here = np.flatnonzero(~np.isnan(t.xy[f, :, 0]))
        if len(here) < 2:
            continue
        i = rng.choice(here)
        d = np.linalg.norm(t.xy[f, here] - t.xy[f, i], axis=1)
        d[here == i] = np.inf
        j = here[np.argmin(d)]
        t.xy[f:, [i, j]] = t.xy[f:, [j, i]]
    if tau_s is not None:
        if shared:  # one camera-level error for every object in the frame, plus a smaller independent part
            common = pos_scale * position_noise(T, 1, t.fps, tau_s, rng)
            t.xy += common + local_scale * position_noise(T, N, t.fps, tau_s, rng)
            if ball:
                t.ball += common[:, 0] + local_scale * position_noise(T, 1, t.fps, tau_s, rng)[:, 0]
        else:
            t.xy += pos_scale * position_noise(T, N, t.fps, tau_s, rng)
            if ball:
                t.ball += pos_scale * position_noise(T, 1, t.fps, tau_s, rng)[:, 0]
    if missing_frac > 0:
        gap = gap_s * t.fps
        for i in range(N):
            present = ~np.isnan(t.xy[:, i, 0])
            if present.any():
                t.xy[drop_frames(present, missing_frac, gap, rng), i] = np.nan
        if ball:
            t.ball[drop_frames(~np.isnan(t.ball[:, 0]), missing_frac, gap, rng)] = np.nan
    return t

"""Closed-form pitch control (after Spearman 2018, simplified so it ports to the browser exactly).

Each player reaches a point after a reaction time (moving with his current velocity meanwhile) and then running
straight at a top speed. A team's control of a point is a logistic function of the gap between the fastest
attacker's and the fastest defender's arrival times (sigma = arrival-time uncertainty).
"""

import numpy as np

REACTION, VMAX, SIGMA = 0.7, 5.0, 0.45
SCALE = np.pi / np.sqrt(3) / SIGMA


def time_to_reach(pos, vel, points):
    """[P, 2] positions and velocities, [N, 2] points -> [P, N] arrival times; NaN players are ignored (inf)."""
    start = pos + vel * REACTION
    d = np.linalg.norm(points[None, :, :] - start[:, None, :], axis=2)
    t = REACTION + d / VMAX
    return np.where(np.isnan(t), np.inf, t)


def control(points, att_pos, att_vel, def_pos, def_vel):
    """Probability that the attacking team controls each point [N]."""
    ta = time_to_reach(att_pos, att_vel, points).min(axis=0)
    td = time_to_reach(def_pos, def_vel, points).min(axis=0)
    return 1.0 / (1.0 + np.exp(-SCALE * np.clip(td - ta, -50, 50)))


def grid(att_pos, att_vel, def_pos, def_vel, step=2.0):
    """Control over the whole pitch: (grid [ny, nx], xs, ys)."""
    xs = np.arange(-52.5 + step / 2, 52.5, step)
    ys = np.arange(-34 + step / 2, 34, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    return control(pts, att_pos, att_vel, def_pos, def_vel).reshape(gx.shape), xs, ys

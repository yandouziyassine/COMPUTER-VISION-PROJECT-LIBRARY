"""Features for the three models, computed from one normalised frame state (team in question attacks +x)."""

import numpy as np

from . import geometry as G
from . import pitch_control as PC
from .match import FPS

PASS_FEATURES = (
    "distance",
    "forward",
    "abs_angle",
    "lane_defenders",
    "receiver_space",
    "passer_pressure",
    "target_control",
    "target_x",
    "target_goal_dist",
)
VALUE_FEATURES = ("x", "abs_y", "goal_dist", "goal_angle", "in_box")
SHOT_FEATURES = ("goal_dist", "goal_angle", "cone_defenders")
LANE_WIDTH = 2.0
BACK = 4  # frames of history for velocities (0.16 s)


def state(m, half: int, frame: int, team: int) -> dict:
    """Causal state: velocities are backward differences over BACK frames (same in training and at inference)."""
    h = m.halves[half]
    direction = G.attack_dirs(h, m.team)[team]
    back = max(0, frame - BACK)
    pos = h.xy[frame]
    vel = np.nan_to_num((pos - h.xy[back]) * (FPS / (frame - back))) if frame > back else np.zeros_like(pos)
    present = ~np.isnan(pos[:, 0])
    att = present & (m.team == team)
    dfn = present & (m.team == 1 - team)  # players with team -1 (not on the match sheet) are neither
    norm = lambda a: G.normalise(a, direction)
    return {
        "att": norm(pos[att]),
        "att_v": norm(vel[att]),
        "def": norm(pos[dfn]),
        "def_v": norm(vel[dfn]),
        "att_idx": np.flatnonzero(att),
        "ball": norm(h.ball[frame, :2]),
        "direction": direction,
    }


def _segment_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def pass_features(s, passer: int, target: int) -> dict:
    a, b = s["att"][passer], s["att"][target]
    d, t = _segment_dist(s["def"], a, b)
    lane = int(np.sum((d < LANE_WIDTH) & (t > 0.05) & (t < 0.95)))
    tc = PC.control(b[None], s["att"], s["att_v"], s["def"], s["def_v"])[0]
    dist = float(np.linalg.norm(b - a))
    gd, _ = G.goal_geometry(b[None])
    return {
        "distance": dist,
        "forward": float(b[0] - a[0]),
        "abs_angle": float(abs(np.arctan2(b[1] - a[1], b[0] - a[0]))),
        "lane_defenders": lane,
        "receiver_space": float(np.min(np.linalg.norm(s["def"] - b, axis=1))),
        "passer_pressure": float(np.min(np.linalg.norm(s["def"] - a, axis=1))),
        "target_control": float(tc),
        "target_x": float(b[0]),
        "target_goal_dist": float(gd[0]),
    }


def value_features(ball) -> dict:
    gd, ga = G.goal_geometry(np.asarray(ball)[None])
    in_box = abs(ball[1]) < 20.16 and ball[0] > G.HALF_LENGTH - 16.5
    return {
        "x": float(ball[0]),
        "abs_y": float(abs(ball[1])),
        "goal_dist": float(gd[0]),
        "goal_angle": float(ga[0]),
        "in_box": float(in_box),
    }


def shot_features(ball, defenders) -> dict:
    """Defenders inside the triangle ball-post-post count as blocking the shot."""
    gd, ga = G.goal_geometry(np.asarray(ball)[None])
    posts = np.array([[G.HALF_LENGTH, G.GOAL_HALF_WIDTH], [G.HALF_LENGTH, -G.GOAL_HALF_WIDTH]])

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def inside(p):
        tri = [ball, posts[0], posts[1]]
        s = [cross(tri[(i + 1) % 3] - tri[i], p - tri[i]) for i in range(3)]
        return all(v >= 0 for v in s) or all(v <= 0 for v in s)

    return {
        "goal_dist": float(gd[0]),
        "goal_angle": float(ga[0]),
        "cone_defenders": int(sum(inside(p) for p in np.atleast_2d(defenders))),
    }

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
# Value model basis (Ridge, so the website can port it): position, goal geometry and three exponentials of the goal
# distance (scales in metres) that let a linear model follow the steep rise of scoring chances near the goal.
VALUE_FEATURES = (
    "x",
    "abs_y",
    "goal_dist",
    "goal_angle",
    "in_box",
    "near_goal",
    "mid_goal",
    "far_goal",
    "x_sq",
    "angle_near",
)
VALUE_BASIS = {
    "x": "x (m, attacking towards +x, goal centre at 52.5)",
    "abs_y": "|y| (m)",
    "goal_dist": "distance to the goal centre (m)",
    "goal_angle": "angle between the posts seen from the point (rad)",
    "in_box": "1 inside the penalty area (|y| < 20.16 and x > 36), else 0",
    "near_goal": "exp(-goal_dist / 8)",
    "mid_goal": "exp(-goal_dist / 20)",
    "far_goal": "exp(-goal_dist / 50)",
    "x_sq": "(x / 52.5)^2",
    "angle_near": "goal_angle * near_goal",
}
SHOT_FEATURES = ("goal_dist", "goal_angle", "cone_defenders")
LANE_WIDTH = 2.0
BACK = 4  # frames of history for velocities (0.16 s at 25 fps)
BACK_S = BACK / FPS


def match_fps(m) -> float:
    """Frame rate of a match: 25 fps (IDSSE) unless its meta says otherwise (SkillCorner: 10 fps)."""
    return float(m.meta.get("fps", FPS)) if getattr(m, "meta", None) else FPS


def state(m, half: int, frame: int, team: int, fps=None, back_frames=None, direction=None) -> dict:
    """Causal state: velocities are backward differences over `back_frames` frames (same in training and at
    inference). Defaults: the match's frame rate and 0.16 s of history (4 frames at 25 fps, 2 at 10 fps, i.e. 0.2 s).
    The website rebuilds it at 12.5 fps with back_frames=2 (0.16 s, as in training) and the attack direction stored
    in its match.json (coach/parity_fixtures.py)."""
    if fps is None:
        fps = match_fps(m)
    if back_frames is None:
        back_frames = max(1, int(round(BACK_S * fps)))
    h = m.halves[half]
    if direction is None:
        direction = G.attack_dirs(h, m.team)[team]
    back = max(0, frame - back_frames)
    pos = h.xy[frame]
    vel = np.nan_to_num((pos - h.xy[back]) * (fps / (frame - back))) if frame > back else np.zeros_like(pos)
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
        "def_idx": np.flatnonzero(dfn),
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
    ball = np.asarray(ball, dtype=float)
    gd, ga = G.goal_geometry(ball[None])
    gd, ga = float(gd[0]), float(ga[0])
    in_box = abs(ball[1]) < 20.16 and ball[0] > G.HALF_LENGTH - 16.5
    near = float(np.exp(-gd / 8.0))
    return {
        "x": float(ball[0]),
        "abs_y": float(abs(ball[1])),
        "goal_dist": gd,
        "goal_angle": ga,
        "in_box": float(in_box),
        "near_goal": near,
        "mid_goal": float(np.exp(-gd / 20.0)),
        "far_goal": float(np.exp(-gd / 50.0)),
        "x_sq": float((ball[0] / G.HALF_LENGTH) ** 2),
        "angle_near": ga * near,
    }


def without_defender(s, row: int) -> dict:
    """Copy of a state with one defender (row of s["def"]) removed."""
    keep = np.arange(len(s["def"])) != row
    return {**s, "def": s["def"][keep], "def_v": s["def_v"][keep], "def_idx": s["def_idx"][keep]}


def lane_loss_point(s, a, b, reach: float = 10.0):
    """Where a pass from a to b is most likely lost: the nearest defender's projection on the lane (clipped to its
    interior), or the lane midpoint when no defender is within `reach` metres of it."""
    d, t = _segment_dist(s["def"], a, b)
    k = int(np.argmin(d))
    if d[k] > reach:
        return (a + b) / 2
    return a + float(np.clip(t[k], 0.1, 0.95)) * (b - a)


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

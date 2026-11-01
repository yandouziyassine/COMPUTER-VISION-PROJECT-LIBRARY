"""Pitch geometry. Normalised frame: the team in question attacks towards +x (goal centre at (52.5, 0))."""

import numpy as np

HALF_LENGTH, HALF_WIDTH, GOAL_HALF_WIDTH = 52.5, 34.0, 3.66


def attack_dirs(half, team) -> tuple:
    """(+1/-1 for home, +1/-1 for guest): a team attacks away from the side where it stands at the half's start."""
    first = half.xy[: min(len(half.t), 250)]  # first 10 s
    home = np.nanmean(first[:, team == 0, 0])
    guest = np.nanmean(first[:, team == 1, 0])
    return (1, -1) if home < guest else (-1, 1)


def normalise(points, direction: int):
    """Rotate by 180° when the team attacks towards -x (keeps left/right consistent)."""
    return np.asarray(points, dtype=float) * (1 if direction == 1 else -1)


def goal_geometry(points):
    """Distance to the goal centre and opening angle between the posts, for normalised points [N, 2]."""
    p = np.atleast_2d(points)
    dx = HALF_LENGTH - p[:, 0]
    dist = np.hypot(dx, p[:, 1])
    a1 = np.arctan2(GOAL_HALF_WIDTH - p[:, 1], dx)
    a2 = np.arctan2(-GOAL_HALF_WIDTH - p[:, 1], dx)
    angle = np.abs(a1 - a2)
    angle = np.where(angle > np.pi, 2 * np.pi - angle, angle)
    return dist, np.where(dx <= 1e-9, np.where(np.abs(p[:, 1]) <= GOAL_HALF_WIDTH, np.pi, 0.0), angle)

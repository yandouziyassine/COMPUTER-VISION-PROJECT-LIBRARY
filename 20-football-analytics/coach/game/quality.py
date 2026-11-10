"""Moment quality filters (spec section 4): a moment is shown only when the data around it is clean."""

import numpy as np

BALL_GAP_M = 2.0
JUMP_M = 1.5
MIN_OPTIONS = 3


def problems(n_att: int, n_def: int, ball_gap: float, lead_up: np.ndarray, n_options: int) -> list:
    """Names of the failed checks, in a fixed order; [] when the moment is usable. lead_up: [T, P, 2] metres."""
    out = []
    if n_att != 11 or n_def != 11:
        out.append("players")
    if not ball_gap <= BALL_GAP_M:
        out.append("ball")
    step = np.linalg.norm(np.diff(lead_up, axis=0), axis=-1)
    if np.nanmax(step, initial=0.0) > JUMP_M:
        out.append("jump")
    if n_options < MIN_OPTIONS:
        out.append("options")
    return out

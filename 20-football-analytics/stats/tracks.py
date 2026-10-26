"""Common input format: positions in metres on an L x W pitch (x along the length, y across)."""

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class Tracks:
    xy: np.ndarray  # (T, N, 2) player positions in metres; NaN = not visible / not on the pitch
    team: np.ndarray  # (N,) 0 or 1
    ball: np.ndarray  # (T, 2) ball position in metres; NaN = ball not seen / out of play
    fps: float = 25.0
    period: np.ndarray | None = None  # (T,) half number; defaults to all 1
    names: list = field(default_factory=list)  # (N,) player identifiers
    pitch: tuple = (105.0, 68.0)
    frame: np.ndarray | None = None  # (T,) original frame numbers; defaults to 0..T-1

    def __post_init__(self):
        self.xy = np.asarray(self.xy, float)
        self.team = np.asarray(self.team, int)
        self.ball = np.asarray(self.ball, float)
        if self.period is None:
            self.period = np.ones(len(self.xy), int)
        if self.frame is None:
            self.frame = np.arange(len(self.xy))
        if not self.names:
            self.names = [f"p{i}" for i in range(self.xy.shape[1])]
        assert self.xy.ndim == 3 and self.xy.shape[2] == 2, "xy must be (T, N, 2)"
        assert len(self.ball) == len(self.xy) == len(self.period) and len(self.team) == self.xy.shape[1]

    @property
    def n_frames(self):
        return len(self.xy)

    def window(self, start, stop):
        """Frames [start, stop) as a new Tracks (arrays are copies)."""
        return replace(
            self,
            xy=self.xy[start:stop].copy(),
            ball=self.ball[start:stop].copy(),
            period=self.period[start:stop].copy(),
            frame=self.frame[start:stop].copy(),
        )

    def copy(self):
        return self.window(0, self.n_frames)


def runs(values):
    """Run-length encoding of a 1-D array: list of (value, start, stop) with stop exclusive."""
    values = np.asarray(values)
    if len(values) == 0:
        return []
    change = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.r_[0, change]
    stops = np.r_[change, len(values)]
    return [(values[a], int(a), int(b)) for a, b in zip(starts, stops)]

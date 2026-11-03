"""Project 20, AI coach: match report (decision quality, missed opportunities). Synthetic matches only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import evaluate as EV  # noqa: E402
from coach import match as M  # noqa: E402


def small_match(frames=400, match_id="SYN"):
    """Two halves at 25 fps, 3 players per team standing still; home attacks +x in half 0 and -x in half 1.
    Events: a completed pass 0 -> 1 (half 0, frame 41), a home goal 1.56 s later, a completed pass 1 -> 2 (half 1)."""
    home = np.array([[-10.0, 0.0], [5.0, 8.0], [15.0, -6.0]])
    guest = np.array([[10.0, 0.0], [-5.0, -8.0], [25.0, 12.0]])
    halves = []
    for h in range(2):
        xy = np.repeat(np.concatenate([home, guest])[None], frames, axis=0).astype(np.float32)
        ball = np.zeros((frames, 3), np.float32)
        ball[:, :2] = home[0]
        if h == 1:  # sides swap at half time
            xy[..., 0] *= -1
            ball[:, 0] *= -1
        halves.append(
            M.Half(
                n=np.arange(frames, dtype=np.int32),
                t=1000.0 + h * 4000 + np.arange(frames) / 25.0,
                xy=xy,
                ball=ball,
                owner=np.zeros(frames, np.int8),
                alive=np.ones(frames, bool),
            )
        )
    m = M.Match(
        match_id,
        list("abcdef"),
        np.array([0, 0, 0, 1, 1, 1], np.int8),
        np.array([1, 2, 3, 1, 2, 3], np.int16),
        list("ABCDEF"),
        halves,
        [],
        {"home": "Home FC", "guest": "Away FC"},
    )

    def ev(kind, half, frame, **kw):
        base = {"type": kind, "half": half, "frame": frame, "t": float(m.halves[half].t[frame]), "team": 0}
        return {**base, "player": 0, "set_piece": None, "cross": False, "x": 0.0, "y": 0.0, **kw}

    m.events = [
        ev("pass", 0, 41, recipient=1, success=True),
        ev("shot", 0, 80, goal=True, xg=0.3),
        ev("pass", 1, 61, player=1, recipient=2, success=True),
    ]
    return m


MODELS = {
    "pass": lambda X_: 1 / (1 + np.exp(-(3.0 - 0.1 * X_[:, 0]))),  # completion falls with distance
    "value": lambda X_: 0.1 * X_[:, 0] / 52.5,  # closer to the goal is better
    "shot": lambda X_: np.full(len(X_), 0.05),
}


def test_pass_decisions_choose_the_recorded_receiver_and_net_xg():
    ds = list(EV.pass_decisions(small_match(), MODELS))
    assert [d["chosen"]["target"] for d in ds] == [1, 2] and [d["sample"]["index"] for d in ds] == [0, 2]
    assert ds[0]["outcome"] == pytest.approx(0.3) and ds[1]["outcome"] == 0.0  # the goal 1.56 s after the first pass
    assert ds[0]["options"] == sorted(ds[0]["options"], key=lambda o: -o["score"])
    assert {o["kind"] for o in ds[0]["options"]} == {"pass", "carry"} and ds[0]["t"] == pytest.approx(1000 + 41 / 25)

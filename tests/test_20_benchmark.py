"""Project 20, AI coach: benchmark against xT and VAEP (coach insight C). Synthetic data only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import match as M  # noqa: E402
from coach import vaep as V  # noqa: E402


def toy_match(match_id="TOY", frames=300):
    """Home players 0, 1; guest 2, 3. Half 0: pass 0 -> 1 at frame 50 (ball reaches 1 at frame 60), failed pass by 1 at
    frame 150 from (7.5, 10), guest goal at frame 200 from (-32.5, 0). Half 1 (sides swapped): kick-off 0 -> 1."""
    home = np.array([[-20.0, 0.0], [-5.0, 10.0]])
    guest = np.array([[10.0, 0.0], [20.0, -5.0]])
    halves = []
    for h in range(2):
        xy = np.repeat(np.concatenate([home, guest])[None], frames, axis=0).astype(np.float32)
        ball = np.zeros((frames, 3), np.float32)
        ball[:, :2] = home[0]
        ball[60:, :2] = home[1]
        if h == 1:
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
        list("abcd"),
        np.array([0, 0, 1, 1], np.int8),
        np.array([1, 2, 1, 2], np.int16),
        list("ABCD"),
        halves,
        [],
        {"home": "H", "guest": "G"},
    )

    def ev(kind, half, frame, x, y, **kw):
        t = float(m.halves[half].t[frame])
        return {"type": kind, "half": half, "frame": frame, "t": t, "x": x, "y": y, "set_piece": None, **kw}

    m.events = [
        ev("pass", 0, 50, 32.5, 34.0, team=0, player=0, recipient=1, success=True, cross=False),
        ev("pass", 0, 150, 60.0, 44.0, team=0, player=1, recipient=2, success=False, cross=False),
        ev("shot", 0, 200, 20.0, 34.0, team=1, player=3, goal=True, xg=0.3),
        {
            **ev("pass", 1, 10, 52.5, 34.0, team=0, player=0, recipient=1, success=True, cross=False),
            "set_piece": "KickOff",
        },
    ]
    return m


def test_actions_types_order_and_carry():
    a = V.actions(toy_match())
    assert list(a["type"]) == ["pass", "carry", "pass", "shot", "kickoff"]
    assert list(a["event"]) == [0, -1, 1, 2, 3]
    assert list(a["result"]) == [1, 1, 0, 1, 1]
    assert list(a["team"]) == [0, 0, 0, 1, 0]
    assert list(a["goal"]) == [0, 0, 0, 1, 0]
    assert np.allclose(a["t"], [2.0, 4.0, 6.0, 8.0, 0.4])
    assert a["xg"][3] == pytest.approx(0.3) and np.isnan(a["xg"][0])


def test_actions_coordinates_attack_plus_x():
    a = V.actions(toy_match())
    start = np.c_[a["x0"], a["y0"]]
    end = np.c_[a["x1"], a["y1"]]
    assert np.allclose(start[0], [-20, 0]) and np.allclose(end[0], [-5, 10])  # completed pass ends at the reception
    assert np.allclose(start[1], [-5, 10]) and np.allclose(end[1], [7.5, 10])  # carry: reception -> next action
    assert np.allclose(end[2], [-32.5, 0])  # failed pass ends where the other team next played the ball
    assert np.allclose(start[3], [32.5, 0]) and np.allclose(end[3], [52.5, 0])  # guest attacks -x: rotated
    assert np.allclose(start[4], [0, 0]) and np.allclose(end[4], [-5, -10])  # half 1: home attacks -x


def test_action_type_mapping():
    assert V.action_type({"type": "shot", "set_piece": "Penalty"}) == "shot_penalty"
    assert V.action_type({"type": "shot", "set_piece": "FreeKick"}) == "shot_freekick"
    assert V.action_type({"type": "pass", "set_piece": "CornerKick", "cross": True}) == "corner"
    assert V.action_type({"type": "pass", "set_piece": None, "cross": True}) == "cross"
    assert V.action_type({"type": "pass", "set_piece": None, "cross": False}) == "pass"

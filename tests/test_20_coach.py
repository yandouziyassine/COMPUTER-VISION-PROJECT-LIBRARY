"""Project 20, AI coach: synthetic tests (storage, parsing, sync, pitch control, features, models, advice)."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import match as M  # noqa: E402


def tiny_match(frames=50):
    """Two halves, 2 players per team, ball moving along x; home attacks +x in half 0."""
    halves = []
    for h in range(2):
        t = 1000.0 + h * 4000 + np.arange(frames) / 25.0
        xy = np.zeros((frames, 4, 2), np.float32)
        xy[:, 0] = [-10, 0]
        xy[:, 1] = [5, 5]
        xy[:, 2] = [10, 0]
        xy[:, 3] = [-5, -5]
        ball = np.zeros((frames, 3), np.float32)
        ball[:, 0] = np.linspace(-10, 10, frames)
        halves.append(
            M.Half(
                n=np.arange(frames, dtype=np.int32) + 10000,
                t=t,
                xy=xy,
                ball=ball,
                owner=np.zeros(frames, np.int8),
                alive=np.ones(frames, bool),
            )
        )
    return M.Match(
        match_id="TEST",
        player_ids=["a", "b", "c", "d"],
        team=np.array([0, 0, 1, 1], np.int8),
        shirt=np.array([1, 2, 1, 2], np.int16),
        names=["A", "B", "C", "D"],
        halves=halves,
        events=[
            {
                "type": "pass",
                "half": 0,
                "t": 1000.4,
                "team": 0,
                "player": 0,
                "recipient": 1,
                "success": True,
                "x": 42.5,
                "y": 34.0,
            }
        ],
    )


def test_save_load_roundtrip(tmp_path):
    m = tiny_match()
    M.save(m, tmp_path / "TEST")
    back = M.load(tmp_path / "TEST")
    assert back.player_ids == m.player_ids and back.events == m.events
    assert np.allclose(back.halves[1].xy, m.halves[1].xy) and np.array_equal(back.team, m.team)


def test_frame_at_time_and_velocity():
    h = tiny_match().halves[0]
    assert M.frame_at(h, 1000.0) == 0 and M.frame_at(h, 1000.41) == 10 and M.frame_at(h, 9e9) == len(h.t) - 1
    v = M.velocity(h.ball[:, :2], 25.0)
    assert np.allclose(v[5:-5, 0], 20 / (49 / 25.0), atol=1e-4)  # 20 m in 49 frames

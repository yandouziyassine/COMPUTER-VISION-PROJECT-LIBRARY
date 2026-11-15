"""Project 20, coach v2: release-based event sync and the direction rule for failed-pass targets (synthetic)."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import datasets as D  # noqa: E402
from coach import match as M  # noqa: E402
from coach import sync  # noqa: E402

FPS = 25


def passing_match(frames=400):
    """One half (plus an empty-ish second one), 3 home players (A, B, C) and 1 guest (D) standing still.

    A (-20, 0) keeps the ball until frame 100, kicks it to B (0, 0): the ball reaches B at frame 120, B keeps it until
    frame 200 and kicks it to C (20, 0), reached at frame 220. The guest D stands far away at (0, 30)."""
    halves = []
    for h in range(2):
        t = 1000.0 + h * 4000 + np.arange(frames) / FPS
        xy = np.zeros((frames, 4, 2), np.float32)
        xy[:, 0] = [-20, 0]
        xy[:, 1] = [0, 0]
        xy[:, 2] = [20, 0]
        xy[:, 3] = [0, 30]
        f = np.arange(frames)
        bx = np.select(
            [f <= 100, f <= 120, f <= 200, f <= 220],
            [-20.0 + 0.5, -20.0 + 0.5 + (f - 100) * 19.5 / 20, 0.5, 0.5 + (f - 200) * 19.5 / 20],
            20.0,
        )
        ball = np.zeros((frames, 3), np.float32)
        ball[:, 0] = bx
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
        match_id="SYNC",
        player_ids=["a", "b", "c", "d"],
        team=np.array([0, 0, 0, 1], np.int8),
        shirt=np.array([1, 2, 3, 1], np.int16),
        names=["A", "B", "C", "D"],
        halves=halves,
    )


def pass_event(m, frame, player, recipient, x, success=True, lag_s=1.0, spot_err=(5.0, 0.0)):
    """A pass logged lag_s early (the hand clock lags the tracking) at a spot off by spot_err metres."""
    return {
        "type": "pass",
        "t": float(m.halves[0].t[frame]) - lag_s,
        "x": x + 52.5 + spot_err[0],
        "y": 34.0 + spot_err[1],
        "team": 0,
        "player": player,
        "recipient": recipient,
        "success": success,
    }


def test_releases_are_the_last_touch_before_the_ball_leaves():
    m = passing_match()
    d = sync.ball_distances(m.halves[0])
    assert sync.releases(d, 0).tolist() == [101]  # last frame within 1.5 m of A (the ball moves ~1 m per frame)
    assert sync.releases(d, 1).tolist() == [201]
    assert sync.releases(d, 2).tolist() == []  # C keeps the ball
    who, at = sync.next_touch(d, 101, 0, 150)
    assert who == 1 and 118 <= at <= 120


def test_release_sync_places_late_logged_passes_with_a_bad_spot():
    m = passing_match()
    m.events = [pass_event(m, 100, 0, 1, -20.0), pass_event(m, 200, 1, 2, 0.0)]
    report = sync.attach_frames(m)
    assert [e["frame"] for e in m.events] == [101, 201]
    assert all(e["sync"] == "release" and e["follow"] == "recipient" for e in m.events)
    assert m.events[0]["reception"] <= 120 and report["passes_kept"] == 2 and report["dropped"] == {}
    assert report["shift_s"][0] == pytest.approx(1.04, abs=0.05)
    q = report["quality"]
    assert q["completed_recipient_next"] == 1.0 and q["completed_recipient_within_2m"] == 1.0
    assert q["ball_speed_jump_median_ms"] > 10
    # the legacy method trusts the spot: 3.5 m between the spot and the ball at the release is over its 3 m limit
    report = sync.attach_frames(m, "legacy")
    assert report["passes_kept"] == 0 and set(report["dropped"]) <= {"far", "no_release", "window_edge"}


def test_release_sync_resets_previous_placement_and_drops_unreceived_passes():
    m = passing_match(frames=200)  # the ball stays with B: C never gets near it
    m.events = [pass_event(m, 100, 0, 2, -20.0)]  # logged "completed" to C
    m.events[0]["frame"], m.events[0]["reception"] = 5, 7  # stale fields of an earlier sync
    sync.attach_frames(m)
    assert m.events[0]["frame"] is None and m.events[0]["drop"] == "no_reception" and "reception" not in m.events[0]


def test_release_sync_failed_pass_and_duplicates():
    m = passing_match()
    m.events = [
        pass_event(m, 200, 1, 3, 0.0, success=False),  # the recorded "recipient" of a failed pass is the opponent
        pass_event(m, 200, 1, 2, 0.0, lag_s=1.1),  # the same release logged twice
    ]
    report = sync.attach_frames(m)
    placed = [e for e in m.events if e["frame"] is not None]
    assert len(placed) == 1 and placed[0]["frame"] == 201
    assert placed[0]["success"]  # the completed reading wins: its recipient touches next (bonus)
    assert [e["drop"] for e in m.events if e["frame"] is None] == ["duplicate"]
    assert report["dropped"] == {"duplicate": 1}


def test_release_sync_without_tracked_actor_and_use_calc_flag():
    m = passing_match()
    shot = {"type": "shot", "t": float(m.halves[0].t[100]), "x": -19.5 + 52.5, "y": 34.0, "team": 0, "player": 0}
    m.events = [{**pass_event(m, 100, 0, 1, -20.0), "player": "a"}, {**shot, "calc_frame": 10101}]
    report = sync.attach_frames(m)
    assert m.events[0]["drop"] == "no_release" and m.events[1]["sync"] == "calc" and report["calc_frames_used"] == 1
    sync.attach_frames(m, use_calc=False)
    assert m.events[1]["sync"] == "release" and m.events[1]["frame"] == 101


def test_direction_rule_follows_the_runner():
    # passer at the origin attacking +x; the ball is cut out 10 m away at (10, 0) heading along +x
    s = {
        "att": np.array([[0.0, 0.0], [20.0, 6.0], [16.0, -4.5]]),
        "att_v": np.array([[0.0, 0.0], [0.0, -4.0], [0.0, 0.0]]),  # player 1 runs into the lane
        "def": np.array([[10.0, 0.0]]),
        "def_v": np.zeros((1, 2)),
    }
    point = np.array([10.0, 0.0])
    ray = D.infer_target(s, 0, [1, 2], point, True, rule="ray")
    direction = D.infer_target(s, 0, [1, 2], point, True, u0=np.array([1.0, 0.0]), speed=15.0, rule="direction")
    assert ray == 2 and direction == 1
    # nobody near the line: both fall back to the teammate nearest the end point (within 10 m)
    far = {**s, "att": np.array([[0.0, 0.0], [12.0, 9.0], [-5.0, 20.0]]), "att_v": np.zeros((3, 2))}
    assert D.infer_target(far, 0, [1, 2], point, True, u0=np.array([1.0, 0.0]), speed=15.0) == 1
    assert D.infer_target(far, 0, [1, 2], np.array([-20.0, -20.0]), False) is None


def test_save_json_only_keeps_the_arrays(tmp_path):
    m = passing_match(frames=50)
    M.save(m, tmp_path / "x")
    before = (tmp_path / "x.npz").stat().st_mtime_ns
    m.events = [{"type": "pass", "t": 1.0, "x": 1.0, "y": 2.0}]
    M.save(m, tmp_path / "x", arrays=False)
    assert (tmp_path / "x.npz").stat().st_mtime_ns == before and M.load(tmp_path / "x").events == m.events


def test_coach_v2_ship_rule():
    from coach import coach_v2 as V

    def res(data, direction, zxc_b, zxc_p):
        g = lambda d, lo: {"all": {"end_to_end": {"delta": d, "ci": [lo, d + 0.01]}}}
        keys = {
            "v2/current vs v1/current": data,
            "v2_direction/current vs v2/current": direction,
            "v2/zxc vs v2/current": None,
        }
        out = {}
        for recipe, z in (("bundesliga", zxc_b), ("pooled", zxc_p)):
            gains = {k: g(*v) for k, v in keys.items() if v is not None}
            gains["v2/zxc vs v2/current"] = g(*z)
            out[recipe] = {"gains": gains}
        return out

    v = V.verdicts(res((0.03, 0.01), (-0.001, -0.002), (0.01, -0.01), (0.005, -0.007)))
    assert v["data v2"]["ship"] and not v["direction target rule"]["ship"] and not v["zone x context"]["ship"]
    v = V.verdicts(res((0.03, -0.004), (0.0, -0.004), (0.01, 0.001), (0.01, 0.001)))
    assert v["data v2"]["ship"] and v["direction target rule"]["ship"] and v["zone x context"]["ship"]


def test_no_shot_option_from_the_touchline():
    from coach import advice as A

    s = {
        "att": np.array([[42.0, 33.0], [30.0, 10.0]]),
        "att_v": np.zeros((2, 2)),
        "def": np.array([[45.0, 20.0]]),
        "def_v": np.zeros((1, 2)),
        "att_idx": np.array([0, 1]),
    }
    models = {
        "pass": lambda X: np.full(len(X), 0.8),
        "value": lambda X: np.full(len(X), 0.01),
        "shot": lambda X: np.full(len(X), 0.5),
    }
    kinds = [o["kind"] for o in A.options(s, 0, models)]
    assert "shot" not in kinds  # 35 m out on the touchline: goal angle 0.065 rad < SHOT_MIN_ANGLE
    s2 = {**s, "att": np.array([[36.0, 4.0], [30.0, 10.0]])}
    assert A.options(s2, 0, models)[0]["kind"] == "shot"

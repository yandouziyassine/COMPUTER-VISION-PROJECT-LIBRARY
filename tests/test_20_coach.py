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


from coach import dfl  # noqa: E402

INFO = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-T" HomeTeamName="Home FC" GuestTeamName="Away FC"/>
<Environment PitchX="105.00" PitchY="68.00"/><Teams>
<Team TeamId="H" Role="home" TeamName="Home FC" PlayerShirtMainColor="#FF0000"><Players>
<Player PersonId="h1" ShirtNumber="9" Shortname="H. Nine" PlayingPosition="STZ" Starting="true"/></Players></Team>
<Team TeamId="G" Role="guest" TeamName="Away FC" PlayerShirtMainColor="#FFFFFF"><Players>
<Player PersonId="g1" ShirtNumber="4" Shortname="G. Four" PlayingPosition="IVL" Starting="true"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>"""

POS = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" TeamId="H" PersonId="h1">
<Frame N="10000" T="2022-08-26T16:32:09.720+00:00" X="-1.00" Y="2.00"/>
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="-1.10" Y="2.00"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="G" PersonId="g1">
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="5.00" Y="-3.00"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="ball">
<Frame N="10000" T="2022-08-26T16:32:09.720+00:00" X="0.00" Y="0.00" Z="0.10" BallPossession="1" BallStatus="1"/>
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="0.50" Y="0.00" Z="0.10" BallPossession="2" BallStatus="0"/>
</FrameSet></Positions></PutDataRequest>"""

EVENTS = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest>
<Event EventTime="2022-08-26T18:32:09.760+02:00" X-Position="51.50" Y-Position="36.00">
 <Play Player="h1" Recipient="h1" Team="H" Evaluation="unsuccessful"><Pass/></Play></Event>
<Event EventTime="2022-08-26T18:32:09.760+02:00" CalculatedFrame="10001" X-Position="95.00" Y-Position="30.00">
 <ShotAtGoal Team="G" TypeOfShot="rightLeg"><SuccessfulShot/></ShotAtGoal></Event>
<Event EventTime="2022-08-26T18:32:09.700+02:00" X-Position="52.50" Y-Position="34.00">
 <KickOff GameSection="firstHalf" TeamRight="G" TeamLeft="H"><Play Player="h1" Team="H" Evaluation="successfullyCompleted"><Pass/></Play></KickOff></Event>
</PutDataRequest>"""


def test_parse_match_information(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    assert info["teams"] == {"H": 0, "G": 1} and info["players"]["g1"] == {
        "team": 1,
        "shirt": 4,
        "name": "G. Four",
        "position": "IVL",
    }
    assert info["meta"]["home"] == "Home FC" and info["meta"]["colors"] == ["#FF0000", "#FFFFFF"]


def test_parse_positions_aligns_frames_and_fills_nan(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    (tmp_path / "pos.xml").write_text(POS, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    ids, halves = dfl.parse_positions(tmp_path / "pos.xml", info)
    h = halves[0]
    assert ids == ["h1", "g1"] and list(h.n) == [10000, 10001]
    assert np.isnan(h.xy[0, 1]).all() and np.allclose(h.xy[1, 1], [5, -3])
    assert list(h.owner) == [0, 1] and list(h.alive) == [True, False]
    assert abs(h.t[1] - h.t[0] - 0.04) < 1e-6


def test_parse_events_passes_shots_and_kickoff_sides(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    (tmp_path / "ev.xml").write_text(EVENTS, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    ev = dfl.parse_events(tmp_path / "ev.xml", info)
    passes = [e for e in ev if e["type"] == "pass"]
    shots = [e for e in ev if e["type"] == "shot"]
    assert len(passes) == 2 and passes[0]["set_piece"] == "KickOff" and passes[0]["recipient"] is None  # time order
    assert passes[1]["success"] is False and passes[1]["recipient"] == "h1"
    assert shots[0]["goal"] is True and shots[0]["team"] == 1 and shots[0]["frame"] == 10001
    assert passes[1]["t"] == pytest.approx(dfl.epoch("2022-08-26T16:32:09.760+00:00"))


from coach import geometry as G  # noqa: E402


def test_attack_direction_from_team_positions():
    m = tiny_match()  # home players at x = -10 and 5, guests at 10 and -5 → home mean -2.5 < guest mean 2.5
    assert G.attack_dirs(m.halves[0], m.team) == (1, -1)


def test_normalise_flips_when_team_attacks_left():
    pts = np.array([[10.0, 5.0]])
    assert np.allclose(G.normalise(pts, -1), [[-10.0, -5.0]]) and np.allclose(G.normalise(pts, 1), pts)


def test_goal_distance_and_angle():
    d, a = G.goal_geometry(np.array([[52.5 - 11.0, 0.0], [52.5, 10.0]]))
    assert d[0] == pytest.approx(11.0) and a[0] == pytest.approx(2 * np.arctan(3.66 / 11.0))
    assert a[1] == pytest.approx(0.0, abs=1e-9)  # on the goal line, outside the posts: no angle


from coach import sync  # noqa: E402


def test_sync_picks_axis_convention_and_attaches_frames():
    m = tiny_match()
    # ball at frame 10 is at x = -10 + 20 * 10 / 49; the event says (X, Y) in 0..105 x 0..68 with Y flipped
    bx = float(m.halves[0].ball[10, 0])
    m.events = [
        {
            "type": "pass",
            "t": float(m.halves[0].t[10]),
            "x": bx + 52.5,
            "y": 34.0 - 3.0,
            "team": 0,
            "player": "a",
            "recipient": "b",
            "success": True,
        }
    ]
    m.halves[0].ball[:, 1] = 3.0
    report = sync.attach_frames(m)
    assert report["axis"] == "flip_y" and report["median_dist_m"] < 0.5
    assert m.events[0]["half"] == 0 and m.events[0]["frame"] == 10


def test_sync_snaps_event_with_hand_timestamp_lead():
    m = tiny_match(frames=200)
    bx = float(m.halves[0].ball[100, 0])
    m.halves[0].ball[:, 1] = 0.0
    # operator stamped the event 1.2 s (30 frames) before the ball is actually there
    m.events = [{"type": "pass", "t": float(m.halves[0].t[70]), "x": bx + 52.5, "y": 34.0, "team": 0}]
    report = sync.attach_frames(m)
    assert m.events[0]["frame"] == 100 and m.events[0]["sync_shift_s"] == pytest.approx(1.2)
    assert report["median_dist_m"] < 0.1 and report["nominal_median_dist_m"] > 2 and report["median_shift_s"] == 1.2


from coach import pitch_control as PC  # noqa: E402


def test_time_to_reach_includes_reaction_and_current_velocity():
    t = PC.time_to_reach(np.array([[0.0, 0.0]]), np.array([[5.0, 0.0]]), np.array([[10.0, 0.0]]))
    # after 0.7 s reaction the player is at x = 3.5, then 6.5 m at 5 m/s
    assert t[0, 0] == pytest.approx(0.7 + 6.5 / 5.0)


def test_control_is_half_when_equidistant_and_high_near_attacker():
    att = np.array([[0.0, 0.0]])
    dfn = np.array([[20.0, 0.0]])
    zero = np.zeros((1, 2))
    pts = np.array([[10.0, 0.0], [1.0, 0.0], [19.0, 0.0]])
    c = PC.control(pts, att, zero, dfn, zero)
    assert c[0] == pytest.approx(0.5, abs=1e-6) and c[1] > 0.95 and c[2] < 0.05


def test_grid_shape():
    g, xs, ys = PC.grid(np.array([[0.0, 0.0]]), np.zeros((1, 2)), np.array([[30.0, 0.0]]), np.zeros((1, 2)), step=5)
    assert g.shape == (len(ys), len(xs)) and 0 <= g.min() <= g.max() <= 1


from coach import features as F  # noqa: E402


def test_state_normalises_and_drops_absent_players():
    m = tiny_match()
    m.halves[0].xy[:, 1] = np.nan  # player b off the pitch
    s = F.state(m, 0, 10, team=1)  # guests attack -x in half 0 → flipped
    assert s["att"].shape == (2, 2) and s["def"].shape == (1, 2)
    assert np.allclose(s["att"][0], [-10.0, 0.0]) and list(s["att_idx"]) == [2, 3]


def test_pass_features_lane_and_pressure():
    s = {
        "att": np.array([[0.0, 0.0], [20.0, 0.0]]),
        "att_v": np.zeros((2, 2)),
        "def": np.array([[10.0, 1.0], [40.0, 30.0]]),
        "def_v": np.zeros((2, 2)),
        "att_idx": np.array([5, 6]),
        "ball": np.array([0.0, 0.0]),
    }
    f = F.pass_features(s, passer=0, target=1)
    assert f["distance"] == pytest.approx(20.0) and f["forward"] == pytest.approx(20.0)
    assert f["lane_defenders"] == 1 and f["receiver_space"] == pytest.approx(np.hypot(10, 1))
    assert 0 < f["target_control"] < 1 and list(F.PASS_FEATURES) == list(f)


def test_value_and_shot_features():
    v = F.value_features(np.array([41.5, 0.0]))
    assert v["goal_dist"] == pytest.approx(11.0) and list(F.VALUE_FEATURES) == list(v)
    s = F.shot_features(np.array([41.5, 0.0]), defenders=np.array([[45.0, 0.0], [0.0, 0.0]]))
    assert s["cone_defenders"] == 1 and list(F.SHOT_FEATURES) == list(s)



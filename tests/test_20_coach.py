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
        if h == 1:  # sides swap at half time: home attacks -x
            xy[..., 0] *= -1
            ball[:, 0] *= -1
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
 <ShotAtGoal Team="G" Player="g1" xG="0.31" TypeOfShot="rightLeg"><SuccessfulShot/></ShotAtGoal></Event>
<Event EventTime="2022-08-26T18:32:10.000+02:00" X-Position="94.00" Y-Position="34.00">
 <Penalty><ShotAtGoal Team="H" Player="h1" xG="0.76"><SavedShot/></ShotAtGoal></Penalty></Event>
<Event EventTime="2022-08-26T18:32:11.000+02:00" X-Position="80.00" Y-Position="30.00">
 <FreeKick><ShotAtGoal Team="H" Player="h1"><ShotWide/></ShotAtGoal></FreeKick></Event>
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
    assert shots[0]["goal"] is True and shots[0]["team"] == 1 and shots[0]["calc_frame"] == 10001
    assert shots[0]["xg"] == pytest.approx(0.31) and shots[0]["set_piece"] is None and shots[0]["player"] == "g1"
    # shots nested in <Penalty> / <FreeKick> are kept and flagged
    assert [s["set_piece"] for s in shots] == [None, "Penalty", "FreeKick"] and shots[1]["xg"] == 0.76
    assert shots[1]["result"] == "SavedShot" and shots[2]["xg"] is None and shots[2]["calc_frame"] is None
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
    assert report["median_dist_m"] < 0.1 and report["nominal_median_dist_m"] > 2 and report["shift_s"][0] == 1.2
    assert report["kept"] == 1 and report["dropped"] == {} and m.events[0]["t_sync"] == float(m.halves[0].t[100])


def _run_events(frames, events, ball_x=None):
    m = tiny_match(frames=frames)
    m.halves[0].ball[:, 1] = 0.0
    if ball_x is not None:
        m.halves[0].ball[:, 0] = ball_x
    base = {"type": "pass", "team": 0, "y": 34.0}
    m.events = [{**base, **e, "t": float(m.halves[0].t[e.pop("nominal")])} for e in events]
    return m, sync.attach_frames(m)


def test_sync_only_snaps_to_release_not_to_later_ball_position():
    # passer a (index 0) stands at (-10, 0); the ball leaves him at frame 55 and passes (0, 0) at frame 75
    ball_x = np.where(np.arange(200) < 55, -10.0, -10.0 + 0.5 * (np.arange(200) - 55))
    ev = {"nominal": 75, "x": 52.5, "player": 0}
    m, report = _run_events(200, [dict(ev)], ball_x)
    assert m.events[0]["frame"] is None and m.events[0]["drop"] == "far" and report["dropped"] == {"far": 1}
    assert m.events[0]["t_sync"] == pytest.approx(m.events[0]["t"])  # unplaced events keep the nominal time
    # same ball and event, actor unknown: nothing ties the snap to the release, so it lands at frame 75
    m, report = _run_events(200, [{**ev, "player": "a"}], ball_x)
    assert m.events[0]["frame"] == 75
    # event at the release itself: kept at the release frame range
    m, report = _run_events(200, [{**ev, "x": 42.5, "nominal": 56}], ball_x)
    assert m.events[0]["frame"] is not None and m.events[0]["frame"] <= 57 and report["kept"] == 1


def test_sync_drops_events_at_the_window_edge_and_uses_calc_frame():
    xs = np.linspace(-10, 10, 200)
    ev = [{"nominal": n, "x": xs[n + 10] + 52.5} for n in (50, 80, 110)]  # three events that lead by 10 frames
    ev.append({"nominal": 100, "x": xs[150] + 52.5})  # true release 40 frames later: outside +-37
    m, report = _run_events(200, ev)
    assert [e["frame"] for e in m.events[:3]] == [60, 90, 120]
    assert m.events[3]["frame"] is None and m.events[3]["drop"] == "window_edge"
    # the provider's CalculatedFrame is trusted when it agrees with the location
    shot = {"type": "shot", "nominal": 20, "x": xs[150] + 52.5, "calc_frame": 10150}
    m, report = _run_events(200, [shot])
    assert m.events[0]["frame"] == 150 and m.events[0]["sync"] == "calc" and report["calc_frames_used"] == 1


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


def test_state_second_half_flips_and_excludes_unknown_team():
    m = tiny_match()
    s = F.state(m, 1, 10, team=0)  # home attacks -x in half 1: its players at +10 and -5 become -10 and 5
    assert np.allclose(s["att"][0], [-10.0, 0.0]) and np.allclose(s["def"][0], [10.0, 0.0])
    m.team[1] = -1  # tracked but not in the match sheet: neither attacker nor defender
    s = F.state(m, 0, 10, team=0)
    assert list(s["att_idx"]) == [0] and s["def"].shape == (2, 2)


def test_state_velocity_is_causal_backward_difference():
    m = tiny_match(frames=50)
    m.halves[0].xy[:, 0, 0] = -10 + 0.1 * np.arange(50)  # 2.5 m/s towards +x
    v = F.state(m, 0, 10, team=0)["att_v"][0]
    assert np.allclose(v, [2.5, 0.0], atol=1e-4)
    m.halves[0].xy[11:, 0, 1] = 99.0  # the future must not change the state at frame 10
    assert np.allclose(F.state(m, 0, 10, team=0)["att_v"][0], v)
    assert np.allclose(F.state(m, 0, 0, team=0)["att_v"][0], 0.0)  # first frame: no history


def test_state_velocity_at_the_site_frame_rate_and_given_direction():
    m = tiny_match(frames=50)
    m.halves[0].xy[:, 0, 0] = -10 + 0.2 * np.arange(50)  # 0.2 m per (12.5 fps) frame = 2.5 m/s
    s = F.state(m, 0, 10, team=0, fps=12.5, back_frames=2)
    assert np.allclose(s["att_v"][0], [2.5, 0.0])  # (x[10] - x[8]) * 12.5 / 2
    flipped = F.state(m, 0, 10, team=0, fps=12.5, back_frames=2, direction=-1)
    assert flipped["direction"] == -1 and np.allclose(flipped["att"][0], -s["att"][0])
    assert np.allclose(flipped["att_v"][0], [-2.5, 0.0])


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


from coach import datasets as D  # noqa: E402


def _pass(m, frame, **kw):
    base = {"type": "pass", "half": 0, "frame": frame, "team": 0, "player": 0, "cross": False, "set_piece": None}
    return {**base, "t": float(m.halves[0].t[frame]), "x": 0, "y": 0, **kw}


def _shot(m, frame, team, xg=None, **kw):
    base = {"type": "shot", "half": 0, "frame": frame, "team": team, "goal": False, "result": "SavedShot"}
    return {**base, "t": float(m.halves[0].t[frame]), "x": 95.0, "y": 34.0, "xg": xg, **kw}


def test_tables_have_labels_and_match_ids():
    m = tiny_match(frames=600)  # 24 s per half
    for h in m.halves:
        h.owner[:] = 0
    m.events = [_pass(m, 10, recipient=1, success=True), _shot(m, 300, 0)]
    p = D.pass_table(m)
    assert p["X"].shape == (1, len(D.F.PASS_FEATURES)) and list(p["y"]) == [1] and list(p["match"]) == ["TEST"]
    assert np.allclose(p["X_noint"], p["X"])  # no interceptor on a completed pass
    v = D.value_table(m, horizons=(10.0, 15.0))
    # the shot at 12 s has no provider xG -> default 0.1; live seconds 0..11 see it within 15 s, seconds 2..11 within 10 s
    assert v["X"].shape[1] == len(D.F.VALUE_FEATURES) and np.allclose(v["y"], v["y_h"][10.0])
    assert v["y_h"][15.0].sum() == pytest.approx(12 * D.XG_DEFAULT) and v["y_h"][10.0].sum() == pytest.approx(10 * 0.1)
    s = D.shot_table(m)
    assert list(s["y"]) == [0] and s["X"].shape == (1, len(D.F.SHOT_FEATURES)) and not s["penalty"].any()


def test_value_labels_are_net_xg_for_the_team_in_possession_on_snapped_time():
    m = tiny_match(frames=600)
    m.halves[0].owner[:] = 1  # guests keep the ball in half 0
    late = _shot(m, 450, 0, xg=0.1)  # home shot at 18 s ...
    late["frame"], late["t_sync"] = None, float(m.halves[0].t[450])  # ... that sync could not place: shifted time
    m.events = [_shot(m, 300, 1, xg=0.3), late, _shot(m, 100, 0, xg=0.76, set_piece="Penalty", t=1e9)]
    y = D.value_table(m)["y"]  # horizon 15 s, one row per second of half 0 (guest possession), then half 1 (owner 0)
    # the penalty (frame 100 = 4 s, nominal t absurd) is counted at its snapped time: against the guests for t < 4 s
    assert y[0] == pytest.approx(0.3 - 0.76) and y[2] == pytest.approx(0.3 - 0.76)
    assert y[3] == pytest.approx(0.3 - 0.76 - 0.1)  # the unplaced home shot (18 s) enters the 15 s horizon at 3 s
    assert y[4] == pytest.approx(0.3 - 0.1) and y[11] == pytest.approx(0.3 - 0.1)
    assert y[12] == pytest.approx(-0.1) and y[18] == 0.0
    t = D.shot_table(m)
    assert (
        list(t["penalty"]) == [False, True] and np.allclose(t["xg"], [0.3, 0.76]) and len(D.drop_penalties(t)["y"]) == 1
    )


def test_failed_pass_target_is_inferred_from_ball_path():
    m = tiny_match(frames=100)  # ball runs from (-10, 0) to (10, 0); player c (guest) waits at (10, 0)
    m.halves[0].xy[:, 1] = [5, 3]  # teammate b stands at (5, 3)
    for h in m.halves:
        h.owner[:] = 0
    m.events = [
        _pass(m, 5, recipient=1, success=True),
        _pass(m, 6, recipient=2, success=False),  # c intercepts at the end of the ball path
    ]
    p = D.pass_table(m)
    assert np.allclose(p["weight"], 1.0)  # table success rate 1/2 = true rate 1/2
    m.events.append({**_pass(m, 30, recipient=2, success=False), "frame": None})  # a failed pass sync could not place
    assert D.completion_rate(m) == pytest.approx(1 / 3) and np.allclose(D.pass_table(m)["weight"], [1.0, 2.0])
    m.events.pop()
    # the ball is within 1.5 m of c from x = 8.5; b is beyond half-way and 3 m from the ray -> target b, c is the interceptor
    assert list(p["y"]) == [1, 0] and p["X"].shape == (2, len(D.F.PASS_FEATURES))
    s = D.F.state(m, 0, 6, 0)
    assert D.intended_target(m, m.events[1], s, {0: 0, 1: 1}) == (1, 2)
    assert (
        p["X"][1, D.F.PASS_FEATURES.index("lane_defenders")]
        >= p["X_noint"][1, D.F.PASS_FEATURES.index("lane_defenders")]
    )
    m.events[0]["team"] = -1  # a pass by a player without a team is skipped
    assert list(D.pass_table(m)["y"]) == [0]
    m.events[0]["team"] = 0
    # no opponent near the ball path and the ball stays alive: unreadable -> skipped; dead ball: nearest teammate
    m.halves[0].xy[:, 2:] = [[50.0, 30.0], [50.0, 30.0]]
    assert D.intended_target(m, m.events[1], s, {0: 0, 1: 1}) is None
    m.halves[0].alive[60:] = False
    assert D.intended_target(m, m.events[1], s, {0: 0, 1: 1}) == (1, None)  # ball dies at (2, 0): b is 4.2 m away


from coach import models as ML  # noqa: E402


def synthetic_table(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1])))
    return {"X": X, "y": (rng.random(n) < p).astype(int), "match": np.repeat(list("ABCDEF"), n // 6)}


def test_logistic_export_matches_sklearn():
    t = synthetic_table()
    model = ML.fit_logistic(t["X"], t["y"])
    w = ML.export_logistic(model, ["a", "b", "c"])
    z = (t["X"] - np.array(w["mean"])) / np.array(w["scale"]) @ np.array(w["coef"]) + w["intercept"]
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict_proba(t["X"])[:, 1], atol=1e-9)


def test_leave_one_match_out_beats_base_rate():
    r = ML.lomo(synthetic_table(), {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate})
    assert r["logistic"]["auc"] > 0.75 and r["logistic"]["log_loss"] < r["base_rate"]["log_loss"]
    assert len(r["logistic"]["per_match"]) == 6 and len(r["logistic"]["reliability"]) == 10
    # sample weights: all ones change nothing, down-weighting the positives lowers the weighted positive rate
    t = synthetic_table()
    ones = ML.lomo({**t, "weight": np.ones(len(t["y"]))}, {"logistic": ML.fit_logistic})["logistic"]
    assert ones["log_loss"] == pytest.approx(r["logistic"]["log_loss"], rel=1e-6)
    half = ML.lomo({**t, "weight": np.where(t["y"] == 1, 0.5, 1.0)}, {"logistic": ML.fit_logistic})["logistic"]
    assert half["positive_rate"] < ones["positive_rate"]


def test_linear_predictor_reproduces_the_fitted_pipelines():
    t = synthetic_table()
    clf = ML.fit_logistic(t["X"], t["y"])
    p = ML.linear_predictor(ML.export_logistic(clf, ["a", "b", "c"]))(t["X"])
    assert np.allclose(p, clf.predict_proba(t["X"])[:, 1], atol=1e-9)
    reg = ML.fit_ridge(t["X"], 0.3 * t["X"][:, 0])
    v = ML.linear_predictor(ML.export_ridge(reg, ["a", "b", "c"]))(t["X"])
    assert np.allclose(v, reg.predict(t["X"]), atol=1e-9)


def test_ridge_export_and_regression_lomo():
    rng = np.random.default_rng(1)
    n = 1200
    x = rng.uniform(-52.5, 52.5, n)
    ay = rng.uniform(0, 34, n)
    X = np.c_[x, ay, np.hypot(52.5 - x, ay)]
    y = 0.002 * (x + 52.5) + rng.normal(0, 0.05, n)
    t = {"X": X, "y": y, "match": np.repeat(list("ABCDEF"), n // 6)}
    model = ML.fit_ridge(X, y)
    w = ML.export_ridge(model, ["x", "abs_y", "goal_dist"])
    z = (X - np.array(w["mean"])) / np.array(w["scale"]) @ np.array(w["coef"]) + w["intercept"]
    assert w["link"] == "identity" and np.allclose(z, model.predict(X), atol=1e-9)
    r = ML.lomo_reg(t, {"mean": ML.fit_mean, "ridge": ML.fit_ridge, "xt": ML.fit_xt_grid, "col": ML.column_reg(0)})
    assert r["ridge"]["mse_skill"] > 0.3 and r["xt"]["mse_skill"] > 0.2 and abs(r["mean"]["mse_skill"]) < 1e-9
    assert len(r["ridge"]["calibration"]) == 10 and set(r["ridge"]["per_match"]) == set("ABCDEF")


from coach import advice as A  # noqa: E402


def test_advice_prefers_open_forward_teammate():
    s = {
        "att": np.array([[20.0, 0.0], [45.0, 0.0], [5.0, 0.0]]),
        "att_v": np.zeros((3, 2)),
        "def": np.array([[20.0, 25.0], [0.0, -25.0]]),
        "def_v": np.zeros((2, 2)),
        "att_idx": np.array([0, 1, 2]),
        "ball": np.array([20.0, 0.0]),
        "direction": 1,
    }  # carrier 32.5 m from goal: the shot option exists
    sig = lambda w: lambda X: 1 / (1 + np.exp(-(X @ w)))
    # pass model: success falls with distance; value model: grows with x; shot model: closer is better
    models = {
        "pass": sig(np.r_[-0.05, 0, 0, -1.0, 0.05, 0, 2.0, 0, 0]),
        "value": lambda X: 0.1 * X[:, 0] / 52.5,
        "shot": sig(np.r_[-0.2, 0, 0]),
    }
    opts = A.options(s, carrier=0, models=models)
    passes = [o for o in opts if o["kind"] == "pass"]
    assert passes[0]["target"] == 1 and {o["kind"] for o in opts} == {"pass", "carry", "shot"}
    assert all(0 <= o["p"] <= 1 for o in opts) and opts == sorted(opts, key=lambda o: -o["score"])
    shot = next(o for o in opts if o["kind"] == "shot")  # shot: xG minus the value of keeping the ball here
    assert shot["score"] == pytest.approx(shot["p"] - 0.1 * 20.0 / 52.5)


def test_advice_turnover_cost_flips_the_order_against_probability_alone():
    # carrier deep in his own half: the short pass back is safe but losing it there is a disaster; the long ball
    # up the pitch is unlikely but losing it near the opponent's box costs nothing
    s = {
        "att": np.array([[-30.0, 0.0], [-35.0, 0.0], [10.0, 0.0]]),
        "att_v": np.zeros((3, 2)),
        "def": np.array([[-32.0, 3.0], [5.0, 2.0]]),
        "def_v": np.zeros((2, 2)),
        "att_idx": np.array([0, 1, 2]),
        "ball": np.array([-30.0, 0.0]),
        "direction": 1,
    }
    models = {
        "pass": lambda X: 1 / (1 + np.exp(-(2.0 - 0.1 * X[:, 0]))),  # completion falls with distance
        "value": lambda X: 0.1 * X[:, 0] / 52.5,
        "shot": lambda X: np.zeros(len(X)),
    }
    passes = [o for o in A.options(s, carrier=0, models=models) if o["kind"] == "pass"]
    by_p = sorted(passes, key=lambda o: -o["p"])
    assert by_p[0]["target"] == 1 and passes[0]["target"] == 2  # p alone says short back-pass, the coach the long ball
    assert passes[0]["score"] > 0 > by_p[0]["score"]
    # the lost-ball point of the long ball is the nearest defender's projection on the lane, mirrored for the opponent
    assert passes[0]["v_lost"] == pytest.approx(0.1 * -5.0 / 52.5)


from coach import export as X  # noqa: E402


def test_pack_match_writes_one_file_per_half_with_its_own_players(tmp_path):
    import json

    m = tiny_match(frames=100)
    m.names[0] = "Müller-Ösil"  # non-ASCII must survive (UTF-8 on disk, whatever the platform default encoding)
    m.halves[0].xy[:7, 1] = np.nan  # player b is not on the pitch for the first frames of half 0
    m.halves[1].xy[:, 3] = np.nan  # player d never plays in half 1: not in that half's file
    m.events[0]["frame"] = 40
    m.events[0]["nominal"] = 41
    m.events[0]["t_sync"] = 1000.0  # sync diagnostics are not shipped
    m.events.append(
        {"type": "shot", "half": 0, "frame": None, "nominal": 61, "team": 0, "drop": "far", "x": 1.0, "y": 2.0}
    )
    (tmp_path / "match.bin").write_bytes(b"old single-file layout")
    X.pack_match(m, tmp_path, fps=10)
    meta = json.loads((tmp_path / "match.json").read_text(encoding="utf-8"))
    assert meta["fps"] == 12.5 and not (tmp_path / "match.bin").exists()
    h0, h1 = meta["halves"]
    assert h0 == {"frames": 50, "slots": [0, 1, 2, 3], "attack": [1, -1]}
    assert h1 == {"frames": 50, "slots": [0, 1, 2], "attack": [-1, 1]}
    raw0 = np.fromfile(tmp_path / "h0.bin", dtype="<i2")
    raw1 = np.fromfile(tmp_path / "h1.bin", dtype="<i2")
    assert raw0.size == 50 * (4 + 1) * 2 and raw1.size == 50 * (3 + 1) * 2
    first = raw0[:10].reshape(-1, 2)
    assert np.allclose(first[0] / 100.0, m.halves[0].xy[0, 0]) and np.allclose(
        first[-1] / 100.0, m.halves[0].ball[0, :2]
    )
    assert (first[1] == -32768).all()  # absent player -> NaN code
    last1 = (
        raw1[-8:].reshape(-1, 2) / 100.0
    )  # half 1, last site frame (source frame 98): players a, b, c, then the ball
    assert np.allclose(last1[2], m.halves[1].xy[98, 2]) and np.allclose(
        last1[3], m.halves[1].ball[98, :2], atol=0.005
    )  # whole cm
    text = (tmp_path / "match.json").read_text(encoding="utf-8")
    assert meta["players"][0]["name"] == "Müller-Ösil" and "ö" in text.lower()
    assert meta["events"][0]["frame"] == 20 and meta["events"][0]["nominal"] == 20 and "t_sync" not in meta["events"][0]
    assert meta["events"][-1]["frame"] is None and meta["events"][-1]["nominal"] == 30
    assert len(meta["owner"][1]) == 50 and set(meta["owner"][0]) == {"1"} and set(meta["alive"][0]) == {"1"}


def test_export_models_leave_the_replay_match_out(tmp_path, monkeypatch):
    import json

    seen = {}

    def fake(kind, n_feat):
        def build(m):
            seen.setdefault(kind, []).append(m.match_id)
            rng = np.random.default_rng(len(seen[kind]))
            X_, y = rng.normal(size=(60, n_feat)), np.arange(60) % 2
            t = {"X": X_, "y": y, "match": np.array([m.match_id] * 60), "weight": np.ones(60)}
            if kind == "value":
                t["y"] = rng.normal(size=60)
            if kind == "shot":
                t["penalty"] = np.zeros(60, bool)
            return t

        return build

    monkeypatch.setattr(X.D, "pass_table", fake("pass", len(X.D.F.PASS_FEATURES)))
    monkeypatch.setattr(X.D, "value_table", fake("value", len(X.D.F.VALUE_FEATURES)))
    monkeypatch.setattr(X.D, "shot_table", fake("shot", len(X.D.F.SHOT_FEATURES)))
    ms = [tiny_match(), tiny_match(), tiny_match()]
    for m, mid in zip(ms, "ABC"):
        m.match_id = mid
    X.export_models(tmp_path, ms, exclude="B")
    w = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert w["trained_without"] == "B" and w["trained_on"] == ["A", "C"] and seen["pass"] == ["A", "C"]
    assert w["value"]["link"] == "identity" and w["value"]["features"] == list(X.D.F.VALUE_FEATURES)
    assert set(w["value"]["basis"]) == set(X.D.F.VALUE_FEATURES) and w["pass"]["link"] == "logit"


def test_advice_summary_buckets_correlations_and_bootstrap():
    pytest.importorskip("matplotlib")
    from coach import evaluate as EV

    rng = np.random.default_rng(0)
    recs = []
    for mid in "ABCD":
        for _ in range(60):
            score = float(rng.normal())
            rank = 1 + int(rng.integers(0, 9))
            recs.append(
                {
                    "match": mid,
                    "success": int(rng.random() < 0.5 + 0.04 * rank),
                    "weight": 1.0,
                    "score": score,
                    "v_now": 0.0,
                    "p": float(rng.random()),
                    "value": float(rng.normal()),
                    "xt": float(rng.normal()),
                    "outcome": score + float(rng.normal(0, 0.1)),  # the outcome follows the coach score
                    "rank_coach": rank,
                    "rank_p": rank,
                    "rank_value": rank,
                    "rank_near": rank,
                    "options": 9,
                }
            )
    s = EV.advice_summary(recs)
    assert s["passes"] == 240 and sum(b["n"] for b in s["completion_by_rank"]) == 240
    assert [b["rank"] for b in s["completion_by_rank"]] == ["1", "2-3", "4-6", "7+"]
    assert s["completion_by_rank"][-1]["completion"] > s["completion_by_rank"][0]["completion"]
    c = s["outcome_correlation"]["coach"]
    assert c["spearman"] > 0.9 and c["ci95"][0] > 0.8 and abs(s["outcome_correlation"]["p_only"]["spearman"]) < 0.3
    assert s["coach_expected_outcome"]["slope"] == pytest.approx(1.0, abs=0.1) and len(s["coach_deciles"]) == 10
    assert s["agreement_context"]["random"]["top3"] == pytest.approx(1 / 3)


from coach import parity_fixtures as PF  # noqa: E402


def _flat_model(names, link, c):
    k = len(names)
    return {
        "features": list(names),
        "link": link,
        "mean": [0.0] * k,
        "scale": [1.0] * k,
        "coef": [c] * k,
        "intercept": 0.0,
    }


def test_parity_fixtures_rebuild_the_site_state_and_rank_options(tmp_path):
    import json

    m = tiny_match(frames=100)
    m.halves[0].ball[30:50, :2] = [-10.0, 0.5]  # the ball at player a's feet around the pass (source frame 40)
    m.events[0]["frame"] = 40
    X.pack_match(m, tmp_path, fps=10)
    models = {
        "trained_without": "TEST",
        "pass": _flat_model(F.PASS_FEATURES, "logit", 0.05),
        "value": _flat_model(F.VALUE_FEATURES, "identity", 0.001),
        "shot": _flat_model(F.SHOT_FEATURES, "logit", -0.1),
    }
    (tmp_path / "models.json").write_text(json.dumps(models), encoding="utf-8")
    meta = json.loads((tmp_path / "match.json").read_text(encoding="utf-8"))
    team_of = np.array([p["team"] for p in meta["players"]])
    h = PF.load_half(tmp_path, meta, 0)
    assert PF.carrier_at(meta, 0, h, team_of, 20) == (0, 0)
    assert PF.carrier_at(meta, 0, h, team_of, 49) is None  # ball ~7 m from the nearest home player
    fx = PF.build(tmp_path, per_half=3)
    assert fx["back_frames"] == 2 and fx["fps"] == 12.5 and fx["models"]["trained_without"] == "TEST"
    assert len(fx["frames"]) == 1  # the tiny match has one pass with a frame
    rec = fx["frames"][0]
    assert (rec["half"], rec["frame"], rec["carrier"], rec["team"], rec["carrier_row"]) == (0, 20, 0, 0, 0)
    st = rec["state"]
    assert st["att_idx"] == [0, 1] and st["def_idx"] == [2, 3] and st["direction"] == 1
    assert np.allclose(st["att"][0], [-10.0, 0.0]) and np.allclose(st["att_v"], 0.0)
    scores = [o["score"] for o in rec["options"]]
    assert scores == sorted(scores, reverse=True) and {"pass", "carry"} <= {o["kind"] for o in rec["options"]}
    assert [p["target"] for p in rec["pass_features"]] == [1] and set(rec["value_features"]) == set(F.VALUE_FEATURES)
    json.dumps(fx, allow_nan=False)  # valid JSON for the browser: no NaN anywhere

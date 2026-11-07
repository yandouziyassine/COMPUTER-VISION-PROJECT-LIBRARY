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
from coach import xt as XT  # noqa: E402


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


def action_table(rows):
    """rows: (type, result, x0, x1, xg, goal) on the centre line y = 0."""
    n = len(rows)
    col = lambda i, dt=float: np.array([r[i] for r in rows], dt)
    return {
        "match": np.array(["T"] * n),
        "half": np.zeros(n, int),
        "t": np.arange(n, dtype=float),
        "team": np.zeros(n, int),
        "player": np.zeros(n, int),
        "type": col(0, "U16"),
        "result": col(1, int),
        "x0": col(2),
        "y0": np.zeros(n),
        "x1": col(3),
        "y1": np.zeros(n),
        "event": np.arange(n),
        "xg": col(4),
        "goal": col(5, int),
    }


def toy_grid():
    """Along the centre row: from every cell 10 completed and 2 failed passes one cell forward (none from the last
    cell); 4 shots from each of the last 4 cells, xG 0.05 .. 0.20 rising towards the goal, one goal per cell."""
    w = 105 / XT.NX
    rows = []
    for cx in range(XT.NX):
        x = -52.5 + (cx + 0.5) * w
        if cx < XT.NX - 1:
            rows += [("pass", 1, x, x + w, np.nan, 0)] * 10 + [("pass", 0, x, x + w, np.nan, 0)] * 2
        if cx >= XT.NX - 4:
            xg = 0.05 * (cx - XT.NX + 5)
            rows += [("shot", 1, x, 52.5, xg, 1)] + [("shot", 0, x, 52.5, xg, 0)] * 3
    return action_table(rows)


def test_xt_converges_to_fixed_point_and_rises_towards_goal():
    model = XT.ExpectedThreat(shrink=0.0).fit(toy_grid())
    assert model.iterations_ < XT.MAX_ITER
    xt = model.xt_
    assert np.allclose(xt, model.s_ * model.g_ + model.m_ * (model.T_ @ xt), atol=1e-5)
    row = model.grid()[:, XT.NY // 2]
    assert np.all(np.diff(row) > 0)
    assert row[-1] == pytest.approx(0.20)  # last cell: shots only, xT = mean xG there
    assert np.allclose(np.delete(model.grid(), XT.NY // 2, axis=1), 0)  # no action elsewhere


def test_xt_goal_source_and_action_values():
    a = toy_grid()
    goals = XT.ExpectedThreat(goal_source="goals", shrink=0.0).fit(a)
    assert goals.g_[XT.cell(np.array([52.0]), np.array([0.0]))[0]] == pytest.approx(0.25)
    model = XT.ExpectedThreat(shrink=0.0).fit(a)
    v = model.action_values(a)
    start = model.xt_[XT.cell(a["x0"], a["y0"])]
    end = model.xt_[XT.cell(a["x1"], a["y1"])]
    ok, failed, shot = a["result"] == 1, a["result"] == 0, a["type"] == "shot"
    assert np.allclose(v[ok & ~shot], (end - start)[ok & ~shot])
    assert np.allclose(v[failed & ~shot], -start[failed & ~shot])
    assert np.isnan(v[shot]).all()


def test_vaep_labels_window_and_halves():
    a = V.actions(toy_match())
    scores, concedes = V.labels(a)
    assert list(scores) == [0, 0, 0, 1, 0]
    assert list(concedes) == [1, 1, 1, 0, 0]  # the kick-off of half 1 is out of the window
    scores, concedes = V.labels(a, k=2)
    assert list(concedes) == [0, 0, 1, 0, 0]


def test_vaep_values_formula():
    a = {
        "match": np.array(["A"] * 4),
        "half": np.array([0, 0, 0, 1]),
        "team": np.array([0, 0, 1, 1]),
        "goal": np.array([0, 0, 0, 0]),
    }
    ps, pc = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.05, 0.1, 0.02, 0.03])
    v = V.values(a, ps, pc)
    # 0: start of the half; 1: same team; 2: other team (previous probabilities swapped); 3: new half
    assert np.allclose(v["value"], [0.05, 0.05, 0.2 + 0.18, 0.37])
    a["goal"] = np.array([0, 1, 0, 0])  # after a goal the previous probabilities are 0
    assert V.values(a, ps, pc)["value"][2] == pytest.approx(0.3 - 0.02)


def test_vaep_features_mirror_previous_opponent_action():
    a = V.actions(toy_match())
    X = V.features(a)
    assert X.shape == (5, len(V.FEATURES))
    f = {k: X[:, i] for i, k in enumerate(V.FEATURES)}
    assert f["a0_x0"][3] == pytest.approx(32.5)
    assert f["a1_x0"][3] == pytest.approx(-7.5)  # failed pass by the other team, seen from the shooter's side
    assert f["team_1"][3] == 0 and f["team_1"][2] == 1
    assert f["a1_x0"][0] == f["a0_x0"][0] and f["dt_1"][0] == 0  # first action: its own "previous"
    assert f["a1_x0"][4] == f["a0_x0"][4]  # a new half starts a new game state
    assert f["goals_against"][4] == 1 and f["goals_for"][3] == 0  # the score before the action


def test_vaep_lomo_held_out():
    tables = {mid: V.actions(toy_match(mid)) for mid in ("A", "B", "C")}
    out = V.lomo(tables)
    for mid, r in out.items():
        assert mid not in r["trained_on"] and len(r["trained_on"]) == 2
        assert len(r["value"]) == len(tables[mid]["type"])
        assert np.allclose(r["value"], V.values(tables[mid], r["p_score"], r["p_concede"])["value"])

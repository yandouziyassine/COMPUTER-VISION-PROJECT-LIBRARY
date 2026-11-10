"""Project 20, AI coach: model lab candidate features and the end-to-end score. Synthetic states only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20-football-analytics")))
from coach import advice as A  # noqa: E402
from coach import features as F  # noqa: E402
from coach import model_lab as L  # noqa: E402


def _state(def_pos, att_v=None):
    att = np.array([[0.0, 0.0], [20.0, 0.0], [10.0, 15.0]])
    return {
        "att": att,
        "att_v": np.zeros((3, 2)) if att_v is None else np.asarray(att_v, float),
        "def": np.atleast_2d(np.asarray(def_pos, float)),
        "def_v": np.zeros((len(np.atleast_2d(def_pos)), 2)),
        "att_idx": np.array([0, 1, 2]),
        "def_idx": np.arange(len(np.atleast_2d(def_pos))) + 3,
        "ball": att[0].copy(),
        "direction": 1,
    }


def test_receiver_moving_to_the_passer_has_positive_speed_to_ball_and_negative_forward_speed():
    s = _state([[-30.0, 30.0]], att_v=[[0, 0], [-4.0, 0.0], [0, 0]])
    e = L.extra_pass_features(s, 0, 1)
    assert e["receiver_to_ball"] == pytest.approx(4.0)
    assert e["receiver_vx"] == pytest.approx(-4.0)
    s2 = _state([[-30.0, 30.0]], att_v=[[0, 0], [0.0, 3.0], [0, 0]])  # running across the lane
    assert L.extra_pass_features(s2, 0, 1)["receiver_to_ball"] == pytest.approx(0.0)


def test_defender_on_the_lane_has_a_small_intercept_margin_and_a_far_one_the_clipped_maximum():
    # pitch control's 0.7 s reaction time: a defender on the lane at 10 m reaches it in 0.8 s, the ball in 0.67 s
    on_lane = L.extra_pass_features(_state([[10.0, 0.5]]), 0, 1)
    off_lane = L.extra_pass_features(_state([[10.0, 6.0]]), 0, 1)
    far = L.extra_pass_features(_state([[-40.0, 30.0]]), 0, 1)
    assert 0 < on_lane["intercept_margin"] < 0.5 and on_lane["intercept_margin"] < off_lane["intercept_margin"]
    assert far["intercept_margin"] == 3.0


def test_defenders_near_target_and_interactions():
    s = _state([[21.0, 1.0], [23.0, -2.0], [5.0, 30.0]])
    base = F.pass_features(s, 0, 1)
    e = L.extra_pass_features(s, 0, 1, base)
    assert e["defenders_near_target"] == 2
    assert e["dist_x_contested"] == pytest.approx(20.0 * (1 - base["target_control"]))
    assert e["forward_x_pressure"] == pytest.approx(20.0 * np.exp(-base["passer_pressure"] / 3.0))
    assert set(e) == set(L.EXTRA_FEATURES)


def test_no_defenders_gives_finite_features():
    s = _state(np.zeros((0, 2)))
    s["def"], s["def_v"] = np.zeros((0, 2)), np.zeros((0, 2))
    e = L.extra_pass_features(s, 0, 1, base={"distance": 20.0, "forward": 20.0, "target_control": 1.0,
                                             "passer_pressure": 50.0})  # fmt: skip
    assert e["intercept_margin"] == 3.0 and e["defenders_near_target"] == 0


def test_value_context_counts_goal_side_opponents_and_team_speed():
    ctx = L.value_context(
        [[0.0, 0.0], [30.0, 0.0]], [[10.0, 0.0], [40.0, 5.0], [-20.0, 0.0]], np.array([[2.0, 0], [4.0, 0]])
    )
    assert ctx.tolist() == [[2.0, 3.0], [1.0, 3.0]]
    V = L.value_rows([[40.0, 0.0], [0.0, 0.0]], [[45.0, 0.0]], np.array([[1.0, 0.0]]))
    I = L.interactions(V)
    assert V.shape == (2, 12) and I.shape == (2, 18)
    # in the box and in the final third: the zone x goal-side terms carry the count, the middle-third term is 0
    assert I[0, 12:15].tolist() == [1.0, 1.0, 0.0] and I[1, 12:15].tolist() == [0.0, 0.0, 1.0]


def test_chosen_score_equals_advice_options_for_the_pro_pass():
    s = _state([[10.0, 3.0], [25.0, -4.0], [5.0, 20.0]], att_v=[[1, 0], [-2.0, 1.0], [0, 3.0]])
    s["def_v"] = np.array([[0.5, 0.0], [-1.0, 0.0], [0.0, 0.0]])
    wp = np.linspace(-0.3, 0.3, len(F.PASS_FEATURES))
    wv = np.linspace(0.001, 0.002, len(F.VALUE_FEATURES))
    models = {
        "pass": lambda X: 1 / (1 + np.exp(-(np.atleast_2d(X) @ wp) / 10)),
        "value": lambda X: np.atleast_2d(X) @ wv,
        "arrival": lambda X: np.atleast_2d(X) @ (2 * wv),
        "shot": lambda X: np.full(len(np.atleast_2d(X)), 0.1),
    }
    opts = A.options(s, 0, models)
    for target in (1, 2):
        r = L.chosen_row(s, 0, target)
        p = models["pass"](r["pass"][: L.N_PASS])[0]
        got = L.chosen_score(
            p,
            models["arrival"](r["target"][: L.N_VALUE])[0],
            models["value"](r["loss"][: L.N_VALUE])[0],
            models["value"](r["now"][: L.N_VALUE])[0],
        )
        want = next(o for o in opts if o["kind"] == "pass" and o["target"] == target)["score"]
        assert got == pytest.approx(want, abs=1e-12)


def test_verdict_needs_every_ci_above_zero_and_no_leak():
    up = {"delta": 0.01, "ci": [0.001, 0.02]}
    across = {"delta": 0.01, "ci": [-0.001, 0.02]}
    assert L.verdict([up, up]) == "ADOPT"
    assert L.verdict([up, across]) == "KEEP CURRENT"
    assert L.verdict([up], leak_ok=False) == "KEEP CURRENT"


def test_boot_diff_on_identical_predictions_is_zero():
    g = np.repeat(["a", "b", "c"], 10)
    d = L.boot_diff(g, lambda i: float(len(i)), lambda i: float(len(i)), reps=20)
    assert d["delta"] == 0 and d["ci"] == [0.0, 0.0]


def test_final_verdicts_reject_non_portable_and_end_to_end_regressions():
    up = {"delta": 0.01, "ci": [0.001, 0.02]}
    down = {"delta": -0.01, "ci": [-0.02, -0.001]}
    res = {
        "verdicts": {
            "pass:logistic_extra": {"verdict": "ADOPT"},
            "pass:hgb_extra_raw": {"verdict": "ADOPT"},
            "value:ridge_zone_x_context": {"verdict": "ADOPT"},
            "value:hgb_context": {"verdict": "ADOPT"},
        },
        "end_to_end_linear": {"pass=logistic_extra": {"gain_spearman": down}},
        "end_to_end": {"value,arrival=ridge_zone_x_context": {"gain_spearman": up}},
    }
    v = L.final_verdicts(res)
    assert v["value:ridge_zone_x_context"]["verdict"] == "ADOPT"
    assert all(
        v[k]["verdict"] == "KEEP CURRENT" for k in ("pass:logistic_extra", "pass:hgb_extra_raw", "value:hgb_context")
    )

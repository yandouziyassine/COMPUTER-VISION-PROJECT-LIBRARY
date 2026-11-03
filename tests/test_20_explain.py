"""Project 20, AI coach: explanations and ghosts (coach insight, part B). Synthetic states only."""

import json
import math
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import advice as A  # noqa: E402
from coach import explain as EX  # noqa: E402
from coach import features as F  # noqa: E402
from coach import ghosts as GH  # noqa: E402
from coach import models as ML  # noqa: E402


def weights(names, coef, mean=None, scale=None, link="logit", intercept=0.5):
    k = len(names)
    return {
        "features": list(names),
        "link": link,
        "mean": list(mean or [0.0] * k),
        "scale": list(scale or [1.0] * k),
        "coef": list(coef),
        "intercept": intercept,
    }


def state(att, deff, att_v=None, def_v=None):
    att, deff = np.array(att, float), np.array(deff, float)
    return {
        "att": att,
        "att_v": np.zeros_like(att) if att_v is None else np.array(att_v, float),
        "def": deff,
        "def_v": np.zeros_like(deff) if def_v is None else np.array(def_v, float),
        "att_idx": np.arange(len(att)),
        "def_idx": np.arange(len(att), len(att) + len(deff)),
        "ball": att[0].copy(),
        "direction": 1,
    }


# ------------------------------------------------------------------ explanations


def test_fmt_rounds_half_away_from_zero_like_the_site():
    assert EX.fmt(2.5, 0) == "3" and EX.fmt(0.25, 1) == "0.3" and EX.fmt(6.25, 1) == "6.3"
    assert EX.fmt(31.4, 0) == "31" and EX.fmt(0.021, 3) == "0.021" and EX.fmt(0.0, 1) == "0.0"
    assert EX.fmt(-0.0004, 3) == "0.000" and EX.fmt(-1.26, 1) == "-1.3" and EX.fmt(0.05, 3) == "0.050"
    assert EX.signed(0.021) == "+0.021" and EX.signed(-0.0042) == "-0.004" and EX.signed(-0.0004) == "+0.000"


def test_contributions_are_exact_and_sum_to_the_logit():
    w = weights(F.PASS_FEATURES, np.linspace(-1, 1, 9), mean=list(range(9)), scale=[2.0] * 9, intercept=0.3)
    x = {f: float(i * 1.5) for i, f in enumerate(F.PASS_FEATURES)}
    cs = EX.contributions(w, x)
    assert [c["feature"] for c in cs] == list(F.PASS_FEATURES)
    assert cs[3]["contribution"] == pytest.approx(w["coef"][3] * (4.5 - 3) / 2.0)
    logit = w["intercept"] + sum(c["contribution"] for c in cs)
    p = ML.linear_predictor(w)(np.array([[x[f] for f in F.PASS_FEATURES]]))[0]
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(p, abs=1e-12)


def pass_weights():
    """Only receiver space (+) and lane defenders (-) matter, distance a little."""
    coef = {"receiver_space": 1.2, "lane_defenders": -0.9, "distance": -0.1}
    mean = {"receiver_space": 9.0, "lane_defenders": 0.3, "distance": 17.0}
    return weights(
        F.PASS_FEATURES,
        [coef.get(f, 0.0) for f in F.PASS_FEATURES],
        mean=[mean.get(f, 0.0) for f in F.PASS_FEATURES],
        scale=[6.0 if f == "receiver_space" else 1.0 for f in F.PASS_FEATURES],
    )


def test_pass_phrases_top_two_terms_and_value_split():
    feats = {f: 0.0 for f in F.PASS_FEATURES}
    feats.update({"receiver_space": 6.2, "lane_defenders": 1.0, "distance": 31.0})
    # contributions: space 1.2 * (6.2 - 9) / 6 = -0.56, lane -0.9 * 0.7 = -0.63, distance -0.1 * 14 = -1.4
    opt = {"kind": "pass", "target": 1, "p": 0.8, "value": 0.031, "v_lost": -0.006, "score": 0.0}
    ex = EX.explain(opt, 0.01, {"pass": pass_weights()}, feats)
    assert ex["phrases"] == [
        "long pass (31 m)",
        "1 defender in the lane",
        "+0.021 xG if it arrives, -0.004 if lost there",
    ]
    assert ex["signs"] == [-1, -1, 0] and not ex["fragile"]
    assert ex["value_terms"] == pytest.approx({"gain": 0.021, "loss": -0.004})
    feats.update({"distance": 10.0, "receiver_space": 20.0})  # short pass helps, big space helps most
    ex = EX.explain(opt, 0.01, {"pass": pass_weights()}, feats)
    assert ex["phrases"][:2] == ["receiver has 20.0 m of space", "short pass (10 m)"] and ex["signs"][:2] == [1, 1]


def test_phrase_variants_count_zero_split_and_degrees():
    w = pass_weights()
    T = EX.PHRASES["pass"]
    assert EX.feature_phrase(T, w, "lane_defenders", 0.0)[0] == "no defender in the lane"
    assert EX.feature_phrase(T, w, "lane_defenders", 3.0)[0] == "3 defenders in the lane"
    assert EX.feature_phrase(T, w, "forward", -12.4)[0] == "12 m backward"
    assert EX.feature_phrase(T, w, "target_x", -20.0)[0] == "receiver 20 m inside his own half"
    assert EX.feature_phrase(T, w, "abs_angle", math.pi / 4)[0] == "45° off straight ahead"
    assert EX.feature_phrase(T, w, "target_control", 0.874)[0] == "87 % pitch control at the receiver"
    assert EX.feature_phrase(T, w, "receiver_space", 6.25)[1] == pytest.approx(0.0, abs=1e-9)  # on a boundary


def test_shot_and_carry_explanations():
    w = {"shot": weights(F.SHOT_FEATURES, [-0.5, 0.3, -0.3], mean=[17.0, 0.4, 1.5], scale=[7.0, 0.24, 0.94])}
    sf = {"goal_dist": 30.0, "goal_angle": 0.2, "cone_defenders": 0.0}
    ex = EX.explain({"kind": "shot", "p": 0.031, "value": 1.0, "score": 0.02}, 0.011, w, sf)
    assert ex["phrases"] == [
        "shot from 30 m",
        "no defender in the shot cone",
        "+0.031 xG for the shot, +0.011 if the ball is kept here",
    ]
    carry = {"kind": "carry", "angle": 0, "p": 0.674, "value": 0.013, "v_lost": 0.004, "score": 0.0}
    ex = EX.explain(carry, 0.01, w)
    assert ex["phrases"] == ["67 % pitch control where the carry ends", "+0.003 xG if kept, -0.014 if lost there"]
    assert ex["contributions"] == [] and ex["intercept"] is None and ex["signs"] == [0, 0]


def advice_case():
    s = state([[20.0, 0.0], [45.0, 0.0], [5.0, 0.0], [30.0, 15.0]], [[20.0, 25.0], [0.0, -25.0], [44.0, 1.5]])
    w = {
        "pass": weights(F.PASS_FEATURES, [-0.05, 0, 0, -1.0, 0.1, 0, 2.0, 0, 0], intercept=0.0),
        "value": weights(F.VALUE_FEATURES, [0.002] + [0.0] * 9, link="identity", intercept=0.0),
        "shot": weights(F.SHOT_FEATURES, [-0.2, 0, 0], intercept=0.0),
    }
    return s, w, {k: ML.linear_predictor(v) for k, v in w.items()}


def test_explain_options_follow_the_advice_order_and_value_terms_rebuild_the_score():
    s, w, pred = advice_case()
    opts = A.options(s, 0, pred)
    v_now = float(A.value_at(pred, [s["att"][0]])[0])
    ex = EX.explain_options(s, 0, opts, w, v_now)
    assert [e["kind"] for e in ex] == [o["kind"] for o in opts] and {"pass", "carry", "shot"} == {
        o["kind"] for o in opts
    }
    for o, e in zip(opts, ex):
        assert 2 <= len(e["phrases"]) <= 3
        if o["kind"] == "shot":
            assert o["score"] == pytest.approx(e["value_terms"]["xg"] - e["value_terms"]["keep"])
        else:
            t = e["value_terms"]
            assert o["score"] == pytest.approx(o["p"] * t["gain"] + (1 - o["p"]) * t["loss"], abs=1e-12)
        if o["kind"] == "pass":
            z = w["pass"]["intercept"] + sum(c["contribution"] for c in e["contributions"])
            assert 1 / (1 + math.exp(-z)) == pytest.approx(o["p"], abs=1e-12)
    json.dumps(ex, allow_nan=False)


# ------------------------------------------------------------------ ghosts


def test_fan_points_follow_the_heading_and_stay_on_the_pitch():
    pts = GH.fan([0.0, 0.0], [0.0, 0.0])  # standing still: towards +x
    assert pts.shape == (GH.FAN_N, 2) and np.allclose(np.hypot(*pts.T), GH.REACH)
    assert np.allclose(pts[3], [GH.REACH, 0.0]) and np.allclose(pts[0], [0.0, -GH.REACH])
    pts = GH.fan([0.0, 0.0], [0.0, 4.0])  # running towards +y
    assert np.allclose(pts[3], [0.0, GH.REACH])
    assert np.all(GH.fan([50.0, 33.0], [3.0, 3.0]) <= [52.5, 34.0])


def marked_case():
    """Carrier at the centre, teammate 1 tightly marked; teammate 2 far back and free; a third teammate unmarked."""
    s = state([[0.0, 0.0], [15.0, 0.0], [-20.0, 10.0]], [[16.0, 0.5], [30.0, -20.0]])
    models = {
        "pass": lambda X: 1 / (1 + np.exp(-(X[:, F.PASS_FEATURES.index("receiver_space")] - 3.0))),
        "value": lambda X: 0.1 * (X[:, 0] + 52.5) / 105,  # >= 0: losing the ball anywhere costs
        "shot": lambda X: np.zeros(len(X)),
    }
    return s, models


def test_ghost_runs_free_the_marked_teammate():
    s, models = marked_case()
    runs = GH.ghost_runs(s, 0, models)
    assert runs and runs == GH.ghost_runs(s, 0, models)  # deterministic
    g = next(r for r in runs if r["player"] == 1)
    assert g["gain"] > GH.RUN_MIN_GAIN and g["score"] - g["score_now"] == pytest.approx(g["gain"])
    assert math.dist(g["to"], s["att"][1]) == pytest.approx(GH.REACH)
    pts = GH.fan(s["att"][1], s["att_v"][1])
    v_now = float(A.value_at(models, [s["att"][0]])[0])
    all_scores = GH.pass_scores([GH.moved(s, "att", 1, p) for p in pts], 0, 1, models, v_now)
    assert g["fan_index"] == int(np.argmax(all_scores)) and g["score"] == pytest.approx(all_scores.max())
    now = next(o for o in A.options(s, 0, models) if o["kind"] == "pass" and o["target"] == 1)
    assert g["score_now"] == pytest.approx(now["score"])  # same formula as advice.options
    assert [r["gain"] for r in runs] == sorted((r["gain"] for r in runs), reverse=True) and len(runs) <= GH.MAX_GHOSTS


def test_no_ghosts_when_position_does_not_matter():
    s, _ = marked_case()
    flat = {
        "pass": lambda X: np.full(len(X), 0.7),
        "value": lambda X: np.zeros(len(X)),
        "shot": lambda X: np.zeros(len(X)),
    }
    assert GH.ghost_runs(s, 0, flat) == [] and GH.ghost_defence(s, 0, flat) == []


def test_ghost_defence_is_greedy_and_lowers_the_best_option():
    s, models = marked_case()
    s["def"][0] = [15.0, 9.0]  # teammate 1 unmarked, a defender 9 m away can close him down
    dfn = GH.ghost_defence(s, 0, models)
    assert 1 <= len(dfn) <= GH.MAX_GHOSTS and dfn == GH.ghost_defence(s, 0, models)
    assert dfn[0]["best_before"] == pytest.approx(GH.best_score(s, 0, models))
    cur = s
    for a, b in zip(dfn, dfn[1:] + [None]):
        assert a["gain"] > GH.DEF_MIN_GAIN and a["best_after"] == pytest.approx(a["best_before"] - a["gain"])
        cur = GH.moved(cur, "def", a["row"], a["to"])
        assert GH.best_score(cur, 0, models) == pytest.approx(a["best_after"])
        if b is not None:
            assert b["best_before"] == a["best_after"] and b["row"] != a["row"]
    assert len({d["row"] for d in dfn}) == len(dfn)

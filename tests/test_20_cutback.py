"""Project 20, AI coach: the cut-back fix (arrival value model). Synthetic data only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import advice as A  # noqa: E402
from coach import datasets as D  # noqa: E402
from coach import export as X  # noqa: E402
from coach import features as F  # noqa: E402
from coach import ghosts as GH  # noqa: E402
from test_20_coach import _pass, _shot, tiny_match  # noqa: E402


def _state():
    return {
        "att": np.array([[46.0, 15.0], [47.0, 2.0], [30.0, -10.0]]),
        "att_v": np.zeros((3, 2)),
        "def": np.array([[44.0, 8.0], [20.0, 20.0]]),
        "def_v": np.zeros((2, 2)),
        "att_idx": np.array([0, 1, 2]),
        "ball": np.array([46.0, 15.0]),
        "direction": 1,
    }


def test_arrival_rows_rebuild_the_value_features_of_the_target_from_the_pass_features():
    s = _state()
    rows = [F.pass_features(s, 0, j) for j in (1, 2)]
    X_pass = np.array([[r[k] for k in F.PASS_FEATURES] for r in rows])
    got = D.arrival_rows(X_pass)
    want = np.array([[F.value_features(s["att"][j])[k] for k in F.VALUE_FEATURES] for j in (1, 2)])
    assert got.shape == (2, len(F.VALUE_FEATURES)) and np.allclose(got, want, atol=1e-5)


def test_arrival_table_has_one_row_per_completed_pass_labelled_with_the_net_xg_after_the_release():
    m = tiny_match(frames=600)
    for h in m.halves:
        h.owner[:] = 0
    m.halves[0].xy[:, 1] = [20.0, 5.0]  # teammate b: his position is the pass target's
    m.events = [
        _pass(m, 10, recipient=1, success=True),
        _pass(m, 200, recipient=1, success=True),
        _shot(m, 300, 0, xg=0.3),  # 12 s: inside 15 s of the first pass (0.4 s) and of the second (8 s)
        _pass(m, 320, recipient=1, success=True),  # after the shot: label 0
        _shot(m, 500, 1, xg=0.1),  # against the team, 20 s: within 15 s of the second pass (-0.1) and the third
    ]
    pt = D.pass_table(m)
    a = D.arrival_table(m, pt)
    assert list(a["match"]) == ["TEST"] * 3 and a["X"].shape == (3, len(F.VALUE_FEATURES))
    assert np.allclose(a["y"], [0.3, 0.2, -0.1])
    b = F.state(m, 0, 10, 0)["att"][1]  # the target in the normalised frame, as the coach sees him
    assert np.allclose(a["X"][0], [F.value_features(b)[k] for k in F.VALUE_FEATURES], atol=1e-5)
    m.events[1]["success"] = False  # a failed pass is not an arrival
    assert len(D.arrival_table(m)["y"]) == 2


def test_advice_values_the_pass_target_with_the_arrival_model_only():
    s = _state()
    sig = lambda w: lambda X_: 1 / (1 + np.exp(-(X_ @ w)))
    base = {
        "pass": sig(np.r_[-0.05, 0, 0, -1.0, 0.05, 0, 2.0, 0, 0]),
        "value": lambda X_: 0.1 * X_[:, 0] / 52.5,
        "shot": sig(np.r_[-0.2, 0, 0]),
    }
    arr = lambda X_: 0.5 * X_[:, 0] / 52.5  # ball arriving is worth more than a ball held at the same spot
    plain = A.options(s, 0, base)
    both = A.options(s, 0, {**base, "arrival": arr})
    pa = {o["target"]: o for o in plain if o["kind"] == "pass"}
    pb = {o["target"]: o for o in both if o["kind"] == "pass"}
    for j, o in pb.items():
        x = s["att"][list(s["att_idx"]).index(j)][0]
        assert o["value"] == pytest.approx(0.5 * x / 52.5) and pa[j]["value"] == pytest.approx(0.1 * x / 52.5)
        assert o["v_lost"] == pytest.approx(pa[j]["v_lost"]) and o["p"] == pytest.approx(pa[j]["p"])
    carries = lambda opts: {o["angle"]: o["value"] for o in opts if o["kind"] == "carry"}
    assert carries(both) == pytest.approx(carries(plain))  # carries and the shot keep the possession value V
    shot = lambda opts: next(o for o in opts if o["kind"] == "shot")
    assert shot(both)["score"] == pytest.approx(shot(plain)["score"])


def test_ghost_pass_scores_use_the_same_arrival_value_as_the_options():
    s = _state()
    models = {
        "pass": lambda X_: np.full(len(X_), 0.7),
        "value": lambda X_: 0.1 * X_[:, 0] / 52.5,
        "arrival": lambda X_: 0.5 * X_[:, 0] / 52.5,
        "shot": lambda X_: np.zeros(len(X_)),
    }
    v_now = float(A.value_at(models, [s["att"][0]])[0])
    o = next(o for o in A.options(s, 0, models) if o["kind"] == "pass" and o["target"] == 1)
    got = GH.pass_scores([s], 0, 1, models, v_now)[0]
    assert got == pytest.approx(o["score"])


def test_fit_models_and_export_carry_the_arrival_model(tmp_path, monkeypatch):
    import json

    nv = len(F.VALUE_FEATURES)

    def tables(m):
        rng = np.random.default_rng(ord(m.match_id))
        mk = lambda n, k, y: {"X": rng.normal(size=(n, k)), "y": y, "match": np.array([m.match_id] * n)}
        return (
            {**mk(60, len(F.PASS_FEATURES), np.arange(60) % 2), "weight": np.ones(60)},
            mk(60, nv, rng.normal(size=60)),
            {**mk(40, len(F.SHOT_FEATURES), np.arange(40) % 2), "penalty": np.zeros(40, bool)},
            mk(30, nv, rng.normal(size=30)),
        )

    monkeypatch.setattr(X.D, "pass_table", lambda m: tables(m)[0])
    monkeypatch.setattr(X.D, "value_table", lambda m: tables(m)[1])
    monkeypatch.setattr(X.D, "shot_table", lambda m: tables(m)[2])
    monkeypatch.setattr(X.D, "arrival_table", lambda m, pt=None: tables(m)[3])
    ms = [tiny_match(), tiny_match(), tiny_match()]
    for m, mid in zip(ms, "ABC"):
        m.match_id = mid
    X.export_models(tmp_path, ms, exclude="B")
    w = json.loads((tmp_path / "models.json").read_text(encoding="utf-8"))
    assert w["arrival"]["link"] == "identity" and w["arrival"]["features"] == list(F.VALUE_FEATURES)
    assert w["arrival"]["basis"] == w["value"]["basis"] and w["arrival"]["horizon_s"] == D.HORIZON
    assert w["arrival"]["coef"] != w["value"]["coef"] and "arrival" in w["advice"]["score"]
    assert set(X.fit_models(X.training_tables(ms[:2]))) == {"pass", "value", "shot", "arrival"}


from coach import cutback as CB  # noqa: E402


def test_rescore_rows_changes_only_the_target_value_and_is_idempotent():
    rows = [
        {
            "match": "A",
            "event": 1,
            "coach_p": 0.6,
            "v_now": 0.03,
            "coach_dv": 0.0,
            "coach_score": -0.012,
            "coach_after": 0.018,
        },
        {
            "match": "A",
            "event": 2,
            "coach_p": 0.9,
            "v_now": 0.00,
            "coach_dv": 0.01,
            "coach_score": 0.002,
            "coach_after": 0.002,
        },
    ]
    arrival = {("A", 1): 0.10, ("A", 2): 0.01}
    new = CB.rescore_rows(rows, arrival)
    assert new[0]["coach_score"] == pytest.approx(-0.012 + 0.6 * (0.10 - 0.03))  # V(target) 0.03 -> 0.10
    assert new[0]["coach_dv"] == pytest.approx(0.07) and new[0]["coach_after"] == pytest.approx(0.018 + 0.042)
    assert new[1]["coach_score"] == pytest.approx(0.002) and new[1]["coach_score_v3"] == 0.002  # no change
    assert rows[0]["coach_score"] == -0.012  # the input is left alone
    twice = CB.rescore_rows(new, {("A", 1): 0.05, ("A", 2): 0.01})  # starts again from the v3 values
    assert twice[0]["coach_score"] == pytest.approx(-0.012 + 0.6 * 0.02) and twice[0]["coach_score_v3"] == -0.012


def test_lomo_never_fits_on_the_held_out_match_and_honours_the_mask():
    seen = []

    class Const:
        def __init__(self, v):
            self.v = v

        def predict(self, X_):
            return np.full(len(X_), self.v)

    def fit(X_, y_):
        seen.append(sorted(set(X_[:, 0])))
        return Const(float(np.mean(y_)))

    groups = np.array(list("AABBCC"))
    X_ = np.array([[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    y_ = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 99.0])
    out = CB.lomo(groups, X_, y_, fit, mask=np.array([1, 1, 1, 1, 1, 0], bool))
    assert seen == [[2.0, 3.0], [1.0, 3.0], [1.0, 2.0]]  # held-out group absent from every fit
    assert np.allclose(out, [19 / 3, 19 / 3, 11 / 3, 11 / 3, 3.0, 3.0])  # C's 99 is masked out of the other folds
    lo, hi = CB.match_bootstrap(groups, lambda i: float(np.mean(y_[i])), reps=50)
    assert lo <= hi

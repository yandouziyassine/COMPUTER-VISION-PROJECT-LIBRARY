"""Project 20, AI coach: match report (decision quality, missed opportunities). Synthetic matches only."""

import json
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import datasets as D  # noqa: E402
from coach import evaluate as EV  # noqa: E402
from coach import export as X  # noqa: E402
from coach import export_report as XR  # noqa: E402
from coach import match as M  # noqa: E402
from coach import report as R  # noqa: E402


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


def test_site_step_matches_the_export_frame_indexing():
    assert R.site_step(25.0) == 2 and R.site_step(10.0) == 1 and R.site_step(25.0, site_fps=25.0) == 1
    for f in (0, 1, 40, 41, 12345):
        assert f // R.site_step(25.0) == X._site_event({"frame": f}, 2)["frame"]


def opt(kind, score, p, **kw):
    return {"kind": kind, "score": score, "p": p, **kw}


def record(opts, chosen_target, frame=41, half=0, success=True, net=0.01, carrier_x=0.0, t=1001.64):
    """An evaluate.pass_decisions record built by hand."""
    opts = sorted(opts, key=lambda o: -o["score"])
    chosen = next(o for o in opts if o["kind"] == "pass" and o["target"] == chosen_target)
    e = {"half": half, "frame": frame, "team": 0, "player": 7, "success": success}
    s = {"att": np.array([[carrier_x, 0.0]])}
    sample = {"index": 3, "event": e, "state": s, "carrier": 0}
    return {"sample": sample, "options": opts, "chosen": chosen, "t": t, "outcome": net}


NO_SHOTS = [np.zeros((0, 3)), np.zeros((0, 3))]


def test_decision_gap_rank_best_and_frames():
    opts = [
        opt("pass", 0.02, 0.6, target=1),
        opt("carry", 0.01, 0.9, angle=30),
        opt("pass", -0.005, 0.95, target=2),
        opt("shot", -0.03, 0.04),
    ]
    row = R.decision_row(record(opts, chosen_target=2), step=2, shots=NO_SHOTS)
    assert row["gap"] == pytest.approx(0.025) and row["rank"] == 3 and row["options"] == 4
    assert row["best"] == {"kind": "pass", "target": 1} and row["p_best"] == 0.6 and row["p_chosen"] == 0.95
    assert (row["frame_src"], row["frame"], row["carrier"], row["target"], row["event"]) == (41, 20, 7, 2, 3)
    assert row["above_median"] == pytest.approx(-0.005 - 0.0025)  # median of 0.02, 0.01, -0.005, -0.03
    assert row["third"] == "middle" and row["outcome"] == "kept" and row["net_xg"] == 0.01


def test_decision_chosen_best_has_zero_gap_and_a_carry_is_keyed_by_angle():
    opts = [opt("pass", 0.03, 0.7, target=4), opt("carry", 0.01, 0.9, angle=-60)]
    row = R.decision_row(record(opts, 4, carrier_x=30.0), 2, NO_SHOTS)
    assert row["gap"] == 0 and row["rank"] == 1 and row["third"] == "attacking"
    opts = [opt("pass", 0.0, 0.7, target=4), opt("carry", 0.01, 0.9, angle=-60)]
    row = R.decision_row(record(opts, 4, carrier_x=-30.0), 2, NO_SHOTS)
    assert row["best"] == {"kind": "carry", "angle": -60} and row["gap"] == pytest.approx(0.01)
    assert row["third"] == "defensive"


def test_gap_is_never_negative_and_rank_counts_better_options():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(2, 12))
        opts = [opt("pass", float(rng.normal(0, 0.02)), float(rng.random()), target=j) for j in range(k)]
        tgt = int(rng.integers(0, k))
        row = R.decision_row(record(opts, tgt), 2, NO_SHOTS)
        assert row["gap"] >= 0 and row["rank"] == 1 + sum(o["score"] > opts[tgt]["score"] for o in opts)


def test_outcome_goal_shot_kept_lost_within_the_horizon():
    shots = np.array([[105.0, 0, 0], [110.0, 1, 1], [120.0, 0, 1]])  # time, team, goal
    assert R.outcome(shots, 100.0, 0, True) == "shot"  # own shot at +5 s; own goal at +20 s is too late
    assert R.outcome(shots, 106.0, 0, False) == "goal"  # own goal at +14 s
    assert R.outcome(shots, 107.0, 0, True, horizon=10.0) == "kept"  # only the opponent's goal in (107, 117]
    assert R.outcome(shots, 121.0, 0, True) == "kept" and R.outcome(shots, 121.0, 0, False) == "lost"
    assert R.outcome(shots, 100.0, 1, False) == "goal" and R.outcome(np.zeros((0, 3)), 0.0, 0, True) == "kept"


def row(carrier=0, team=0, rank=1, gap=0.0, third="middle", p_best=0.8, above=0.01, kind="pass"):
    return {
        "carrier": carrier,
        "team": team,
        "rank": rank,
        "gap": gap,
        "third": third,
        "p_best": p_best,
        "above_median": above,
        "best": {"kind": kind},
    }


def test_summarise_shares_gaps_and_too_few():
    rows = [row(rank=1), row(rank=2, gap=0.01), row(rank=3, gap=0.02), row(rank=5, gap=0.03)]
    expected = {"n": 4, "too_few": True, "top1": 0.25, "top3": 0.75, "mean_gap": 0.015, "total_gap": 0.06}
    assert R.summarise(rows) == expected
    many = R.summarise(rows * 5)
    assert many["n"] == 20 and many["too_few"] is False and many["total_gap"] == pytest.approx(0.3)
    assert R.summarise([])["n"] == 0 and R.summarise([])["top1"] is None and R.summarise([])["too_few"] is True


def test_players_and_teams_aggregate_with_thirds():
    m = small_match()
    rows = [row(carrier=0)] * 25 + [row(carrier=1, rank=4, gap=0.02, third="attacking")] * 5
    rows += [row(carrier=3, team=1, third="defensive")] * 2
    players = R.by_player(m, rows)
    assert set(players) == {"0", "1", "3"} and players["0"]["n"] == 25 and players["0"]["too_few"] is False
    assert players["1"]["too_few"] is True and players["1"]["total_gap"] == pytest.approx(0.1)
    assert (players["0"]["name"], players["1"]["shirt"], players["3"]["team"]) == ("A", 2, 1)
    teams = R.by_team(m, rows)
    assert teams["0"]["n"] == 30 and teams["0"]["name"] == "Home FC" and teams["1"]["n"] == 2
    assert teams["0"]["by_third"]["attacking"]["n"] == 5 and teams["0"]["by_third"]["defensive"]["n"] == 0
    assert teams["0"]["top1"] == pytest.approx(25 / 30, abs=1e-3)


def test_highlights_filter_on_best_p_and_rank_one():
    rows = [
        row(rank=2, gap=0.05, p_best=0.3),  # the coach asks for a miracle: not a missed opportunity
        row(rank=2, gap=0.04, p_best=0.6),
        row(rank=1, gap=0.0, above=0.02),
        row(rank=3, gap=0.01, p_best=0.5),
        row(rank=1, above=0.05),
        row(rank=2, gap=0.03, p_best=0.9, above=0.09),  # far above the median but not the coach's best
    ]
    h = R.highlights(rows)
    assert [e["decision"] for e in h["missed"]] == [1, 5, 3] and h["missed"][0]["gap"] == 0.04
    assert [e["decision"] for e in h["best"]] == [4, 2]
    big = [row(rank=2, gap=0.001 * i, p_best=0.9) for i in range(1, 30)]
    assert len(R.highlights(big)["missed"]) == 10 and R.highlights(big)["missed"][0]["gap"] == pytest.approx(0.029)


def test_build_report_on_a_synthetic_match():
    rep = R.build(small_match(), MODELS, trained_on=["OTHER"])
    assert rep["trained_without"] == "SYN" and rep["fps"] == 12.5 and rep["fps_src"] == 25.0
    d0, d1 = rep["decisions"]
    assert (d0["half"], d0["frame_src"], d0["frame"], d0["event"], d0["outcome"]) == (0, 41, 20, 0, "goal")
    assert (d1["half"], d1["frame"], d1["carrier"], d1["target"], d1["outcome"]) == (1, 30, 1, 2, "kept")
    assert all(d["gap"] >= 0 and 1 <= d["rank"] <= d["options"] for d in rep["decisions"])
    assert rep["players"]["0"]["n"] == 1 and rep["players"]["0"]["too_few"] is True
    assert rep["summary"]["n"] == 2 and rep["summary"]["open_play_passes"] == 2 and len(rep["caveats"]) >= 4
    assert set(rep["highlights"]) == {"missed", "best"} and len(json.dumps(rep)) < 300_000


def test_build_refuses_models_trained_on_the_match():
    with pytest.raises(AssertionError):
        R.build(small_match(), MODELS, trained_on=["A", "SYN"])


def fake_tables(mid, rng):
    nP, nV, nS = len(R.F.PASS_FEATURES), len(R.F.VALUE_FEATURES), len(R.F.SHOT_FEATURES)
    Xp, Xs = rng.normal(size=(80, nP)), rng.normal(size=(30, nS))
    ids = lambda n: np.array([mid] * n)
    return {
        "pass": {"X": Xp, "y": (Xp[:, 0] > 0).astype(int), "match": ids(80), "weight": np.ones(80)},
        "value": {"X": rng.normal(size=(80, nV)), "y": rng.normal(size=80), "match": ids(80)},
        "shot": {"X": Xs, "y": (Xs[:, 0] > 0).astype(int), "match": ids(30), "penalty": np.zeros(30, bool)},
    }


def test_held_out_models_never_see_the_target_match():
    rng = np.random.default_rng(0)
    tables = {mid: fake_tables(mid, rng) for mid in ("A", "B", "C")}
    Xe = rng.normal(size=(40, len(R.F.SHOT_FEATURES)))
    extra = {"X": Xe, "y": (Xe[:, 1] > 0).astype(int), "match": np.array(["B"] * 10 + ["s1"] * 30)}
    models, fitted, trained_on = R.held_out_models("B", tables, extra)
    assert trained_on == ["A", "C", "s1"] and set(models) == {"pass", "value", "shot"}
    ref = X.fit_models(
        {k: D.stack([tables["A"][k], tables["C"][k]]) for k in ("pass", "value", "shot")},
        {k: v[10:] for k, v in extra.items()},
    )
    for k in ("pass", "shot"):
        Xk = tables["B"][k]["X"]
        assert np.allclose(fitted[k].predict_proba(Xk), ref[k].predict_proba(Xk))
    assert np.allclose(models["value"](tables["B"]["value"]["X"]), ref["value"].predict(tables["B"]["value"]["X"]))


def test_report_frames_match_the_site_export_and_the_size_limit(tmp_path, monkeypatch):
    m = small_match()
    X.pack_match(m, tmp_path)  # the real site export: match.json with events at 12.5 fps
    site = json.loads((tmp_path / "match.json").read_text(encoding="utf-8"))
    rep = R.build(m, MODELS, trained_on=["OTHER"])
    XR.check_against_site(rep, site)
    assert [site["events"][d["event"]]["frame"] for d in rep["decisions"]] == [20, 30]
    site["events"][2]["frame"] = 31
    with pytest.raises(AssertionError):
        XR.check_against_site(rep, site)
    path = XR.write(rep, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["decisions"] == rep["decisions"]
    monkeypatch.setattr(XR, "MAX_BYTES", 100)
    with pytest.raises(AssertionError):
        XR.write(rep, tmp_path)


def test_held_out_summary_pools_matches():
    reps = [R.build(small_match(match_id=mid), MODELS, trained_on=["OTHER"]) for mid in ("A1", "A2")]
    s = R.held_out_summary(reps)
    assert set(s["matches"]) == {"A1", "A2"} and s["pooled"]["n"] == 4 and s["pooled"]["open_play_passes"] == 4
    assert sum(s["rank_counts"].values()) == 4 and sum(s["gap_histogram"]["counts"]) == 4
    assert s["pooled"]["outcome_share"] == {"goal": 0.5, "shot": 0.0, "kept": 0.5, "lost": 0.0}

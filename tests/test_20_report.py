"""Project 20, AI coach: match report (decision quality, missed opportunities). Synthetic matches only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import evaluate as EV  # noqa: E402
from coach import export as X  # noqa: E402
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

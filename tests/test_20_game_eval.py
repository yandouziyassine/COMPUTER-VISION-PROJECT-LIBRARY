"""Project 20, Playmaker content audit (coach/game/evaluate_content.py): synthetic candidates only, no D: access."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from conftest import project_dir

sys.path.insert(0, str(project_dir("20-football-analytics")))

from coach.game import evaluate_content as EC  # noqa: E402


def moment(points, scores=None, pro=0, ai=None, shot=True, kind="kept", net_xg=0.0, match="M1", league="A-League"):
    """Carrier at (0, 0); 3 teammates: k0 near (5 m, x=5), k1 far forward (x=40), k2 far back (x=-30);
    k3 carry angle 0, k4 carry angle 60, k5 shot (if shot). points/scores are per option in k order."""
    players = [{"side": 0, "shirt": i + 1, "name": f"P{i}"} for i in range(4)]
    release = [[0, 0], [500, 0], [4000, 1000], [-3000, 0]]
    opts = [
        {"k": 0, "kind": "pass", "player": 1, "to": release[1]},
        {"k": 1, "kind": "pass", "player": 2, "to": release[2]},
        {"k": 2, "kind": "pass", "player": 3, "to": release[3]},
        {"k": 3, "kind": "carry", "angle": 0, "to": [700, 0]},
        {"k": 4, "kind": "carry", "angle": 60, "to": [350, 600]},
    ]
    if shot:
        opts.append({"k": 5, "kind": "shot", "to": [5250, 0]})
    scores = scores if scores is not None else [-i for i in range(len(opts))]
    public = {
        "id": f"{match}-0-{len(points)}",
        "league": league,
        "carrier": 0,
        "players": players,
        "lead_up": [release],
        "ball": [[0, 0]],
        "options": opts,
    }
    answer = {
        "options": [{"k": o["k"], "score": scores[o["k"]], "points": points[o["k"]]} for o in opts],
        "ai_pick": int(np.argmax(scores)) if ai is None else ai,
        "pro_pick": pro,
        "outcome": {"kind": kind, "net_xg_15s": net_xg},
    }
    return public, answer


def test_bucket_of():
    assert [EC.bucket_of(p) for p in (100, 99, 50, 49, 0)] == ["100", "50-99", "50-99", "0-49", "0-49"]


def test_strategy_pickers_read_geometry():
    pub, _ = moment([100, 90, 72, 54, 36, 18])
    assert EC.pick_most_advanced(pub) == 1
    assert EC.pick_nearest(pub) == 0
    assert EC.pick_longest(pub) == 1  # 41.2 m vs 30 m
    assert EC.pick_shoot(pub) == 5
    assert EC.pick_carry_forward(pub) == 3
    no_shot, _ = moment([100, 90, 72, 54, 36], shot=False)
    assert EC.pick_shoot(no_shot) == 1  # falls back to the most advanced pass


def test_pickers_fall_back_to_option_target_when_tracking_missing():
    pub, _ = moment([100, 90, 72, 54, 36, 18])
    pub["lead_up"][-1][2] = None  # far-forward teammate not tracked at release: use the option's "to"
    assert EC.pick_most_advanced(pub) == 1
    pub["options"][1]["to"] = [-5000, 0]
    assert EC.pick_most_advanced(pub) == 0


def test_strategy_table_and_exploit_flag():
    # shot always 100, the pro's pass never: "shoot_if_possible" is an exploit, the pro is not flagged
    ms = [moment([30, 60, 0, 90, 45, 100], pro=0, ai=5) for _ in range(4)]
    t = EC.strategy_table(ms)
    assert t["shoot_if_possible"] == {"mean": 100.0, "full": 1.0, "exploit": True}
    assert t["pro"]["mean"] == 30.0 and not t["pro"]["exploit"]
    assert t["ai"]["mean"] == 100.0 and not t["ai"]["exploit"]
    assert t["random"]["mean"] == pytest.approx(np.mean([30, 60, 0, 90, 45, 100]), abs=0.01)
    assert t["random"]["full"] == pytest.approx(1 / 6, abs=1e-4)
    assert t["most_advanced_pass"]["mean"] == 60.0 and not t["most_advanced_pass"]["exploit"]
    assert t["carry_forward"]["mean"] == 90.0 and t["carry_forward"]["exploit"]  # mean >= 70


def test_close_calls():
    ms = [moment([100, 100, 0, 50, 20, 10], pro=1), moment([100, 0, 10, 20, 30, 40], pro=2)]
    c = EC.close_calls(ms)
    assert c["n_options_at_100"] == {"1": 1, "2": 1}
    assert c["mean_options_at_100"] == 1.5
    assert c["pro_scores_100"] == 0.5


def rows_from(spec):
    """spec: [(match, points, score, net_xg, kind)] -> candidate rows via candidate_row."""
    out = []
    for m, pts, sc, xg, kind in spec:
        pub, ans = moment([pts, 0, 0, 0, 0, 0], scores=[sc, -1, -2, -3, -4, -5], kind=kind, net_xg=xg, match=m)
        out.append(EC.candidate_row({"public": pub, "answer": ans, "features": {"match": m}}))
    return out


def test_bucket_stats():
    rows = rows_from(
        [
            ("A", 100, 0.03, 0.10, "shot"),
            ("A", 100, 0.02, 0.00, "lost"),
            ("A", 70, 0.01, 0.02, "kept"),
            ("B", 20, 0.00, -0.02, "lost"),
            ("B", 10, -0.01, 0.00, "goal"),
        ]
    )
    s = EC.bucket_stats(rows)
    assert s["100"]["n"] == 2 and s["100"]["net_xg"] == pytest.approx(0.05)
    assert s["100"]["p_shot"] == 0.5 and s["100"]["completion"] == 0.5
    assert s["50-99"] == {"n": 1, "net_xg": pytest.approx(0.02), "p_shot": 0.0, "completion": 1.0}
    assert s["0-49"]["n"] == 2 and s["0-49"]["p_shot"] == 0.5 and s["0-49"]["completion"] == 0.5


def test_match_bootstrap_resamples_whole_matches():
    # every row of match A has net xG 1, of match B 0: a match resample can only give means k/n_matches-ish values
    rows = rows_from([("A", 100, 0.1, 1.0, "kept")] * 3 + [("B", 10, 0.0, 0.0, "kept")] * 3)
    ci = EC.match_bootstrap(rows, lambda a: float(a["net_xg"].mean()), n_boot=200, seed=1)
    assert ci["value"] == 0.5
    assert ci["lo"] == 0.0 and ci["hi"] == 1.0
    again = EC.match_bootstrap(rows, lambda a: float(a["net_xg"].mean()), n_boot=200, seed=1)
    assert again == ci  # fixed seed


def test_validity_spearman_and_cis():
    rng = np.random.default_rng(0)
    spec = []
    for m in "ABCDEF":
        for _ in range(30):
            pts = int(rng.integers(0, 101))
            spec.append((m, pts, pts / 1000, pts / 1000 + rng.normal(0, 0.01), "kept"))
    v = EC.validity(rows_from(spec), n_boot=100, seed=0)
    assert v["n"] == 180 and v["matches"] == 6
    sp = v["spearman_points_netxg"]
    assert sp["value"] > 0.8 and sp["lo"] <= sp["value"] <= sp["hi"]
    assert v["spearman_score_netxg"]["value"] == pytest.approx(sp["value"], abs=0.05)
    for b in EC.BUCKETS:
        lo, hi = v["buckets"][b]["net_xg_ci"]
        assert lo <= v["buckets"][b]["net_xg"] <= hi


def test_rescore_and_trivial_hit():
    pub, ans = moment([100, 90, 60, 30, 0, 100], scores=[5, 4, 3, 2, 1, 6])
    assert EC.trivial_hit(pub, ans)  # nearest (k0) and shot (k5) score 100
    pub2, ans2 = moment([0, 30, 100, 60, 90, 45], scores=[1, 2, 6, 3, 5, 0])
    assert not EC.trivial_hit(pub2, ans2)
    same = EC.rescore(ans, 1.0)
    assert [o["points"] for o in same["options"]] == [100, 90, 60, 30, 0, 100]
    steep = EC.rescore(ans, 2.0)
    assert [o["points"] for o in steep["options"]] == [100, 90, 40, 10, 0, 100]

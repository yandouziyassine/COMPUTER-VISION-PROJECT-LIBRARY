"""Project 20, Playmaker content (the daily football decision game). Synthetic matches only."""

import json
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
jsonschema = pytest.importorskip("jsonschema")

from conftest import project_dir

sys.path.insert(0, str(project_dir("20-football-analytics")))

from coach import match as M  # noqa: E402
from coach import stamp  # noqa: E402


def test_generated_by_has_commit_and_date():
    g = stamp.generated_by()
    assert set(g) == {"commit", "date"}
    assert len(g["commit"]) in (7, 40) or g["commit"] == "unknown"
    assert len(g["date"]) == 10


from coach import evaluate as EV  # noqa: E402
from coach import report as R  # noqa: E402
from coach.game import inputs as GI  # noqa: E402
from test_20_report import fake_tables, small_match  # noqa: E402


def eleven_match(match_id="SYA", frames=300):
    """11 v 11 standing still at 25 fps, two halves (home attacks +x in half 0, -x in half 1). Events: a completed
    pass 0 -> 1 at frame 120 of half 0 (ball at player 0), a home goal 1.6 s later, a completed pass 1 -> 2 at frame
    150 of half 1 (ball at player 1)."""
    rng = np.random.default_rng(5)
    home = np.column_stack([rng.uniform(-40, 30, 11), rng.uniform(-30, 30, 11)])
    home[0], home[1], home[2] = [5.0, 0.0], [20.0, 10.0], [25.0, -12.0]
    guest = np.column_stack([rng.uniform(-30, 45, 11), rng.uniform(-30, 30, 11)])
    halves = []
    for h in range(2):
        xy = np.repeat(np.concatenate([home, guest])[None], frames, axis=0).astype(np.float32)
        ball = np.zeros((frames, 3), np.float32)
        ball[:, :2] = home[0] if h == 0 else home[1]
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
        [f"p{i}" for i in range(22)],
        np.array([0] * 11 + [1] * 11, np.int8),
        np.arange(1, 23, dtype=np.int16),
        [f"Player {i}" for i in range(22)],
        halves,
        [],
        {"home": "Home FC", "guest": "Away FC"},
    )

    def ev(kind, half, frame, **kw):
        base = {"type": kind, "half": half, "frame": frame, "t": float(m.halves[half].t[frame]), "team": 0}
        return {**base, "player": 0, "set_piece": None, "cross": False, "x": 0.0, "y": 0.0, **kw}

    m.events = [
        ev("pass", 0, 120, recipient=1, success=True),
        ev("shot", 0, 160, goal=True, xg=0.3),
        ev("pass", 1, 150, player=1, recipient=2, success=True),
    ]
    return m


@pytest.fixture(scope="module")
def two_matches():
    """A synthetic match to score (SYA) and random training tables for three matches (models need both classes)."""
    a, b = eleven_match("SYA"), small_match(match_id="SYB")
    rng = np.random.default_rng(3)
    tables = {}
    for mid in ("SYA", "SYB", "SYC"):
        t = fake_tables(mid, rng)
        t["arrival"] = {
            "X": rng.normal(size=(40, len(R.F.VALUE_FEATURES))),
            "y": rng.normal(size=40),
            "match": t["value"]["match"][:40],
        }
        tables[mid] = t
    return a, b, tables


def test_fast_scores_equal_advice_options(two_matches):
    a, _, tables = two_matches
    models, _, _ = R.held_out_models("SYA", tables)
    n = 0
    for d in EV.pass_decisions(a, models):
        p = d["sample"]
        inp = GI.inputs(p["state"], p["carrier"])
        fast = GI.scores(inp, models)
        by_key = {GI.key(o): o["score"] for o in d["options"]}
        assert [GI.key(o) for o in GI.options(inp)] == sorted(by_key, key=GI.order)
        for o, s in zip(GI.options(inp), fast):
            assert s == pytest.approx(by_key[GI.key(o)], abs=1e-12)
        n += 1
    assert n > 0


from coach.game import scoring as GS  # noqa: E402


def test_share_top_counts_ties_for_every_tied_option():
    refit = np.array([[0.1, 0.3, 0.2], [0.3, 0.3, 0.0], [0.0, 0.1, 0.2], [0.5, 0.1, 0.1]])
    assert GS.share_top(refit).tolist() == [0.5, 0.5, 0.25]


def test_points_full_for_main_top_and_close_calls_linear_for_the_rest():
    main = np.array([0.30, 0.25, 0.10, 0.05, -0.20])
    share = np.array([0.6, 0.25, 0.1, 0.05, 0.0])
    assert GS.points(main, share).tolist() == [100, 100, 90, 45, 0]


def test_points_single_remaining_option_gets_45():
    assert GS.points(np.array([0.3, 0.1]), np.array([1.0, 0.0])).tolist() == [100, 45]


def test_points_main_top_is_full_even_with_low_share():
    assert GS.points(np.array([0.3, 0.2, 0.1]), np.array([0.1, 0.9, 0.0])).tolist() == [100, 100, 45]


from coach.game import refits as GR  # noqa: E402


def test_refits_never_use_the_target_and_are_reproducible(two_matches):
    _, _, tables = two_matches
    tables3 = tables
    sets1 = GR.refit_models("SYA", tables3, n=3, seed=7)
    sets2 = GR.refit_models("SYA", tables3, n=3, seed=7)
    assert len(sets1) == 3
    for (m1, used1), (m2, used2) in zip(sets1, sets2):
        assert "SYA" not in used1 and used1 == used2
        x = np.zeros((1, 9))
        assert m1["pass"](x)[0] == pytest.approx(m2["pass"](x)[0])


from coach.game import quality as GQ  # noqa: E402


def frames(n=10, players=22, step=0.3):
    xy = np.zeros((n, players, 2))
    xy[:, :, 0] = np.arange(players)[None, :] * 2.0
    xy[:, :, 0] += np.arange(n)[:, None] * step
    return xy


def test_quality_accepts_a_clean_moment():
    assert GQ.problems(n_att=11, n_def=11, ball_gap=1.0, lead_up=frames(), n_options=8) == []


def test_quality_rejects_missing_players_far_ball_jumps_and_few_options():
    xy = frames()
    xy[5, 3, 0] += 4.0
    got = GQ.problems(n_att=10, n_def=11, ball_gap=3.0, lead_up=xy, n_options=2)
    assert got == ["players", "ball", "jump", "options"]


def test_quality_ignores_nan_players_in_jump_check():
    xy = frames()
    xy[2:4, 7] = np.nan
    assert GQ.problems(n_att=11, n_def=11, ball_gap=0.5, lead_up=xy, n_options=5) == []


from coach.game import moments as GM  # noqa: E402

SCHEMAS = project_dir("20-football-analytics") / "coach" / "game" / "schemas"


def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(two_matches):
    a, _, tables = two_matches
    tables3 = tables
    models, fitted, trained_on = R.held_out_models("SYA", tables3)
    refits = GR.refit_models("SYA", tables3, n=4, seed=1)
    return GM.match_moments(a, (models, fitted, trained_on), refits, league="2. Bundesliga", require_full=False)


def test_moment_files_match_schemas_and_hide_answers(built):
    assert built, "the synthetic match must give at least one moment"
    for c in built:
        jsonschema.validate(c["public"], schema("moment"))
        jsonschema.validate(c["answer"], schema("answer"))
        text = json.dumps(c["public"])
        for banned in ("score", "points", "share_top", "pro_pick", "ai_pick", "outcome", "reasons", "net_xg"):
            assert banned not in text


def test_answers_are_held_out_and_consistent(built):
    for c in built:
        a = c["answer"]
        assert a["models"]["trained_without"] == "SYA"
        ks = [o["k"] for o in a["options"]]
        assert ks == [o["k"] for o in c["public"]["options"]]
        best = max(a["options"], key=lambda o: o["score"])
        assert a["ai_pick"] == best["k"] and best["points"] == 100
        assert a["pro_pick"] in ks
        assert all(0 <= o["points"] <= 100 for o in a["options"])


from coach.game import days as GD  # noqa: E402


def feat(i, match, x, difficulty, shot=False, iqr=0.01, v=0.01):
    return {
        "id": f"M{i}",
        "match": match,
        "x": x,
        "difficulty": difficulty,
        "shot_after": shot,
        "iqr": iqr,
        "v_now": v,
        "all_best": False,
        "obvious": False,
        "cutback": False,
        "ai_is_pro": False,
    }


def pool(n=200):
    rng = np.random.default_rng(0)
    zones = [-40.0, 0.0, 0.0, 40.0, 40.0]
    return [
        feat(
            i,
            f"G{i % 13}",
            zones[i % 5] + rng.normal(0, 3),
            float(rng.uniform(0, 1)),
            shot=(i % 7 == 0),
            iqr=float(rng.uniform(0, 0.02)),
            v=float(rng.uniform(0, 0.03)),
        )
        for i in range(n)
    ]


def test_interest_drops_all_best_and_obvious():
    f = pool(10)
    f[0]["all_best"] = True
    f[1]["obvious"] = True
    kept = GD.rank_pool(f, size=10)
    assert "M0" not in [k["id"] for k in kept] and "M1" not in [k["id"] for k in kept]


def test_daily_sets_respect_zones_difficulty_matches_and_shots():
    days = GD.daily_sets(GD.rank_pool(pool(), size=200), n_days=8)
    assert len(days) == 8
    used = set()
    prev_matches = set()
    for d in days:
        assert len(d) == 5 and not used & {m["id"] for m in d}
        used |= {m["id"] for m in d}
        assert [GD.zone(m["x"]) for m in d].count("build") == 1
        assert sorted(GD.zone(m["x"]) for m in d) == ["build", "final", "final", "middle", "middle"]
        assert [m["difficulty"] for m in d] == sorted(m["difficulty"] for m in d)
        assert len({m["match"] for m in d}) == 5
        assert any(m["shot_after"] for m in d)
        assert not prev_matches & {m["match"] for m in d}
        prev_matches = {m["match"] for m in d}


from coach.game import build as GB  # noqa: E402


def test_write_release_layout_and_validation(tmp_path, built):
    feats = [{**c["features"], "id": c["public"]["id"]} for c in built]
    for i, f in enumerate(feats):
        f.update(x=[-40.0, 0.0, 0.0, 40.0, 40.0][i % 5], match=f"G{i}", shot_after=True)
    cands = {c["public"]["id"]: c for c in built}
    days = [[f for f in feats[:5]]] if len(feats) >= 5 else []
    out = GB.write_release(tmp_path, cands, days, start="2026-12-01")
    if days:
        assert (out / "public" / "days" / "2026-12-01.json").exists()
    for mid in cands:
        assert (out / "public" / "moments" / f"{mid}.json").exists()
        assert (out / "answers" / f"{mid}.json").exists()
    assert GB.validate_release(out) == []

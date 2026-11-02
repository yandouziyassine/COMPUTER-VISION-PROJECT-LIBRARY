"""Project 20, AI coach: player profiles (shrinkage, k choice, format, providers, held-out evaluation). Synthetic."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import profiles as PR  # noqa: E402


def test_shrink_small_samples_stay_near_zero_large_ones_near_raw():
    small = PR.shrink((5, 2.0, 1.0), k=30.0)  # 5 passes, 2 more completions than expected
    large = PR.shrink((2000, 80.0, 400.0), k=30.0)
    assert small["n"] == 5 and abs(small["logit"]) < 0.1 and small["raw"] == pytest.approx(2.0)
    assert large["logit"] == pytest.approx(80 / 430) and large["raw"] == pytest.approx(0.2)
    assert PR.shrink((3, 1.0, 0.5), k=0.0)["logit"] == pytest.approx(2.0)


def test_offsets_apply_and_missing_players_unchanged():
    keys = np.array(["a", "a", "b", ""])
    y = np.array([1, 1, 0, 1])
    p = np.full(4, 0.5)
    table = PR.offsets(keys, y, p, k=1.0)
    assert set(table) == {"a", "b"} and table["a"]["n"] == 2
    assert table["a"]["logit"] == pytest.approx(1.0 / (0.5 + 1.0)) and table["b"]["logit"] < 0
    off = PR.offset_vector(np.array(["a", "zz"]), table)
    q = PR.apply(np.array([0.5, 0.7]), off, a=1.0)
    assert q[0] > 0.5 and q[1] == pytest.approx(0.7)


def _synthetic(effects, matches=6, rows=300, seed=0):
    rng = np.random.default_rng(seed)
    players = np.array([f"p{i}" for i in range(len(effects))])
    keys = rng.choice(players, size=matches * rows)
    base = rng.uniform(0.4, 0.95, size=len(keys))
    eff = np.array([effects[int(k[1:])] for k in keys])
    y = (rng.uniform(size=len(keys)) < PR.sigmoid(PR.logit(base) + eff)).astype(int)
    groups = np.repeat(np.arange(matches), rows)
    return groups, keys, y, base


def test_choose_k_finds_real_player_effects():
    groups, keys, y, base = _synthetic([1.0, -1.0, 0.8, -0.8, 0.0, 0.5])
    sel = PR.choose_k(groups, keys, y, base)
    assert sel["a"] > 0 and sel["log_loss"] < sel["base_log_loss"] - 0.01 and sel["coverage"] == 1.0


def test_choose_k_without_effects_gains_nothing():
    groups, keys, y, base = _synthetic([0.0] * 6, seed=1)
    sel = PR.choose_k(groups, keys, y, base)
    assert sel["log_loss"] >= sel["base_log_loss"] - 0.002
    assert sel["a"] == 0.0 or sel["k"] >= 30


class FakeApi(PR.ApiProvider):
    """Test double for a live API: canned season records, no network, no key."""

    def __init__(self, ids, records):
        super().__init__(ids, k_finishing=10.0)
        self.records, self.calls = records, []

    def fetch(self, api_id):
        self.calls.append(api_id)
        return self.records.get(api_id)


def _players():
    return [
        {"id": "dfl:A", "name": "A. One", "team": 0},
        {"id": "dfl:B", "name": "B. Two", "team": 1, "position": "STZ"},
        {"id": "dfl:C", "name": "C. Three", "team": 1},
    ]


def test_build_open_data_and_api_providers(tmp_path):
    keys = np.array(["dfl:A"] * 50 + ["dfl:B"] * 5)
    y = np.r_[np.ones(45), np.zeros(5), np.ones(5)]
    open_data = PR.OpenDataProvider({"pass_skill": (keys, y, np.full(55, 0.8), None)}, {"pass_skill": 10.0})
    record = {"shots": 20, "goals": 5, "xg": 3.0, "passes": 100, "passes_completed": 80}
    api = FakeApi({"dfl:B": 77, "dfl:C": 78}, {77: record})
    prof = PR.build(_players(), [open_data, api], built_without=["J03WOH"], k={"pass": 10.0})
    assert set(prof["players"]) == {"0", "1"}  # C: mapped to the API, but no record -> no profile
    assert prof["players"]["0"]["pass_skill"]["n"] == 50 and "finishing" not in prof["players"]["0"]
    b = prof["players"]["1"]
    assert b["position"] == "STZ" and b["finishing"]["n"] == 20 and b["finishing"]["logit"] > 0
    assert b["pass_accuracy"] == {"value": 0.8, "n": 100}
    assert prof["mapping"] == {"dfl:A": 0, "dfl:B": 1} and api.calls == [77, 78]
    assert PR.validate(prof) == []
    PR.save(prof, tmp_path / "profiles.json")
    back = PR.load(tmp_path / "profiles.json")
    assert back["built_without"] == ["J03WOH"] and back["players"]["0"]["pass_skill"]["n"] == 50


def test_validate_rejects_bad_fields(tmp_path):
    prof = {"version": 1, "built_without": [], "mapping": {"dfl:A": 0}, "players": {"0": {"id": "dfl:A"}}}
    assert PR.validate(prof) == []
    prof["players"]["0"]["pass_skill"] = {"logit": 0.1, "n": 0, "raw": 0.1}
    prof["players"]["0"]["rating"] = {"value": 7}
    errs = PR.validate(prof)
    assert any("pass_skill" in e for e in errs) and any("rating" in e for e in errs)
    with pytest.raises(ValueError):
        PR.save(prof, tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()


def test_api_provider_is_abstract():
    with pytest.raises(TypeError):
        PR.ApiProvider({})


def test_skillcorner_descriptors(tmp_path):
    (tmp_path / PR.SC_AGGREGATES["passing"]).write_text(
        "player_id,player_birthdate,performance_included_count,pass_count_dangerous_attempted_p30tip,"
        "pass_pct_completed,pass_avgxpass_attempted,pass_count_attempted_total\n"
        "211,1993-06-02,13,5.66,75.76,74.46,198.0\n",
        encoding="utf-8",
    )
    (tmp_path / PR.SC_AGGREGATES["physical"]).write_text(
        "player_id,psv99,count_match\n211,28.83,24\n", encoding="utf-8"
    )
    d = PR.skillcorner_descriptors(tmp_path)
    p = d["skillcorner:211"]
    assert p["dangerous_passes_p30"] == {"value": 5.66, "n_matches": 13}
    assert p["completion_vs_xpass"] == {"value": pytest.approx(0.013), "n_passes": 198}
    assert p["top_speed_kmh"] == {"value": 28.83, "n_matches": 24} and "runs_p30" not in p
    assert not any("birth" in k for k in p)


from coach import models as ML  # noqa: E402
from coach import profiles_eval as PE  # noqa: E402


def _pass_table(effects, matches=5, rows=250, seed=0):
    rng = np.random.default_rng(seed)
    n = matches * rows
    X = rng.normal(size=(n, 2))
    keys = np.array([f"p{i}" for i in rng.integers(0, len(effects), n)])
    z = 1.0 + 1.2 * X[:, 0] + np.array([effects[int(k[1:])] for k in keys])
    y = (rng.uniform(size=n) < PR.sigmoid(z)).astype(int)
    match = np.repeat([f"M{i}" for i in range(matches)], rows)
    return {"X": X, "y": y, "match": match, "weight": np.ones(n)}, keys


def test_lomo_profiles_improves_with_real_effects_and_reports_coverage():
    table, keys = _pass_table([1.5, -1.5, 1.0, -1.0])
    r = PE.lomo_profiles(table, keys, fit=ML.fit_logistic)
    s = PE.summarize(table, r)
    assert s["coverage"] == 1.0 and set(s["folds"]) == {f"M{i}" for i in range(5)}
    assert s["with"]["log_loss"] < s["without"]["log_loss"]
    assert s["log_loss_diff"]["ci95"][1] < 0 and s["decision"] == "advice"


def test_lomo_profiles_fixed_base_and_unseen_player():
    table, keys = _pass_table([0.0, 0.0])
    keys[table["match"] == "M0"] = "only_in_M0"  # never in a training set when M0 is held out
    r = PE.lomo_profiles(table, keys, base=np.full(len(keys), 0.7))
    m0 = table["match"] == "M0"
    assert not r["has"][m0].any() and np.allclose(r["p1"][m0], 0.7)


def test_bootstrap_diff_sign():
    y = np.array([1, 0] * 50)
    match = np.repeat(["a", "b", "c", "d"], 25)
    good, bad = np.where(y == 1, 0.9, 0.1), np.full(100, 0.5)
    d = PE.bootstrap_diff(match, y, bad, good, np.ones(100))
    assert d["mean"] < 0 and d["ci95"][1] < 0


def test_check_target_mapping():
    assert PE.check_target_mapping([5, 6, 5], [0, 1, 0]) == 2
    with pytest.raises(ValueError):
        PE.check_target_mapping([5, 6, 5], [0, 1, 1])


def test_idsse_keys_follow_the_derived_rows():
    info = {
        "player_ids": ["P0", "P1"],
        "events": [
            {"type": "pass", "player": 1, "team": 0, "half": 0},
            {"type": "shot", "player": 0, "team": 0, "half": 0},
            {"type": "shot", "player": 1, "team": 1, "half": None},  # not a shot row (no half)
        ],
    }
    d = {"pass": {"event_idx": np.array([0])}, "shot": {"y": np.array([0])}}
    pk, sk = PE.idsse_keys(d, info)
    assert list(pk) == ["dfl:P1"] and list(sk) == ["dfl:P0"]


from coach import export_profiles as EP  # noqa: E402


def _part(mid, players, rng, n=200, ns=12):
    return {
        "match": mid,
        "pass": {
            "X": rng.normal(size=(n, 2)),
            "y": rng.integers(0, 2, n),
            "match": np.full(n, mid),
            "weight": np.ones(n),
        },
        "pass_keys": np.array(rng.choice(players, n), dtype=str),
        "shot": {
            "X": rng.normal(size=(ns, 3)),
            "y": rng.integers(0, 2, ns),
            "match": np.full(ns, mid),
            "xg": rng.uniform(0.05, 0.3, ns),
            "penalty": np.zeros(ns, bool),
            "placed": np.ones(ns, bool),
        },
        "shot_keys": np.array(rng.choice(players, ns), dtype=str),
        "info": {"player_ids": [p[4:] for p in players], "names": ["A", "B"], "meta": {"home": "H", "guest": "G"}},
        "team": np.array([0, 1], np.int8),
    }


def test_export_profiles_built_without_the_match(tmp_path):
    rng = np.random.default_rng(3)
    parts = [
        _part(m, ["dfl:A", "dfl:B"] if m != "M0" else ["dfl:A", "dfl:ONLY0"], rng) for m in ("M0", "M1", "M2", "M3")
    ]
    prof = EP.build("M0", parts, decision={"pass": "display-only", "shot": "display-only"})
    assert prof["built_without"] == ["M0"] and "M0" not in prof["trained_on"]
    assert set(prof["players"]) == {"0"} and prof["mapping"] == {"dfl:A": 0}  # ONLY0 plays only in M0
    assert prof["a"] == {"pass": 0.0, "shot": 0.0} and prof["advice_effect"] == "display-only"
    assert prof["teams"]["0"] == {"name": "H", "matches_in_profiles": 3}
    EP.write(prof, tmp_path)
    assert PR.load(tmp_path / "profiles.json")["players"]["0"]["pass_skill"]["n"] > 0

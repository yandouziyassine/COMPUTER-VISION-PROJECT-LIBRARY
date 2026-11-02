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

"""Project 14 W6 (uncertainty): sparsification curves and AUSE."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_module  # noqa: E402

w6 = load_module("14", "w6_uncertainty.py")


def test_perfect_uncertainty_has_zero_ause_and_random_has_more():
    rng = np.random.default_rng(0)
    err = rng.exponential(0.1, 5000)
    assert w6.ause(err, err) == pytest.approx(0.0, abs=1e-12)
    assert w6.ause(err, rng.random(5000)) > 0.01


def test_sparsification_starts_at_the_mean_and_falls_for_a_good_ranking():
    err = np.linspace(0, 1, 1000)
    curve = w6.sparsification(err, err)
    assert curve[0] == pytest.approx(err.mean()) and np.all(np.diff(curve) <= 1e-12)

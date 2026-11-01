"""Project 13: operating-threshold selection (threshold.py), pure NumPy, no data or model needed."""

import numpy as np
import pytest
from conftest import load_defs

t = load_defs(
    "13",
    "threshold.py",
    ["threshold_for_recall", "stratified_halves", "rates", "evaluate_category"],
    {"np": np},
)


def test_threshold_for_recall_small_set_keeps_every_defect():
    # Under 100 defects, a 99 % target allows zero misses: the threshold is the lowest defect score.
    scores = np.array([0.9, 0.4, 0.7, 0.55])
    assert t.threshold_for_recall(scores, 0.99) == 0.4
    assert t.threshold_for_recall(scores, 1.0) == 0.4


def test_threshold_for_recall_allows_the_permitted_misses():
    scores = np.arange(200, dtype=float)  # 200 defects, 99 % -> may miss 2
    th = t.threshold_for_recall(scores, 0.99)
    assert th == 2.0
    assert (scores >= th).mean() >= 0.99
    assert t.threshold_for_recall(np.arange(10, dtype=float), 0.75) == 2.0  # floor(2.5) misses


def test_stratified_halves_splits_every_group():
    groups = np.array(["good"] * 10 + ["crack"] * 7 + ["hole"] * 1)
    mask = t.stratified_halves(groups, np.random.default_rng(0))
    assert mask[:10].sum() == 5
    assert mask[10:17].sum() in (3, 4)
    assert mask[17:].sum() in (0, 1)


def test_rates():
    scores = np.array([0.1, 0.5, 0.9, 0.2, 0.8])
    labels = np.array([0, 0, 1, 1, 1])
    recall, far = t.rates(scores, labels, 0.5)
    assert recall == pytest.approx(2 / 3)
    assert far == pytest.approx(1 / 2)


def test_evaluate_category_perfect_separation():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0, 1, 40), rng.uniform(2, 3, 60)])
    labels = np.array([0] * 40 + [1] * 60)
    defect = np.array(["good"] * 40 + ["a"] * 30 + ["b"] * 30)
    r = t.evaluate_category(scores, labels, defect, 0.99, n_splits=3)
    assert r["n_heldout_evaluations"] == 6
    assert r["heldout_false_alarm_mean"] == 0.0
    assert r["oracle_false_alarm"] == 0.0
    # The held-out half can contain a defect scored below the calibration half's lowest one.
    assert 0.9 <= r["heldout_recall_mean"] <= 1.0

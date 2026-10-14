"""Project 09: rule-based empty-facing detection on synthetic product boxes.
evaluate.py imports Ultralytics at import time, so only find_gaps is loaded (see conftest.load_defs)."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_defs  # noqa: E402

find_gaps = load_defs("09", "evaluate.py", ["find_gaps"], {"np": np}).find_gaps


def row(y, xs, w=20, h=40):
    return [[x, y, x + w, y + h] for x in xs]


def test_too_few_boxes_returns_nothing():
    assert find_gaps(np.array(row(0, [0, 20, 40, 100]), float)) == []


def test_tightly_packed_row_has_no_gap():
    assert find_gaps(np.array(row(0, range(0, 200, 20)), float)) == []


def test_single_gap_in_one_row():
    # Products 20 px wide; a 60 px hole (> 1.5 x 20) between x=60 and x=120.
    boxes = np.array(row(0, [0, 20, 40, 120, 140, 160]), float)
    assert find_gaps(boxes) == [(60.0, 0.0, 120.0, 40.0)]


def test_gap_below_threshold_is_ignored():
    boxes = np.array(row(0, [0, 20, 40, 85, 105, 125]), float)  # 25 px hole < 30 px
    assert find_gaps(boxes) == []
    assert len(find_gaps(boxes, gap_factor=1.0)) == 1


def test_gaps_found_per_shelf_row():
    top = row(0, range(0, 120, 20))  # full shelf
    bottom = row(100, [0, 20, 40, 140, 160])  # hole between 60 and 140
    boxes = np.array(bottom + top, float)  # order must not matter
    gaps = find_gaps(boxes)
    assert gaps == [(60.0, 100.0, 140.0, 140.0)]


def test_rows_with_fewer_than_three_products_are_skipped():
    top = row(0, range(0, 120, 20))
    sparse = row(100, [0, 200])  # only 2 products: no gap reported
    assert find_gaps(np.array(top + sparse, float)) == []

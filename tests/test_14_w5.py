"""Project 14 W5 (focal sweep): zoom geometry and scale statistics."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from conftest import load_module  # noqa: E402

w5 = load_module("14", "w5_focal.py")


def test_zoom_in_crops_the_centre_and_keeps_depths():
    rgb = np.zeros((480, 640, 3), np.uint8)
    gt = np.tile(np.arange(640, dtype=np.float32), (480, 1))
    inp, gt_view, box = w5.zoom_view(rgb, gt, 2.0)
    assert inp.shape == rgb.shape and gt_view.shape == (240, 320) and box == (0, 480, 0, 640)
    assert gt_view[0, 0] == 160  # centre crop starts a quarter of the width in


def test_zoom_out_shrinks_into_a_border_and_maps_back():
    rgb = np.full((480, 640, 3), 200, np.uint8)
    gt = np.full((480, 640), 3.0, np.float32)
    inp, gt_view, box = w5.zoom_view(rgb, gt, 0.5)
    y0, y1, x0, x1 = box
    assert (y1 - y0, x1 - x0) == (240, 320) and inp[0, 0].sum() == 0 and inp[240, 320].sum() > 0
    pred = np.zeros((480, 640), np.float32)
    pred[y0:y1, x0:x1] = 3.0
    back = w5.pred_on_gt(pred, box, gt_view.shape)
    assert back.shape == gt.shape and np.allclose(back[5:-5, 5:-5], 3.0)


def test_scale_stats_reports_the_scale_error():
    gt = np.random.default_rng(0).uniform(1, 5, (100, 100))
    s = w5.scale_stats(gt * 1.27, gt)
    assert s["scale"] == pytest.approx(1.27) and s["absrel"] == pytest.approx(0.27)

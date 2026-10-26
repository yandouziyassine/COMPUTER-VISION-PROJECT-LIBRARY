"""Project 14: depth metrics, alignment and geometry on synthetic data (no model, no dataset)."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_defs, load_module  # noqa: E402

m = load_module("14", "metrics.py")
K = (500.0, 500.0, 320.0, 240.0)


def test_alignment_recovers_scale_and_shift_in_disparity():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 8, (48, 64)).astype(np.float32)
    pred_disp = (1 / gt - 0.05) / 3.0  # model output = disparity with unknown scale/shift
    aligned = m.align_disparity(pred_disp, gt, np.ones_like(gt, bool))
    assert np.allclose(aligned, gt, rtol=1e-4)


def test_absrel_and_delta1_match_hand_computation():
    gt = np.array([[1.0, 2.0], [4.0, 5.0]])
    pred = np.array([[1.1, 2.0], [3.6, 5.0]])  # ratios 1.1, 1, 1.11, 1: all within 1.25
    mask = np.ones_like(gt, bool)
    assert m.abs_rel(pred, gt, mask) == pytest.approx((0.1 + 0 + 0.1 + 0) / 4)
    assert m.delta1(pred, gt, mask) == pytest.approx(1.0)
    pred[1, 0] = 5.5  # 5.5 / 4 = 1.375 > 1.25
    assert m.delta1(pred, gt, mask) == pytest.approx(0.75)


def test_distance_between_backprojected_pixels():
    depth = np.full((480, 640), 2.0)
    # two pixels on the same row, 100 px apart, at 2 m: 100 * 2 / 500 = 0.4 m apart
    assert m.distance((300, 240), (400, 240), depth, K) == pytest.approx(0.4)


def test_plane_fit_and_plane_depth():
    fx, fy, cx, cy = K
    v, u = np.mgrid[100:200, 200:300]
    z = 1.0 / (1 / 3.0 + 0.0002 * (u - cx))  # tilted wall: for a plane, 1/depth is linear in pixels
    pts = m.backproject(u.ravel(), v.ravel(), z.ravel(), K)
    n, d = m.fit_plane(pts)
    assert np.allclose(m.plane_depth(u, v, n, d, K), z, atol=1e-6)


def test_eval_mask_applies_eigen_crop_and_range():
    gt = np.full((480, 640), 3.0)
    gt[100, 100] = 0.0
    gt[200, 200] = 12.0
    mask = m.eval_mask(gt)
    assert not mask[0, 0] and not mask[100, 100] and not mask[200, 200] and mask[300, 300]


def test_poster_quad_lies_inside_the_wall():
    cv2 = pytest.importorskip("cv2")
    poster = load_defs(
        "14",
        "poster_test.py",
        ["wall_plane", "orient_quad", "poster_quad"],
        {"np": np, "cv2": cv2, "backproject": m.backproject, "fit_plane": m.fit_plane},
    )
    z = np.full((480, 640), 3.0)  # fronto-parallel wall 3 m away
    wall = np.zeros((480, 640), bool)
    wall[100:400, 150:550] = True
    quad = poster.poster_quad(wall, z, K)
    assert quad is not None
    qx, qy = quad[:, 0].round().astype(int), quad[:, 1].round().astype(int)
    assert wall[qy.clip(0, 479), qx.clip(0, 639)].all()


def test_intrinsics_from_fov_and_from_35mm_focal():
    meas = load_defs("14", "measure.py", ["DEFAULT_FOV", "intrinsics_from_fov", "intrinsics_from_35mm"], {"np": np})
    fx, fy, cx, cy = meas.intrinsics_from_fov(640, 480, 90.0)
    assert fx == pytest.approx(320.0) and (cx, cy) == (320.0, 240.0)
    fx, *_ = meas.intrinsics_from_35mm(4000, 3000, 36.0)  # 36 mm equivalent = sensor width
    assert fx == pytest.approx(4000.0)

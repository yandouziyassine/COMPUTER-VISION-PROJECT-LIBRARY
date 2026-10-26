"""Project 14 W7 (posters at scale): cue-ablation helpers."""

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from conftest import load_module  # noqa: E402

w7 = load_module("14", "w7_posters.py")


def test_phase_scramble_keeps_colour_statistics_but_not_the_image():
    rng = np.random.default_rng(0)
    img = np.full((64, 64, 3), 60, np.uint8)
    img[16:48, 16:48] = 200  # one bright object
    s = w7.phase_scramble(img, rng)
    assert s.shape == img.shape
    assert abs(s.mean() - img.mean()) < 8  # DC term kept (up to clipping)
    assert np.corrcoef(s[..., 0].ravel(), img[..., 0].ravel())[0, 1] < 0.5


def test_upright_quad_keeps_centre_and_area():
    quad = np.float32([[100, 100], [300, 120], [300, 260], [100, 280]])
    up = w7.upright_quad(quad)
    area = lambda q: 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1)))  # noqa: E731
    assert area(up) == pytest.approx(area(quad), rel=1e-4)
    assert np.allclose(up.mean(axis=0), quad.mean(axis=0))
    assert up[0, 1] == up[1, 1] and up[0, 0] == up[3, 0]  # axis-aligned


def test_paste_with_and_without_frame():
    rgb = np.zeros((120, 160, 3), np.uint8)
    pic = np.full((30, 40, 3), 100, np.uint8)
    quad = np.float32([[40, 30], [120, 30], [120, 90], [40, 90]])
    framed, m1 = w7.paste(rgb, quad, pic, True)
    plain, m2 = w7.paste(rgb, quad, pic, False)
    assert (
        m1.sum() == pytest.approx(m2.sum(), rel=0.05)
        and framed[31, 41].mean() > 200
        and plain[31, 41].mean() == pytest.approx(100, abs=2)
    )

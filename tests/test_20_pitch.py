"""Project 20: pitch landmark layout and image → pitch homography on a synthetic camera view."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from conftest import load_module  # noqa: E402

pitch = load_module("20", "pitch.py")


def test_layout_is_symmetric_like_flip_idx():
    flip = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7,
            0, 1, 2, 3, 4, 5, 31, 30]
    p = pitch.PITCH_POINTS
    mirrored = np.stack([pitch.LENGTH - p[:, 0], p[:, 1]], axis=1)
    assert np.allclose(mirrored, p[flip])


def test_homography_recovers_pitch_positions():
    # A made-up broadcast camera: pitch cm -> image px
    cam = np.array([[0.09, 0.03, 300.0], [0.0, 0.05, 250.0], [0.0, 0.00004, 1.0]])
    img = cv2.perspectiveTransform(pitch.PITCH_POINTS.reshape(-1, 1, 2), cam).reshape(-1, 2)
    kp = np.concatenate([img, np.ones((32, 1))], axis=1)
    kp[20:, 2] = 0.1                                   # only the left half is visible / confident
    H, used = pitch.image_to_pitch(kp)
    assert H is not None and len(used) == 20
    player_cm = np.array([[3000.0, 2000.0], [7000.0, 5000.0]])
    player_px = cv2.perspectiveTransform(player_cm.reshape(-1, 1, 2), cam).reshape(-1, 2)
    assert np.allclose(pitch.project(H, player_px), player_cm, atol=5)   # within 5 cm


def test_too_few_landmarks_gives_no_homography():
    kp = np.zeros((32, 3))
    kp[:3, 2] = 0.9
    H, used = pitch.image_to_pitch(kp)
    assert H is None and len(used) == 3


def test_feet_are_bottom_centre():
    assert np.allclose(pitch.feet([[10, 20, 30, 80]]), [[20, 80]])

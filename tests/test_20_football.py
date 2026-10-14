"""Project 20: team assignment by shirt color and ball possession, on synthetic frames."""

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("sklearn")

from conftest import load_module  # noqa: E402

analytics = load_module("20", "analytics.py")

GRASS = (40, 140, 40)  # BGR
RED, BLUE = (30, 30, 200), (200, 60, 30)


def pitch_with_players(colors, seed=0):
    """Green pitch with one 30x80 player box per color; small per-player color jitter."""
    rng = np.random.default_rng(seed)
    img = np.full((300, 60 * len(colors) + 40, 3), GRASS, np.uint8)
    boxes = []
    for i, c in enumerate(colors):
        x, y = 20 + 60 * i, 100
        jitter = np.clip(np.array(c) + rng.integers(-8, 9, 3), 0, 255)
        img[y : y + 80, x : x + 30] = jitter
        boxes.append([x, y, x + 30, y + 80])
    return img, np.array(boxes, float)


def test_torso_color_ignores_grass_and_empty_boxes():
    img, boxes = pitch_with_players([RED])
    lab_red = cv2.cvtColor(np.uint8([[RED]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    assert np.linalg.norm(analytics.torso_color(img, boxes[0]) - lab_red) < 15
    assert analytics.torso_color(img, [5, 5, 5, 5]) is None  # empty crop
    assert analytics.torso_color(img, [0, 0, 20, 60]) is None  # only grass


def test_assign_teams_splits_two_shirt_colors():
    colors = [RED, BLUE, RED, BLUE, RED, BLUE, RED, BLUE]
    img, boxes = pitch_with_players(colors)
    labels, centers = analytics.assign_teams(img, boxes)
    assert centers.shape == (2, 3)
    red = {labels[i] for i, c in enumerate(colors) if c == RED}
    blue = {labels[i] for i, c in enumerate(colors) if c == BLUE}
    assert len(red) == 1 and len(blue) == 1 and red != blue
    assert -1 not in red | blue


def test_assign_teams_needs_four_players():
    img, boxes = pitch_with_players([RED, BLUE, RED])
    labels, centers = analytics.assign_teams(img, boxes)
    assert centers is None
    assert labels.tolist() == [-1, -1, -1]


PLAYERS = np.array(
    [
        [100, 100, 130, 180],  # feet at (115, 180), team 0
        [300, 100, 330, 180],  # feet at (315, 180), team 1
    ],
    float,
)
TEAMS = np.array([0, 1])


def test_possession_goes_to_closest_player():
    ball = [305, 170, 315, 180]  # at the feet of player 1
    assert analytics.possession(ball, PLAYERS, TEAMS) == (1, 1)
    ball = [110, 175, 120, 185]
    assert analytics.possession(ball, PLAYERS, TEAMS) == (0, 0)


def test_possession_none_when_ball_is_far():
    far_ball = [700, 600, 710, 610]  # far more than 1.5 player heights from anyone
    assert analytics.possession(far_ball, PLAYERS, TEAMS) == (None, None)


def test_possession_distance_threshold_scales_with_player_height():
    ball = [205, 175, 215, 185]  # 95 px from player 0's feet; player height 80
    assert analytics.possession(ball, PLAYERS, TEAMS) == (0, 0)  # 95 < 1.5 * 80
    assert analytics.possession(ball, PLAYERS, TEAMS, max_dist_factor=1.0) == (None, None)


def test_possession_without_ball_or_players():
    assert analytics.possession(None, PLAYERS, TEAMS) == (None, None)
    assert analytics.possession([0, 0, 1, 1], np.zeros((0, 4)), np.array([])) == (None, None)

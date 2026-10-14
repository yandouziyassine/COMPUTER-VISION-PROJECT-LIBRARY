"""Pitch geometry: 32 field landmarks, image → pitch homography, and a top-down tactical map.

Landmark numbering follows Roboflow's open football-field keypoint layout (the dataset's `flip_idx` matches it,
and it was checked by drawing the labels on training frames). Coordinates are in centimeters on a 120 m × 70 m
pitch, x along the length (left goal line = 0), y across (far touchline = 0).
"""

import cv2
import numpy as np

LENGTH, WIDTH = 12000, 7000
PEN_W, PEN_L, GOAL_W, GOAL_L, CIRCLE_R, SPOT = 4100, 2015, 1832, 550, 915, 1100

PITCH_POINTS = np.array([
    (0, 0), (0, (WIDTH - PEN_W) / 2), (0, (WIDTH - GOAL_W) / 2), (0, (WIDTH + GOAL_W) / 2),
    (0, (WIDTH + PEN_W) / 2), (0, WIDTH),                                                     # 0-5 left goal line
    (GOAL_L, (WIDTH - GOAL_W) / 2), (GOAL_L, (WIDTH + GOAL_W) / 2),                           # 6-7 left goal box
    (SPOT, WIDTH / 2),                                                                        # 8 left penalty spot
    (PEN_L, (WIDTH - PEN_W) / 2), (PEN_L, (WIDTH - GOAL_W) / 2), (PEN_L, (WIDTH + GOAL_W) / 2),
    (PEN_L, (WIDTH + PEN_W) / 2),                                                             # 9-12 left penalty box
    (LENGTH / 2, 0), (LENGTH / 2, WIDTH / 2 - CIRCLE_R), (LENGTH / 2, WIDTH / 2 + CIRCLE_R),
    (LENGTH / 2, WIDTH),                                                                      # 13-16 halfway line
    (LENGTH - PEN_L, (WIDTH - PEN_W) / 2), (LENGTH - PEN_L, (WIDTH - GOAL_W) / 2),
    (LENGTH - PEN_L, (WIDTH + GOAL_W) / 2), (LENGTH - PEN_L, (WIDTH + PEN_W) / 2),            # 17-20 right penalty box
    (LENGTH - SPOT, WIDTH / 2),                                                               # 21 right penalty spot
    (LENGTH - GOAL_L, (WIDTH - GOAL_W) / 2), (LENGTH - GOAL_L, (WIDTH + GOAL_W) / 2),         # 22-23 right goal box
    (LENGTH, 0), (LENGTH, (WIDTH - PEN_W) / 2), (LENGTH, (WIDTH - GOAL_W) / 2), (LENGTH, (WIDTH + GOAL_W) / 2),
    (LENGTH, (WIDTH + PEN_W) / 2), (LENGTH, WIDTH),                                           # 24-29 right goal line
    (LENGTH / 2 - CIRCLE_R, WIDTH / 2), (LENGTH / 2 + CIRCLE_R, WIDTH / 2),                   # 30-31 centre circle
], dtype=np.float32)


def image_to_pitch(keypoints, min_conf=0.3, min_points=4, ransac_cm=150.0):
    """keypoints: (32, 3) array of (x, y, confidence) in image pixels.
    Returns (3x3 homography image→pitch cm, indices used) or (None, used) if too few reliable landmarks.
    min_conf and ransac_cm were chosen on the validation frames (see README)."""
    used = np.flatnonzero(keypoints[:, 2] >= min_conf)
    if len(used) < min_points:
        return None, used
    H, _ = cv2.findHomography(keypoints[used, :2].astype(np.float32), PITCH_POINTS[used], cv2.RANSAC, ransac_cm)
    return H, used


def project(H, points_xy):
    """Image points (N, 2) → pitch coordinates in cm (N, 2)."""
    if len(points_xy) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(np.asarray(points_xy, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def feet(boxes):
    """Bottom-centre of each (x1, y1, x2, y2) box: where a player touches the grass."""
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def draw_pitch(scale=0.08, margin=40):
    """Empty top-down pitch image (BGR). scale: pixels per cm."""
    w, h = int(LENGTH * scale) + 2 * margin, int(WIDTH * scale) + 2 * margin
    img = np.full((h, w, 3), (60, 120, 40), np.uint8)

    def pt(x, y):
        return int(margin + x * scale), int(margin + y * scale)

    white, t = (235, 235, 235), 2
    cv2.rectangle(img, pt(0, 0), pt(LENGTH, WIDTH), white, t)
    cv2.line(img, pt(LENGTH / 2, 0), pt(LENGTH / 2, WIDTH), white, t)
    cv2.circle(img, pt(LENGTH / 2, WIDTH / 2), int(CIRCLE_R * scale), white, t)
    for x0, sign in ((0, 1), (LENGTH, -1)):
        cv2.rectangle(img, pt(x0, (WIDTH - PEN_W) / 2), pt(x0 + sign * PEN_L, (WIDTH + PEN_W) / 2), white, t)
        cv2.rectangle(img, pt(x0, (WIDTH - GOAL_W) / 2), pt(x0 + sign * GOAL_L, (WIDTH + GOAL_W) / 2), white, t)
        cv2.circle(img, pt(x0 + sign * SPOT, WIDTH / 2), 3, white, -1)
    return img, (lambda x, y: pt(x, y))


def tactical_map(pitch_xy, teams, ball_xy=None, colors=None, scale=0.08):
    """Top-down map: one dot per player, colored by team (-1 = unassigned), plus the ball."""
    colors = colors or {0: (255, 80, 0), 1: (0, 80, 255), -1: (200, 200, 200)}
    img, pt = draw_pitch(scale)
    for (x, y), t in zip(pitch_xy, teams):
        if -500 <= x <= LENGTH + 500 and -500 <= y <= WIDTH + 500:   # ignore wild projections
            cv2.circle(img, pt(x, y), 9, colors.get(int(t), colors[-1]), -1)
            cv2.circle(img, pt(x, y), 9, (20, 20, 20), 1)
    if ball_xy is not None:
        cv2.circle(img, pt(*ball_xy), 6, (0, 230, 255), -1)
    return img

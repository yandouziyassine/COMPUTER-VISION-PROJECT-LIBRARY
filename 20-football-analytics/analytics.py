"""Match analytics on top of the detector: team assignment by jersey color, ball possession.

Team assignment needs no labels: each player's torso color (grass pixels masked out) is clustered
into two groups with k-means. Players far from both team colors (referee, goalkeepers) are marked
"other".
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def torso_color(img, box):
    """Mean Lab color of the shirt area: upper-middle part of the box, grass pixels removed."""
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    crop = img[y1 + int(0.15 * h) : y1 + int(0.5 * h), x1 + int(0.25 * w) : x2 - int(0.25 * w)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grass = (hsv[..., 0] > 30) & (hsv[..., 0] < 90) & (hsv[..., 1] > 50)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = lab[~grass.reshape(-1)]
    return keep.mean(axis=0) if len(keep) >= 5 else None


def assign_teams(img, boxes, outlier_factor=2.5):
    """Returns (labels, centers): label 0/1 per player = team, -1 = other (referee, keeper) or unknown."""
    feats = [torso_color(img, b) for b in boxes]
    idx = [i for i, f in enumerate(feats) if f is not None]
    labels = np.full(len(boxes), -1)
    if len(idx) < 4:
        return labels, None
    X = np.array([feats[i] for i in idx])
    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    d = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    cut = np.median(d) * outlier_factor
    for j, i in enumerate(idx):
        labels[i] = km.labels_[j] if d[j] <= cut else -1
    return labels, km.cluster_centers_


def possession(ball_box, player_boxes, team_labels, max_dist_factor=1.5):
    """Team of the player whose feet are closest to the ball, if close enough; else None."""
    if ball_box is None or len(player_boxes) == 0:
        return None, None
    bx, by = (ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2
    feet = np.stack([(player_boxes[:, 0] + player_boxes[:, 2]) / 2, player_boxes[:, 3]], axis=1)
    dist = np.linalg.norm(feet - [bx, by], axis=1)
    k = int(np.argmin(dist))
    player_h = player_boxes[k, 3] - player_boxes[k, 1]
    if dist[k] > max_dist_factor * player_h:
        return None, None
    return int(team_labels[k]), k

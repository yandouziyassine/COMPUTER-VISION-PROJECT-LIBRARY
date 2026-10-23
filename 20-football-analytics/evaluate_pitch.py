"""Evaluate the pitch-landmark model and the image → pitch homography on the held-out test frames, and draw
tactical maps (players from the main detector, teams from jersey color, positions from the homography).

Metrics:
- landmark error: distance between predicted and labeled landmarks, in pixels (1920 px wide frames)
- pitch error: build the homography from *predicted* landmarks, project the *labeled* landmark pixels with it,
  and compare with their true position on the pitch: the error a player position would have, in meters.

Usage:
    python evaluate_pitch.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

from analytics import assign_teams  # noqa: E402
from pitch import PITCH_POINTS, feet, image_to_pitch, project, tactical_map  # noqa: E402
from train_pitch import PITCH_DATA, RUN  # noqa: E402

OUT = output_dir("football")
DOCS = Path(__file__).resolve().parent / "docs"
PLAYER_WEIGHTS = OUT / "runs" / "yolo11n_football" / "weights" / "best.pt"


def load_label(path, w, h):
    v = np.array(path.read_text().split(), float)
    k = v[5:].reshape(32, 3)
    k[:, 0] *= w
    k[:, 1] *= h
    return k


def main():
    model = YOLO(str(RUN / "weights" / "best.pt"))
    players = YOLO(str(PLAYER_WEIGHTS))
    images = sorted((PITCH_DATA / "test" / "images").glob("*.jpg"))
    px_err, pitch_err, oracle_err, n_ok, panels = [], [], [], 0, []
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = load_label(PITCH_DATA / "test" / "labels" / (p.stem + ".txt"), w, h)
        r = model.predict(img, imgsz=640, device="cpu", verbose=False)[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            continue
        kp = np.concatenate([r.keypoints.xy[0].numpy(), r.keypoints.conf[0].numpy()[:, None]], axis=1)
        vis = gt[:, 2] > 0
        # Floor: the same measurement with a homography fitted on the *labeled* landmarks (label noise, lens).
        Hg, _ = cv2.findHomography(gt[vis, :2].astype(np.float32), PITCH_POINTS[vis], 0)
        oracle_err += (np.linalg.norm(project(Hg, gt[vis, :2]) - PITCH_POINTS[vis], axis=1) / 100).tolist()
        both = vis & (kp[:, 2] >= 0.5)
        px_err += np.linalg.norm(kp[both, :2] - gt[both, :2], axis=1).tolist()
        H, used = image_to_pitch(kp)
        if H is None:
            continue
        n_ok += 1
        pitch_err += (np.linalg.norm(project(H, gt[vis, :2]) - PITCH_POINTS[vis], axis=1) / 100).tolist()  # cm → m
        if len(panels) < 3 and i % 8 == 0:  # three frames spread over the test set
            panels.append(draw_panel(img, kp, H, players))
    res = {
        "test_frames": len(images),
        "homography_found": n_ok,
        "landmark_error_px": {
            "median": round(float(np.median(px_err)), 1),
            "p90": round(float(np.percentile(px_err, 90)), 1),
            "n": len(px_err),
        },
        "pitch_error_m": {
            "median": round(float(np.median(pitch_err)), 2),
            "p90": round(float(np.percentile(pitch_err, 90)), 2),
            "n": len(pitch_err),
        },
        "pitch_error_m_from_labeled_landmarks": {
            "median": round(float(np.median(oracle_err)), 2),
            "p90": round(float(np.percentile(oracle_err, 90)), 2),
        },
    }
    print(json.dumps(res, indent=1))
    results = json.loads((DOCS / "results.json").read_text()) if (DOCS / "results.json").exists() else {}
    results["pitch"] = res
    (DOCS / "results.json").write_text(json.dumps(results, indent=1))
    if panels:
        cv2.imwrite(str(DOCS / "tactical_map.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_panel(img, kp, H, players):
    """Broadcast frame with landmarks | top-down map of the same moment."""
    frame = img.copy()
    r = players.predict(img, imgsz=960, device="cpu", verbose=False, conf=0.3)[0]
    boxes, cls = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int)
    pl, ball = boxes[cls == 1], boxes[cls == 0]
    teams = assign_teams(img, pl)[0] if len(pl) else np.zeros(0, int)
    for x, y, c in kp:
        if c >= 0.5:
            cv2.circle(frame, (int(x), int(y)), 9, (0, 255, 255), -1)
    for b, t in zip(pl, teams):
        col = {0: (255, 80, 0), 1: (0, 80, 255)}.get(int(t), (200, 200, 200))
        cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 3)
    ball_xy = project(H, feet(ball[:1]))[0] if len(ball) else None
    m = tactical_map(project(H, feet(pl)), teams, ball_xy)
    fh = 540
    frame = cv2.resize(frame, (int(frame.shape[1] * fh / frame.shape[0]), fh))
    m = cv2.resize(m, (int(m.shape[1] * fh / m.shape[0]), fh))
    return np.hstack([frame, m])


if __name__ == "__main__":
    main()

"""Pose-based fall detection: video → per-frame skeleton (YOLO11n-pose) → motion/posture features.

The key difficulty: *lying down* is not a fall. People lie on beds and floors on purpose, so a
detector must combine posture (horizontal body) with dynamics (a fast, uncontrolled descent).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = data_dir("fall")
OUT = output_dir("fall")
DOCS = Path(__file__).resolve().parent / "docs"
FPS = 30  # UR Fall frame rate; other sources pass their own fps (see video_fps)

# COCO keypoint indices
L_SH, R_SH, L_HIP, R_HIP, NOSE = 5, 6, 11, 12, 0
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
]

_model = None


def pose_model():
    global _model
    if _model is None:
        _model = YOLO(str(OUT / "yolo11n-pose.pt"))  # downloaded into cv-data on first use
    return _model


def read_frames(path, rgb_half=True):
    """UR Fall videos are depth | RGB side by side (640x240): keep the RGB half."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f[:, f.shape[1] // 2 :] if rgb_half else f)
    return frames


def video_fps(path, default=FPS):
    """Frame rate of a video file (falls back to 30 if the container doesn't say)."""
    fps = cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FPS)
    return fps if fps and fps > 1 else default


def extract_poses(frames):
    """Largest detected person per frame: keypoints (17, 3) [x, y, conf] and box; NaN if nobody."""
    kps = np.full((len(frames), 17, 3), np.nan, np.float32)
    boxes = np.full((len(frames), 4), np.nan, np.float32)
    for i, f in enumerate(frames):
        r = pose_model()(f, imgsz=320, conf=0.25, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        b = r.boxes.xyxy.numpy()
        k = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
        boxes[i] = b[k]
        kps[i] = np.concatenate([r.keypoints.xy.numpy()[k], r.keypoints.conf.numpy()[k][:, None]], axis=1)
    return kps, boxes


def _ffill(a):
    """Forward-fill NaNs along time (missed detections keep the last known value), then back-fill the start."""
    a = a.copy()
    for t in range(1, len(a)):
        if np.isnan(a[t]).any():
            a[t] = np.where(np.isnan(a[t]), a[t - 1], a[t])
    for t in range(len(a) - 2, -1, -1):
        if np.isnan(a[t]).any():
            a[t] = np.where(np.isnan(a[t]), a[t + 1], a[t])
    return np.nan_to_num(a)


def features(kps, boxes, frame_h=240, fps=FPS):
    """Per-frame features. Posture: torso angle, box aspect, heights. Dynamics: how fast those changed.
    Time windows are defined in seconds and converted to frames with `fps`, so any frame rate works."""
    s = fps / FPS  # 1.0 at 30 fps
    n = lambda frames_at_30fps: max(1, round(frames_at_30fps * s))  # noqa: E731
    T = len(kps)
    present = ~np.isnan(boxes[:, 0])
    boxes = _ffill(boxes)
    xy = kps[..., :2].copy()
    xy[kps[..., 2] < 0.3] = np.nan  # ignore low-confidence keypoints
    xy = _ffill(xy.reshape(T, -1)).reshape(T, 17, 2)

    sh = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    torso = sh - hip
    angle = np.degrees(np.arctan2(np.abs(torso[:, 0]), -torso[:, 1] + 1e-6))  # 0 = upright, 90 = horizontal
    angle = np.where(np.linalg.norm(torso, axis=1) < 3, np.nan, angle)
    angle = _ffill(angle[:, None])[:, 0]

    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    aspect = w / np.maximum(h, 1)
    hip_y = hip[:, 1] / frame_h
    head_y = xy[:, NOSE, 1] / frame_h
    top_y = boxes[:, 1] / frame_h
    height = h / frame_h

    def delta(x, lag):  # change over the last `lag` frames (per second)
        d = np.zeros_like(x)
        d[lag:] = (x[lag:] - x[:-lag]) * fps / lag
        return d

    def past_max(x, win):  # max over the last `win` frames
        return np.array([x[max(0, t - win + 1) : t + 1].max() for t in range(T)])

    v_head = delta(head_y, n(5))  # + = moving down; 5 frames = 1/6 s
    v_top = delta(top_y, n(5))
    v_angle = delta(angle, n(5))
    feats = np.stack(
        [
            angle,
            aspect,
            hip_y,
            head_y,
            top_y,
            height,
            present.astype(np.float32),
            v_head,
            v_top,
            v_angle,
            delta(height, n(5)),
            past_max(v_head, n(45)),
            past_max(v_top, n(45)),
            past_max(np.abs(v_angle), n(45)),  # dynamics within 1.5 s
            past_max(angle, n(15)),
            height / np.maximum(past_max(height, n(60)), 1e-3),  # shrink vs last 2 s
        ],
        axis=1,
    )
    names = [
        "torso_angle",
        "aspect",
        "hip_y",
        "head_y",
        "top_y",
        "height",
        "present",
        "v_head",
        "v_top",
        "v_angle",
        "v_height",
        "max_v_head_1.5s",
        "max_v_top_1.5s",
        "max_v_angle_1.5s",
        "max_angle_0.5s",
        "height_vs_2s_max",
    ]
    return np.nan_to_num(feats), names


def alarm_frames(prob, threshold=0.5, min_frames=10):
    """Raise an alarm when the per-frame fall probability stays above `threshold` for `min_frames` in a row.
    Returns the frame index of the first alarm, or None."""
    run = 0
    for t, p in enumerate(prob):
        run = run + 1 if p > threshold else 0
        if run >= min_frames:
            return t
    return None


def draw_skeleton(frame, kp, color=(0, 255, 0)):
    for a, b in SKELETON:
        if kp[a, 2] > 0.3 and kp[b, 2] > 0.3:
            cv2.line(frame, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), color, 2)
    for x, y, c in kp:
        if c > 0.3:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    return frame

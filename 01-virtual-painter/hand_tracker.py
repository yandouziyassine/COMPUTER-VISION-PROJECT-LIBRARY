# ─────────────────────────────────────────────
#  hand_tracker.py  —  MediaPipe hand tracking
# ─────────────────────────────────────────────

from collections import deque
import numpy as np
import mediapipe as mp
import cv2

from config import SMOOTHING_WINDOW
# MediaPipe landmark indices
TIP_IDS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
PIP_IDS  = [3, 6, 10, 14, 18]   # corresponding PIP joints


class HandTracker:
    """
    Wraps MediaPipe Hands.
    Detects a single hand, returns smoothed 21-landmark arrays.
    """

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._mp_draw = mp.solutions.drawing_utils
        # Separate smoothing buffer per landmark (21 × 2)
        self._buffers = [deque(maxlen=SMOOTHING_WINDOW) for _ in range(21)]

    # ── public ────────────────────────────────────────────────────────────

    def get_landmarks(self, frame_rgb):
        """
        Run detection on an RGB frame.
        Returns list of (x_px, y_px) tuples for all 21 landmarks,
        smoothed over the rolling window. Returns None if no hand detected.
        """
        h, w = frame_rgb.shape[:2]
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            # Flush buffers so old positions don't persist
            for buf in self._buffers:
                buf.clear()
            return None

        raw = results.multi_hand_landmarks[0]

        # Convert normalised → pixel, push into buffers
        smoothed = []
        for i, lm in enumerate(raw.landmark):
            px, py = int(lm.x * w), int(lm.y * h)
            self._buffers[i].append((px, py))
            xs = [p[0] for p in self._buffers[i]]
            ys = [p[1] for p in self._buffers[i]]
            smoothed.append((int(np.mean(xs)), int(np.mean(ys))))

        return smoothed

    def count_fingers(self, landmarks, handedness="Right"):
        """
        Returns the number of extended fingers (0–5).
        landmarks: list of (x, y) from get_landmarks().
        """
        if landmarks is None:
            return 0

        count = 0

        # Thumb  —  compare tip X to IP joint X (handedness-aware)
        thumb_tip = landmarks[TIP_IDS[0]]
        thumb_ip  = landmarks[TIP_IDS[0] - 1]
        if handedness == "Right":
            if thumb_tip[0] < thumb_ip[0]:
                count += 1
        else:
            if thumb_tip[0] > thumb_ip[0]:
                count += 1

        # Fingers 2–5  —  tip Y < PIP Y → extended
        for tip_id, pip_id in zip(TIP_IDS[1:], PIP_IDS[1:]):
            if landmarks[tip_id][1] < landmarks[pip_id][1]:
                count += 1

        return count

    def draw_skeleton(self, frame_bgr, landmarks):
        """Overlay a subtle hand skeleton for debugging / idle feedback."""
        if landmarks is None:
            return
        h, w = frame_bgr.shape[:2]
        # Reconstruct a NormalizedLandmarkList-like structure for mp_draw
        # Easier: just draw dots + key connections manually
        connections = self._mp_hands.HAND_CONNECTIONS
        for start, end in connections:
            p1 = landmarks[start]
            p2 = landmarks[end]
            cv2.line(frame_bgr, p1, p2, (80, 80, 80), 1, cv2.LINE_AA)
        for pt in landmarks:
            cv2.circle(frame_bgr, pt, 3, (160, 160, 160), -1, cv2.LINE_AA)

    def close(self):
        self._hands.close()
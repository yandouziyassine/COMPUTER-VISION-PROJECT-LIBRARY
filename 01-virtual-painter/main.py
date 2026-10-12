#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  main.py  —  Finger Paint — entry point
# ─────────────────────────────────────────────
#
#  Gestures:
#    ✊  Fist (0 fingers)   →  IDLE  — move freely, no drawing
#    ☝️  1 finger           →  DRAW  — paint with index fingertip
#    ✌️  2 fingers          →  ERASE — circular eraser at midpoint
#    ✌️ on color swatch     →  SELECT — change active drawing color
#
#  Keyboard shortcuts:
#    c  → clear canvas
#    e  → toggle eraser mode
#    q / ESC → quit
# ─────────────────────────────────────────────

import time
import cv2
import numpy as np

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    ERASER_RADIUS, BTN_ERASER_RADIUS, STROKE_THICKNESS,
    WINDOW_NAME,
)
from hand_tracker     import HandTracker
from gesture_detector import GestureDetector, Mode
from canvas           import Canvas
from toolbar          import Toolbar


def draw_hud(frame, mode: Mode, fps: float, eraser_mode: bool):
    """Overlay mode label and FPS counter."""
    mode_colors = {
        Mode.DRAW:  (100, 220, 100),
        Mode.ERASE: (100, 180, 255),
        Mode.IDLE:  (160, 160, 160),
    }
    mode_labels = {
        Mode.DRAW:  "DRAW",
        Mode.ERASE: "ERASE",
        Mode.IDLE:  "IDLE",
    }
    color = mode_colors[mode]
    label = mode_labels[mode]
    if eraser_mode and mode == Mode.IDLE:
        label = "ERASER MODE"
        color = (100, 220, 100)

    cv2.putText(frame, label,         (14, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} fps", (frame.shape[1] - 80, frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


def draw_cursor(frame, state, mode: Mode, eraser_mode: bool, radius: int):
    """Render fingertip dot or eraser circle."""
    if state.index_tip is None:
        return

    if mode == Mode.DRAW and not eraser_mode:
        cv2.circle(frame, state.index_tip, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, state.index_tip, 7, (0, 0, 0),       1,  cv2.LINE_AA)

    elif mode == Mode.ERASE or eraser_mode:
        center = state.midpoint if mode == Mode.ERASE else state.index_tip
        if center:
            r = radius
            cv2.circle(frame, center, r, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, center, 4, (80, 80, 255), -1, cv2.LINE_AA)


def main():
    # ── Initialise hardware / trackers ────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera resolution: {actual_w}×{actual_h}")

    tracker  = HandTracker()
    detector = GestureDetector()
    canvas   = Canvas(actual_w, actual_h)
    toolbar  = Toolbar(actual_w)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_pos   = None   # previous action point for stroke continuity
    prev_mode  = Mode.IDLE

    t_prev = time.perf_counter()
    fps    = 0.0

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Dropped frame.")
            continue

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # ── Hand tracking ─────────────────────────────────────────────────
        rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks  = tracker.get_landmarks(rgb)
        fingers    = tracker.count_fingers(landmarks)
        state      = detector.update(landmarks, fingers)
        mode       = state.mode

        # ── Toolbar interaction (2-finger midpoint OR index tip in toolbar) ─
        if landmarks:
            probe = state.midpoint if mode == Mode.ERASE else state.index_tip
            if probe and toolbar.point_in_toolbar(probe):
                hit = toolbar.hit_test(*probe)
                if hit:
                    if hit.kind == "color" and mode == Mode.ERASE:
                        toolbar.active_color = hit.value
                        toolbar.eraser_mode  = False
                    elif hit.kind == "clear":
                        canvas.clear()
                    elif hit.kind == "eraser":
                        toolbar.eraser_mode = not toolbar.eraser_mode
                # Don't draw while in toolbar
                prev_pos = None

            else:
                # ── Canvas actions ────────────────────────────────────────
                # Reset continuity on mode change
                if mode != prev_mode:
                    prev_pos = None

                if mode == Mode.DRAW and not toolbar.eraser_mode:
                    if prev_pos and state.position:
                        canvas.draw_stroke(
                            prev_pos, state.position,
                            toolbar.active_color,
                            STROKE_THICKNESS,
                        )
                    prev_pos = state.position

                elif mode == Mode.ERASE or toolbar.eraser_mode:
                    r = BTN_ERASER_RADIUS if toolbar.eraser_mode else ERASER_RADIUS
                    center = state.index_tip if toolbar.eraser_mode else state.midpoint
                    canvas.erase(center, r)
                    prev_pos = None

                else:
                    prev_pos = None
        else:
            prev_pos = None

        prev_mode = mode

        # ── Render ────────────────────────────────────────────────────────
        canvas.composite(frame)

        # Optional: skeleton in IDLE
        if mode == Mode.IDLE:
            tracker.draw_skeleton(frame, landmarks)

        toolbar.render(frame)

        # FPS
        t_now = time.perf_counter()
        fps   = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now

        draw_hud(frame, mode, fps, toolbar.eraser_mode)
        draw_cursor(frame, state, mode, toolbar.eraser_mode,
                    BTN_ERASER_RADIUS if toolbar.eraser_mode else ERASER_RADIUS)

        cv2.imshow(WINDOW_NAME, frame)

        # ── Keyboard shortcuts ────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break
        elif key == ord('c'):
            canvas.clear()
        elif key == ord('e'):
            toolbar.eraser_mode = not toolbar.eraser_mode

    # ── Cleanup ───────────────────────────────────────────────────────────
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Bye!")


if __name__ == "__main__":
    main()
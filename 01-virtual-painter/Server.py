#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  Server.py  —  FastAPI web interface
#  Run with:  python Server.py
#  Then open: http://localhost:8000
# ─────────────────────────────────────────────

import threading
import time
import cv2
import numpy as np
import os
import signal

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    ERASER_RADIUS,
    BTN_ERASER_RADIUS,
    STROKE_THICKNESS,
)
from hand_tracker import HandTracker
from gesture_detector import GestureDetector, Mode
from canvas import Canvas
from toolbar import Toolbar

# ── Shared state ──────────────────────────────────────────────────────────────


class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.fps = 0.0
        self.mode = "IDLE"
        self.color_name = "Blue"
        self.eraser_mode = False
        self.running = False
        self.camera_active = True  # <-- Ajout de l'état de la caméra


state = AppState()

# ── Pipeline thread ───────────────────────────────────────────────────────────


def pipeline_loop():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        state.running = False
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {actual_w}x{actual_h}")

    tracker = HandTracker()
    detector = GestureDetector()
    canvas = Canvas(actual_w, actual_h)
    toolbar = Toolbar(actual_w)

    prev_pos = None
    prev_mode = Mode.IDLE
    t_prev = time.perf_counter()
    fps = 0.0

    from config import PALETTE

    color_map = {bgr: name for name, bgr in PALETTE}

    while state.running:
        # --- GESTION DE L'ÉTAT DE LA CAMÉRA ---
        with state.lock:
            cam_active = state.camera_active

        if not cam_active:
            if cap.isOpened():
                cap.release()  # Libère la ressource physique
                print("[INFO] Caméra désactivée par l'utilisateur.")

            # Génère une image noire avec un message
            blank_frame = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)
            cv2.putText(
                blank_frame,
                "Camera en pause",
                (actual_w // 2 - 140, actual_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            _, jpeg = cv2.imencode(".jpg", blank_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state.lock:
                state.latest_frame = jpeg.tobytes()

            time.sleep(0.1)  # Repos du processeur
            continue

        # Si la caméra doit être active mais est actuellement fermée
        if not cap.isOpened():
            print("[INFO] Réactivation de la caméra...")
            cap.open(CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not cap.isOpened():
                print("[ERROR] Impossible de rouvrir la caméra.")
                time.sleep(1)
                continue
        # --------------------------------------

        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = tracker.get_landmarks(rgb)
        fingers = tracker.count_fingers(landmarks)
        gs = detector.update(landmarks, fingers)
        mode = gs.mode

        # Toolbar interaction
        if landmarks:
            probe = gs.midpoint if mode == Mode.ERASE else gs.index_tip
            if probe and toolbar.point_in_toolbar(probe):
                hit = toolbar.hit_test(*probe)
                if hit:
                    if hit.kind == "color" and mode == Mode.ERASE:
                        toolbar.active_color = hit.value
                        toolbar.eraser_mode = False
                    elif hit.kind == "clear":
                        canvas.clear()
                    elif hit.kind == "eraser":
                        toolbar.eraser_mode = not toolbar.eraser_mode
                prev_pos = None
            else:
                if mode != prev_mode:
                    prev_pos = None

                if mode == Mode.DRAW and not toolbar.eraser_mode:
                    if prev_pos and gs.position:
                        canvas.draw_stroke(prev_pos, gs.position, toolbar.active_color, STROKE_THICKNESS)
                    prev_pos = gs.position
                elif mode == Mode.ERASE or toolbar.eraser_mode:
                    r = BTN_ERASER_RADIUS if toolbar.eraser_mode else ERASER_RADIUS
                    center = gs.index_tip if toolbar.eraser_mode else gs.midpoint
                    canvas.erase(center, r)
                    prev_pos = None
                else:
                    prev_pos = None
        else:
            prev_pos = None

        prev_mode = mode

        # Render
        canvas.composite(frame)
        if mode == Mode.IDLE:
            tracker.draw_skeleton(frame, landmarks)
        toolbar.render(frame)

        # FPS
        t_now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # HUD
        mode_labels = {Mode.DRAW: "DRAW", Mode.ERASE: "ERASE", Mode.IDLE: "IDLE"}
        mode_colors = {Mode.DRAW: (100, 220, 100), Mode.ERASE: (100, 180, 255), Mode.IDLE: (160, 160, 160)}
        cv2.putText(
            frame,
            mode_labels[mode],
            (14, actual_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            mode_colors[mode],
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"{fps:.0f} fps",
            (actual_w - 80, actual_h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        # Cursor
        if gs.index_tip:
            if mode == Mode.DRAW and not toolbar.eraser_mode:
                cv2.circle(frame, gs.index_tip, 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, gs.index_tip, 7, (0, 0, 0), 1, cv2.LINE_AA)
            elif mode == Mode.ERASE or toolbar.eraser_mode:
                center = gs.midpoint if mode == Mode.ERASE else gs.index_tip
                if center:
                    r = BTN_ERASER_RADIUS if toolbar.eraser_mode else ERASER_RADIUS
                    cv2.circle(frame, center, r, (80, 80, 255), 2, cv2.LINE_AA)

        # Encode to JPEG
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with state.lock:
            state.latest_frame = jpeg.tobytes()
            state.fps = fps
            state.mode = mode_labels[mode]
            state.color_name = color_map.get(toolbar.active_color, "Custom")
            state.eraser_mode = toolbar.eraser_mode

    if cap.isOpened():
        cap.release()
    tracker.close()
    print("[INFO] Pipeline stopped.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="Finger Paint Stream")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def mjpeg_generator():
    """Yield MJPEG frames for the /video_feed endpoint."""
    while True:
        with state.lock:
            frame = state.latest_frame
        if frame is None:
            time.sleep(0.02)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def get_status():
    with state.lock:
        return {
            "fps": round(state.fps, 1),
            "mode": state.mode,
            "color": state.color_name,
            "eraser_mode": state.eraser_mode,
            "running": state.running,
        }


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return f.read()


# ── Routes d'actions ────────────────────────────────────────────────────────


@app.post("/toggle_camera")
def toggle_camera():
    """Endpoint pour activer/désactiver la webcam."""
    with state.lock:
        state.camera_active = not state.camera_active
        active = state.camera_active
    return {"camera_active": active}


@app.post("/quit")
def quit_app():
    """Endpoint pour arrêter le serveur depuis l'interface web."""
    print("Signal de fermeture reçu depuis l'interface web. Arrêt du serveur...")

    with state.lock:
        state.running = False

    os.kill(os.getpid(), signal.SIGINT)
    return {"message": "Serveur en cours d'arrêt..."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    state.running = True
    thread = threading.Thread(target=pipeline_loop, daemon=True)
    thread.start()
    print("[INFO] Open http://localhost:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")

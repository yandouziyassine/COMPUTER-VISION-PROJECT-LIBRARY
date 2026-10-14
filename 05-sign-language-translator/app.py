"""Webcam fingerspelling translator: show static ASL letters to the camera, they are typed as text.

MediaPipe finds the hand, the crop is converted to the dataset's format (grayscale, square, 28x28) and the CNN
reads the letter. A letter is typed when it stays the top prediction for HOLD frames; no hand for ~1.5 s types
a space. Keys: Backspace deletes, C clears, Esc quits.

Usage:
    python app.py [--camera 0]
"""

import argparse
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from asl import DATA, LETTERS, hand_crop_to_input, load_model, predict_proba

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HOLD, MIN_CONF, SPACE_AFTER = 12, 0.8, 1.5


def hand_landmarker():
    path = DATA / "hand_landmarker.task"
    if not path.exists():
        print("downloading the MediaPipe hand model (7 MB)…", flush=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    opts = vision.HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=1)
    return vision.HandLandmarker.create_from_options(opts)


def hand_box(landmarks, w, h, margin=0.25):
    xs = np.array([p.x for p in landmarks]) * w
    ys = np.array([p.y for p in landmarks]) * h
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) * (0.5 + margin)
    return [int(np.clip(v, 0, lim)) for v, lim in ((cx - half, w), (cy - half, h), (cx + half, w), (cy + half, h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    model, detector = load_model("cnn_aug"), hand_landmarker()
    cap = cv2.VideoCapture(args.camera)
    text, streak, last_letter, last_hand = "", 0, None, time.time()
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect_for_video(rgb, int((time.time() - t_start) * 1000))
        label = ""
        if res.hand_landmarks:
            last_hand = time.time()
            x1, y1, x2, y2 = hand_box(res.hand_landmarks[0], w, h)
            gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if gray.size:
                probs = predict_proba(model, hand_crop_to_input(gray)[None])[0]
                k = int(probs.argmax())
                label = f"{LETTERS[k]} {probs[k]:.0%}"
                streak = streak + 1 if (k == last_letter and probs[k] >= MIN_CONF) else 0
                last_letter = k
                if streak == HOLD:                    # typed once per hold
                    text += LETTERS[k]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (31, 91, 255), 2)
                cv2.rectangle(frame, (x1, y1 - int(6 + 22 * min(streak, HOLD) / HOLD)), (x1 + 8, y1), (31, 91, 255), -1)
        elif text and not text.endswith(" ") and time.time() - last_hand > SPACE_AFTER:
            text += " "
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.rectangle(frame, (0, h - 60), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, text[-40:] + "_", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.imshow("Fingerspelling translator (Esc to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == 8:
            text = text[:-1]
        if key in (ord("c"), ord("C")):
            text = ""
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

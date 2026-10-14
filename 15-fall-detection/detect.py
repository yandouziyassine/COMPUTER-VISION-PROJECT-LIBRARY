"""Run the trained fall detector on any video and save an annotated copy.

Usage:
    python detect.py my_video.mp4                      # full frame
    python detect.py fall-05-cam0.mp4 --ur-fall        # UR Fall format (depth | RGB): RGB half only
    python detect.py 0                                 # webcam (press q to quit)

Requires fall_model.joblib, produced by `python evaluate.py`.
"""

import argparse

import cv2
import joblib
import numpy as np

from fall import FPS, OUT, alarm_frames, draw_skeleton, extract_poses, features, read_frames, video_fps


def alarm_len(fps):
    """The alarm needs the probability high for 1/3 s: 10 frames at 30 fps."""
    return max(1, round(10 * fps / FPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--ur-fall", action="store_true", help="input is a UR Fall depth|RGB video")
    ap.add_argument("--out", default=str(OUT / "detected.mp4"))
    args = ap.parse_args()
    bundle = joblib.load(OUT / "fall_model.joblib")
    model = bundle["model"]

    if args.video == "0":  # live webcam: score a rolling 3-second window
        cap, kps_all, boxes_all = cv2.VideoCapture(0), [], []
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        window = max(1, round(3 * fps))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            k, b = extract_poses([f])
            kps_all.append(k[0])
            boxes_all.append(b[0])
            kps_all, boxes_all = kps_all[-window:], boxes_all[-window:]
            F, _ = features(np.array(kps_all), np.array(boxes_all), frame_h=f.shape[0], fps=fps)
            p = model.predict_proba(F)[:, 1]
            draw_skeleton(f, np.nan_to_num(k[0]))
            if alarm_frames(p[-round(0.5 * fps) :], 0.5, alarm_len(fps)) is not None:
                cv2.putText(f, "FALL ALARM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            cv2.imshow("fall detection", f)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return

    fps = video_fps(args.video)
    frames = read_frames(args.video, rgb_half=args.ur_fall)
    kps, boxes = extract_poses(frames)
    F, _ = features(kps, boxes, frame_h=frames[0].shape[0], fps=fps)
    prob = model.predict_proba(F)[:, 1]
    alarm = alarm_frames(prob, 0.5, alarm_len(fps))
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for t, f in enumerate(frames):
        draw_skeleton(f, np.nan_to_num(kps[t]))
        cv2.rectangle(f, (0, h - 8), (int(prob[t] * w), h), (0, 0, 255) if prob[t] > 0.5 else (0, 200, 0), -1)
        if alarm is not None and t >= alarm:
            cv2.putText(f, "FALL ALARM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        out.write(f)
    out.release()
    print(f"alarm at {alarm / fps:.1f} s" if alarm is not None else "no fall detected", "→", args.out)


if __name__ == "__main__":
    main()

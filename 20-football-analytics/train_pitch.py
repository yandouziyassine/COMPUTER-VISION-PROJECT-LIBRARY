"""Fine-tune YOLO11n-pose to find the 32 pitch landmarks (corners, box corners, halfway line, centre circle).

Data: "Football pitch keypoints detection" (Kaggle, MIT; a Roboflow export of Bundesliga broadcast frames),
222 train / 30 valid / 24 test images, one "pitch" object with 32 keypoints per image.

Usage:
    kaggle datasets download danielmachniak/football-pitch-keypoints-detection -p ../../cv-data/football/pitch2 --unzip
    python train_pitch.py --epochs 60
    python train_pitch.py --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

PITCH_DATA = data_dir("football") / "pitch2" / "football-pitch-keypoints-detection"
OUT = output_dir("football")
RUN = OUT / "runs" / "yolo11n_pitch"
FLIP_IDX = [24, 25, 26, 27, 28, 29, 22, 23, 21, 17, 18, 19, 20, 13, 14, 15, 16, 9, 10, 11, 12, 8, 6, 7,
            0, 1, 2, 3, 4, 5, 31, 30]


def write_yaml():
    path = PITCH_DATA.parent / "pitch.yaml"
    path.write_text(
        f"path: {PITCH_DATA.as_posix()}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"kpt_shape: [32, 3]\nflip_idx: {FLIP_IDX}\nnames:\n  0: pitch\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.resume:
        YOLO(str(RUN / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(str(OUT / "yolo11n-pose.pt")).train(     # downloaded there on first use
        data=str(write_yaml()),
        epochs=args.epochs,
        imgsz=640,
        batch=8,
        device="cpu",
        workers=0,
        mosaic=0.0,          # a mosaic of four pitches breaks the one-pitch-per-image geometry
        degrees=0.0,
        project=str(OUT / "runs"),
        name="yolo11n_pitch",
        exist_ok=True,
        seed=0,
        plots=False,
    )


if __name__ == "__main__":
    main()

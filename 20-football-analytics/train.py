"""Fine-tune YOLO11n to detect players and the ball in broadcast football frames, on CPU.

The ball is tiny (a few pixels), so training uses 960 px instead of the usual 640.

Usage:
    python prepare.py
    python train.py --epochs 15
    python train.py --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

DATA = data_dir("football")
OUT = output_dir("football")
RUN = OUT / "runs" / "yolo11n_football"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.resume:
        YOLO(str(RUN / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(str(OUT / "yolo11n.pt")).train(
        data=str(DATA / "football.yaml"),
        epochs=args.epochs,
        imgsz=960,
        batch=8,
        device="cpu",
        workers=0,
        project=str(OUT / "runs"),
        name="yolo11n_football",
        exist_ok=True,
        plots=True,
        verbose=False,
        seed=0,
    )


if __name__ == "__main__":
    main()

"""Fine-tune YOLO11n on densely packed retail shelves (SKU-110K subset), on CPU.

Usage:
    python prepare.py
    python train.py --epochs 30
    python train.py --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

DATA = data_dir("shelf")
OUT = output_dir("shelf")
RUN = OUT / "runs" / "yolo11n_shelf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.resume:
        YOLO(str(RUN / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(str(OUT / "yolo11n.pt")).train(
        data=str(DATA / "shelf.yaml"),
        epochs=args.epochs,
        imgsz=960,
        batch=8,
        device="cpu",
        workers=0,
        max_det=1000,  # shelves hold 100–400+ products; the default cap of 300 would cut them off
        project=str(OUT / "runs"),
        name="yolo11n_shelf",
        exist_ok=True,
        plots=True,
        verbose=False,
        seed=0,
    )


if __name__ == "__main__":
    main()

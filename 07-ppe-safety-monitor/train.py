"""Fine-tune YOLO11n (COCO-pretrained) to detect heads with / without a hard hat, on CPU.

Usage:
    python prepare.py
    python train.py --epochs 10
    python train.py --resume          # continue an interrupted run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

DATA = data_dir("ppe")
OUT = output_dir("ppe")
RUN = OUT / "runs" / "yolo11n_ppe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.resume:
        YOLO(str(RUN / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(str(OUT / "yolo11n.pt")).train(
        data=str(DATA / "ppe.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=16,
        device="cpu",
        workers=0,
        project=str(OUT / "runs"),
        name="yolo11n_ppe",
        exist_ok=True,
        plots=True,
        verbose=False,
        seed=0,
    )


if __name__ == "__main__":
    main()

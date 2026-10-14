"""Fine-tune YOLO11n (COCO-pretrained) to detect license plates, on CPU.

Usage:
    python prepare.py
    python train.py --epochs 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

from ultralytics import YOLO  # noqa: E402

DATA = data_dir("plates")
OUT = output_dir("plates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run from last.pt")
    args = ap.parse_args()
    if args.resume:  # Ultralytics saves last.pt every epoch, with optimizer state
        YOLO(str(OUT / "runs" / "yolo11n_plates" / "weights" / "last.pt")).train(resume=True)
        return
    model = YOLO(str(OUT / "yolo11n.pt"))  # start from COCO weights: transfer learning
    model.train(
        data=str(DATA / "plates.yaml"),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=16,
        device="cpu",
        workers=0,
        project=str(OUT / "runs"),
        name="yolo11n_plates",
        exist_ok=True,
        plots=True,
        verbose=False,
        seed=0,
        patience=50,
    )


if __name__ == "__main__":
    main()

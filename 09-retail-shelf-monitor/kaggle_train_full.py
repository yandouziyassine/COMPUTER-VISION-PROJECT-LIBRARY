"""Kaggle GPU job for project 09: YOLO11n on the full SKU-110K train split (8,219 images).

The repo's local test set (488 images) comes from the official *validation* split, so model selection here
uses 500 images of the official *test* split instead: the local test images are never seen."""

import random
import subprocess
import sys
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175"], check=True)
from ultralytics import YOLO

SRC = next(Path("/kaggle/input").rglob("SKU110K_fixed"))
WORK = Path("/kaggle/working")
train = sorted(str(p) for p in (SRC / "images" / "train").glob("*.jpg"))
test = sorted(str(p) for p in (SRC / "images" / "test").glob("*.jpg"))
random.Random(0).shuffle(test)
(WORK / "train.txt").write_text("\n".join(train))
(WORK / "select.txt").write_text("\n".join(test[:500]))
(WORK / "sku.yaml").write_text(f"path: {WORK}\ntrain: {WORK}/train.txt\nval: {WORK}/select.txt\nnames:\n  0: object\n")
print("train", len(train), "select", 500, flush=True)
YOLO("yolo11n.pt").train(
    data=str(WORK / "sku.yaml"),
    epochs=40,
    imgsz=960,
    batch=16,
    device=0,
    workers=4,
    max_det=1000,
    project=str(WORK / "runs"),
    name="yolo11n_shelf_full",
    exist_ok=True,
    seed=0,
    plots=False,
)

"""Kaggle GPU job for project 06: YOLO11n on the FULL plate train split (6,176 images).
Selection on the full official valid split (1,765 images); the test split is never touched here,
evaluation on it happens locally. Same labels as prepare.py: every box is class 0, license_plate."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

ROOT = Path("/kaggle/temp/plates")


def export(split_zip, out_split):
    path = hf_hub_download("keremberke/license-plate-object-detection", f"data/{split_zip}", repo_type="dataset")
    z = zipfile.ZipFile(path)
    d = json.loads(z.read("_annotations.coco.json"))
    items = {im["id"]: (im, []) for im in d["images"]}
    for a in d["annotations"]:
        items[a["image_id"]][1].append(a["bbox"])
    img_dir, lab_dir = ROOT / out_split / "images", ROOT / out_split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for k, (im, boxes) in enumerate(sorted(items.values(), key=lambda t: t[0]["file_name"])):
        w, h = im["width"], im["height"]
        (img_dir / f"{k:05d}.jpg").write_bytes(z.read(im["file_name"]))
        (lab_dir / f"{k:05d}.txt").write_text(
            "\n".join(
                f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}" for x, y, bw, bh in boxes
            )
        )
    return len(items)


n_tr, n_va = export("train.zip", "train"), export("valid.zip", "val")
print("train", n_tr, "val", n_va, flush=True)
(ROOT / "plates.yaml").write_text(f"path: {ROOT}\ntrain: train/images\nval: val/images\nnames:\n  0: license_plate\n")
YOLO("yolo11n.pt").train(
    data=str(ROOT / "plates.yaml"),
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    project="/kaggle/working/runs",
    name="yolo11n_plates_full",
    exist_ok=True,
    seed=0,
    plots=False,
)

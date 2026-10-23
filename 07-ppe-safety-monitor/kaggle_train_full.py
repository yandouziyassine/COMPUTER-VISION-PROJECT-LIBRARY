"""Kaggle GPU job for project 07: YOLO11n on the FULL Hard Hat Workers train/valid splits.
Same class mapping by name as prepare.py in the repo; evaluation on the test split happens locally."""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.175", "huggingface_hub"], check=True)
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

NAMES = ["hardhat", "no-hardhat"]
ROOT = Path("/kaggle/temp/ppe")


def export(split_zip, out_split):
    z = zipfile.ZipFile(hf_hub_download("keremberke/hard-hat-detection", f"data/{split_zip}", repo_type="dataset"))
    d = json.loads(z.read("_annotations.coco.json"))
    to_idx = {c["id"]: NAMES.index(c["name"]) for c in d["categories"] if c["name"] in NAMES}
    items = {im["id"]: (im, []) for im in d["images"]}
    for a in d["annotations"]:
        if a["category_id"] in to_idx:
            items[a["image_id"]][1].append((to_idx[a["category_id"]], a["bbox"]))
    img_dir, lab_dir = ROOT / out_split / "images", ROOT / out_split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for k, (im, anns) in enumerate(sorted(items.values(), key=lambda t: t[0]["file_name"])):
        w, h = im["width"], im["height"]
        (img_dir / f"{k:05d}.jpg").write_bytes(z.read(im["file_name"]))
        (lab_dir / f"{k:05d}.txt").write_text(
            "\n".join(
                f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                for c, (x, y, bw, bh) in anns
            )
        )
    return len(items)


n_tr, n_va = export("train.zip", "train"), export("valid.zip", "val")
print("train", n_tr, "val", n_va, flush=True)
(ROOT / "ppe.yaml").write_text(
    f"path: {ROOT}\ntrain: train/images\nval: val/images\nnames:\n  0: hardhat\n  1: no-hardhat\n"
)
YOLO("yolo11n.pt").train(
    data=str(ROOT / "ppe.yaml"),
    epochs=50,
    imgsz=640,
    batch=32,
    device=0,
    workers=4,
    project="/kaggle/working/runs",
    name="yolo11n_ppe_full",
    exist_ok=True,
    seed=0,
    plots=False,
)

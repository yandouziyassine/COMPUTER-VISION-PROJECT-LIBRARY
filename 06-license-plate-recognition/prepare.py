"""Convert the Roboflow plate dataset (COCO JSON) to the YOLO layout and write the dataset YAMLs.

Only the subsets actually used are copied, into yolo/<split>/images + yolo/<split>/labels.
Why not write labels next to the original images? The dataset lives on a FAT32 USB key, and FAT32
allows ~65k directory entries per folder; long filenames use several entries each, so 6,176 images
+ 6,176 label files overflow the train folder ("Permission denied" on write).

Usage:
    python prepare.py --train-subset 1500 --val-subset 300
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

DATA = data_dir("plates")
YOLO_DIR = DATA / "yolo"


def load(split):
    d = json.loads((DATA / split / "_annotations.coco.json").read_text())
    items = {im["id"]: (im, []) for im in d["images"]}
    for a in d["annotations"]:
        items[a["image_id"]][1].append(a["bbox"])
    return sorted(items.values(), key=lambda t: t[0]["file_name"])


def export(items, split, out_split):
    img_dir, lab_dir = YOLO_DIR / out_split / "images", YOLO_DIR / out_split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for k, (im, boxes) in enumerate(items):
        w, h = im["width"], im["height"]
        name = f"{out_split}_{k:05d}"  # short names: FAT32-friendly
        shutil.copyfile(DATA / split / im["file_name"], img_dir / f"{name}.jpg")
        lines = [f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}" for x, y, bw, bh in boxes]
        (lab_dir / f"{name}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subset", type=int, default=1500)
    ap.add_argument("--val-subset", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(0)
    train = rng.sample(load("train"), args.train_subset)
    val = rng.sample(load("valid"), args.val_subset)
    test = load("test")
    for items, split, out in ((train, "train", "train"), (val, "valid", "val"), (test, "test", "test")):
        export(items, split, out)
    for name, val_split in (("plates.yaml", "val"), ("plates_test.yaml", "test")):
        (DATA / name).write_text(
            f"path: {YOLO_DIR.as_posix()}\ntrain: train/images\nval: {val_split}/images\nnames:\n  0: license_plate\n"
        )
    print(f"train {len(train)} | val {len(val)} | test {len(test)} images exported")


if __name__ == "__main__":
    main()

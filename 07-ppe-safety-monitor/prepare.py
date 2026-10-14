"""Extract a YOLO-format subset of the hard-hat dataset straight from the downloaded zips.

The full train split (13,782 images with long names) would overflow a FAT32 folder, so only the
subset used is extracted, with short names, into yolo/<split>/images + labels.

Usage:
    python prepare.py --train-subset 1500 --val-subset 300
"""

import argparse
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

DATA = data_dir("ppe")
YOLO_DIR = DATA / "yolo"
NAMES = ["hardhat", "no-hardhat"]


def export(zip_name, out_split, n, rng):
    z = zipfile.ZipFile(DATA / zip_name)
    d = json.loads(z.read("_annotations.coco.json"))
    # Map COCO category ids to our class indices *by name*, so a reordered dataset can't swap classes.
    to_idx = {c["id"]: NAMES.index(c["name"]) for c in d["categories"] if c["name"] in NAMES}
    items = {im["id"]: (im, []) for im in d["images"]}
    for a in d["annotations"]:
        if a["category_id"] in to_idx:
            items[a["image_id"]][1].append((to_idx[a["category_id"]], a["bbox"]))
    items = sorted(items.values(), key=lambda t: t[0]["file_name"])
    if n and n < len(items):
        items = rng.sample(items, n)
    img_dir, lab_dir = YOLO_DIR / out_split / "images", YOLO_DIR / out_split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for k, (im, anns) in enumerate(items):
        w, h = im["width"], im["height"]
        name = f"{out_split}_{k:05d}"
        (img_dir / f"{name}.jpg").write_bytes(z.read(im["file_name"]))
        (lab_dir / f"{name}.txt").write_text(
            "\n".join(
                f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                for c, (x, y, bw, bh) in anns
            )
        )
    return len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subset", type=int, default=1500)
    ap.add_argument("--val-subset", type=int, default=300)
    args = ap.parse_args()
    rng = random.Random(0)
    n = {
        s: export(z, s, k, rng)
        for z, s, k in (
            ("train.zip", "train", args.train_subset),
            ("valid.zip", "val", args.val_subset),
            ("test.zip", "test", 0),
        )
    }
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(NAMES))
    for yaml_name, val in (("ppe.yaml", "val"), ("ppe_test.yaml", "test")):
        (DATA / yaml_name).write_text(
            f"path: {YOLO_DIR.as_posix()}\ntrain: train/images\nval: {val}/images\nnames:\n{names}"
        )
    print(n)


if __name__ == "__main__":
    main()

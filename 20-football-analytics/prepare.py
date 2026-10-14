"""Extract the football dataset (players + ball, COCO JSON inside zips) to the YOLO layout.

Usage:
    python prepare.py
"""

import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

DATA = data_dir("football")
YOLO_DIR = DATA / "yolo"
NAMES = ["ball", "player"]
SOURCE_NAMES = {"football": "ball", "player": "player"}  # dataset category name → our class name


def export(zip_name, split):
    z = zipfile.ZipFile(DATA / zip_name)
    d = json.loads(z.read("_annotations.coco.json"))
    # Map category ids by name, so a reordered dataset can't swap classes.
    to_idx = {c["id"]: NAMES.index(SOURCE_NAMES[c["name"]]) for c in d["categories"] if c["name"] in SOURCE_NAMES}
    items = {im["id"]: (im, []) for im in d["images"]}
    for a in d["annotations"]:
        if a["category_id"] in to_idx:
            items[a["image_id"]][1].append((to_idx[a["category_id"]], a["bbox"]))
    img_dir, lab_dir = YOLO_DIR / split / "images", YOLO_DIR / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for k, (im, anns) in enumerate(sorted(items.values(), key=lambda t: t[0]["file_name"])):
        w, h = im["width"], im["height"]
        name = f"{split}_{k:04d}"
        (img_dir / f"{name}.jpg").write_bytes(z.read(im["file_name"]))
        (lab_dir / f"{name}.txt").write_text(
            "\n".join(
                f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                for c, (x, y, bw, bh) in anns
            )
        )
    return len(items)


def main():
    n = {s: export(f"{z}.zip", s) for z, s in (("train", "train"), ("valid", "val"), ("test", "test"))}
    names = "".join(f"  {i}: {c}\n" for i, c in enumerate(NAMES))
    for yaml_name, v in (("football.yaml", "val"), ("football_test.yaml", "test")):
        (DATA / yaml_name).write_text(
            f"path: {YOLO_DIR.as_posix()}\ntrain: train/images\nval: {v}/images\nnames:\n{names}"
        )
    print(n)


if __name__ == "__main__":
    main()

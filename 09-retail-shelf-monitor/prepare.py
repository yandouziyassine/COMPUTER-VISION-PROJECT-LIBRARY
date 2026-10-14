"""Export SKU-110K shards (parquet, COCO xywh boxes) to the YOLO layout.

- train: train shard 00008 (219 shelf photos)
- val:   first 100 images of the validation shard
- test:  remaining validation images (~488)
Images are shrunk to a max side of 1280 px (originals are up to 4160 px): it saves disk space on the
USB key and CPU time, and YOLO trains at 960 px anyway. Labels are normalized, so they don't change.

Usage:
    python prepare.py
"""

import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

DATA = data_dir("shelf")
YOLO_DIR = DATA / "yolo"
MAX_SIDE = 1280


def export(rows, split):
    img_dir, lab_dir = YOLO_DIR / split / "images", YOLO_DIR / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    n_boxes = 0
    for k, r in enumerate(rows):
        try:
            img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        except OSError:  # a few SKU-110K JPEGs are known to be corrupt
            continue
        w, h = r["width"], r["height"]
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        name = f"{split}_{k:04d}"
        img.save(img_dir / f"{name}.jpg", quality=90)
        lines = [
            f"0 {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
            for x, y, bw, bh in r["objects"]["bbox"]
        ]
        (lab_dir / f"{name}.txt").write_text("\n".join(lines))
        n_boxes += len(lines)
    return n_boxes


def main():
    train = pq.read_table(DATA / "train-00008-of-00009.parquet").to_pylist()
    val = pq.read_table(DATA / "validation-00000-of-00001.parquet").to_pylist()
    counts = {"train": export(train, "train"), "val": export(val[:100], "val"), "test": export(val[100:], "test")}
    for name, v in (("shelf.yaml", "val"), ("shelf_test.yaml", "test")):
        (DATA / name).write_text(
            f"path: {YOLO_DIR.as_posix()}\ntrain: train/images\nval: {v}/images\nnames:\n  0: product\n"
        )
    print({"images": {"train": len(train), "val": 100, "test": len(val) - 100}, "boxes": counts})


if __name__ == "__main__":
    main()

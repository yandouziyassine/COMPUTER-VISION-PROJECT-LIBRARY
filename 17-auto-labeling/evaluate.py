"""How good are zero-shot auto-labels? Grounding DINO (text prompt only, never trained on these
labels) vs YOLO11n (trained on COCO), scored with the official COCO mAP on a 10-class subset.

Also renders an open-vocabulary demo with SAM masks on prompts that are *not* COCO classes.

Usage:
    python evaluate.py --n 200
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from autolabel import GroundingDino, Segmenter  # noqa: E402

COCO_DIR = data_dir("clip-search")
OUT = output_dir("autolabel")
DOCS = Path(__file__).resolve().parent / "docs"
CLASSES = ["person", "bicycle", "car", "bus", "dog", "cat", "horse", "bird", "bottle", "chair"]
DEMO = [
    ("000000000785.jpg", ["ski pole", "helmet", "goggles", "glove"]),
    ("000000000724.jpg", ["stop sign", "tree", "street lamp", "curb"]),
    ("000000001000.jpg", ["tennis racket", "cap", "child", "fence"]),
]


def coco_eval(gt, dets, img_ids, cat_ids):
    if not dets:
        return {"mAP50-95": 0.0, "mAP50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools is chatty
        e = COCOeval(gt, gt.loadRes(dets), "bbox")
        e.params.imgIds, e.params.catIds = img_ids, cat_ids
        e.evaluate()
        e.accumulate()
        e.summarize()
    return {
        "mAP50-95": round(float(e.stats[0]), 4),
        "mAP50": round(float(e.stats[1]), 4),
        "AR100": round(float(e.stats[8]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)

    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(COCO_DIR / "annotations" / "instances_val2017.json"))
    name2cat = {c["name"]: c["id"] for c in gt.loadCats(gt.getCatIds())}
    cat_ids = [name2cat[c] for c in CLASSES]
    img_ids = sorted({i for c in cat_ids for i in gt.getImgIds(catIds=[c])})[: args.n]

    dino = GroundingDino()
    yolo = YOLO(str(OUT / "yolo11n.pt"))
    coco_to_yolo = {cid: k for k, cid in enumerate(gt.getCatIds())}  # YOLO uses contiguous 0..79
    yolo_classes = [coco_to_yolo[c] for c in cat_ids]
    yolo_to_coco = {v: k for k, v in coco_to_yolo.items()}

    dino_dets, yolo_dets, t_dino, t_yolo = [], [], [], []
    for k, iid in enumerate(img_ids):
        info = gt.loadImgs(iid)[0]
        img = Image.open(COCO_DIR / "val2017" / info["file_name"]).convert("RGB")

        t0 = time.time()
        # Low box threshold: mAP needs the full precision/recall curve, not just confident boxes.
        for name, score, (x1, y1, x2, y2) in dino.detect(img, CLASSES, box_threshold=0.15, text_threshold=0.15):
            dino_dets.append(
                {"image_id": iid, "category_id": name2cat[name], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}
            )
        t_dino.append(time.time() - t0)

        t0 = time.time()
        r = yolo(np.asarray(img)[:, :, ::-1], conf=0.01, classes=yolo_classes, verbose=False)[0]
        for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            yolo_dets.append(
                {"image_id": iid, "category_id": yolo_to_coco[int(c)], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s}
            )
        t_yolo.append(time.time() - t0)
        if k % 25 == 0:
            print(
                f"{k}/{len(img_ids)} | DINO {np.mean(t_dino):.2f}s/img | YOLO {np.mean(t_yolo) * 1000:.0f}ms/img",
                flush=True,
            )

    res = {
        "n_images": len(img_ids),
        "classes": CLASSES,
        "grounding_dino_tiny_zero_shot": coco_eval(gt, dino_dets, img_ids, cat_ids)
        | {"s_per_image_cpu": round(float(np.mean(t_dino)), 2)},
        "yolo11n_supervised": coco_eval(gt, yolo_dets, img_ids, cat_ids)
        | {"s_per_image_cpu": round(float(np.mean(t_yolo)), 3)},
    }
    # Per-class AP50 for the zero-shot model shows which concepts transfer well from text.
    per_class = {}
    for name in CLASSES:
        per_class[name] = coco_eval(
            gt, [d for d in dino_dets if d["category_id"] == name2cat[name]], img_ids, [name2cat[name]]
        )["mAP50"]
    res["grounding_dino_per_class_AP50"] = per_class
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    # Open-vocabulary demo with SAM masks, on concepts COCO doesn't have.
    sam = Segmenter()
    fig, axes = plt.subplots(1, len(DEMO), figsize=(5.5 * len(DEMO), 4.6))
    rng = np.random.default_rng(3)
    for ax, (fname, prompt) in zip(axes, DEMO):
        img = Image.open(COCO_DIR / "val2017" / fname).convert("RGB")
        dets = dino.detect(img, prompt, box_threshold=0.30)
        masks = sam.masks(img, [d[2] for d in dets])
        ax.imshow(img)
        for (name, score, (x1, y1, x2, y2)), m in zip(dets, masks):
            color = rng.uniform(0.2, 1.0, 3)
            ax.imshow(np.dstack([np.broadcast_to(color, m.shape + (3,)), m * 0.45]))
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=color, lw=2))
            ax.text(x1, y1 - 3, f"{name} {score:.2f}", color="white", fontsize=8, backgroundcolor=(*color, 0.8))
        ax.set_title('prompt: "' + ", ".join(prompt) + '"', fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "open_vocab_demo.jpg", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()

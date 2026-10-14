"""Evaluate the shelf detector: mAP, product counting accuracy, and a rule-based empty-gap demo.

1. Detection mAP on the test images (SKU-110K validation shard, unseen in training).
2. Counting: predicted product count vs true count per image (MAE, relative error).
3. Empty-gap demo (no ground truth for gaps exists in SKU-110K, so this is qualitative): products
   are grouped into shelf rows by vertical position; a horizontal gap wider than 1.5 typical product
   widths between neighbors in a row is flagged as a possible empty facing.

Usage:
    python evaluate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = data_dir("shelf")
OUT = output_dir("shelf")
DOCS = Path(__file__).resolve().parent / "docs"
WEIGHTS = OUT / "runs" / "yolo11n_shelf" / "weights" / "best.pt"
TEST = DATA / "yolo" / "test"


def find_gaps(boxes, gap_factor=1.5):
    """boxes: (N, 4) xyxy. Returns gap rectangles (x1, y1, x2, y2) between products in the same row."""
    if len(boxes) < 5:
        return []
    h_med = np.median(boxes[:, 3] - boxes[:, 1])
    w_med = np.median(boxes[:, 2] - boxes[:, 0])
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    order = np.argsort(cy)
    rows, cur = [], [order[0]]
    for i in order[1:]:  # same row = vertical centers within half a product height
        if cy[i] - cy[cur[-1]] < 0.5 * h_med:
            cur.append(i)
        else:
            rows.append(cur)
            cur = [i]
    rows.append(cur)
    gaps = []
    for r in rows:
        if len(r) < 3:
            continue
        b = boxes[r][np.argsort(boxes[r][:, 0])]
        for left, right in zip(b[:-1], b[1:]):
            if right[0] - left[2] > gap_factor * w_med:
                gaps.append((left[2], min(left[1], right[1]), right[0], max(left[3], right[3])))
    return gaps


def main():
    DOCS.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))
    m = model.val(
        data=str(DATA / "shelf_test.yaml"),
        imgsz=960,
        batch=4,
        device="cpu",
        max_det=1000,
        plots=False,
        verbose=False,
        project=str(OUT / "runs"),
        name="val",
        exist_ok=True,
    )
    res = {
        "detection": {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
            "precision": round(float(m.box.mp), 4),
            "recall": round(float(m.box.mr), 4),
            "ms_per_image": round(float(m.speed["inference"]), 1),
        }
    }

    images = sorted((TEST / "images").glob("*.jpg"))
    true_n, pred_n, gap_counts = [], [], []
    for p in images:
        true_n.append(len((TEST / "labels" / f"{p.stem}.txt").read_text().strip().splitlines()))
        r = model(str(p), imgsz=960, conf=0.35, max_det=1000, verbose=False)[0]
        pred_n.append(len(r.boxes))
        gap_counts.append(len(find_gaps(r.boxes.xyxy.numpy())))
    true_n, pred_n = np.array(true_n), np.array(pred_n)
    err = pred_n - true_n
    res["counting"] = {
        "images": len(images),
        "mean_true_count": round(float(true_n.mean()), 1),
        "max_true_count": int(true_n.max()),
        "MAE": round(float(np.abs(err).mean()), 1),
        "mean_relative_error_pct": round(100 * float((np.abs(err) / np.maximum(true_n, 1)).mean()), 1),
        "bias_mean_pred_minus_true": round(float(err.mean()), 1),
        "conf_threshold": 0.35,
    }
    res["gap_demo"] = {
        "images_with_flagged_gaps": int((np.array(gap_counts) > 0).sum()),
        "note": "rule-based, no ground truth in SKU-110K",
    }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(true_n, pred_n, s=10, alpha=0.6)
    lim = [0, max(true_n.max(), pred_n.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect count")
    ax.set_xlabel("true products in image")
    ax.set_ylabel("predicted products")
    ax.legend()
    ax.set_title(f"Counting on {len(images)} test shelves (MAE {res['counting']['MAE']})")
    fig.tight_layout()
    fig.savefig(DOCS / "counting.png", dpi=100)
    plt.close(fig)

    # Examples: the image with the most flagged gaps + two typical ones
    order = np.argsort(gap_counts)[::-1]
    picks = [images[order[0]], images[len(images) // 3], images[2 * len(images) // 3]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    for ax, p in zip(axes, picks):
        img = cv2.imread(str(p))
        r = model(str(p), imgsz=960, conf=0.35, max_det=1000, verbose=False)[0]
        b = r.boxes.xyxy.numpy()
        for x1, y1, x2, y2 in b.astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for x1, y1, x2, y2 in np.array(find_gaps(b)).astype(int).reshape(-1, 4):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        true = len((TEST / "labels" / f"{p.stem}.txt").read_text().strip().splitlines())
        ax.set_title(f"{len(b)} detected / {true} true products", fontsize=11)
    fig.suptitle("Green: detected products · Red: possible empty facings (rule-based)")
    fig.tight_layout()
    fig.savefig(DOCS / "examples.jpg", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()

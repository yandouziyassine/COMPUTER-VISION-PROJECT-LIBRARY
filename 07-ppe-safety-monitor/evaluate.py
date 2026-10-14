"""Evaluate the PPE detector two ways:

1. Object level: COCO-style mAP per class (hardhat / no-hardhat) on the full test split.
2. Site-alert level: for each image, raise an alert if any "no-hardhat" head is detected.
   Compared with the ground truth "is anyone in this image missing a hard hat?", giving the
   precision / recall a safety manager actually cares about.

Usage:
    python evaluate.py                       # the laptop model (1,500 training images, CPU)
    python evaluate.py --model full          # the Kaggle GPU model (all 13,782 training images)
"""

import argparse
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

DATA = data_dir("ppe")
OUT = output_dir("ppe")
DOCS = Path(__file__).resolve().parent / "docs"
RUNS = {"subset": "yolo11n_ppe", "full": "yolo11n_ppe_full"}
TEST = DATA / "yolo" / "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RUNS), default="subset")
    args = ap.parse_args()
    suffix = "" if args.model == "subset" else f"_{args.model}"
    DOCS.mkdir(exist_ok=True)
    model = YOLO(str(OUT / "runs" / RUNS[args.model] / "weights" / "best.pt"))
    m = model.val(
        data=str(DATA / "ppe_test.yaml"),
        imgsz=640,
        batch=16,
        device="cpu",
        plots=False,
        verbose=False,
        project=str(OUT / "runs"),
        name="val",
        exist_ok=True,
    )
    per_class = {
        model.names[c]: {"AP50": round(float(m.box.ap50[i]), 4), "AP50-95": round(float(m.box.ap[i]), 4)}
        for i, c in enumerate(m.box.ap_class_index)
    }
    res = {
        "object_level": {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
            "per_class": per_class,
            "ms_per_image": round(float(m.speed["inference"]), 1),
        }
    }

    # Site alerts, swept over the confidence threshold for the no-hardhat class
    images = sorted((TEST / "images").glob("*.jpg"))
    truth = np.array(
        [
            any(ln.startswith("1 ") for ln in (TEST / "labels" / f"{p.stem}.txt").read_text().splitlines())
            for p in images
        ]
    )
    max_conf = []
    for p in images:
        r = model(str(p), conf=0.05, verbose=False)[0]
        c = r.boxes.conf.numpy()[r.boxes.cls.numpy() == 1]
        max_conf.append(float(c.max()) if len(c) else 0.0)
    max_conf = np.array(max_conf)
    sweep = []
    for thr in (0.25, 0.4, 0.5, 0.6):
        alert = max_conf > thr
        tp, fp, fn = int((alert & truth).sum()), int((alert & ~truth).sum()), int((~alert & truth).sum())
        sweep.append(
            {
                "conf_threshold": thr,
                "alert_precision": round(tp / max(tp + fp, 1), 3),
                "alert_recall": round(tp / max(tp + fn, 1), 3),
                "TP": tp,
                "FP": fp,
                "FN": fn,
            }
        )
    res["site_alerts"] = {"images": len(images), "images_with_violation": int(truth.sum()), "sweep": sweep}
    print(json.dumps(res, indent=1))
    (DOCS / f"results{suffix}.json").write_text(json.dumps(res, indent=1))

    # Demo: violation images first (where an alert should fire), then compliant ones
    viol = [p for p, t in zip(images, truth) if t][::15][:4]
    ok = [p for p, t in zip(images, truth) if not t][::40][:2]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
    for ax, p in zip(axes.ravel(), viol + ok):
        r = model(str(p), conf=0.4, verbose=False)[0]
        img = cv2.imread(str(p))
        for (x1, y1, x2, y2), c, k in zip(
            r.boxes.xyxy.numpy().astype(int), r.boxes.conf.numpy(), r.boxes.cls.numpy().astype(int)
        ):
            color = (0, 200, 0) if k == 0 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img, f"{model.names[k]} {c:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )
        if (r.boxes.cls.numpy() == 1).any():
            cv2.putText(img, "ALERT: missing hard hat", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / f"examples{suffix}.jpg", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()

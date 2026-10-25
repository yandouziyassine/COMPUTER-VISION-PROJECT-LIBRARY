"""Evaluate both stages of license plate recognition.

1. Detection (find the plate): mAP on the 882-image test split for
     - YOLO11n fine-tuned here on 1,500 images (CPU)
     - a public YOLO11n plate detector (morsetechlab/yolov11-license-plate-detection, v1n)
2. Recognition (read the plate): RapidOCR (OpenVINO) on UK plates with known registrations,
   clean and augmented (blur, angle, lighting): exact-match rate and character accuracy.
3. End-to-end demo on test photos: detect → crop → read.

Usage:
    python evaluate.py
    python evaluate.py --model full   # the Kaggle GPU detector (all 6,176 training images): detection
                                      # mAP + end-to-end figure into results_full.json (OCR does not
                                      # depend on the detector, so it is not re-run)
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402
import common.ov_compat  # noqa: E402,F401  (lets rapidocr_openvino run on current OpenVINO)
from rapidocr_openvino import RapidOCR  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = data_dir("plates")
OUT = output_dir("plates")
DOCS = Path(__file__).resolve().parent / "docs"
RUNS = {"subset": "yolo11n_plates", "full": "yolo11n_plates_full"}
FINETUNED = OUT / "runs" / RUNS["subset"] / "weights" / "best.pt"
PRETRAINED = OUT / "pretrained_plate_v1n.pt"

ocr = RapidOCR()


def clean(s):
    return "".join(ch for ch in s.upper() if ch.isascii() and ch.isalnum())


def read_plate(crop_bgr):
    """OCR a plate crop; join text boxes left to right.

    The text detector needs margin around the text: on a tight crop it splits the plate into
    single characters. So the crop is upscaled if small and padded with a border first."""
    h = crop_bgr.shape[0]
    if h < 64:
        crop_bgr = cv2.resize(crop_bgr, None, fx=64 / h, fy=64 / h, interpolation=cv2.INTER_CUBIC)
    pad = crop_bgr.shape[0] // 2
    crop_bgr = cv2.copyMakeBorder(crop_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    res, _ = ocr(crop_bgr, use_cls=False)
    if not res:
        return ""
    res = sorted(res, key=lambda r: min(p[0] for p in r[0]))
    # Drop the country band ("UK"/"GB") that sits left of the registration on UK plates.
    parts = [clean(r[1]) for r in res if clean(r[1]) not in ("UK", "GB")]
    return "".join(parts)


TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B", "4": "A", "7": "T"}
TO_DIGIT = {v: k for k, v in TO_LETTER.items()} | {"D": "0", "Q": "0"}


def uk_syntax_fix(text):
    """Current UK format is LLDDLLL (e.g. MY69JKI). If the read has 7 characters, force letters
    and digits into their positions, fixing the classic OCR confusions (O/0, I/1, S/5, B/8...)."""
    if len(text) != 7:
        return text
    return "".join((TO_DIGIT.get(c, c) if i in (2, 3) else TO_LETTER.get(c, c)) for i, c in enumerate(text))


def char_accuracy(pred, truth):
    """1 - normalized Levenshtein distance."""
    d = np.arange(len(truth) + 1)
    for i, c in enumerate(pred, 1):
        prev, d[0] = d[0], i
        for j, t in enumerate(truth, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (c != t))
    return max(0.0, 1 - d[-1] / max(len(truth), 1))


def detection_metrics(weights):
    m = YOLO(str(weights)).val(
        data=str(DATA / "plates_test.yaml"),
        imgsz=640,
        batch=16,
        device="cpu",
        plots=False,
        verbose=False,
        project=str(OUT / "runs"),
        name="val",
        exist_ok=True,
    )
    return {
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "ms_per_image": round(float(m.speed["inference"]), 1),
    }


def ocr_metrics(n=300):
    table = pq.read_table(DATA / "uk_plates_0.parquet", columns=["vrn", "front_plate", "augmented_front_plate"]).slice(
        0, n
    )
    rows = table.to_pylist()
    res, examples = {}, []
    for col in ("front_plate", "augmented_front_plate"):
        exact, chars, exact_fix, chars_fix, t = [], [], [], [], []
        for k, r in enumerate(rows):
            img = cv2.cvtColor(np.array(Image.open(io.BytesIO(r[col]["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
            t0 = time.time()
            pred = read_plate(img)
            t.append(time.time() - t0)
            truth = clean(r["vrn"])
            fixed = uk_syntax_fix(pred)
            exact.append(pred == truth)
            chars.append(char_accuracy(pred, truth))
            exact_fix.append(fixed == truth)
            chars_fix.append(char_accuracy(fixed, truth))
            if col == "augmented_front_plate" and k < 6:
                examples.append((img, truth, fixed))
        res[col] = {
            "n": len(rows),
            "exact_match_raw": round(float(np.mean(exact)), 3),
            "char_accuracy_raw": round(float(np.mean(chars)), 3),
            "exact_match_with_uk_syntax": round(float(np.mean(exact_fix)), 3),
            "char_accuracy_with_uk_syntax": round(float(np.mean(chars_fix)), 3),
            "ms_per_plate": round(1000 * float(np.mean(t)), 1),
        }
    return res, examples


def end_to_end_demo(weights=FINETUNED, out_name="end_to_end.jpg", n=6):
    model = YOLO(str(weights))
    imgs = sorted((DATA / "yolo" / "test" / "images").glob("*.jpg"))[::97][:n]
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 7.5))
    for ax, p in zip(axes.ravel(), imgs):
        img = cv2.imread(str(p))
        r = model(img, conf=0.4, verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.numpy().astype(int), r.boxes.conf.numpy()):
            pad = int(0.08 * (x2 - x1))
            crop = img[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
            text = read_plate(crop) or "?"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{text} ({c:.2f})", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    fig.suptitle("End to end on test photos: fine-tuned YOLO11n → crop → RapidOCR (no ground-truth text)")
    fig.tight_layout()
    fig.savefig(DOCS / out_name, dpi=70)
    plt.close(fig)


def main_full():
    weights = OUT / "runs" / RUNS["full"] / "weights" / "best.pt"
    res = {
        "detection_test_882_images": {
            "yolo11n_full_6176_imgs_kaggle_t4": detection_metrics(weights),
            "pretrained_morsetechlab_v1n": detection_metrics(PRETRAINED),
        }
    }
    print(json.dumps(res, indent=1), flush=True)
    (DOCS / "results_full.json").write_text(json.dumps(res, indent=1))
    end_to_end_demo(weights, "end_to_end_full.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(RUNS), default="subset")
    ap.add_argument("--ocr-only", action="store_true")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    if args.model == "full":
        main_full()
        return
    prev = DOCS / "results.json"
    if args.ocr_only and prev.exists():  # reuse detection numbers, redo OCR + figures
        res = {"detection_test_882_images": json.loads(prev.read_text())["detection_test_882_images"]}
    else:
        res = {
            "detection_test_882_images": {
                "yolo11n_finetuned_here_1500_imgs_cpu": detection_metrics(FINETUNED),
                "pretrained_morsetechlab_v1n": detection_metrics(PRETRAINED),
            }
        }
    print(json.dumps(res, indent=1), flush=True)
    res["ocr_uk_plates"], examples = ocr_metrics()
    print(json.dumps(res["ocr_uk_plates"], indent=1), flush=True)
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(2, 3, figsize=(12, 4.2))
    for ax, (img, truth, pred) in zip(axes.ravel(), examples):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(f"true {truth} | read {pred or '—'}", fontsize=10, color="green" if pred == truth else "red")
    fig.suptitle("OCR on augmented UK plates (synthetic)")
    fig.tight_layout()
    fig.savefig(DOCS / "ocr_examples.jpg", dpi=80)
    plt.close(fig)
    end_to_end_demo()


if __name__ == "__main__":
    main()

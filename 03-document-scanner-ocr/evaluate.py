"""Evaluate the scanner + OCR pipeline on SROIE receipts photographed "in the wild".

SROIE receipts are flat scans, so realistic phone photos are generated from them: each receipt
is warped with a random perspective onto a random COCO photo (table/background), with uneven
lighting, a shadow, blur, noise and JPEG compression. Because we generate the photo, we know
the true page corners, which lets us measure the scanner exactly.

Measured:
    corner detection   success rate + mean corner error (as % of the page diagonal)
    field extraction   date / total accuracy on: original scan · raw photo · photo after the scanner

Usage:
    python evaluate.py --n 100
"""

import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from receipt import normalize_amount, normalize_date, read_receipt  # noqa: E402
from scanner import scan  # noqa: E402

DATA = data_dir("receipts")
OUT = output_dir("receipts")
DOCS = Path(__file__).resolve().parent / "docs"
BACKGROUNDS = data_dir("clip-search") / "val2017"  # COCO photos, shared with project 16


def load_receipts(n):
    df = pd.read_parquet(DATA / "sroie_test.parquet")
    for _, row in df.head(n).iterrows():
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
        yield row["key"], img, row["entities"]


def synth_photo(receipt, background, rng, canvas=(1600, 1200)):
    """Place the receipt on the background with a random perspective; return photo + true corners."""
    W, H = canvas
    bg = cv2.resize(background, (W, H))
    h, w = receipt.shape[:2]
    target_h = rng.uniform(0.70, 0.88) * H
    rw, rh = w * target_h / h, target_h
    cx, cy = W / 2 + rng.uniform(-0.12, 0.12) * W, H / 2 + rng.uniform(-0.04, 0.04) * H
    rect = np.array([[-rw / 2, -rh / 2], [rw / 2, -rh / 2], [rw / 2, rh / 2], [-rw / 2, rh / 2]])
    a = np.radians(rng.uniform(-12, 12))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    quad = rect @ rot.T + [cx, cy]
    quad += rng.uniform(-0.06, 0.06, size=(4, 2)) * [rw, rh]  # perspective tilt
    quad = quad.astype(np.float32)

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(receipt, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    photo = np.where(mask[..., None] > 0, warped, bg).astype(np.float32)

    # Uneven lighting (linear gradient) and a soft shadow.
    gx, gy = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    t = rng.uniform(0, 2 * np.pi)
    light = 0.75 + 0.35 * (np.cos(t) * gx + np.sin(t) * gy)
    photo *= light[..., None]
    shadow = np.zeros((H, W), np.float32)
    x0 = int(rng.uniform(0.2, 0.8) * W)
    cv2.fillPoly(shadow, [np.array([[x0, 0], [x0 + 300, 0], [x0 + 100, H], [x0 - 200, H]], np.int32)], 1.0)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 40)
    photo *= (1 - 0.35 * shadow)[..., None]

    photo = cv2.GaussianBlur(photo, (0, 0), rng.uniform(0.3, 1.2))
    photo += rng.normal(0, 4, photo.shape)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR), quad


def table_background(rng, size=(1200, 1600)):
    """A plain, darker surface with some texture, like a desk: the setup scanner apps ask for."""
    base = rng.integers(25, 95, size=3)
    bg = np.ones((*size, 3)) * base
    bg += cv2.GaussianBlur(rng.normal(0, 12, size), (0, 0), 3)[..., None]
    return np.clip(bg, 0, 255).astype(np.uint8)


def corner_error(found, true):
    """Mean corner distance as a fraction of the page diagonal."""
    diag = np.linalg.norm(true[2] - true[0])
    return float(np.mean(np.linalg.norm(found - true, axis=1)) / diag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    bg_paths = sorted(BACKGROUNDS.glob("*.jpg"))

    rows, examples = [], {"table": [], "coco": []}
    t_scan, t_ocr = [], []
    for k, (key, receipt, ent) in enumerate(load_receipts(args.n)):
        gt_date, gt_total = normalize_date(ent["date"] or ""), normalize_amount(ent["total"] or "")
        t1 = time.time()
        fields, _ = read_receipt(receipt)
        t_ocr.append(time.time() - t1)
        row = {"key": key, "scan_date_ok": fields["date"] == gt_date, "scan_total_ok": fields["total"] == gt_total}

        # Two settings: a plain table (the normal use case) and a random COCO photo (stress test).
        for mode in ("table", "coco"):
            bg = table_background(rng) if mode == "table" else cv2.imread(str(bg_paths[rng.integers(len(bg_paths))]))
            photo, true_quad = synth_photo(receipt, bg, rng)
            t0 = time.time()
            scanned, corners = scan(photo)
            t_scan.append(time.time() - t0)
            err = corner_error(corners, true_quad) if corners is not None else None
            row[f"{mode}_corner_err"] = err
            row[f"{mode}_corner_ok"] = err is not None and err < 0.03
            for name, img in (("photo", photo), ("scanned", scanned)):
                t1 = time.time()
                fields, _ = read_receipt(img)
                t_ocr.append(time.time() - t1)
                row[f"{mode}_{name}_date_ok"] = fields["date"] == gt_date
                row[f"{mode}_{name}_total_ok"] = fields["total"] == gt_total
            if len(examples[mode]) < 2:
                examples[mode].append((photo, corners, scanned))
        rows.append(row)
        if k % 10 == 0:
            print(f"{k}/{args.n} | table err {row['table_corner_err']} | coco err {row['coco_corner_err']}", flush=True)

    df = pd.DataFrame(rows)
    pct = lambda s: round(100 * float(s.mean()), 1) if len(s) else None  # noqa: E731
    res = {
        "n_receipts": len(df),
        "scanner_ms_per_photo": round(1000 * float(np.mean(t_scan)), 1),
        "ocr_ms_per_image": round(1000 * float(np.mean(t_ocr)), 1),
        "clean_scan": {"date_acc": pct(df["scan_date_ok"]), "total_acc": pct(df["scan_total_ok"])},
    }
    for mode in ("table", "coco"):
        errs = df[f"{mode}_corner_err"].dropna()
        ok = df[f"{mode}_corner_ok"]
        res[mode] = {
            "corner_success_pct": pct(ok),
            "corner_median_err_pct_diag": round(100 * float(errs.median()), 2) if len(errs) else None,
            "raw_photo": {"date_acc": pct(df[f"{mode}_photo_date_ok"]), "total_acc": pct(df[f"{mode}_photo_total_ok"])},
            "scanned": {
                "date_acc": pct(df[f"{mode}_scanned_date_ok"]),
                "total_acc": pct(df[f"{mode}_scanned_total_ok"]),
            },
            # What the scanner adds when it found the page correctly:
            "when_corners_ok": {
                "n": int(ok.sum()),
                "raw_photo_total_acc": pct(df.loc[ok, f"{mode}_photo_total_ok"]),
                "scanned_total_acc": pct(df.loc[ok, f"{mode}_scanned_total_ok"]),
                "raw_photo_date_acc": pct(df.loc[ok, f"{mode}_photo_date_ok"]),
                "scanned_date_acc": pct(df.loc[ok, f"{mode}_scanned_date_ok"]),
            },
        }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))
    df.to_csv(OUT / "per_receipt.csv", index=False)
    examples = examples["table"] + examples["coco"]

    fig, axes = plt.subplots(2, len(examples), figsize=(3.6 * len(examples), 8.5))
    for j, (photo, corners, scanned) in enumerate(examples):
        shown = photo.copy()
        if corners is not None:
            cv2.polylines(shown, [corners.astype(np.int32)], True, (0, 255, 0), 8)
        axes[0, j].imshow(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB))
        axes[0, j].axis("off")
        axes[1, j].imshow(cv2.cvtColor(scanned, cv2.COLOR_BGR2RGB))
        axes[1, j].axis("off")
    axes[0, 0].set_title(
        "synthetic photo (2 on a table, 2 on random COCO scenes) + detected page", loc="left", fontsize=10
    )
    axes[1, 0].set_title("scanner output", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "examples.jpg", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()

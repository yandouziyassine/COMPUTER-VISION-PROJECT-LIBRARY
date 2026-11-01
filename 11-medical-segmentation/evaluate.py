"""Evaluate a trained U-Net: TTA, a screening threshold tuned on validation, and a second center.

1. Kvasir-SEG test split (100 images): Dice / IoU with and without flip TTA, at threshold 0.5.
2. Threshold for screening: swept on the Kvasir *validation* split (never the test split), chosen
   to maximize pixel-level F2 (sensitivity weighted 4x over precision); then reported on test.
3. CVC-ClinicDB (612 frames, another hospital, never used for training or tuning): same metrics.

Outputs: docs/results<tag>.json and charts only (no dataset images: their licences forbid it).

Usage:
    python evaluate.py                                   # baseline best.pt
    python evaluate.py --tag r18_352 --size 352          # a variant trained with --tag
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from polyp import (
    CVC_ROOT,
    DOCS,
    ENCODER,
    OUT,
    SIZE,
    PolypDataset,
    cvc_pairs,
    kvasir_pairs,
    load_trained,
    pick_threshold,
    predict_probs,
    screening_metrics,
    split,
)

THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)


@torch.no_grad()
def collect(model, ds, tta):
    """Probability maps and masks for a whole dataset, plus CPU milliseconds per image."""
    probs, masks = [], []
    t0 = time.time()
    for x, y in DataLoader(ds, batch_size=16):
        probs.append(predict_probs(model, x, tta))
        masks.append(y)
    ms = (time.time() - t0) / len(ds) * 1000
    return torch.cat(probs), torch.cat(masks), ms


def summary(probs, masks, thr):
    """Rounded screening metrics plus the per-image Dice distribution."""
    m = screening_metrics(probs, masks, thr)
    pred = (probs > thr).float()
    inter = (pred * masks).flatten(1).sum(1)
    dice = ((2 * inter + 1e-6) / (pred.flatten(1).sum(1) + masks.flatten(1).sum(1) + 1e-6)).numpy()
    m["dice_median"] = float(np.median(dice))
    m["share_dice_above_0.8"] = float((dice > 0.8).mean())
    m["share_dice_below_0.5"] = float((dice < 0.5).mean())
    return {k: round(v, 4) for k, v in m.items()}, dice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="checkpoint suffix used at training time")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--no-cvc", action="store_true", help="skip the CVC-ClinicDB evaluation")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    DOCS.mkdir(exist_ok=True)
    torch.set_grad_enabled(False)
    model = load_trained(f"best{suffix}.pt", args.encoder)
    s = split()
    res = {"checkpoint": f"best{suffix}.pt", "encoder": args.encoder, "size": args.size}

    # 1. Same-center test split, with and without TTA, at the default threshold.
    test_ds = PolypDataset(kvasir_pairs(s["test"]), size=args.size)
    p_plain, y_test, ms_plain = collect(model, test_ds, tta=False)
    p_tta, _, ms_tta = collect(model, test_ds, tta=True)
    res["kvasir_test_t0.5"], dice_plain = summary(p_plain, y_test, 0.5)
    res["kvasir_test_t0.5_tta"], dice_tta = summary(p_tta, y_test, 0.5)
    res["ms_per_image_cpu"] = round(ms_plain, 1)
    res["ms_per_image_cpu_tta"] = round(ms_tta, 1)
    # Paired view: on how many test images does TTA help / hurt (by more than 0.01 Dice)?
    diff = dice_tta - dice_plain
    res["tta_paired"] = {
        "mean_gain": round(float(diff.mean()), 4),
        "images_better": int((diff > 0.01).sum()),
        "images_worse": int((diff < -0.01).sum()),
    }

    # 2. Screening threshold chosen on validation only (with TTA, the deployed setting).
    val_ds = PolypDataset(kvasir_pairs(s["val"]), size=args.size)
    p_val, y_val, _ = collect(model, val_ds, tta=True)
    thr, sweep = pick_threshold(p_val, y_val, THRESHOLDS, beta=2.0)
    res["screening_threshold_from_val"] = thr
    res["val_sweep"] = [{k: round(v, 4) for k, v in r.items()} for r in sweep]
    res["kvasir_test_tuned_tta"], _ = summary(p_tta, y_test, thr)
    test_sweep = [screening_metrics(p_tta, y_test, t) for t in THRESHOLDS]  # for the chart only

    # 3. Second center: CVC-ClinicDB, never seen in training or tuning.
    dice_cvc = None
    if not args.no_cvc and (CVC_ROOT / "Original").exists():
        cvc_ds = PolypDataset(cvc_pairs(), size=args.size)
        c_plain, y_cvc, _ = collect(model, cvc_ds, tta=False)
        c_tta, _, _ = collect(model, cvc_ds, tta=True)
        res["n_cvc"] = len(cvc_ds)
        res["cvc_t0.5"], _ = summary(c_plain, y_cvc, 0.5)
        res["cvc_t0.5_tta"], dice_cvc = summary(c_tta, y_cvc, 0.5)
        res["cvc_tuned_tta"], _ = summary(c_tta, y_cvc, thr)
        del c_plain, c_tta, y_cvc
    res["n_test"], res["n_val"] = len(test_ds), len(val_ds)

    out = {k: v for k, v in res.items() if k != "val_sweep"}
    print(json.dumps(out, indent=1))
    (DOCS / f"results{suffix}.json").write_text(json.dumps(res, indent=1))

    if args.tag:  # charts only for the baseline, to keep docs/ small
        return
    plot_threshold(sweep, test_sweep, thr)
    plot_dice_hist(dice_tta, dice_cvc)
    plot_training()


def plot_threshold(val_sweep, test_sweep, thr):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, rows, title in [(axes[0], val_sweep, "Validation (used to choose)"), (axes[1], test_sweep, "Test")]:
        t = [r["threshold"] for r in rows]
        ax.plot(t, [r["pixel_sensitivity"] for r in rows], label="sensitivity (pixels)")
        ax.plot(t, [r["pixel_precision"] for r in rows], label="precision (pixels)")
        ax.plot(t, [r["dice_mean"] for r in rows], label="mean Dice")
        ax.axvline(thr, color="grey", ls="--", lw=1)
        ax.axvline(0.5, color="grey", ls=":", lw=1)
        ax.set_title(title)
        ax.set_xlabel("decision threshold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower center", fontsize=8)
    axes[0].text(thr, 1.0, f" chosen {thr:.2f}", color="grey", fontsize=8, va="top")
    axes[0].text(0.5, 1.0, " default 0.5", color="grey", fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(DOCS / "threshold_sweep.png", dpi=100)
    plt.close(fig)


def plot_dice_hist(dice_kvasir, dice_cvc):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    bins = np.linspace(0, 1, 21)
    ax.hist(dice_kvasir, bins=bins, density=True, alpha=0.6, label=f"Kvasir-SEG test (n={len(dice_kvasir)})")
    if dice_cvc is not None:
        ax.hist(dice_cvc, bins=bins, density=True, alpha=0.6, label=f"CVC-ClinicDB (n={len(dice_cvc)})")
    ax.set_xlabel("per-image Dice (TTA, threshold 0.5)")
    ax.set_ylabel("density")
    ax.set_title("Same center vs. another hospital")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "dice_by_center.png", dpi=100)
    plt.close(fig)


def plot_training():
    hist = json.loads((OUT / "history.json").read_text())
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.plot(ep, [h["train_loss"] for h in hist])
    a1.set_title("Training loss (Dice + BCE)")
    a1.set_xlabel("epoch")
    a2.plot(ep, [h["val_dice"] for h in hist], label="Dice")
    a2.plot(ep, [h["val_iou"] for h in hist], label="IoU")
    a2.set_title("Validation")
    a2.set_xlabel("epoch")
    a2.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "training_curves.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()

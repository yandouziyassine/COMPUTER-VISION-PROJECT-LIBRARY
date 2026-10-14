"""Test-set Dice / IoU for the trained U-Net, plus README figures.

Outputs (docs/): results.json, predictions.jpg, training_curves.png
"""

import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from polyp import DOCS, OUT, PolypDataset, dice_iou, load_trained, split


def main():
    DOCS.mkdir(exist_ok=True)
    model = load_trained()
    names = split()["test"]
    ds = PolypDataset(names)

    dices, ious, preds = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in DataLoader(ds, batch_size=16):
            out = model(x)
            d, i = dice_iou(out, y)
            dices += d
            ious += i
            preds.append((out.sigmoid() > 0.5).float())
    ms = (time.time() - t0) / len(ds) * 1000
    preds = torch.cat(preds)
    dices, ious = np.array(dices), np.array(ious)

    res = {
        "n_test": len(ds),
        "dice_mean": round(float(dices.mean()), 4),
        "iou_mean": round(float(ious.mean()), 4),
        "dice_median": round(float(np.median(dices)), 4),
        "share_dice_above_0.8": round(float((dices > 0.8).mean()), 3),
        "share_dice_below_0.5": round(float((dices < 0.5).mean()), 3),
        "ms_per_image_cpu_256px": round(ms, 1),
    }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    # Figure: 3 typical cases + the 2 worst, so failures are visible too.
    order = np.argsort(dices)
    picks = list(order[[len(order) // 4, len(order) // 2, 3 * len(order) // 4]]) + list(order[:2])
    fig, axes = plt.subplots(3, len(picks), figsize=(2.8 * len(picks), 8.4))
    for j, k in enumerate(picks):
        img, mask = ds.data[k]
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"Dice {dices[k]:.2f}" + (" (worst)" if j >= 3 else ""), fontsize=9)
        axes[1, j].imshow(img)
        axes[1, j].imshow(np.ma.masked_where(mask == 0, mask), cmap="autumn", alpha=0.5)
        p = preds[k, 0].numpy()
        axes[2, j].imshow(img)
        axes[2, j].imshow(np.ma.masked_where(p == 0, p), cmap="winter", alpha=0.5)
        for r in range(3):
            axes[r, j].axis("off")
    for r, label in enumerate(["input", "expert mask", "U-Net prediction"]):
        axes[r, 0].text(-0.12, 0.5, label, transform=axes[r, 0].transAxes, rotation=90, va="center")
    fig.tight_layout()
    fig.savefig(DOCS / "predictions.jpg", dpi=80)
    plt.close(fig)

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

"""Train a U-Net (ResNet-18 encoder, ImageNet-pretrained) to segment polyps on CPU.

Usage:
    python train.py --epochs 20
"""

import argparse
import json
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from polyp import OUT, PolypDataset, build_model, dice_iou, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  # the augmentations in polyp.py use Python's random module

    s = split()
    train_dl = DataLoader(PolypDataset(s["train"], augment=True), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(PolypDataset(s["val"]), batch_size=32)
    print(f"train {len(s['train'])} | val {len(s['val'])} | test {len(s['test'])}", flush=True)

    model = build_model()
    # Dice loss handles the foreground/background imbalance (polyps are often <20 % of the image);
    # BCE keeps per-pixel gradients well-behaved early in training.
    dice_loss = smp.losses.DiceLoss("binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_dl))

    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_dl:
            opt.zero_grad()
            out = model(x)
            loss = dice_loss(out, y) + bce(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(x)
            n += len(x)

        model.eval()
        dices, ious = [], []
        with torch.no_grad():
            for x, y in val_dl:
                d, i = dice_iou(model(x), y)
                dices += d
                ious += i
        val_dice, val_iou = float(np.mean(dices)), float(np.mean(ious))
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_dice": val_dice, "val_iou": val_iou})
        flag = ""
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), OUT / "best.pt")
            flag = " *"
        print(
            f"epoch {epoch:2d} | loss {loss_sum / n:.3f} | val Dice {val_dice:.3f} IoU {val_iou:.3f} | "
            f"{time.time() - t0:.0f}s{flag}",
            flush=True,
        )
        # Saved every epoch so an interrupted run keeps its history.
        (OUT / "history.json").write_text(json.dumps(history, indent=1))

    print(f"best val Dice {best:.3f}")


if __name__ == "__main__":
    main()

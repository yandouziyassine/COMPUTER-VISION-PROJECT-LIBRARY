"""Train a U-Net (ImageNet-pretrained encoder, ResNet-18 by default) to segment polyps on CPU.

Usage:
    python train.py --epochs 20                                   # baseline -> best.pt
    python train.py --epochs 12 --size 352 --tag r18_352          # variant  -> best_r18_352.pt
"""

import argparse
import json
import random
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

from polyp import ENCODER, OUT, SIZE, PolypDataset, build_model, dice_iou, kvasir_pairs, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--encoder", default=ENCODER, help="any segmentation_models_pytorch encoder name")
    ap.add_argument("--size", type=int, default=SIZE, help="square input size in pixels")
    ap.add_argument("--tag", default="", help="suffix for checkpoint/history files (empty = baseline)")
    args = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)  # the augmentations in polyp.py use Python's random module
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt, hist_path = OUT / f"best{suffix}.pt", OUT / f"history{suffix}.json"

    s = split()
    train_ds = PolypDataset(kvasir_pairs(s["train"]), augment=True, size=args.size)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(PolypDataset(kvasir_pairs(s["val"]), size=args.size), batch_size=32)
    print(f"train {len(s['train'])} | val {len(s['val'])} | test {len(s['test'])} | {args}", flush=True)

    model = build_model(args.encoder)
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
                d, i = dice_iou(model(x).sigmoid(), y)
                dices += d
                ious += i
        val_dice, val_iou = float(np.mean(dices)), float(np.mean(ious))
        secs = time.time() - t0
        history.append(
            {"epoch": epoch, "train_loss": loss_sum / n, "val_dice": val_dice, "val_iou": val_iou, "seconds": secs}
        )
        flag = ""
        if val_dice > best:
            best = val_dice
            torch.save(model.state_dict(), ckpt)
            flag = " *"
        print(
            f"epoch {epoch:2d} | loss {loss_sum / n:.3f} | val Dice {val_dice:.3f} IoU {val_iou:.3f} | {secs:.0f}s{flag}",
            flush=True,
        )
        # Saved every epoch so an interrupted run keeps its history.
        hist_path.write_text(json.dumps(history, indent=1))

    print(f"best val Dice {best:.3f}")


if __name__ == "__main__":
    main()

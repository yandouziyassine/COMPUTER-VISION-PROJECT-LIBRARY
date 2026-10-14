"""Fine-tune ResNet-18 (ImageNet-pretrained) on EuroSAT RGB, on CPU.

Usage:
    python train.py --epochs 6
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eurosat import OUT, TileDataset, build_model, load_split, predict, train_tf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)  # performance cores only: faster on this hybrid CPU (see project 19)

    tr_img, tr_lab = load_split("train")
    va_img, va_lab = load_split("validation")
    print(f"train {len(tr_lab)} | val {len(va_lab)}", flush=True)
    dl = DataLoader(TileDataset(tr_img, tr_lab, train_tf), batch_size=64, shuffle=True)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.15)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    history, best = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        model.eval()
        acc = float((predict(model, va_img) == np.array(va_lab)).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": acc})
        (OUT / "history.json").write_text(json.dumps(history, indent=1))
        flag = ""
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "best.pt")
            flag = " *"
        print(f"epoch {epoch} | loss {tot / n:.3f} | val acc {acc:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
    print(f"best val acc {best:.4f}")


if __name__ == "__main__":
    main()

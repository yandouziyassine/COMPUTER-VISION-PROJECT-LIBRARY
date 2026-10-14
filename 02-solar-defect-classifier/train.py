"""Fine-tune EfficientNet-B0 on the solar panel defect dataset (CPU-friendly).

Usage:
    python train.py --epochs 15
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from solar import CLASSES, OUT_DIR, SolarDataset, build_model, class_counts, eval_tf, make_split, train_tf


def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            loss_sum += criterion(out, y).item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    torch.manual_seed(0)

    split = make_split()
    counts = class_counts(split["train"])
    print(f"train {len(split['train'])} | val {len(split['val'])} | test {len(split['test'])}")
    print("train counts:", dict(zip(CLASSES, counts)), flush=True)

    train_dl = DataLoader(SolarDataset(split["train"], train_tf), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(SolarDataset(split["val"], eval_tf), batch_size=64)

    model = build_model(pretrained=True)
    # Inverse-frequency class weights: Physical-Damage has ~3x fewer images than Clean.
    weights = torch.tensor([sum(counts) / (len(counts) * c) for c in counts], dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.15
    )

    history, best_acc = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n = time.time(), 0.0, 0
        for x, y in train_dl:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
        val_loss, val_acc = evaluate(model, val_dl)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss, "val_acc": val_acc})
        flag = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": CLASSES}, OUT_DIR / "best.pt")
            flag = " *"
        print(
            f"epoch {epoch:2d} | train_loss {loss_sum / n:.3f} | val_loss {val_loss:.3f} "
            f"| val_acc {val_acc:.3f} | {time.time() - t0:.0f}s{flag}",
            flush=True,
        )

    with open(OUT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"best val_acc {best_acc:.3f} -> {OUT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()

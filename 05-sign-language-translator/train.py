"""Train the sign-letter CNN from scratch on CPU, with and without data augmentation (an ablation).

Usage:
    python train.py --aug      # -> cnn_aug.pt   (the model the app uses)
    python train.py            # -> cnn_noaug.pt (same network, no augmentation)

Validation: 10 % of the training CSV. Caveat measured in evaluate.py: the CSV was built by augmenting a few
hundred original photos, so train and validation share near-duplicates and validation accuracy is optimistic.
"""

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from asl import OUT, SignCNN, SignDataset, load_split, predict_proba, train_aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug", action="store_true", help="random affine + brightness/contrast augmentation")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    name = "cnn_aug" if args.aug else "cnn_noaug"
    torch.manual_seed(0)
    torch.set_num_threads(4)  # performance cores only: faster on this hybrid CPU (see project 19)

    images, labels = load_split("train")
    perm = np.random.default_rng(0).permutation(len(labels))
    va, tr = perm[: len(perm) // 10], perm[len(perm) // 10 :]
    dl = DataLoader(SignDataset(images[tr], labels[tr], train_aug if args.aug else None), batch_size=128, shuffle=True)
    print(f"{name}: train {len(tr)} | val {len(va)}", flush=True)

    model = SignCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl), pct_start=0.15)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    history, best = [], -1.0
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
        acc = float((predict_proba(model, images[va]).argmax(1) == labels[va]).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": acc})
        (OUT / f"{name}_history.json").write_text(json.dumps(history, indent=1))
        flag = ""
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), OUT / f"{name}.pt")
            flag = " *"
        print(f"epoch {epoch} | loss {tot / n:.3f} | val acc {acc:.4f} | {time.time() - t0:.0f}s{flag}", flush=True)
    print(f"best val acc {best:.4f}")


if __name__ == "__main__":
    main()

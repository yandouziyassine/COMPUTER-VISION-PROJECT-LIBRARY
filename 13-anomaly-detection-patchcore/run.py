"""Fit PatchCore on the defect-free training images of each MVTec AD category, then evaluate
on the test set (good + defective images, with pixel masks).

Usage:
    python run.py --categories bottle metal_nut grid

Outputs:
    docs/<category>_examples.png   heatmaps on defective test images
    docs/results.json              image- and pixel-level AUROC per category
    ../cv-data/mvtec/outputs/      memory banks
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from patchcore import PatchCore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

MVTEC = data_dir("mvtec")
OUT = output_dir("mvtec")
DOCS = Path(__file__).resolve().parent / "docs"
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
img_tf = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
)
mask_tf = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
    ]
)


class MVTecSplit(Dataset):
    def __init__(self, category, split):
        root = MVTEC / category / split
        self.items = []
        for defect_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for img in sorted(defect_dir.glob("*.png")):
                mask = None
                if defect_dir.name != "good":
                    mask = MVTEC / category / "ground_truth" / defect_dir.name / f"{img.stem}_mask.png"
                self.items.append((img, defect_dir.name, mask))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, defect, mask_path = self.items[i]
        x = img_tf(Image.open(img_path).convert("RGB"))
        if mask_path is not None and mask_path.exists():
            m = (mask_tf(Image.open(mask_path).convert("L")) > 0.5).float()[0]
        else:
            m = torch.zeros(SIZE, SIZE)
        return x, int(defect != "good"), m


def denorm(x):
    return (x.permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)


def save_examples(category, test_ds, scores, maps, n=5):
    """Top row: defective test images. Middle: ground-truth mask. Bottom: PatchCore heatmap."""
    bad = [i for i, (_, d, _) in enumerate(test_ds.items) if d != "good"]
    picks = bad[:: max(1, len(bad) // n)][:n]
    fig, axes = plt.subplots(3, len(picks), figsize=(2.6 * len(picks), 8))
    vmax = float(maps.max())
    for j, i in enumerate(picks):
        x, _, m = test_ds[i]
        defect = test_ds.items[i][1]
        axes[0, j].imshow(denorm(x))
        axes[0, j].set_title(f"{defect}\nscore {scores[i]:.2f}", fontsize=9)
        axes[1, j].imshow(m, cmap="gray")
        axes[2, j].imshow(denorm(x))
        axes[2, j].imshow(maps[i], cmap="jet", alpha=0.5, vmin=0, vmax=vmax)
        for r in range(3):
            axes[r, j].axis("off")
    axes[0, 0].text(-0.15, 0.5, "input", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.15, 0.5, "true defect", transform=axes[1, 0].transAxes, rotation=90, va="center")
    axes[2, 0].text(-0.15, 0.5, "PatchCore", transform=axes[2, 0].transAxes, rotation=90, va="center")
    fig.suptitle(f"MVTec AD — {category} (trained on good images only)")
    fig.tight_layout()
    fig.savefig(DOCS / f"{category}_examples.png", dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["bottle", "metal_nut", "grid"])
    ap.add_argument("--coreset-ratio", type=float, default=0.1)
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    torch.manual_seed(0)
    results = {}

    for cat in args.categories:
        t0 = time.time()
        model = PatchCore(coreset_ratio=args.coreset_ratio)
        train_dl = DataLoader(MVTecSplit(cat, "train"), batch_size=16)
        n_patches, n_bank = model.fit(train_dl)
        fit_s = time.time() - t0
        model.save(OUT / f"{cat}_memory_bank.pt")

        test_ds = MVTecSplit(cat, "test")
        scores, maps, labels, masks = [], [], [], []
        t1 = time.time()
        for x, y, m in DataLoader(test_ds, batch_size=8):
            s, a = model.predict(x, SIZE)
            scores.append(s)
            maps.append(a)
            labels.append(y)
            masks.append(m)
        scores, maps = torch.cat(scores).numpy(), torch.cat(maps).numpy()
        labels, masks = torch.cat(labels).numpy(), torch.cat(masks).numpy()
        ms_per_img = (time.time() - t1) / len(test_ds) * 1000

        img_auroc = roc_auc_score(labels, scores)
        pix_auroc = roc_auc_score(masks.ravel().astype(int), maps.ravel())
        results[cat] = {
            "image_auroc": round(float(img_auroc), 4),
            "pixel_auroc": round(float(pix_auroc), 4),
            "n_train": len(train_dl.dataset),
            "n_test": len(test_ds),
            "patches_seen": n_patches,
            "memory_bank": n_bank,
            "fit_seconds": round(fit_s, 1),
            "ms_per_test_image": round(ms_per_img, 1),
        }
        print(
            f"{cat}: image AUROC {img_auroc:.3f} | pixel AUROC {pix_auroc:.3f} | "
            f"bank {n_bank}/{n_patches} | fit {fit_s:.0f}s | {ms_per_img:.0f} ms/img",
            flush=True,
        )
        save_examples(cat, test_ds, scores, maps)

    with open(DOCS / "results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

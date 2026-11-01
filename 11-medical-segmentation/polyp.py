"""Shared pieces for polyp segmentation: split, datasets, model, TTA, metrics."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

ROOT = data_dir("kvasir") / "Kvasir-SEG"
# Second center (Hospital Clinic, Barcelona), used only for evaluation. See download_data.py.
CVC_ROOT = data_dir("kvasir") / "cvc-clinicdb" / "CVC-ClinicDB"
OUT = output_dir("kvasir")
DOCS = Path(__file__).resolve().parent / "docs"
SIZE = 256
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ENCODER = "resnet18"


def split(seed=42):
    """Fixed 80 / 10 / 10 split of the 1,000 Kvasir-SEG images."""
    names = sorted(p.name for p in (ROOT / "images").glob("*.jpg"))
    random.Random(seed).shuffle(names)
    return {"train": names[:800], "val": names[800:900], "test": names[900:]}


def kvasir_pairs(names):
    return [(ROOT / "images" / n, ROOT / "masks" / n) for n in names]


def cvc_pairs():
    """All 612 CVC-ClinicDB frames (image and mask share the file name)."""
    imgs = sorted((CVC_ROOT / "Original").glob("*.png"), key=lambda p: int(p.stem))
    return [(p, CVC_ROOT / "Ground Truth" / p.name) for p in imgs]


def load_pair(img_path, mask_path, size=SIZE):
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0, (np.asarray(mask) > 127).astype(np.float32)


class PolypDataset(Dataset):
    """Images are resized once and kept in RAM (1,000 images at 256 px is ~800 MB as float32)."""

    def __init__(self, pairs, augment=False, size=SIZE):
        self.data = [load_pair(i, m, size) for i, m in pairs]
        self.names = [Path(i).name for i, _ in pairs]
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, mask = self.data[i]
        if self.augment:
            # Geometric augmentations must hit image and mask identically.
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)
            # Photometric augmentation on the image only (endoscope lighting varies).
            img = np.clip(img * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08), 0, 1)
        x = torch.from_numpy(((img - MEAN) / STD).transpose(2, 0, 1).astype(np.float32).copy())
        y = torch.from_numpy(mask[None].copy())
        return x, y


def build_model(encoder=ENCODER, pretrained=True):
    import segmentation_models_pytorch as smp  # imported here so the pure helpers below stay light

    return smp.Unet(encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def load_trained(ckpt="best.pt", encoder=ENCODER):
    model = build_model(encoder, pretrained=False)
    model.load_state_dict(torch.load(OUT / ckpt, map_location="cpu"))
    return model.eval()


# Flips that are their own inverse: predict on the flipped image, flip the prediction back.
TTA_FLIPS = ((), (3,), (2,), (2, 3))  # identity, horizontal, vertical, both


def predict_probs(model, x, tta=False):
    """Polyp probability map (N, 1, H, W). With `tta`, the mean over the 4 flip variants."""
    if not tta:
        return model(x).sigmoid()
    probs = torch.zeros(x.shape[0], 1, *x.shape[2:])
    for dims in TTA_FLIPS:
        if dims:
            probs += model(x.flip(dims)).sigmoid().flip(dims)
        else:
            probs += model(x).sigmoid()
    return probs / len(TTA_FLIPS)


def dice_iou(probs, target, thr=0.5, eps=1e-6):
    """Per-image Dice and IoU of the mask `probs > thr` (returned as lists).

    An image where both prediction and ground truth are empty scores 1 (eps / eps)."""
    pred = (probs > thr).float()
    inter = (pred * target).flatten(1).sum(1)
    total = pred.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = (2 * inter + eps) / (total + eps)
    iou = (inter + eps) / (total - inter + eps)
    return dice.tolist(), iou.tolist()


def screening_metrics(probs, target, thr, miss_recall=0.5):
    """Dice plus sensitivity-oriented numbers at one threshold, over a whole set of images.

    - pixel sensitivity / precision: micro-averaged over all pixels of all images
    - missed polyps: share of images where less than `miss_recall` of the polyp pixels are found
    - false-alarm area: predicted area outside the polyp, as a share of the image"""
    pred = (probs > thr).float()
    tp = (pred * target).flatten(1).sum(1)
    fp = (pred * (1 - target)).flatten(1).sum(1)
    fn = ((1 - pred) * target).flatten(1).sum(1)
    dice, iou = dice_iou(probs, target, thr)
    gt = target.flatten(1).sum(1)
    img_recall = torch.where(gt > 0, tp / gt.clamp(min=1), torch.ones_like(gt))
    return {
        "threshold": round(float(thr), 3),
        "dice_mean": float(np.mean(dice)),
        "iou_mean": float(np.mean(iou)),
        "pixel_sensitivity": float(tp.sum() / (tp.sum() + fn.sum()).clamp(min=1)),
        "pixel_precision": float(tp.sum() / (tp.sum() + fp.sum()).clamp(min=1)),
        "missed_polyps": float((img_recall < miss_recall).float().mean()),
        "false_alarm_area": float((fp / target[0].numel()).mean()),
    }


def f_beta(precision, recall, beta=2.0):
    """F-beta score; beta=2 weighs sensitivity 4x as much as precision (screening)."""
    if precision + recall == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def pick_threshold(probs, target, thresholds, beta=2.0):
    """Threshold maximizing pixel-level F-beta on a validation set. Returns (best, sweep rows)."""
    rows = [screening_metrics(probs, target, t) for t in thresholds]
    for r in rows:
        r["f_beta"] = f_beta(r["pixel_precision"], r["pixel_sensitivity"], beta)
    best = max(rows, key=lambda r: r["f_beta"])
    return best["threshold"], rows

"""Shared pieces for polyp segmentation on Kvasir-SEG: split, dataset, model, metrics."""

import random
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

ROOT = data_dir("kvasir") / "Kvasir-SEG"
OUT = output_dir("kvasir")
DOCS = Path(__file__).resolve().parent / "docs"
SIZE = 256
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
ENCODER = "resnet18"


def split(seed=42):
    """Fixed 80 / 10 / 10 split of the 1,000 images."""
    names = sorted(p.name for p in (ROOT / "images").glob("*.jpg"))
    random.Random(seed).shuffle(names)
    return {"train": names[:800], "val": names[800:900], "test": names[900:]}


def load_pair(name):
    img = Image.open(ROOT / "images" / name).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
    mask = Image.open(ROOT / "masks" / name).convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0, (np.asarray(mask) > 127).astype(np.float32)


class PolypDataset(Dataset):
    """All images are resized once and kept in RAM (the dataset lives on a slow USB key)."""

    def __init__(self, names, augment=False):
        self.data = [load_pair(n) for n in names]
        self.names = names
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


def build_model(pretrained=True):
    return smp.Unet(ENCODER, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


def dice_iou(logits, target, eps=1e-6):
    """Per-image Dice and IoU at threshold 0.5 (returned as lists)."""
    pred = (logits.sigmoid() > 0.5).float()
    inter = (pred * target).flatten(1).sum(1)
    total = pred.flatten(1).sum(1) + target.flatten(1).sum(1)
    dice = (2 * inter + eps) / (total + eps)
    iou = (inter + eps) / (total - inter + eps)
    return dice.tolist(), iou.tolist()


def load_trained():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu"))
    return model.eval()

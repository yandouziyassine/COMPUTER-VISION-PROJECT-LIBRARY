"""Shared pieces for the solar panel defect classifier: data split, transforms, model."""

import csv
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import timm
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

DATA_ROOT = data_dir("solar") / "Faulty_solar_panel"
OUT_DIR = output_dir("solar")
DOCS_DIR = Path(__file__).resolve().parent / "docs"
CLASSES = ["Bird-drop", "Clean", "Dusty", "Electrical-damage", "Physical-Damage", "Snow-Covered"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}
IMG_SIZE = 224
MODEL_NAME = "efficientnet_b0"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def list_images():
    items = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((DATA_ROOT / cls).iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, label))
    return items


def make_split(seed=42, val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test split, saved to CSV so every script sees the same split."""
    split_file = OUT_DIR / "split.csv"
    if split_file.exists():
        with open(split_file, newline="") as f:
            rows = list(csv.DictReader(f))
        return {
            s: [(Path(r["path"]), int(r["label"])) for r in rows if r["split"] == s] for s in ("train", "val", "test")
        }

    rng = random.Random(seed)
    by_class = {}
    for p, y in list_images():
        by_class.setdefault(y, []).append(p)
    split = {"train": [], "val": [], "test": []}
    for y, paths in by_class.items():
        rng.shuffle(paths)
        n_test = round(len(paths) * test_frac)
        n_val = round(len(paths) * val_frac)
        split["test"] += [(p, y) for p in paths[:n_test]]
        split["val"] += [(p, y) for p in paths[n_test : n_test + n_val]]
        split["train"] += [(p, y) for p in paths[n_test + n_val :]]

    with open(split_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for s, items in split.items():
            for p, y in items:
                w.writerow([str(p), y, s])
    return split


def class_counts(items):
    c = Counter(y for _, y in items)
    return [c.get(i, 0) for i in range(len(CLASSES))]


train_tf = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)

eval_tf = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE + 32),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def load_rgb(path, max_side=320):
    """Open an image as RGB and shrink it so the whole dataset fits in RAM."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


class SolarDataset(Dataset):
    """Images are decoded once and kept in memory: the USB key is slow for random reads."""

    def __init__(self, items, tf):
        self.images = [load_rgb(p) for p, _ in items]
        self.labels = [y for _, y in items]
        self.tf = tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


def build_model(pretrained=True):
    return timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=len(CLASSES))


def load_trained(path=None):
    model = build_model(pretrained=False)
    state = torch.load(path or OUT_DIR / "best.pt", map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model

"""EuroSAT RGB (Sentinel-2 satellite tiles, 64x64 px, 10 m/pixel): data loading and model."""

import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import timm
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

DATA = data_dir("eurosat")
OUT = output_dir("eurosat")
DOCS = Path(__file__).resolve().parent / "docs"
CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
SIZE = 128  # tiles are 64x64; upsampling 2x lets ImageNet filters see familiar scales
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_split(split):
    """Returns (list of PIL images, labels). The whole split fits in RAM (tiles are tiny)."""
    t = pq.read_table(DATA / f"{split}.parquet").to_pylist()
    img_key = next(k for k in t[0] if isinstance(t[0][k], dict) and "bytes" in t[0][k])
    lab_key = next(k for k in t[0] if k in ("label", "labels", "class"))
    return [Image.open(io.BytesIO(r[img_key]["bytes"])).convert("RGB") for r in t], [int(r[lab_key]) for r in t]


train_tf = transforms.Compose(
    [
        transforms.Resize(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # satellite images have no "up"
        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)
eval_tf = transforms.Compose([transforms.Resize(SIZE), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class TileDataset(Dataset):
    def __init__(self, images, labels, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


def build_model(pretrained=True):
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=len(CLASSES))


def load_trained():
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(OUT / "best.pt", map_location="cpu"))
    return m.eval()


def predict(model, images, batch=256):
    preds = []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.stack([eval_tf(im) for im in images[i : i + batch]])
            preds.append(model(x).argmax(1).numpy())
    return np.concatenate(preds)

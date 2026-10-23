"""Sign Language MNIST (American Sign Language letters, 28x28 grayscale): data loading, model, preprocessing."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

DATA = data_dir("asl")
OUT = output_dir("asl")
DOCS = Path(__file__).resolve().parent / "docs"
# Labels are 0-25 for A-Z, but J (9) and Z (25) need motion and are absent: 24 static letters.
LABELS = [i for i in range(26) if i not in (9, 25)]
LETTERS = [chr(65 + i) for i in LABELS]
MEAN, STD = 0.5, 0.25


def load_split(split):
    """Returns (images uint8 (N, 28, 28), class indices 0-23)."""
    raw = np.loadtxt(DATA / f"sign_mnist_{split}.csv", delimiter=",", skiprows=1, dtype=np.int64)
    to_idx = {lab: i for i, lab in enumerate(LABELS)}
    y = np.array([to_idx[v] for v in raw[:, 0]])
    return raw[:, 1:].reshape(-1, 28, 28).astype(np.uint8), y


def to_tensor(images):
    """uint8 (N, 28, 28) -> normalized float tensor (N, 1, 28, 28)."""
    return (torch.from_numpy(images).float().div(255).unsqueeze(1) - MEAN) / STD


# Hand photos from a real camera are never perfectly centered, upright or evenly lit.
train_aug = v2.Compose(
    [
        v2.RandomAffine(degrees=12, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.3, contrast=0.3),
    ]
)


class SignDataset(Dataset):
    """uint8 images; optional per-image augmentation (each image gets its own random transform)."""

    def __init__(self, images, labels, aug=None):
        self.images, self.labels, self.aug = torch.from_numpy(images), torch.from_numpy(labels), aug

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i][None]  # (1, 28, 28) uint8
        if self.aug is not None:
            x = self.aug(x)
        return (x.float() / 255 - MEAN) / STD, self.labels[i]


class SignCNN(nn.Module):
    """Three conv blocks (28 -> 14 -> 7 -> 3) and a small classifier: ~0.4 M parameters."""

    def __init__(self, n_classes=len(LABELS)):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


@torch.no_grad()
def predict_proba(model, images, batch=1024):
    """uint8 images (N, 28, 28) -> class probabilities (N, 24)."""
    model.eval()
    x = to_tensor(images)
    return torch.cat([model(x[i : i + batch]).softmax(1) for i in range(0, len(x), batch)]).numpy()


def load_model(name="cnn_aug"):
    model = SignCNN()
    model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location="cpu"))
    return model.eval()


def hand_crop_to_input(gray_crop):
    """A grayscale hand crop (any size) -> uint8 (28, 28), the dataset's format: square, centered, resized."""
    import cv2

    h, w = gray_crop.shape
    side = max(h, w)
    canvas = np.full((side, side), int(np.median(gray_crop)), np.uint8)  # pad with the background tone
    canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = gray_crop
    return cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

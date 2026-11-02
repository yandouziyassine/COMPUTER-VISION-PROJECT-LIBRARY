"""Minimal PatchCore implementation (Roth et al., "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022), written from scratch in PyTorch for CPU.

Idea: a model pretrained on ImageNet already describes local image patches well. Collect the
patch features of *defect-free* training images into a memory bank, then score a test patch by
its distance to the nearest patch in the bank. Patches unlike anything seen in good parts are
anomalies. No defect images are needed for training.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


def local_average(f):
    """3x3 mean with zero padding: the same values as F.avg_pool2d(f, 3, 1, 1) (count_include_pad=True), but
    written as Pad + unpadded AvgPool. The Intel NPU compiler (OpenVINO 2026.4) miscomputes a padded AvgPool
    with count_include_pad on large maps (every second block of 32 channels wrong); the split form is exact."""
    return F.avg_pool2d(F.pad(f, (1, 1, 1, 1)), 3, 1)


class FeatureExtractor(nn.Module):
    """Mid-level WideResNet-50 features (layer2 + layer3): local enough for small defects, semantic
    enough to ignore pixel noise. A module so the same code is exported to ONNX / OpenVINO."""

    def __init__(self, backbone):
        super().__init__()
        self.b = backbone

    def forward(self, x):
        b = self.b
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f2 = b.layer2(b.layer1(x))
        f3 = b.layer3(f2)
        # Locally aware patch features: average each position with its 3x3 neighborhood (zeros outside).
        f2 = local_average(f2)
        f3 = local_average(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # (B, 1536, H, W)


class PatchCore:
    def __init__(self, coreset_ratio=0.1, proj_dim=128, device="cpu"):
        self.coreset_ratio = coreset_ratio
        self.proj_dim = proj_dim
        self.device = device
        backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval().to(device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.memory_bank = None
        self.feature_hw = None

    @torch.no_grad()
    def _features(self, x):
        """(B*H*W, 1536) patch features, one row per 8x8-pixel patch."""
        feats = FeatureExtractor(self.backbone)(x)  # (B, 1536, H, W)
        self.feature_hw = feats.shape[-2:]
        return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])

    @torch.no_grad()
    def _greedy_coreset(self, feats):
        """Pick a small subset of patches that still covers the whole feature space
        (iterative farthest-point sampling on a random projection, as in the paper)."""
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        proj = torch.randn(feats.shape[1], self.proj_dim, device=feats.device) / self.proj_dim**0.5
        z = feats @ proj
        # Squared distances via |a|^2 + |b|^2 - 2 a.b: one matrix-vector product per step instead of a
        # cdist call. Same farthest-point order as Euclidean distances (squaring is monotonic), ~6x faster.
        sq = (z * z).sum(1)
        selected = torch.empty(n_keep, dtype=torch.long)
        selected[0] = torch.randint(len(z), (1,)).item()
        min_dist = sq - 2 * (z @ z[selected[0]]) + sq[selected[0]]
        for i in range(1, n_keep):
            idx = int(torch.argmax(min_dist))
            selected[i] = idx
            torch.minimum(min_dist, sq - 2 * (z @ z[idx]) + sq[idx], out=min_dist)
        return feats[selected]

    @torch.no_grad()
    def fit(self, loader):
        t0 = time.perf_counter()
        feats = torch.cat([self._features(x.to(self.device)) for x, *_ in loader])
        t1 = time.perf_counter()
        self.memory_bank = self._greedy_coreset(feats)
        self.timing = {"features_s": round(t1 - t0, 1), "coreset_s": round(time.perf_counter() - t1, 1)}
        return len(feats), len(self.memory_bank)

    @torch.no_grad()
    def predict(self, x, out_size):
        """Returns (image scores (B,), anomaly maps (B, out_size, out_size))."""
        feats = self._features(x.to(self.device))
        dists = torch.cat([torch.cdist(chunk, self.memory_bank).min(dim=1).values for chunk in feats.split(4096)])
        h, w = self.feature_hw
        maps = dists.reshape(x.shape[0], 1, h, w)
        scores = maps.flatten(1).max(dim=1).values
        maps = F.interpolate(maps, size=(out_size, out_size), mode="bilinear", align_corners=False)
        maps = gaussian_blur(maps, sigma=4)
        return scores.cpu(), maps.squeeze(1).cpu()

    def save(self, path):
        torch.save({"memory_bank": self.memory_bank, "coreset_ratio": self.coreset_ratio}, path)

    def load(self, path):
        self.memory_bank = torch.load(path, map_location=self.device)["memory_bank"]


def gaussian_blur(maps, sigma):
    k = int(4 * sigma + 1) | 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = (g / g.sum()).to(maps)
    maps = F.conv2d(F.pad(maps, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    maps = F.conv2d(F.pad(maps, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1))
    return maps

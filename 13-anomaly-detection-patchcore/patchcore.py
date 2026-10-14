"""Minimal PatchCore implementation (Roth et al., "Towards Total Recall in Industrial Anomaly
Detection", CVPR 2022), written from scratch in PyTorch for CPU.

Idea: a model pretrained on ImageNet already describes local image patches well. Collect the
patch features of *defect-free* training images into a memory bank, then score a test patch by
its distance to the nearest patch in the bank. Patches unlike anything seen in good parts are
anomalies. No defect images are needed for training.
"""

import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


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
        """Mid-level features (layer2 + layer3): local enough for small defects,
        semantic enough to ignore pixel noise."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        f2 = b.layer2(b.layer1(x))
        f3 = b.layer3(f2)
        # Locally aware patch features: average each position with its 3x3 neighborhood.
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([f2, f3], dim=1)  # (B, 1536, H, W)
        self.feature_hw = feats.shape[-2:]
        return feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1])  # (B*H*W, 1536)

    @torch.no_grad()
    def _greedy_coreset(self, feats):
        """Pick a small subset of patches that still covers the whole feature space
        (iterative farthest-point sampling on a random projection, as in the paper)."""
        n_keep = max(1, int(len(feats) * self.coreset_ratio))
        proj = torch.randn(feats.shape[1], self.proj_dim, device=feats.device) / self.proj_dim**0.5
        z = feats @ proj
        selected = [torch.randint(len(z), (1,)).item()]
        min_dist = torch.cdist(z, z[selected]).squeeze(1)
        for _ in range(n_keep - 1):
            idx = int(torch.argmax(min_dist))
            selected.append(idx)
            min_dist = torch.minimum(min_dist, torch.cdist(z, z[idx : idx + 1]).squeeze(1))
        return feats[selected]

    @torch.no_grad()
    def fit(self, loader):
        feats = torch.cat([self._features(x.to(self.device)) for x, *_ in loader])
        self.memory_bank = self._greedy_coreset(feats)
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

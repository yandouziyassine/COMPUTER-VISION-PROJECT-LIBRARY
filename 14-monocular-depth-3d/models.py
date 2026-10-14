"""Depth models compared in this project, behind one interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common.paths  # noqa: E402,F401  (model downloads go to cv-data, not C:)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import pipeline  # noqa: E402

# name: (Hugging Face id, output kind). "disparity" = relative inverse depth; "metric" = meters.
MODELS = {
    "da2-small": ("depth-anything/Depth-Anything-V2-Small-hf", "disparity"),
    "da2-base": ("depth-anything/Depth-Anything-V2-Base-hf", "disparity"),
    "dpt-hybrid": ("Intel/dpt-hybrid-midas", "disparity"),
    "da2-metric-indoor": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "metric"),
}
RELATIVE = [k for k, (_, kind) in MODELS.items() if kind == "disparity"]


def load(name):
    torch.set_num_threads(4)
    return pipeline("depth-estimation", model=MODELS[name][0], device="cpu")


@torch.inference_mode()
def predict(estimator, rgb):
    """rgb uint8 (H, W, 3) -> float32 (H, W): disparity or meters depending on the model."""
    out = estimator(Image.fromarray(rgb))["predicted_depth"]
    out = out.reshape(1, 1, *out.shape[-2:]).float()
    return F.interpolate(out, size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].numpy()

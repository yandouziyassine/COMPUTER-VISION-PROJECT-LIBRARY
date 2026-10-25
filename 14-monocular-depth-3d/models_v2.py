"""v2 model zoo (8 models) behind one interface: load(name, device) -> predictor; predictor(rgb) -> float32 (H, W).

Output kind decides the protocol alignment (configs/protocol.yaml):
  disparity    affine-invariant inverse depth  (Depth Anything V2 S/B/L, DPT-Hybrid)
  affine_depth affine-invariant depth          (Marigold LCM)
  metric       meters                          (Depth Pro, MoGe-2, DA-V2 Metric-Indoor)

Revisions are pinned so cached predictions can be traced in the manifest. Heavy imports happen inside load(),
so this file can be imported (and MODELS read) without torch/transformers/diffusers/moge installed.
Inference for v2 runs on a Kaggle T4 (kaggle_infer_v2.py); v1's models.py stays as the CPU path."""

import numpy as np

# name: (source, repo id, revision, kind, licence)
MODELS = {
    "da2-small": ("hf-pipeline", "depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f3", "disparity", "Apache-2.0"),
    "da2-base": ("hf-pipeline", "depth-anything/Depth-Anything-V2-Base-hf", "b1958afc87", "disparity", "CC BY-NC 4.0"),
    "da2-large": (
        "hf-pipeline",
        "depth-anything/Depth-Anything-V2-Large-hf",
        "7581137eff",
        "disparity",
        "CC BY-NC 4.0",
    ),
    "dpt-hybrid": ("hf-pipeline", "Intel/dpt-hybrid-midas", "11eaf7a1cf", "disparity", "Apache-2.0"),
    "da2-metric-indoor": (
        "hf-pipeline",
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
        "8078d68a9c",
        "metric",
        "not stated (Small family Apache-2.0)",
    ),
    "depth-pro": (
        "depth-pro",
        "apple/DepthPro-hf",
        "de816c8ce7",
        "metric",
        "Apple ML Research licence (non-commercial)",
    ),
    "moge2-large": ("moge2", "Ruicheng/moge-2-vitl", "39c4d5e957", "metric", "MIT"),
    "marigold-lcm": ("marigold", "prs-eth/marigold-depth-lcm-v1-0", "04a73502f7", "affine_depth", "Apache-2.0"),
}
NEW_IN_V2 = ["da2-large", "depth-pro", "moge2-large", "marigold-lcm"]
MARIGOLD_ENSEMBLE, MARIGOLD_STEPS, MARIGOLD_RES = 10, 1, 768  # "LCM 10 x 1" row of the Marigold journal paper
MOGE_GIT = "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"


def kind(name):
    return MODELS[name][3]


def _resize(out, hw):
    import torch.nn.functional as F

    out = out.reshape(1, 1, *out.shape[-2:]).float()
    return F.interpolate(out, size=hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()


def load(name, device="cuda"):
    import torch
    from PIL import Image

    source, repo, rev, _, _ = MODELS[name]
    full_rev = _full_revision(repo, rev)

    if source == "hf-pipeline":
        from transformers import pipeline

        est = pipeline("depth-estimation", model=repo, revision=full_rev, device=0 if device == "cuda" else -1)

        @torch.inference_mode()
        def predict(rgb):
            return _resize(est(Image.fromarray(rgb))["predicted_depth"], rgb.shape[:2])

    elif source == "depth-pro":
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast

        proc = DepthProImageProcessorFast.from_pretrained(repo, revision=full_rev)
        net = DepthProForDepthEstimation.from_pretrained(repo, revision=full_rev, torch_dtype=torch.float16)
        net = net.to(device).eval()

        @torch.inference_mode()
        def predict(rgb):
            inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device, torch.float16)
            post = proc.post_process_depth_estimation(net(**inputs), target_sizes=[rgb.shape[:2]])
            return post[0]["predicted_depth"].float().cpu().numpy()  # meters, focal predicted by the model

    elif source == "moge2":
        from moge.model.v2 import MoGeModel

        net = MoGeModel.from_pretrained(repo, revision=full_rev).to(device).eval()

        @torch.inference_mode()
        def predict(rgb):
            x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
            d = net.infer(x, apply_mask=False)["depth"].float().cpu().numpy()  # meters, focal predicted
            bad = ~np.isfinite(d) | (d <= 0)
            if bad.any():
                d[bad] = np.nanmax(np.where(bad, np.nan, d))
            return d

    elif source == "marigold":
        import diffusers

        pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            repo, revision=full_rev, variant="fp16", torch_dtype=torch.float16
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        @torch.inference_mode()
        def predict(rgb):
            out = pipe(
                Image.fromarray(rgb),
                num_inference_steps=MARIGOLD_STEPS,
                ensemble_size=MARIGOLD_ENSEMBLE,
                processing_resolution=MARIGOLD_RES,
                match_input_resolution=True,
                batch_size=MARIGOLD_ENSEMBLE,
                generator=torch.Generator(device=device).manual_seed(0),
            )
            return np.asarray(out.prediction).reshape(rgb.shape[:2]).astype(np.float32)  # affine depth in [0, 1]

    else:
        raise ValueError(source)
    return predict


def _full_revision(repo, short):
    """Short sha -> full commit sha of the Hugging Face repo (pinning needs the full hash)."""
    from huggingface_hub import HfApi

    full = HfApi().model_info(repo).sha
    if not full.startswith(short):
        print(f"warning: {repo} head is {full[:10]}, pinned {short}; using head", flush=True)
    return full

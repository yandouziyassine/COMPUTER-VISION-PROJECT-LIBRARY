"""W8 · Can it be fixed? Mirror-aware fine-tuning of Depth Anything V2 Small on NYU (Kaggle T4).

Inside a mirror the Kinect measures the reflected room, so a model trained on sensor depth learns to see "through"
mirrors. Four fine-tunes on the NYU labeled training split (795 images, mirror images oversampled x4), same schedule:
  plain    target = sensor depth everywhere (the reflection inside mirrors)
  ignore   mirror pixels removed from the loss
  plane    inside mirrors the target is the mirror plane fitted to the sensor depth of the frame / wall around it
           (only mirrors whose surrounding ring fits a plane: a visible boundary)
  lora     'plane' targets, LoRA (rank 8 on attention query/value) instead of full fine-tuning
Loss: scale-and-shift-invariant L1 in disparity (MiDaS), the model's native output.
Evaluation on the 654 test images, aligned on non-mirror pixels (least squares in disparity):
  mirror AbsRel against the mirror plane, mirror planarity (relative RMS distance of the predicted 3D points to
  their own best plane), and AbsRel on every other class against the sensor (regression check).
Weights stay on Kaggle (never committed); only metrics are written.

Usage (Kaggle): python kaggle_push_v2.py --name depth-w8-mirror --script w8_mirror_finetune.py --commit <sha>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
VARIANTS = ["plain", "ignore", "plane", "lora"]
SIZE = (476, 644)  # multiples of the 14-px ViT patch, NYU aspect ratio
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def ssi_l1(pred, target, mask):
    """Scale-and-shift-invariant L1 (torch): per image, least-squares align pred to target on mask, mean |residual|."""
    import torch

    losses = []
    for p, t, m in zip(pred, target, mask):
        p, t = p[m], t[m]
        if p.numel() < 100:
            continue
        A = torch.stack([p, torch.ones_like(p)], 1)
        sol = torch.linalg.lstsq(A.detach().float(), t.unsqueeze(1).float()).solution.squeeze(1)
        losses.append(torch.mean(torch.abs(sol[0] * p + sol[1] - t)) / t.mean())
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def mirror_targets(data, i, K, mirror_id):
    """(disparity target, loss mask, plane target or None) for one NYU image."""
    import cv2

    from w9_build_study import surface_depth_from_ring

    raw = data.raw(i).astype(np.float32)
    lab = data.labels(i)
    valid = raw > 0
    mirror = lab == mirror_id
    plane = None
    if mirror.sum() > 300:
        nl, comp = cv2.connectedComponents(mirror.astype(np.uint8))
        plane = np.full(raw.shape, np.nan, np.float32)
        for k in range(1, nl):
            region = comp == k
            if region.sum() < 300:
                continue
            p = surface_depth_from_ring(region, raw, K)
            if p is not None:
                plane[region] = p[region]
    return raw, valid, mirror, plane


def planarity(depth, region, K):
    """Relative RMS distance of the back-projected points in `region` to their own best-fit plane."""
    from metrics import backproject, fit_plane

    v, u = np.nonzero(region)
    pts = backproject(u, v, depth[v, u], K)
    n, d = fit_plane(pts)
    return float(np.sqrt(np.mean((pts @ n + d) ** 2)) / np.median(pts[:, 2]))


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.53.0", "peft==0.15.2"], check=True)
    src = Path("/tmp/repo")
    if not src.exists():
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/yandouziyassine/COMPUTER-VISION-PROJECT-LIBRARY.git", str(src)],
            check=True,
        )
    subprocess.run(["git", "-C", str(src), "checkout", "-q", JOB["commit"]], check=True)
    os.environ["CV_DATA"] = "/tmp/cv-data"
    sys.path[:0] = [str(src / "14-monocular-depth-3d"), str(src)]
    import cv2
    import torch
    import torch.nn.functional as F
    from scipy.io import loadmat
    from transformers import AutoModelForDepthEstimation

    import datasets_v2 as D
    import models_v2 as M
    import protocol as P

    root = Path("/tmp/cv-data/depth/nyu")
    D.fetch_nyu(root)
    from nyu import K, NYU

    data = NYU()
    mirror_id = data.class_id("mirror")
    train = [int(i) - 1 for i in loadmat(root / "splits.mat")["trainNdxs"].ravel()]
    test = data.test_indices()
    print("caching", flush=True)
    cache = {}
    for i in train + test:
        raw, valid, mirror, plane = mirror_targets(data, i, K, mirror_id)
        cache[i] = (data.rgb(i), raw, valid, mirror, plane, data.labels(i))
    train_list = [i for i in train for _ in range(4 if cache[i][3].sum() > 300 else 1)]
    n_mirror_train = sum(cache[i][3].sum() > 300 for i in train)
    n_plane_train = sum(cache[i][4] is not None and np.isfinite(cache[i][4]).any() for i in train)
    print("train", len(train), "mirror images", n_mirror_train, "with plane", n_plane_train, flush=True)

    def to_input(rgbs, flip):
        x = np.stack([cv2.resize(r, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_LINEAR) for r in rgbs]) / 255.0
        x = (x - MEAN) / STD
        x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
        return torch.flip(x, [3]) if flip else x

    def predict(net, rgb):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = net(pixel_values=to_input([rgb], False).cuda()).predicted_depth
        return (
            F.interpolate(out[:, None].float(), size=(480, 640), mode="bilinear", align_corners=False)[0, 0]
            .cpu()
            .numpy()
        )

    proto = P.load_protocol()
    ds = proto["datasets"]["nyu"]
    other_classes = {
        n: data.class_id(n)
        for n in [
            "wall",
            "floor",
            "ceiling",
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "picture",
            "television",
            "monitor",
        ]
        if n in data.names
    }

    def evaluate(net):
        net.eval()
        mirror_abs, mirror_plan, per_class, other = [], [], {n: [] for n in other_classes}, []
        for i in test:
            rgb, raw, valid, mirror, plane, lab = cache[i]
            fit_mask = P.valid_mask(raw, ds) & ~mirror
            d = P.align(predict(net, rgb), raw, fit_mask, "lsq_disparity", 10.0, 1e-3)
            other.append(float(np.mean(np.abs(d[fit_mask] - raw[fit_mask]) / raw[fit_mask])))
            for n, c in other_classes.items():
                m = fit_mask & (lab == c)
                if m.sum() > 200:
                    per_class[n].append(float(np.mean(np.abs(d[m] - raw[m]) / raw[m])))
            if plane is not None:
                m = np.isfinite(plane) & mirror
                if m.sum() > 300:
                    mirror_abs.append(float(np.mean(np.abs(d[m] - plane[m]) / plane[m])))
                    mirror_plan.append(planarity(d, m, K))
        return {
            "mirror_absrel_vs_plane": float(np.mean(mirror_abs)),
            "mirror_planarity": float(np.mean(mirror_plan)),
            "mirror_images": len(mirror_abs),
            "absrel_non_mirror": float(np.mean(other)),
            "absrel_by_class": {n: float(np.mean(v)) for n, v in per_class.items() if v},
            "per_image_mirror_absrel": mirror_abs,
            "per_image_non_mirror": other,
        }

    _, repo, rev, _, _ = M.MODELS["da2-small"]
    full_rev = M._full_revision(repo, rev)
    results = {"commit": JOB["commit"], "train_images": len(train), "mirror_train_images": int(n_mirror_train)}
    base = AutoModelForDepthEstimation.from_pretrained(repo, revision=full_rev).cuda()
    results["pretrained"] = evaluate(base)
    print("pretrained", {k: v for k, v in results["pretrained"].items() if not k.startswith("per_")}, flush=True)
    del base
    epochs, bs = JOB.get("epochs", 5), 4
    for variant in JOB.get("variants") or VARIANTS:
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        net = AutoModelForDepthEstimation.from_pretrained(repo, revision=full_rev).cuda()
        if variant == "lora":
            from peft import LoraConfig, get_peft_model

            net = get_peft_model(
                net, LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], modules_to_save=["head"])
            )
            opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
        else:
            backbone = [p for n, p in net.named_parameters() if "backbone" in n]
            rest = [p for n, p in net.named_parameters() if "backbone" not in n]
            opt = torch.optim.AdamW([{"params": backbone, "lr": 5e-6}, {"params": rest, "lr": 5e-5}])
        scaler = torch.cuda.amp.GradScaler()
        t0 = time.time()
        for ep in range(epochs):
            net.train()
            order = rng.permutation(train_list)
            for b in range(0, len(order) - bs + 1, bs):
                ids = order[b : b + bs]
                flip = bool(rng.random() < 0.5)
                tgt, msk = [], []
                for i in ids:
                    _, raw, valid, mirror, plane, _ = cache[i]
                    depth, m = raw.copy(), valid.copy()
                    if variant == "ignore":
                        m &= ~mirror
                    elif variant in ("plane", "lora") and plane is not None:
                        has = np.isfinite(plane)
                        depth[has], m[has] = plane[has], True
                        m &= ~(mirror & ~has)  # mirror parts without a fitted plane are ignored
                    disp = np.where(m & (depth > 0), 1.0 / np.maximum(depth, 1e-3), 0)
                    tgt.append(cv2.resize(disp, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_NEAREST))
                    msk.append(cv2.resize(m.astype(np.uint8), (SIZE[1], SIZE[0]), interpolation=cv2.INTER_NEAREST) > 0)
                tgt, msk = torch.tensor(np.stack(tgt)).cuda(), torch.tensor(np.stack(msk)).cuda()
                if flip:
                    tgt, msk = torch.flip(tgt, [2]), torch.flip(msk, [2])
                x = to_input([cache[i][0] for i in ids], flip).cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    out = net(pixel_values=x).predicted_depth
                out = F.interpolate(out[:, None].float(), size=SIZE, mode="bilinear", align_corners=False)[:, 0]
                loss = ssi_l1(out, tgt, msk)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.save(
                {k: v.cpu() for k, v in net.state_dict().items() if "lora" in k or variant != "lora"},
                f"/kaggle/working/w8_{variant}_last.pt",
            )
            print(variant, "epoch", ep + 1, f"loss {loss.item():.4f}", f"{time.time() - t0:.0f}s", flush=True)
        results[variant] = evaluate(net)
        results[variant]["train_minutes"] = round((time.time() - t0) / 60, 1)
        print(variant, {k: v for k, v in results[variant].items() if not k.startswith("per_")}, flush=True)
        Path("/kaggle/working/w8_results.json").write_text(json.dumps(results))
        del net, opt
        torch.cuda.empty_cache()
    for f in Path("/kaggle/working").glob("w8_*_last.pt"):
        f.unlink()  # weights are not kept as kernel output (never redistributed)


if __name__ == "__main__":
    main()

"""W5 · Scale and the camera: does metric depth survive a change of focal length?

A photo taken with a longer lens is simulated by cropping the centre and enlarging it (zoom z > 1: focal x z);
a wider lens by shrinking the photo inside a black border (z < 1: focal x z, the border is visible to the model).
Depths in the scene do not change, so a perfect metric model gives the same meters at every zoom. Each metric
model runs three ways: estimating the focal itself, given the true focal, and given a wrong focal (the
unzoomed camera's, as when EXIF and crop disagree). Runs on a Kaggle T4 (pushed with kaggle_push_v2.py --script).

Usage (Kaggle): python kaggle_push_v2.py --name depth-w5-focal --script w5_focal.py --commit <sha>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ZOOMS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
METRIC_MODELS = ["depth-pro", "moge2-large", "da2-metric-indoor"]
JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))


def zoom_view(rgb, gt, z):
    """Simulate a focal length z times the original. Returns (model_input (H, W, 3), gt_view, box), where box =
    (y0, y1, x0, x1) is the part of model_input that shows gt_view (prediction there is compared with gt_view)."""
    import cv2

    h, w = rgb.shape[:2]
    if z >= 1:
        ch, cw = int(round(h / z)), int(round(w / z))
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = rgb[y0 : y0 + ch, x0 : x0 + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR), gt[y0 : y0 + ch, x0 : x0 + cw], (0, h, 0, w)
    sh, sw = int(round(h * z)), int(round(w * z))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros_like(rgb)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0 : y0 + sh, x0 : x0 + sw] = small
    return canvas, gt, (y0, y0 + sh, x0, x0 + sw)


def pred_on_gt(pred, box, gt_shape):
    """Cut the prediction to `box` and resize it to the GT view."""
    import cv2

    y0, y1, x0, x1 = box
    return cv2.resize(pred[y0:y1, x0:x1], (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)


def scale_stats(pred, gt, max_depth=10.0):
    """AbsRel without alignment and the median scale ratio pred / gt (1 = right scale) on valid GT."""
    m = (gt > 1e-3) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if m.sum() < 100:
        return None
    return {
        "absrel": float(np.mean(np.abs(pred[m] - gt[m]) / gt[m])),
        "scale": float(np.median(pred[m] / gt[m])),
    }


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers==4.53.0",
            "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a",
        ],
        check=True,
    )
    src = Path("/tmp/repo")
    if not src.exists():
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/yandouziyassine/COMPUTER-VISION-PROJECT-LIBRARY.git", str(src)],
            check=True,
        )
    subprocess.run(["git", "-C", str(src), "checkout", "-q", JOB["commit"]], check=True)
    os.environ["CV_DATA"] = "/tmp/cv-data"
    sys.path[:0] = [str(src / "14-monocular-depth-3d"), str(src)]
    import datasets_v2 as D
    import models_v2 as M

    root = Path("/tmp/cv-data/depth/nyu")
    root.mkdir(parents=True, exist_ok=True)
    for url, fn in [(D.NYU_URL, "nyu_depth_v2_labeled.mat"), (D.NYU_SPLIT_URL, "splits.mat")]:
        if not (root / fn).exists():
            subprocess.run(["curl", "-sL", "-o", str(root / fn), url], check=True)
    from nyu import K, NYU

    data = NYU()
    idx = data.test_indices()[:: JOB.get("every", 3)]
    fx0 = K[0]
    out = {"commit": JOB["commit"], "zooms": ZOOMS, "fx0": fx0, "images": len(idx), "rows": []}
    for name in JOB.get("models") or METRIC_MODELS:
        predict, t0 = M.load_with_focal(name, "cuda"), time.time()
        for i in idx:
            rgb, raw = data.rgb(i), data.raw(i).astype(np.float32)
            for z in ZOOMS:
                inp, gt_view, box = zoom_view(rgb, raw, z)
                modes = {"own": None, "true": fx0 * z, "wrong": fx0} if name != "da2-metric-indoor" else {"own": None}
                for mode, fx in modes.items():
                    if mode == "wrong" and z == 1.0:
                        continue  # identical to "true"
                    pred, fx_used = predict(inp, fx)
                    s = scale_stats(pred_on_gt(pred, box, gt_view.shape), gt_view)
                    if s:
                        out["rows"].append(
                            {
                                "model": name,
                                "image": int(i),
                                "zoom": z,
                                "mode": mode,
                                "fx_true": fx0 * z,
                                "fx_used": fx_used,
                                **s,
                            }
                        )
        print(name, f"{time.time() - t0:.0f}s", flush=True)
        Path("/kaggle/working/w5_focal.json").write_text(json.dumps(out))


def report(path):
    """Summarise the Kaggle output: per model x mode x zoom, mean AbsRel (bootstrap CI over images), median scale
    ratio and, for 'own', the focal length the model estimated relative to the true one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x)
        m = x[rng.integers(0, len(x), (1000, len(x)))].mean(axis=1)
        return [
            round(float(x.mean()), 4),
            round(float(np.quantile(m, 0.025)), 4),
            round(float(np.quantile(m, 0.975)), 4),
        ]

    res = {"images": d["images"], "zooms": d["zooms"], "fx0": d["fx0"], "models": {}}
    for model in dict.fromkeys(r["model"] for r in rows):
        res["models"][model] = {}
        for mode in ("own", "true", "wrong"):
            per_zoom = {}
            for z in d["zooms"]:
                r = [x for x in rows if x["model"] == model and x["mode"] == mode and x["zoom"] == z]
                if not r:
                    continue
                e = {
                    "absrel": ci([x["absrel"] for x in r]),
                    "median_scale": round(float(np.median([x["scale"] for x in r])), 3),
                }
                if mode == "own" and r[0]["fx_used"] is not None:
                    e["focal_estimate_ratio"] = round(float(np.median([x["fx_used"] / x["fx_true"] for x in r])), 3)
                per_zoom[str(z)] = e
            if per_zoom:
                res["models"][model][mode] = per_zoom
    out = Path(__file__).resolve().parent / "docs"
    (out / "w5_focal.json").write_text(json.dumps(res, indent=1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    styles = {"own": "-", "true": "--", "wrong": ":"}
    for k, (model, modes) in enumerate(res["models"].items()):
        for mode, pz in modes.items():
            zs = [float(z) for z in pz]
            axes[0].plot(
                zs,
                [pz[z]["median_scale"] for z in pz],
                styles[mode],
                marker="o",
                color=f"C{k}",
                label=f"{model} · {mode} focal",
            )
            axes[1].plot(zs, [pz[z]["absrel"][0] for z in pz], styles[mode], marker="o", color=f"C{k}")
    axes[0].axhline(1, color="gray", lw=0.8)
    axes[0].set_ylabel("median predicted / true depth (1 = right scale)")
    axes[1].set_ylabel("AbsRel, no alignment")
    for ax in axes:
        ax.set_xlabel("simulated focal length (x original); < 1 = wider lens (black border), > 1 = zoom")
        ax.set_xscale("log", base=2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Metric depth vs camera focal length, NYU ({res['images']} images)")
    fig.tight_layout()
    fig.savefig(out / "w5_focal.png", dpi=110)
    return res


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--report":
        report(sys.argv[2])
    else:
        main()

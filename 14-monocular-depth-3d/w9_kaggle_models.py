"""Kaggle T4 job for W9: every model answers every "which point is closer?" pair of the human study.

Input: a private Kaggle dataset with the study folder (img/*.jpg) and docs/w9_pairs.json (built by w9_build_study.py).
Output (/kaggle/working/w9_model_answers.json): per model, per pair, the model's value at A and B (median of a 5 x 5
window) and its answer. Ordinal answers need no alignment: a larger disparity, or a smaller depth, is closer.
Pushed by kaggle_push_v2.py --script w9_kaggle_models.py (JOB carries the commit and dataset path)."""

import json
import os

import numpy as np
import subprocess
import sys
import time
from pathlib import Path

JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
REPO = "https://github.com/yandouziyassine/COMPUTER-VISION-PROJECT-LIBRARY.git"
PIP = [
    "transformers==4.53.0",
    "diffusers==0.34.0",
    "accelerate",
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a",
]


def value_at(pred, xy, r=2):
    x, y = xy
    win = pred[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    return float(np.median(win))


TRANSFORMS = ["none", "rot180", "gray", "blur"]
BLUR_SIGMA = 0.004  # fraction of the image width; the web page uses the same fraction of the displayed width


def transform(rgb, xy_a, xy_b, name):
    """Cue ablations shared with the web page: rotate 180 degrees (tests the light-from-above / ground-below
    priors), grayscale (no colour), Gaussian blur (no fine texture). Returns image and moved points."""
    import cv2

    h, w = rgb.shape[:2]
    if name == "none":
        return rgb, xy_a, xy_b
    if name == "rot180":
        flip = lambda p: [w - 1 - p[0], h - 1 - p[1]]  # noqa: E731
        return np.ascontiguousarray(rgb[::-1, ::-1]), flip(xy_a), flip(xy_b)
    if name == "gray":
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return np.stack([g] * 3, -1), xy_a, xy_b
    if name == "blur":
        return cv2.GaussianBlur(rgb, (0, 0), BLUR_SIGMA * w), xy_a, xy_b
    raise ValueError(name)


def closer(kind, va, vb):
    """'a' or 'b': larger disparity is closer; smaller depth is closer."""
    if kind == "disparity":
        return "a" if va > vb else "b"
    return "a" if va < vb else "b"


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PIP], check=True)
    src = Path("/tmp/repo")
    if not src.exists():
        subprocess.run(["git", "clone", "-q", REPO, str(src)], check=True)
    subprocess.run(["git", "-C", str(src), "checkout", "-q", JOB["commit"]], check=True)
    sys.path[:0] = [str(src / "14-monocular-depth-3d"), str(src)]
    import cv2

    import models_v2 as M

    root = next(Path("/kaggle/input").rglob("w9_pairs.json")).parent
    img_dir = next(Path("/kaggle/input").rglob("img"))
    pairs = json.loads((root / "w9_pairs.json").read_text())["pairs"]
    by_image = {}
    for p in pairs:
        by_image.setdefault(p["image"], []).append(p)
    out = {"commit": JOB["commit"], "models": {}}
    for name in JOB["models"]:
        predict, t0 = M.load(name, "cuda"), time.time()
        by_t = {}
        for tname in JOB.get("transforms") or ["none"]:
            ans = {}
            for img, ps in by_image.items():
                rgb = cv2.cvtColor(cv2.imread(str(img_dir / img)), cv2.COLOR_BGR2RGB)
                preds = {}
                for p in ps:
                    timg, a, b = transform(rgb, p["a"], p["b"], tname)
                    if "pred" not in preds:
                        preds["pred"] = np.asarray(predict(timg), np.float32)
                    va, vb = value_at(preds["pred"], a), value_at(preds["pred"], b)
                    ans[p["id"]] = {"a": va, "b": vb, "answer": closer(M.kind(name), va, vb)}
            by_t[tname] = ans
            acc = np.mean([ans[p["id"]]["answer"] == p["truth"] for p in pairs])
            print(name, tname, f"accuracy {acc:.3f}", flush=True)
        out["models"][name] = {
            "kind": M.kind(name),
            "seconds": round(time.time() - t0, 1),
            "answers": by_t["none"],
            "by_transform": by_t,
        }
        Path("/kaggle/working/w9_model_answers.json").write_text(json.dumps(out))
        del predict
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

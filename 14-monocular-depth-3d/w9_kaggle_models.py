"""Kaggle T4 job for W9: every model answers every "which point is closer?" pair of the human study.

Input: a private Kaggle dataset with the study folder (img/*.jpg) and docs/w9_pairs.json (built by w9_build_study.py).
Output (/kaggle/working/w9_model_answers.json): per model, per pair, the model's value at A and B (median of a 5 x 5
window) and its answer. Ordinal answers need no alignment: a larger disparity, or a smaller depth, is closer.
Pushed by kaggle_push_v2.py --script w9_kaggle_models.py (JOB carries the commit and dataset path)."""

import json
import os
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
    import numpy as np

    x, y = xy
    win = pred[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    return float(np.median(win))


def closer(kind, va, vb):
    """'a' or 'b': larger disparity is closer; smaller depth is closer."""
    if kind == "disparity":
        return "a" if va > vb else "b"
    return "a" if va < vb else "b"


def main():
    import numpy as np

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
        ans = {}
        for img, ps in by_image.items():
            rgb = cv2.cvtColor(cv2.imread(str(img_dir / img)), cv2.COLOR_BGR2RGB)
            pred = np.asarray(predict(rgb), np.float32)
            for p in ps:
                va, vb = value_at(pred, p["a"]), value_at(pred, p["b"])
                ans[p["id"]] = {"a": va, "b": vb, "answer": closer(M.kind(name), va, vb)}
        out["models"][name] = {"kind": M.kind(name), "seconds": round(time.time() - t0, 1), "answers": ans}
        acc = np.mean([ans[p["id"]]["answer"] == p["truth"] for p in pairs])
        print(name, f"accuracy {acc:.3f}", flush=True)
        Path("/kaggle/working/w9_model_answers.json").write_text(json.dumps(out))
        del predict
        import torch

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

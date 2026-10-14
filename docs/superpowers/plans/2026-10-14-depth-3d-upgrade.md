# Project 14 Depth Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn project 14 into an in-depth study: where monocular depth fails (NYU benchmark, failure map by material, poster illusion test), how well one photo can measure the world (metric depth + distances), and a "what you see vs what the computer sees" figure.

**Architecture:** Small focused modules in `14-monocular-depth-3d/`: `nyu.py` (data access), `metrics.py` (pure NumPy math, unit-tested), `models.py` (model zoo wrapper). Each experiment is one script that reads cached predictions written once by `benchmark.py` (so the slow inference runs once), writes its numbers into `docs/results.json` under its own key and saves one figure. The existing `depth3d.py` (point clouds, z-buffer rendering) is reused for 3D views.

**Tech Stack:** Python 3.10 shared venv, PyTorch CPU, Hugging Face Transformers (`depth-estimation` pipeline), h5py + SciPy (NYU `.mat`), OpenCV (homography warps), NumPy, Matplotlib, Gradio 6 (measuring app), pytest.

**Facts the engineer needs:**
- NYU labeled file: `D:\cv-data\depth\nyu\nyu_depth_v2_labeled.mat` (HDF5, open with h5py). Arrays are stored **transposed**: `images (1449, 3, 640, 480)`, `depths`/`rawDepths (1449, 640, 480)` float32 meters, `labels (1449, 640, 480)` uint16 (0 = unlabeled, else 1-based index into `names`). Transpose to `(480, 640)`.
- `depths` are hole-filled (in-painted); `rawDepths` are the sensor's real measurements (0 = no return). The Kinect itself fails on glass and is fooled by mirrors (it measures the reflection), so per-class analysis uses **raw-valid pixels only** and reports sensor coverage.
- Official split: `D:\cv-data\depth\nyu\splits.mat`, key `testNdxs` (654, 1-based).
- Class ids (1-based, from `names`): wall 21, floor 11, ceiling 4, table 19, door 28, cabinet 3, bed 157, chair 5, sofa 83, window 59, mirror 122, glass 612, television 172, monitor 49, picture 64, whiteboard 45, blinds 80, curtain 89. Look them up by name in code, don't hard-code.
- NYU RGB intrinsics (official toolbox): fx 518.8579, fy 519.4696, cx 325.5824, cy 253.7362.
- Standard evaluation: Eigen crop rows 45:471, cols 41:601; valid depth 0.001–10 m; relative models are aligned per image by least-squares scale + shift **in disparity space**.
- COCO val2017 images (poster content): `D:\cv-data\clip-search\val2017\`.
- Run everything from `14-monocular-depth-3d/` with `..\.venv\Scripts\python.exe`. CPU: `torch.set_num_threads(4)`.
- Long jobs (benchmark ≈ 1 h) run detached with checkpoints (repo CLAUDE.md rule).

---

## File structure

| File | Responsibility |
|---|---|
| `14-monocular-depth-3d/nyu.py` | open NYU, test indices, per-image rgb/depth/raw/labels, class ids, intrinsics, eval mask |
| `14-monocular-depth-3d/metrics.py` | alignment, AbsRel, δ<1.25, back-projection, 3D distance, plane fit, plane depth |
| `14-monocular-depth-3d/models.py` | model registry, loading, prediction at NYU resolution |
| `14-monocular-depth-3d/benchmark.py` | run all models on the 654 test images once, cache predictions, write `benchmark` results |
| `14-monocular-depth-3d/failures.py` | error by object class, distance and image region; sensor coverage; figure |
| `14-monocular-depth-3d/poster_test.py` | poster composites on flat walls, illusion score; figure |
| `14-monocular-depth-3d/illusions.py` | download 6 Commons illusion photos with credits; depth + side-view gallery |
| `14-monocular-depth-3d/see_vs_infer.py` | Part C panels: photo / raw numbers / inferred depth / truth / 3D side view |
| `14-monocular-depth-3d/measure.py` | metric depth + intrinsics (EXIF or FOV) + distance between two pixels |
| `14-monocular-depth-3d/measure_eval.py` | distance error on NYU pairs, with and without one reference length; figure |
| `14-monocular-depth-3d/measure_app.py` | Gradio app: click two points → meters ± uncertainty |
| `tests/test_14_depth.py` | unit tests for `metrics.py` and the poster geometry |
| `14-monocular-depth-3d/README.md`, `requirements.txt` | rewritten write-up; add h5py, scipy, opencv-python, gradio |

---

### Task 1: Metrics module (TDD)

**Files:**
- Create: `14-monocular-depth-3d/metrics.py`
- Test: `tests/test_14_depth.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Project 14: depth metrics, alignment and geometry on synthetic data (no model, no dataset)."""

import pytest

np = pytest.importorskip("numpy")

from conftest import load_module  # noqa: E402

m = load_module("14", "metrics.py")
K = (500.0, 500.0, 320.0, 240.0)


def test_alignment_recovers_scale_and_shift_in_disparity():
    rng = np.random.default_rng(0)
    gt = rng.uniform(1, 8, (48, 64)).astype(np.float32)
    pred_disp = (1 / gt - 0.05) / 3.0                  # model output = disparity with unknown scale/shift
    aligned = m.align_disparity(pred_disp, gt, np.ones_like(gt, bool))
    assert np.allclose(aligned, gt, rtol=1e-4)


def test_absrel_and_delta1_match_hand_computation():
    gt = np.array([[1.0, 2.0], [4.0, 5.0]])
    pred = np.array([[1.1, 2.0], [3.6, 5.0]])           # ratios 1.1, 1, 1.11, 1: all within 1.25
    mask = np.ones_like(gt, bool)
    assert m.abs_rel(pred, gt, mask) == pytest.approx((0.1 + 0 + 0.1 + 0) / 4)
    assert m.delta1(pred, gt, mask) == pytest.approx(1.0)
    pred[1, 0] = 5.5                                      # 5.5 / 4 = 1.375 > 1.25
    assert m.delta1(pred, gt, mask) == pytest.approx(0.75)


def test_distance_between_backprojected_pixels():
    depth = np.full((480, 640), 2.0)
    # two pixels on the same row, 100 px apart, at 2 m: 100 * 2 / 500 = 0.4 m apart
    assert m.distance((300, 240), (400, 240), depth, K) == pytest.approx(0.4)


def test_plane_fit_and_plane_depth():
    fx, fy, cx, cy = K
    v, u = np.mgrid[100:200, 200:300]
    z = 3.0 + 0.002 * (u - cx)                         # tilted wall
    pts = m.backproject(u.ravel(), v.ravel(), z.ravel(), K)
    n, d = m.fit_plane(pts)
    assert np.allclose(m.plane_depth(u, v, n, d, K), z, atol=1e-6)


def test_eval_mask_applies_eigen_crop_and_range():
    gt = np.full((480, 640), 3.0)
    gt[100, 100] = 0.0
    gt[200, 200] = 12.0
    mask = m.eval_mask(gt)
    assert not mask[0, 0] and not mask[100, 100] and not mask[200, 200] and mask[300, 300]
```

- [ ] **Step 2: Run to verify failure**

Run: `..\.venv\Scripts\python.exe -m pytest ..\tests\test_14_depth.py -q` (from `14-monocular-depth-3d`)
Expected: errors, `metrics.py` not found.

- [ ] **Step 3: Implement `metrics.py`**

```python
"""Depth evaluation math: alignment, standard metrics, pinhole geometry. Pure NumPy, unit-tested."""

import numpy as np

EIGEN_CROP = (slice(45, 471), slice(41, 601))   # standard NYU evaluation crop (480 x 640 images)
MIN_DEPTH, MAX_DEPTH = 1e-3, 10.0


def eval_mask(gt):
    """Valid ground-truth pixels inside the Eigen crop."""
    mask = np.zeros(gt.shape, bool)
    mask[EIGEN_CROP] = True
    return mask & (gt > MIN_DEPTH) & (gt < MAX_DEPTH)


def align_disparity(pred_disp, gt_depth, mask):
    """Relative models predict disparity up to scale and shift. Fit s, t (least squares) so that
    s * pred + t ~ 1 / gt on the mask, then return the aligned *depth* for every pixel."""
    x = pred_disp[mask].astype(np.float64)
    y = 1.0 / gt_depth[mask].astype(np.float64)
    A = np.stack([x, np.ones_like(x)], axis=1)
    (s, t), *_ = np.linalg.lstsq(A, y, rcond=None)
    disp = s * pred_disp + t
    return 1.0 / np.clip(disp, 1.0 / MAX_DEPTH, None)


def abs_rel(pred, gt, mask):
    return float(np.mean(np.abs(pred[mask] - gt[mask]) / gt[mask]))


def delta1(pred, gt, mask):
    r = np.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return float(np.mean(r < 1.25))


def backproject(u, v, z, K):
    """Pixels (u, v) with depth z (meters) -> 3D points (N, 3) in the camera frame. K = (fx, fy, cx, cy)."""
    fx, fy, cx, cy = K
    u, v, z = (np.asarray(a, np.float64) for a in (u, v, z))
    return np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=-1)


def distance(p1, p2, depth, K):
    """3D distance (meters) between two pixels (u, v) of a depth map."""
    a = backproject(p1[0], p1[1], depth[int(p1[1]), int(p1[0])], K)
    b = backproject(p2[0], p2[1], depth[int(p2[1]), int(p2[0])], K)
    return float(np.linalg.norm(a - b))


def fit_plane(points):
    """Least-squares plane n . p + d = 0 (unit normal) through 3D points (N, 3)."""
    c = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def plane_depth(u, v, n, d, K):
    """Depth at which each pixel's viewing ray hits the plane."""
    fx, fy, cx, cy = K
    rx, ry = (np.asarray(u, np.float64) - cx) / fx, (np.asarray(v, np.float64) - cy) / fy
    return -d / (n[0] * rx + n[1] * ry + n[2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `..\.venv\Scripts\python.exe -m pytest ..\tests\test_14_depth.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add 14-monocular-depth-3d/metrics.py tests/test_14_depth.py
git commit -m "Project 14: depth metrics and geometry helpers with tests"
```

---

### Task 2: NYU data access

**Files:**
- Create: `14-monocular-depth-3d/nyu.py`

- [ ] **Step 1: Implement `nyu.py`**

```python
"""NYU Depth V2 labeled subset (1,449 Kinect RGB-D images, 894 object classes): data access.

The .mat file is HDF5 with transposed arrays; everything returned here is (480, 640)."""

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

ROOT = data_dir("depth") / "nyu"
K = (518.8579, 519.4696, 325.5824, 253.7362)   # RGB camera intrinsics (official NYU toolbox)


class NYU:
    def __init__(self):
        self.f = h5py.File(ROOT / "nyu_depth_v2_labeled.mat", "r")
        refs = self.f["names"][0]
        self.names = ["".join(chr(c) for c in self.f[r][:].ravel()) for r in refs]

    def test_indices(self):
        """The official 654 test images, as 0-based indices."""
        return [int(i) - 1 for i in loadmat(ROOT / "splits.mat")["testNdxs"].ravel()]

    def rgb(self, i):
        return np.ascontiguousarray(self.f["images"][i].transpose(2, 1, 0))       # (480, 640, 3) uint8

    def depth(self, i):
        return np.ascontiguousarray(self.f["depths"][i].T)                        # hole-filled, meters

    def raw(self, i):
        return np.ascontiguousarray(self.f["rawDepths"][i].T)                     # sensor only, 0 = no data

    def labels(self, i):
        return np.ascontiguousarray(self.f["labels"][i].T)                        # 0 = unlabeled

    def class_id(self, name):
        return self.names.index(name) + 1
```

- [ ] **Step 2: Smoke-check it**

Run:
```bash
..\.venv\Scripts\python.exe -c "from nyu import NYU; d=NYU(); t=d.test_indices(); print(len(t), d.rgb(t[0]).shape, d.depth(t[0]).max(), d.class_id('mirror'))"
```
Expected: `654 (480, 640, 3) <a value between 2 and 10> 122`

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/nyu.py
git commit -m "Project 14: NYU Depth V2 data access"
```

---

### Task 3: Model registry

**Files:**
- Create: `14-monocular-depth-3d/models.py`

- [ ] **Step 1: Implement `models.py`**

```python
"""Depth models compared in this project, behind one interface."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import pipeline

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
```

- [ ] **Step 2: Smoke-check each model once**

```bash
..\.venv\Scripts\python.exe -c "from nyu import NYU; import models as M; d=NYU(); im=d.rgb(d.test_indices()[0]); [print(n, M.predict(M.load(n), im).shape) for n in M.MODELS]"
```
Expected: four lines ending in `(480, 640)` (first run downloads ~1 GB of weights into `cv-data/.cache`).

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/models.py
git commit -m "Project 14: depth model registry"
```

---

### Task 4: Benchmark with cached predictions

**Files:**
- Create: `14-monocular-depth-3d/benchmark.py`

- [ ] **Step 1: Implement `benchmark.py`**

```python
"""Run every model once on the 654 NYU test images, cache the predictions, report AbsRel and delta<1.25.

Relative models are aligned per image (scale + shift in disparity). The metric model is reported as is
(true meters) and with a per-image median scale. Resumable: predictions are written to a memory-mapped
.npy per model and a progress file records how many images are done.

Usage:
    python benchmark.py                 # all models, ~1 h on CPU
    python benchmark.py --models da2-small
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import models as M
from metrics import abs_rel, align_disparity, delta1, eval_mask
from nyu import NYU

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import output_dir  # noqa: E402

OUT = output_dir("depth")
PREDS = OUT / "preds"
DOCS = Path(__file__).resolve().parent / "docs"


def cached(name, n):
    """Memory-mapped (n, 480, 640) float16 predictions for one model, created on first use."""
    PREDS.mkdir(exist_ok=True)
    path = PREDS / f"{name}.npy"
    if not path.exists():
        np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(n, 480, 640)).flush()
    return np.load(path, mmap_mode="r+")


def progress(name):
    p = PREDS / f"{name}.done"
    return int(p.read_text()) if p.exists() else 0


def run_model(data, name, idx):
    preds, start = cached(name, len(idx)), progress(name)
    if start < len(idx):
        est, t0 = M.load(name), time.time()
        for k in range(start, len(idx)):
            preds[k] = M.predict(est, data.rgb(idx[k])).astype(np.float16)
            if (k + 1) % 20 == 0 or k + 1 == len(idx):
                preds.flush()
                (PREDS / f"{name}.done").write_text(str(k + 1))
                print(f"{name} {k + 1}/{len(idx)} {(time.time() - t0) / (k + 1 - start):.2f}s/img", flush=True)
    return preds


def score(data, name, preds, idx):
    kind = M.MODELS[name][1]
    rows = {"absrel": [], "delta1": []}
    extra = {"absrel_median_scaled": [], "delta1_median_scaled": []}
    for k, i in enumerate(idx):
        gt = data.depth(i)
        mask = eval_mask(gt)
        p = preds[k].astype(np.float32)
        d = align_disparity(p, gt, mask) if kind == "disparity" else p
        rows["absrel"].append(abs_rel(d, gt, mask))
        rows["delta1"].append(delta1(d, gt, mask))
        if kind == "metric":
            s = np.median(gt[mask]) / np.median(p[mask])
            extra["absrel_median_scaled"].append(abs_rel(p * s, gt, mask))
            extra["delta1_median_scaled"].append(delta1(p * s, gt, mask))
    out = {k: round(float(np.mean(v)), 4) for k, v in {**rows, **extra}.items() if v}
    np.save(OUT / f"per_image_absrel_{name}.npy", np.array(rows["absrel"], np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(M.MODELS))
    args = ap.parse_args()
    data = NYU()
    idx = data.test_indices()
    results_path = DOCS / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    bench = results.setdefault("benchmark", {})
    for name in args.models:
        preds = run_model(data, name, idx)
        bench[name] = score(data, name, preds, idx)
        print(name, bench[name], flush=True)
        results_path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry run on one model, 20 images**

Temporarily run with a small slice to check the pipeline end to end:
```bash
..\.venv\Scripts\python.exe -c "import benchmark as B, models as M; from nyu import NYU; d=NYU(); idx=d.test_indices()[:20]; p=B.run_model(d,'da2-small',idx); print(B.score(d,'da2-small',p,idx))"
```
Expected: AbsRel around 0.05–0.10 and delta1 above 0.9 (published Depth Anything V2 Small on NYU: AbsRel ≈ 0.05–0.06 after alignment). Then delete the 20-image cache: `del D:\cv-data\depth\outputs\preds\da2-small.*`.

- [ ] **Step 3: Full run, detached**

```powershell
Start-Process cmd.exe -ArgumentList "/c","cd /d D:\COMPUTER-VISION-PROJECT-LIBRARY\14-monocular-depth-3d && ..\.venv\Scripts\python.exe benchmark.py > D:\cv-data\depth\outputs\benchmark.log 2>&1 && echo done > D:\cv-data\depth\outputs\benchmark.done" -WindowStyle Hidden
```
Expected after ~1 h: `docs/results.json` has `benchmark` with 4 models.

- [ ] **Step 4: Commit**

```bash
git add 14-monocular-depth-3d/benchmark.py 14-monocular-depth-3d/docs/results.json
git commit -m "Project 14: NYU benchmark of three relative and one metric depth model"
```

---

### Task 5: Failure map by class, distance and region

**Files:**
- Create: `14-monocular-depth-3d/failures.py`

- [ ] **Step 1: Implement `failures.py`**

```python
"""Where do depth models fail? Error by object class, distance and image region on the NYU test set.

Only pixels the Kinect actually measured (raw depth > 0) are scored: on glass and mirrors the sensor
itself often fails or measures the reflection, and the hole-filled depth there is interpolation, not truth.
Sensor coverage per class is reported because it is a finding in itself.

Usage:
    python failures.py            # after benchmark.py, ~10 min
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

import models as M
from benchmark import DOCS, cached
from metrics import EIGEN_CROP, align_disparity, eval_mask
from nyu import NYU

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REFLECTIVE = ["mirror", "window", "glass", "television", "monitor", "picture", "whiteboard", "blinds", "curtain"]
ORDINARY = ["wall", "floor", "ceiling", "table", "door", "cabinet", "bed", "chair", "sofa"]
DIST_BINS = [(0, 2), (2, 4), (4, 10)]
BORDER = 40   # px band inside the Eigen crop


def main():
    data = NYU()
    idx = data.test_indices()
    classes = REFLECTIVE + ORDINARY
    ids = {c: data.class_id(c) for c in classes}
    results_path = DOCS / "results.json"
    results = json.loads(results_path.read_text())
    out = results.setdefault("failures", {})
    region = np.zeros((480, 640), bool)
    region[EIGEN_CROP] = True
    inner = np.zeros_like(region)
    inner[45 + BORDER:471 - BORDER, 41 + BORDER:601 - BORDER] = True
    border = region & ~inner
    for name in M.RELATIVE:
        preds = cached(name, len(idx))
        err_sum = {c: 0.0 for c in classes}
        err_n = {c: 0 for c in classes}
        labeled_n = {c: 0 for c in classes}
        dist = {b: [0.0, 0] for b in DIST_BINS}
        reg = {"border": [0.0, 0], "centre": [0.0, 0]}
        for k, i in enumerate(idx):
            gt, raw, lab = data.depth(i), data.raw(i), data.labels(i)
            mask = eval_mask(gt)
            pred = align_disparity(preds[k].astype(np.float32), gt, mask)
            ok = mask & (raw > 0)
            rel = np.abs(pred - gt) / np.maximum(gt, 1e-3)
            for c in classes:
                cm = (lab == ids[c]) & region
                labeled_n[c] += int(cm.sum())
                sel = cm & ok
                err_sum[c] += float(rel[sel].sum())
                err_n[c] += int(sel.sum())
            for lo, hi in DIST_BINS:
                sel = ok & (gt >= lo) & (gt < hi)
                dist[(lo, hi)][0] += float(rel[sel].sum())
                dist[(lo, hi)][1] += int(sel.sum())
            for rname, rm in (("border", border), ("centre", inner)):
                sel = ok & rm
                reg[rname][0] += float(rel[sel].sum())
                reg[rname][1] += int(sel.sum())
        out[name] = {
            "absrel_by_class": {c: round(err_sum[c] / err_n[c], 4) for c in classes if err_n[c] > 1000},
            "sensor_coverage_by_class": {c: round(err_n[c] / max(labeled_n[c], 1), 3) for c in classes},
            "pixels_by_class": {c: err_n[c] for c in classes},
            "absrel_by_distance_m": {f"{lo}-{hi}": round(s / max(n, 1), 4) for (lo, hi), (s, n) in dist.items()},
            "absrel_by_region": {r: round(s / max(n, 1), 4) for r, (s, n) in reg.items()},
        }
        print(name, json.dumps(out[name]["absrel_by_class"]), flush=True)
    results_path.write_text(json.dumps(results, indent=1))
    plot(out)


def plot(out):
    names = list(out)
    classes = [c for c in REFLECTIVE + ORDINARY if all(c in out[n]["absrel_by_class"] for n in names)]
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(13, 4.2))
    for j, n in enumerate(names):
        ax.bar(x + (j - 1) * 0.27, [out[n]["absrel_by_class"][c] for c in classes], 0.27, label=n)
    split = sum(c in REFLECTIVE for c in classes) - 0.5
    ax.axvline(split, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, [f"{c}\n{out[names[0]]['sensor_coverage_by_class'][c]:.0%} measured" for c in classes], fontsize=8)
    ax.set_ylabel("AbsRel (lower is better)")
    ax.set_title("Depth error by surface: reflective / transparent / pictures (left) vs ordinary surfaces (right)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "failure_by_class.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `..\.venv\Scripts\python.exe failures.py`
Expected: three printed dictionaries; `docs/failure_by_class.png` written. Sanity: wall/floor AbsRel in the 0.04–0.10 range.

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/failures.py 14-monocular-depth-3d/docs/results.json 14-monocular-depth-3d/docs/failure_by_class.png
git commit -m "Project 14: depth error by surface class, distance and region"
```

---

### Task 6: Poster illusion test (TDD for the geometry)

**Files:**
- Create: `14-monocular-depth-3d/poster_test.py`
- Modify: `tests/test_14_depth.py` (append)

- [ ] **Step 1: Append the failing test**

```python
def test_poster_quad_lies_inside_the_wall():
    poster = load_module("14", "poster_test.py")
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:480, 0:640]
    z = np.full((480, 640), 3.0)                       # fronto-parallel wall 3 m away
    wall = np.zeros((480, 640), bool)
    wall[100:400, 150:550] = True
    quad = poster.poster_quad(wall, z, K)
    assert quad is not None
    qx, qy = quad[:, 0].round().astype(int), quad[:, 1].round().astype(int)
    assert wall[qy.clip(0, 479), qx.clip(0, 639)].all()
```

- [ ] **Step 2: Run to verify failure**

Run: `..\.venv\Scripts\python.exe -m pytest ..\tests\test_14_depth.py -q -k poster`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `poster_test.py`**

```python
"""The poster test: an optical illusion with a known answer.

Take NYU test images with a large flat wall, paste a COCO photo onto the wall as a framed poster with the
correct perspective. The true depth is unchanged: still a flat plane. A model that "sees into" the poster
predicts depth that bulges in or out of the wall. Illusion score = mean relative deviation from the wall plane
inside the poster, compared with the same region before pasting.

Usage:
    python poster_test.py --n 100       # ~15 min for three models
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

import models as M
from benchmark import DOCS
from metrics import align_disparity, backproject, eval_mask, fit_plane, plane_depth
from nyu import K, NYU

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COCO = data_dir("clip-search") / "val2017"


def wall_plane(wall, depth, K):
    """Plane (n, d) through the wall pixels and its median residual (m)."""
    v, u = np.nonzero(wall)
    pts = backproject(u, v, depth[v, u], K)
    n, d = fit_plane(pts)
    return n, d, float(np.median(np.abs(pts @ n + d)))


def poster_quad(wall, depth, K, fill=0.7):
    """Image corners of a rectangle lying on the wall plane, centred on the point deepest inside the wall
    mask and small enough to stay on the wall. None if the wall is too small."""
    dist = cv2.distanceTransform(wall.astype(np.uint8), cv2.DIST_L2, 5)
    r = float(dist.max())
    if r < 40:
        return None
    vy, ux = np.unravel_index(int(dist.argmax()), dist.shape)
    n, d, _ = wall_plane(wall, depth, K)
    centre = backproject(ux, vy, depth[vy, ux], K)
    up = np.array([0.0, -1.0, 0.0])
    right = np.cross(up, n)
    right /= np.linalg.norm(right)
    upw = np.cross(n, right)
    upw /= np.linalg.norm(upw)
    fx, fy, cx, cy = K

    def project(p):
        return np.array([fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy])

    half = 2.0
    for _ in range(60):                                  # shrink until all corners are well inside the mask
        corners = [centre + sx * half * right + sy * half * 0.75 * upw for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        quad = np.array([project(c) for c in corners], np.float32)
        if np.all(np.linalg.norm(quad - [ux, vy], axis=1) <= fill * r):
            return quad
        half *= 0.92
    return None


def composite(rgb, quad, picture):
    """Warp `picture` into `quad` with a white frame; returns the new image and the poster mask."""
    h, w = picture.shape[:2]
    framed = cv2.copyMakeBorder(picture, h // 12, h // 12, h // 12, h // 12, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    fh, fw = framed.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), quad)
    warped = cv2.warpPerspective(framed, H, (rgb.shape[1], rgb.shape[0]))
    mask = cv2.warpPerspective(np.ones((fh, fw), np.uint8), H, (rgb.shape[1], rgb.shape[0])) > 0
    out = rgb.copy()
    out[mask] = warped[mask]
    return out, mask


def deviation(pred_disp, gt, poster, n, d):
    """Align on everything *except* the poster, then measure relative deviation from the wall plane inside it."""
    mask = eval_mask(gt) & ~poster
    depth = align_disparity(pred_disp, gt, mask)
    v, u = np.nonzero(poster)
    plane = plane_depth(u, v, n, d, K)
    return float(np.mean(np.abs(depth[v, u] - plane) / plane)), float(np.mean((depth[v, u] - plane) / plane))


def candidates(data, idx, n):
    wall_id, out = data.class_id("wall"), []
    for i in idx:
        lab, gt = data.labels(i), data.depth(i)
        wall = (lab == wall_id) & eval_mask(gt)
        if wall.mean() < 0.15:
            continue
        _, _, resid = wall_plane(wall, gt, K)
        quad = poster_quad(wall, gt, K) if resid < 0.02 else None
        if quad is not None:
            out.append((i, wall, quad))
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    data = NYU()
    cands = candidates(data, data.test_indices(), args.n)
    pics = sorted(COCO.glob("*.jpg"))
    rng = np.random.default_rng(0)
    pics = [pics[j] for j in rng.choice(len(pics), len(cands), replace=False)]
    print(f"{len(cands)} flat-wall images", flush=True)
    res, examples = {}, []
    for name in M.RELATIVE:
        est = M.load(name)
        before, after, signed = [], [], []
        for (i, wall, quad), pic in zip(cands, pics):
            rgb, gt = data.rgb(i), data.depth(i)
            n, d, _ = wall_plane(wall, gt, K)
            comp, poster = composite(rgb, quad, cv2.cvtColor(cv2.imread(str(pic)), cv2.COLOR_BGR2RGB))
            b, _ = deviation(M.predict(est, rgb), gt, poster, n, d)
            pa = M.predict(est, comp)
            a, s = deviation(pa, gt, poster, n, d)
            before.append(b)
            after.append(a)
            signed.append(s)
            if name == "da2-small":
                examples.append((a - b, comp, pa, poster))
        res[name] = {"images": len(cands), "deviation_before": round(float(np.mean(before)), 4),
                     "deviation_with_poster": round(float(np.mean(after)), 4),
                     "illusion_score": round(float(np.mean(np.array(after) - np.array(before))), 4),
                     "share_worse_with_poster": round(float(np.mean(np.array(after) > np.array(before))), 3),
                     "mean_signed_deviation": round(float(np.mean(signed)), 4)}
        print(name, res[name], flush=True)
    results_path = DOCS / "results.json"
    results = json.loads(results_path.read_text())
    results["poster_test"] = res
    results_path.write_text(json.dumps(results, indent=1))
    examples.sort(key=lambda e: -e[0])
    np.save(Path(__file__).resolve().parent / "docs" / "_poster_best.npy", examples[0][1])  # used by see_vs_infer.py
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.6))
    for col, (_, comp, pa, poster) in enumerate(examples[:4]):
        axes[0, col].imshow(comp)
        axes[0, col].contour(poster, levels=[0.5], colors="orange", linewidths=0.8)
        axes[1, col].imshow(pa, cmap="inferno")
        for ax in axes[:, col]:
            ax.axis("off")
    axes[0, 0].set_title("photo with a flat poster (orange)", loc="left")
    axes[1, 0].set_title("predicted depth (bright = near)", loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / "poster_test.jpg", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

Note: `_poster_best.npy` is a scratch file for Task 8; add `14-monocular-depth-3d/docs/_*.npy` to `.gitignore` in Step 5.

- [ ] **Step 4: Run the test, then the experiment**

Run: `..\.venv\Scripts\python.exe -m pytest ..\tests\test_14_depth.py -q`
Expected: `6 passed`.
Run: `..\.venv\Scripts\python.exe poster_test.py --n 100`
Expected: three result lines; `docs/poster_test.jpg`. If fewer than 30 flat walls are found, lower the 0.15 wall share to 0.10.

- [ ] **Step 5: Commit**

```bash
printf '14-monocular-depth-3d/docs/_*.npy\n' >> .gitignore
git add .gitignore 14-monocular-depth-3d/poster_test.py tests/test_14_depth.py 14-monocular-depth-3d/docs/results.json 14-monocular-depth-3d/docs/poster_test.jpg
git commit -m "Project 14: poster illusion test with a known flat answer"
```

---

### Task 7: Real illusion gallery

**Files:**
- Create: `14-monocular-depth-3d/illusions.py`

- [ ] **Step 1: Implement `illusions.py`**

```python
"""Depth models on real optical illusions (Wikimedia Commons, open licenses; credits written to docs/).

Usage:
    python illusions.py
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

import models as M
from benchmark import DOCS
from depth3d import backproject, render_view
from nyu import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = ROOT.parent / "illusions"
UA = {"User-Agent": "cv-portfolio-research/1.0 (github.com/yandouziyassine)"}
TITLES = {
    "ames_room": ("File:Ames room forced perspective.jpg", "Two people the same height look giant and tiny: the room is not a box"),
    "street_art": ("File:3D street painting.jpg", "A flat painting on the ground that looks like a hole"),
    "den_helder": ("File:Den Helder - Julianaplein, 3D street art 2020.jpg", "3D street art seen from its intended viewpoint"),
    "colonnade": ("File:Colonnade, trompe l'oeil mural by John Pugh.jpg", "A flat wall painted as a colonnade"),
    "cinema_vox": ("File:Film crew in trompe l'oeil mural, Cinéma Vox, Chamonix-Mont-Blanc, 2007.jpg", "Painted film crew on a flat facade"),
    "mirrors": ("File:Man photographing himself in hotel bathroom with mirrors all around to generate illusion (cropped twice).jpg", "Facing mirrors: an endless corridor that does not exist"),
}


def fetch():
    DIR.mkdir(parents=True, exist_ok=True)
    q = {"action": "query", "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": 1280,
         "titles": "|".join(t for t, _ in TITLES.values()), "format": "json"}
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(q)
    pages = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))["query"]["pages"].values()
    by_title = {p["title"]: p["imageinfo"][0] for p in pages}
    credits = []
    for key, (title, caption) in TITLES.items():
        info = by_title[title]
        path = DIR / f"{key}.jpg"
        if not path.exists():
            req = urllib.request.Request(info["thumburl"], headers=UA)
            path.write_bytes(urllib.request.urlopen(req).read())
        md = info["extmetadata"]
        artist = md.get("Artist", {}).get("value", "unknown")
        credits.append(f"- **{key}**: [{title}]({info['descriptionurl']}), {md.get('LicenseShortName', {}).get('value')}, "
                       f"by {artist}")
    (DOCS / "illusion_credits.md").write_text("# Illusion photo credits\n\n" + "\n".join(credits) + "\n", encoding="utf-8")


def main():
    fetch()
    est = M.load("da2-small")
    fig, axes = plt.subplots(len(TITLES), 3, figsize=(12, 3.3 * len(TITLES)))
    for row, (key, (_, caption)) in enumerate(TITLES.items()):
        img = Image.open(DIR / f"{key}.jpg").convert("RGB")
        img.thumbnail((960, 960))
        disp = M.predict(est, np.asarray(img))
        pts, cols, f = backproject(img, disp)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(caption, fontsize=9, loc="left")
        axes[row, 1].imshow(disp, cmap="inferno")
        axes[row, 1].set_title("what the model infers (bright = near)", fontsize=9, loc="left")
        axes[row, 2].imshow(render_view(pts, cols, f, img.size, 35))
        axes[row, 2].set_title("its 3D, seen from the side", fontsize=9, loc="left")
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "illusions.jpg", dpi=75)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `..\.venv\Scripts\python.exe illusions.py`
Expected: 6 images in `D:\cv-data\depth\illusions\`, `docs/illusions.jpg`, `docs/illusion_credits.md` with 6 lines. If a title fails (renamed file), search Commons again and replace it in `TITLES`.

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/illusions.py 14-monocular-depth-3d/docs/illusions.jpg 14-monocular-depth-3d/docs/illusion_credits.md
git commit -m "Project 14: depth models on real optical illusions"
```

---

### Task 8: Part C, what you see vs what the computer sees

**Files:**
- Create: `14-monocular-depth-3d/see_vs_infer.py`

- [ ] **Step 1: Implement `see_vs_infer.py`**

```python
"""Part C figure: what you see / what the computer receives / what it infers / the truth / its 3D.

Rows: the NYU test image the model gets most right, the image with the most mirror pixels, the poster
composite with the strongest illusion, and the Ames room photo.

Usage:
    python see_vs_infer.py          # after benchmark.py, poster_test.py and illusions.py
"""

from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

import models as M
from benchmark import DOCS, OUT
from depth3d import backproject as bp_render
from depth3d import render_view, save_sweep_gif
from illusions import DIR as ILLUSIONS
from metrics import align_disparity, eval_mask
from nyu import NYU

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pixel_panel(ax, rgb, y, x, size=8):
    """The numbers a computer actually receives: an 8x8 block of brightness values."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))[y:y + size, x:x + size]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for (r, c), val in np.ndenumerate(gray):
        ax.text(c, r, str(val), ha="center", va="center", fontsize=6, color="red" if val > 127 else "yellow")
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    data = NYU()
    idx = data.test_indices()
    absrel = np.load(OUT / "per_image_absrel_da2-small.npy")
    mirror = data.class_id("mirror")
    best = idx[int(np.argmin(absrel))]
    most_mirror = max(idx, key=lambda i: int((data.labels(i) == mirror).sum()))
    est = M.load("da2-small")
    rows = [("A scene the model gets right", data.rgb(best), data.depth(best)),
            ("A mirror: the room inside it is not real", data.rgb(most_mirror), data.depth(most_mirror)),
            ("A flat poster on a wall (Part A test)", np.load(DOCS / "_poster_best.npy"), None),
            ("The Ames room illusion", np.asarray(Image.open(ILLUSIONS / "ames_room.jpg").convert("RGB")), None)]
    titles = ["What you see", "What the computer receives", "What it infers (bright = near)",
              "The truth (Kinect) / error", "Its 3D, seen from the side"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(19, 3.6 * len(rows)))
    for r, (label, rgb, gt) in enumerate(rows):
        disp = M.predict(est, rgb)
        h, w = rgb.shape[:2]
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(label, fontsize=10)
        pixel_panel(axes[r, 1], rgb, h // 2, w // 2)
        axes[r, 2].imshow(disp, cmap="inferno")
        if gt is not None:
            mask = eval_mask(gt)
            depth = align_disparity(disp, gt, mask)
            err = np.where(mask, np.abs(depth - gt) / np.maximum(gt, 1e-3), np.nan)
            axes[r, 3].imshow(gt, cmap="inferno_r")
            axes[r, 3].imshow(err, cmap="Reds", vmin=0, vmax=0.3, alpha=0.6)
        else:
            axes[r, 3].text(0.5, 0.5, "no sensor data:\na flat poster / a trick room", ha="center", va="center")
        img = Image.fromarray(rgb)
        pts, cols, f = bp_render(img, disp)
        axes[r, 4].imshow(render_view(pts, cols, f, img.size, 40))
        if r == 0:
            save_sweep_gif(pts, cols, f, img.size, DOCS / "see_vs_infer.gif", width=360)
    for c, t in enumerate(titles):
        axes[0, c].set_title(t, fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(DOCS / "see_vs_infer.jpg", dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and look at it**

Run: `..\.venv\Scripts\python.exe see_vs_infer.py`
Expected: `docs/see_vs_infer.jpg` (4 rows × 5 panels) and `docs/see_vs_infer.gif`. Open the JPG and check every panel is filled and readable.

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/see_vs_infer.py 14-monocular-depth-3d/docs/see_vs_infer.jpg 14-monocular-depth-3d/docs/see_vs_infer.gif
git commit -m "Project 14: what you see vs what the computer sees"
```

---

### Task 9: Measuring from one photo

**Files:**
- Create: `14-monocular-depth-3d/measure.py`, `14-monocular-depth-3d/measure_eval.py`
- Modify: `tests/test_14_depth.py` (append)

- [ ] **Step 1: Append the failing test**

```python
def test_intrinsics_from_fov_and_from_35mm_focal():
    meas = load_module("14", "measure.py")
    fx, fy, cx, cy = meas.intrinsics_from_fov(640, 480, 90.0)
    assert fx == pytest.approx(320.0) and (cx, cy) == (320.0, 240.0)
    fx, *_ = meas.intrinsics_from_35mm(4000, 3000, 36.0)          # 36 mm equivalent = sensor width
    assert fx == pytest.approx(4000.0)
```

- [ ] **Step 2: Implement `measure.py`**

```python
"""Measure the real world from one photo: metric depth + camera intrinsics + two pixels -> meters."""

import numpy as np
from PIL import Image

import models as M
from metrics import distance

DEFAULT_FOV = 60.0


def intrinsics_from_fov(w, h, fov_deg=DEFAULT_FOV):
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    return (float(f), float(f), w / 2, h / 2)


def intrinsics_from_35mm(w, h, focal_35mm):
    """EXIF FocalLengthIn35mmFilm: a full-frame sensor is 36 mm wide."""
    f = focal_35mm / 36.0 * w
    return (float(f), float(f), w / 2, h / 2)


def intrinsics(img):
    """From EXIF when the photo has it, else the default field of view. Returns (K, source)."""
    focal = img.getexif().get_ifd(0x8769).get(0xA405)            # FocalLengthIn35mmFilm
    if focal:
        return intrinsics_from_35mm(*img.size, float(focal)), f"EXIF ({focal} mm equivalent)"
    return intrinsics_from_fov(*img.size), f"assumed {DEFAULT_FOV:.0f}° field of view"


class Measurer:
    def __init__(self):
        self.est = M.load("da2-metric-indoor")

    def depth(self, img):
        return M.predict(self.est, np.asarray(img.convert("RGB")))

    def measure(self, img: Image.Image, p1, p2, depth=None, reference=None):
        """Distance in meters between pixels p1 and p2. reference = (q1, q2, true_meters) rescales the depth
        so a known length in the photo comes out exact."""
        depth = self.depth(img) if depth is None else depth
        K, source = intrinsics(img)
        scale = 1.0
        if reference:
            q1, q2, true_m = reference
            scale = true_m / distance(q1, q2, depth, K)
        return distance(p1, p2, depth * scale, K), source
```

- [ ] **Step 3: Implement `measure_eval.py`**

```python
"""How accurate is "measure from one photo"? Random pixel pairs on the NYU test set, true distance from the
Kinect depth, predicted distance from the metric model's cached depth, with the true camera intrinsics.

Usage:
    python measure_eval.py          # after benchmark.py (uses the cached da2-metric-indoor predictions)
"""

import json

import matplotlib
import numpy as np

from benchmark import DOCS, cached
from metrics import distance, eval_mask
from nyu import K, NYU

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BINS = [(0.3, 1.0), (1.0, 2.0), (2.0, 3.0)]


def pairs(gt, rng, n=40):
    v, u = np.nonzero(eval_mask(gt))
    out = []
    while len(out) < n:
        a, b = rng.integers(len(u), size=2)
        p1, p2 = (u[a], v[a]), (u[b], v[b])
        d = distance(p1, p2, gt, K)
        if BINS[0][0] <= d < BINS[-1][1]:
            out.append((p1, p2, d))
    return out


def main():
    data = NYU()
    idx = data.test_indices()
    preds = cached("da2-metric-indoor", len(idx))
    rng = np.random.default_rng(0)
    plain, calibrated, truth = [], [], []
    for k, i in enumerate(idx):
        gt = data.depth(i)
        pred = preds[k].astype(np.float32)
        ps = pairs(gt, rng)
        q1, q2, ref = ps[0]                                          # the "known object" in this photo
        scale = ref / distance(q1, q2, pred, K)
        for p1, p2, d in ps[1:]:
            pd = distance(p1, p2, pred, K)
            plain.append(abs(pd - d) / d)
            calibrated.append(abs(pd * scale - d) / d)
            truth.append(d)
    plain, calibrated, truth = map(np.array, (plain, calibrated, truth))
    res = {"pairs": int(len(truth)), "images": len(idx), "model": "da2-metric-indoor", "intrinsics": "true NYU"}
    for name, err in (("metric_model_only", plain), ("with_one_known_length", calibrated)):
        res[name] = {"median_rel_error": round(float(np.median(err)), 3),
                     "p90_rel_error": round(float(np.percentile(err, 90)), 3),
                     "by_distance_m": {f"{lo}-{hi}": round(float(np.median(err[(truth >= lo) & (truth < hi)])), 3)
                                       for lo, hi in BINS}}
    print(json.dumps(res, indent=1))
    results_path = DOCS / "results.json"
    results = json.loads(results_path.read_text())
    results["measure"] = res
    results_path.write_text(json.dumps(results, indent=1))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, err in (("metric model only", plain), ("+ one known length in the photo", calibrated)):
        s = np.sort(err)
        ax.plot(s * 100, np.arange(1, len(s) + 1) / len(s), label=name)
    ax.set_xlim(0, 60)
    ax.set_xlabel("relative error on the measured distance (%)")
    ax.set_ylabel("share of measurements")
    ax.set_title(f"Measuring from one photo: {len(truth)} distances on NYU test")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "measure_error.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and the evaluation**

Run: `..\.venv\Scripts\python.exe -m pytest ..\tests\test_14_depth.py -q` → `7 passed`.
Run: `..\.venv\Scripts\python.exe measure_eval.py` → JSON with both settings; `docs/measure_error.png`.

- [ ] **Step 5: Commit**

```bash
git add 14-monocular-depth-3d/measure.py 14-monocular-depth-3d/measure_eval.py tests/test_14_depth.py 14-monocular-depth-3d/docs/results.json 14-monocular-depth-3d/docs/measure_error.png
git commit -m "Project 14: metric measurement from one photo and its measured accuracy"
```

---

### Task 10: Measuring app

**Files:**
- Create: `14-monocular-depth-3d/measure_app.py`

- [ ] **Step 1: Implement `measure_app.py`**

```python
"""Click two points on a photo, get the distance in meters (with the error measured on NYU).

Usage:
    python measure_app.py           # opens http://127.0.0.1:7860
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from measure import Measurer

RESULTS = json.loads((Path(__file__).resolve().parent / "docs" / "results.json").read_text())["measure"]
P90 = RESULTS["metric_model_only"]["p90_rel_error"]
P90_REF = RESULTS["with_one_known_length"]["p90_rel_error"]
measurer = Measurer()


def on_upload(img):
    return img, [], None, "Click two points."


def on_click(state_img, points, depth, evt: gr.SelectData):
    if state_img is None:
        return None, points, depth, "Upload a photo first."
    img = Image.fromarray(state_img)
    if depth is None:
        depth = measurer.depth(img)
    points = (points + [tuple(evt.index)])[-2:]
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas)
    for x, y in points:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 91, 31), width=3)
    if len(points) < 2:
        return np.asarray(canvas), points, depth, "Click a second point."
    draw.line([points[0], points[1]], fill=(255, 91, 31), width=3)
    d, source = measurer.measure(img, points[0], points[1], depth=depth)
    msg = (f"**{d:.2f} m** (±{P90 * 100:.0f} % for 9 measurements out of 10 on NYU indoor scenes).  \n"
           f"Camera: {source}. Indoor model: outdoor scenes will be less accurate.")
    return np.asarray(canvas), points, depth, msg


with gr.Blocks(title="Measure from one photo") as demo:
    gr.Markdown("# Measure from one photo\nUpload an indoor photo, click two points. Metric depth from Depth Anything V2 "
                f"(indoor); with one known length in the photo the 90th-percentile error drops to ±{P90_REF * 100:.0f} %.")
    img_state, pts_state, depth_state = gr.State(), gr.State([]), gr.State()
    with gr.Row():
        inp = gr.Image(type="numpy", label="Photo (click two points)")
        out = gr.Markdown()
    inp.upload(on_upload, inp, [img_state, pts_state, depth_state, out])
    inp.select(on_click, [img_state, pts_state, depth_state], [inp, pts_state, depth_state, out])

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Manual check**

Run: `..\.venv\Scripts\python.exe measure_app.py`, open the URL, upload `D:\cv-data\clip-search\val2017\000000000139.jpg` (a living room), click the two ends of the TV. Expected: a distance around 1 m and the uncertainty line.

- [ ] **Step 3: Commit**

```bash
git add 14-monocular-depth-3d/measure_app.py
git commit -m "Project 14: click-to-measure Gradio app"
```

---

### Task 11: Write-up and propagation

**Files:**
- Modify: `14-monocular-depth-3d/README.md`, `14-monocular-depth-3d/requirements.txt`, `README.md` (root row 14), `THEORY.md` (chapter 45), `DATA_LICENSES.md` (NYU row status, illusion photos, model licenses), `PROGRESS.md`
- Modify (website repo `D:\yandouziyassine.github.io`): `data/projects.js` card 14 (depth 3, new image `images/cv-14.jpg` from `see_vs_infer.jpg`), then copy `data/projects.js` to `D:\hf-spaces\cv-library-hub\projects.js` and upload it.

- [ ] **Step 1: README** — structure: title; one-paragraph pitch ("computers have trouble seeing in 3D: here is exactly where"); **What you see vs what the computer sees** (`see_vs_infer.jpg`); **Part A** benchmark table (4 models: AbsRel, δ1, ms/img), failure-by-class figure + table with sensor coverage and the mirror caveat, distance/region numbers, poster test table + figure, illusion gallery + credits link; **Part B** measurement table (median/p90, with/without one known length, by distance) + figure + app usage; How it works (alignment, plane fit, homography poster, metric back-projection); Data & licenses; Run it (every script in order with times); What I learned (from the numbers, not guesses); Next steps; Tech stack. Every number must come from `docs/results.json`.
- [ ] **Step 2: requirements.txt** — add `h5py`, `scipy`, `opencv-python`, `gradio`.
- [ ] **Step 3: Propagate** — root README row 14 (★ → ★★★, headline metrics), THEORY chapter 45 one paragraph with the measured findings (keep heading text unchanged so anchors stay valid), DATA_LICENSES (NYU: research use, local only; illusion photos with licenses; DA-V2 Base CC BY-NC 4.0), PROGRESS entry, website card + HF hub sync.
- [ ] **Step 4: Verify** — `..\.venv\Scripts\python.exe -m ruff check ..` and `..\.venv\Scripts\python.exe -m pytest .. -q` from the project folder: all pass.
- [ ] **Step 5: Commit and push** both repos (library: `git -c credential.helper= -c "credential.helper=!gh auth git-credential" push origin main`; website the same), upload the hub `projects.js`.

---

## Self-review notes

- Spec coverage: A1 → Task 4; A2 → Task 5; A3 → Task 6; A4 → Task 7; B1–B2 → Task 9 `measure.py`; B3–B4 → Task 9 `measure_eval.py`; B5 → Task 10; C → Task 8; deliverables/tests/propagation → Tasks 1, 6, 9, 11.
- Names used across tasks: `cached`, `OUT`, `DOCS` (benchmark.py); `align_disparity`, `eval_mask`, `backproject`, `distance`, `fit_plane`, `plane_depth`, `EIGEN_CROP` (metrics.py); `NYU`, `K`, `ROOT` (nyu.py); `MODELS`, `RELATIVE`, `load`, `predict` (models.py); `poster_quad` (poster_test.py); `DIR` (illusions.py) — consistent.
- Known judgment calls: per-class errors use raw-valid pixels; the illusion score aligns on pixels outside the poster so the poster cannot bend the alignment.

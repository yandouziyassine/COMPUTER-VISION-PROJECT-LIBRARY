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
        (PREDS / f"{name}.seconds_per_image").write_text(f"{(time.time() - t0) / (len(idx) - start):.3f}")
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
    spi = PREDS / f"{name}.seconds_per_image"
    if spi.exists():
        out["seconds_per_image_cpu"] = round(float(spi.read_text()), 2)
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

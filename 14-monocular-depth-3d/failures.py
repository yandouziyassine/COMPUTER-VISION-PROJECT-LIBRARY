"""Where do depth models fail? Error by object class, distance and image region on the NYU test set.

Only pixels the Kinect actually measured (raw depth > 0) are scored: on glass and mirrors the sensor
itself often fails or measures the reflection, and the hole-filled depth there is interpolation, not truth.
Sensor coverage per class is reported because it is a finding in itself.

Usage:
    python failures.py            # after benchmark.py, ~10 min
"""

import json

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
BORDER = 40  # px band inside the Eigen crop


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
    inner[45 + BORDER : 471 - BORDER, 41 + BORDER : 601 - BORDER] = True
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
    cov = out[names[0]]["sensor_coverage_by_class"]
    ax.set_xticks(x, [f"{c} ({cov[c]:.0%})" for c in classes], fontsize=8, rotation=35, ha="right")
    ax.set_xlabel("surface (share of its pixels the Kinect could measure)")
    ax.set_ylabel("AbsRel (lower is better)")
    ax.set_title("Depth error by surface: reflective / transparent / pictures (left) vs ordinary surfaces (right)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "failure_by_class.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if "--plot-only" in sys.argv:  # redraw the figure from docs/results.json
        plot(json.loads((DOCS / "results.json").read_text())["failures"])
    else:
        main()

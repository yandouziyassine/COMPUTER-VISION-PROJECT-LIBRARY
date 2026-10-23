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
        q1, q2, ref = ps[0]  # the "known object" in this photo
        scale = ref / distance(q1, q2, pred, K)
        for p1, p2, d in ps[1:]:
            pd = distance(p1, p2, pred, K)
            plain.append(abs(pd - d) / d)
            calibrated.append(abs(pd * scale - d) / d)
            truth.append(d)
    plain, calibrated, truth = map(np.array, (plain, calibrated, truth))
    res = {"pairs": int(len(truth)), "images": len(idx), "model": "da2-metric-indoor", "intrinsics": "true NYU"}
    for name, err in (("metric_model_only", plain), ("with_one_known_length", calibrated)):
        res[name] = {
            "median_rel_error": round(float(np.median(err)), 3),
            "p90_rel_error": round(float(np.percentile(err, 90)), 3),
            "by_distance_m": {
                f"{lo}-{hi}": round(float(np.median(err[(truth >= lo) & (truth < hi)])), 3) for lo, hi in BINS
            },
        }
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

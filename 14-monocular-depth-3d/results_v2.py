"""Aggregate the v2 Kaggle outputs into docs/results_v2.json + docs/results_v2.png, and apply the reproduction gate.

Inputs: a folder with scores/<model>__<dataset>.json (per-image metrics written on Kaggle by kaggle_infer_v2.py
using protocol.py). Confidence intervals are recomputed here (bootstrap over images, protocol.yaml).
--recheck-nyu re-scores the downloaded NYU prediction cache locally against the local .mat, to confirm that the
Kaggle scores and the local evaluation agree.

Usage:
    python results_v2.py --runs ../../work/kaggle14/out [--recheck-nyu]
"""

import argparse
import json
from pathlib import Path

import numpy as np

import models_v2 as M
import protocol as P

DOCS = Path(__file__).resolve().parent / "docs"
DATASETS = ["nyu", "kitti", "eth3d", "diode", "ibims"]
GATE_TOL = 0.003

# One published number per new model, with the alignment that paper used (reproduction gate, protocol W1).
PUBLISHED = {
    "da2-large": ("nyu", "lsq_disparity", 0.045, "Depth Anything V2 paper, Table 2 (ViT-L, NYU-D)"),
    "marigold-lcm": ("nyu", "lsq_depth", 0.058, "Marigold journal paper (arXiv 2505.09358), Table: v1.0 LCM, 10 x 1"),
    "depth-pro": ("nyu", "none", 0.107, "MoGe-2 paper (arXiv 2507.02546), Table B.4, metric depth w/o GT intrinsics"),
    "moge2-large": (
        "nyu",
        "none",
        0.0733,
        "MoGe-2 paper (arXiv 2507.02546), Table B.4, metric depth w/o GT intrinsics",
    ),
}


def load_runs(folder):
    runs = {}
    for f in sorted(Path(folder).rglob("scores/*.json")):
        r = json.loads(f.read_text())
        runs[(r["model"], r["dataset"])] = r
    return runs


def recheck_nyu(folder, proto):
    """Re-score cached NYU predictions with the local .mat; return max |local - kaggle| AbsRel per model."""
    from nyu import NYU

    data = NYU()
    ids = {f"nyu_{i:04d}": i for i in data.test_indices()}
    gaps = {}
    for model in M.MODELS:
        files = sorted(Path(folder).rglob(f"preds/{model}__nyu__*.npz"))
        if not files:
            continue
        kag = json.loads(next(Path(folder).rglob(f"scores/{model}__nyu.json")).read_text())["per_image"]
        worst = 0.0
        for f in files:
            z = np.load(f)
            for k in z.files[:20]:  # 20 images per chunk is enough to catch a mismatch
                gt = data.raw(ids[k]).astype(np.float32)
                s = P.score_image(z[k].astype(np.float32), gt, proto["datasets"]["nyu"], M.kind(model), proto)
                a = proto["alignment"][M.kind(model)][0]
                worst = max(worst, abs(s[a]["absrel"] - kag[k][a]["absrel"]))
        gaps[model] = worst
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--recheck-nyu", action="store_true")
    a = ap.parse_args()
    proto = P.load_protocol()
    runs = load_runs(a.runs)
    table = {}
    for (model, ds), r in runs.items():
        s = P.summarize(list(r["per_image"].values()), proto)
        s["seconds_per_image_t4"] = r["seconds_per_image_t4"]
        s["primary_alignment"] = proto["alignment"][M.kind(model)][0]
        table.setdefault(model, {})[ds] = s
    gate = {}
    for model, (ds, align, pub, src) in PUBLISHED.items():
        got = table.get(model, {}).get(ds, {}).get(align, {}).get("absrel")
        gate[model] = {
            "dataset": ds,
            "alignment": align,
            "published_absrel": pub,
            "source": src,
            "ours_absrel": got[0] if got else None,
            "gap": round(got[0] - pub, 4) if got else None,
            "passed": bool(got and abs(got[0] - pub) <= GATE_TOL),
        }
    out = {
        "protocol_version": proto["version"],
        "bootstrap": proto["bootstrap"],
        "note": "values are [mean, 95% CI low, 95% CI high] over images; gate tolerance +-0.003 AbsRel",
        "gate": gate,
        "results": table,
    }
    if a.recheck_nyu:
        out["nyu_local_vs_kaggle_max_abs_diff"] = recheck_nyu(a.runs, proto)
    (DOCS / "results_v2.json").write_text(json.dumps(out, indent=1))
    figure(table, gate)
    for m, g in gate.items():
        print("gate", m, g["ours_absrel"], "vs", g["published_absrel"], "PASS" if g["passed"] else "FAIL")


def figure(table, gate):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in M.MODELS if m in table]
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.45 * len(models) + 2), sharey=True)
    for ax, metric, label in zip(axes, ["absrel", "delta1"], ["AbsRel (lower is better)", "δ1 (higher is better)"]):
        for j, ds in enumerate(DATASETS):
            ys, xs, lo, hi = [], [], [], []
            for i, m in enumerate(models):
                s = table[m].get(ds)
                if not s:
                    continue
                v = s[s["primary_alignment"]][metric]
                ys.append(i + (j - 2) * 0.15)
                xs.append(v[0])
                lo.append(v[0] - v[1])
                hi.append(v[2] - v[0])
            ax.errorbar(xs, ys, xerr=[lo, hi], fmt="o", ms=4, capsize=2, label=ds)
        ax.set_title(label)
        ax.grid(axis="x", alpha=0.3)
    names = [m + ("" if m not in gate or gate[m]["passed"] else "  (gate failed)") for m in models]
    axes[0].set_yticks(range(len(models)), names)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle("v2: 8 models x 5 datasets, headline alignment per model kind, 95 % bootstrap CIs")
    fig.tight_layout()
    fig.savefig(DOCS / "results_v2.png", dpi=110)


if __name__ == "__main__":
    main()

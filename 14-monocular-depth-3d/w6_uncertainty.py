"""W6 · Does the model know when it's wrong?

Two uncertainty signals per pixel, scored on NYU test images (raw sensor depth, Eigen crop):
  disagreement  spread (std of log depth) across the relative models after each is aligned to the sensor;
                the reference prediction is DA-V2 Large
  ensemble      Marigold LCM's own spread over 10 random seeds (diffusers `output_uncertainty`), reference = Marigold
Sparsification: remove the most uncertain pixels first and track AbsRel of the rest; the oracle removes the truly
worst pixels first. AUSE = area between the two curves (0 = uncertainty ranks errors perfectly). Also AbsRel at
90 % coverage, and the uncertainty on mirrors / glass compared with other surfaces.

Usage:
    python w6_uncertainty.py --preds <folder with preds/*__nyu__*.npz> [--marigold <w6_marigold folder>]
    (Kaggle) python kaggle_push_v2.py --name depth-w6-marigold --script w6_uncertainty.py --commit <sha>
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parent / "docs" if "__file__" in globals() else Path(".")
JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
RELATIVE = ["da2-small", "da2-base", "da2-large", "dpt-hybrid", "marigold-lcm"]
FRACTIONS = np.linspace(0, 0.95, 20)


def sparsification(err, unc, fractions=FRACTIONS):
    """Mean error of the pixels kept after removing the `f` most uncertain ones, for each f."""
    order = np.argsort(-unc, kind="stable")
    e = err[order]
    n = len(e)
    return np.array([e[int(round(f * n)) :].mean() for f in fractions])


def ause(err, unc, fractions=FRACTIONS):
    """Area between the uncertainty-ranked and the oracle sparsification curves (trapezoid over fractions)."""
    gap = sparsification(err, unc, fractions) - sparsification(err, err, fractions)
    return float(np.sum((gap[1:] + gap[:-1]) / 2 * np.diff(fractions)))


def kaggle_marigold():
    """Marigold LCM, 10 seeds, with its ensemble uncertainty, on every 3rd NYU test image."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "diffusers==0.34.0", "accelerate"], check=True)
    src = Path("/tmp/repo")
    if not src.exists():
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/yandouziyassine/COMPUTER-VISION-PROJECT-LIBRARY.git", str(src)],
            check=True,
        )
    subprocess.run(["git", "-C", str(src), "checkout", "-q", JOB["commit"]], check=True)
    os.environ["CV_DATA"] = "/tmp/cv-data"
    sys.path[:0] = [str(src / "14-monocular-depth-3d"), str(src)]
    import diffusers
    import torch
    from PIL import Image

    import datasets_v2 as D
    import models_v2 as M

    root = Path("/tmp/cv-data/depth/nyu")
    D.fetch_nyu(root)
    from nyu import NYU

    data = NYU()
    _, repo, rev, _, _ = M.MODELS["marigold-lcm"]
    pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
        repo, revision=M._full_revision(repo, rev), variant="fp16", torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    out = {}
    for i in data.test_indices()[::3]:
        rgb = data.rgb(i)
        r = pipe(
            Image.fromarray(rgb),
            num_inference_steps=M.MARIGOLD_STEPS,
            ensemble_size=M.MARIGOLD_ENSEMBLE,
            processing_resolution=M.MARIGOLD_RES,
            match_input_resolution=True,
            batch_size=M.MARIGOLD_ENSEMBLE,
            output_uncertainty=True,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        out[f"nyu_{i:04d}_pred"] = np.asarray(r.prediction).reshape(480, 640).astype(np.float16)
        out[f"nyu_{i:04d}_unc"] = np.asarray(r.uncertainty).reshape(480, 640).astype(np.float16)
    np.savez_compressed("/kaggle/working/w6_marigold.npz", **out)


def analyze(preds_dir, marigold_dir):
    import models_v2 as M
    import protocol as P
    from nyu import NYU

    data = NYU()
    proto = P.load_protocol()
    ds = proto["datasets"]["nyu"]
    preds = {m: {} for m in RELATIVE}
    for m in RELATIVE:
        for f in sorted(Path(preds_dir).rglob(f"preds/{m}__nyu__*.npz")):
            z = np.load(f)
            preds[m].update({k: z[k] for k in z.files})
    mari = None
    if marigold_dir:
        mari = np.load(next(Path(marigold_dir).rglob("w6_marigold.npz")))
    special = {
        "mirror": [data.class_id("mirror")],
        "glass": [data.class_id(n) for n in ("window", "glass")],
    }
    acc = {"disagreement": {"err": [], "unc": [], "surf": []}, "ensemble": {"err": [], "unc": [], "surf": []}}
    rng = np.random.default_rng(0)
    for i in data.test_indices():
        key = f"nyu_{i:04d}"
        gt, lab = data.raw(i).astype(np.float32), data.labels(i)
        mask = P.valid_mask(gt, ds)
        logs = []
        for m in RELATIVE:
            method = "lsq_disparity" if M.kind(m) == "disparity" else "lsq_depth"
            logs.append(np.log(P.align(preds[m][key].astype(np.float32), gt, mask, method, 10.0, 1e-3)))
        logs = np.stack(logs)
        ref = np.exp(logs[RELATIVE.index("da2-large")])
        surf = np.zeros(gt.shape, np.int8)
        surf[np.isin(lab, special["mirror"])] = 1
        surf[np.isin(lab, special["glass"])] = 2
        v, u = np.nonzero(mask)
        pick = rng.choice(len(v), min(4000, len(v)), replace=False)
        v, u = v[pick], u[pick]
        acc["disagreement"]["err"].append(np.abs(ref[v, u] - gt[v, u]) / gt[v, u])
        acc["disagreement"]["unc"].append(logs[:, v, u].std(axis=0))
        acc["disagreement"]["surf"].append(surf[v, u])
        if mari is not None and f"{key}_pred" in mari.files:
            d = P.align(mari[f"{key}_pred"].astype(np.float32), gt, mask, "lsq_depth", 10.0, 1e-3)
            acc["ensemble"]["err"].append(np.abs(d[v, u] - gt[v, u]) / gt[v, u])
            acc["ensemble"]["unc"].append(mari[f"{key}_unc"].astype(np.float32)[v, u])
            acc["ensemble"]["surf"].append(surf[v, u])
    res = {}
    curves = {}
    for name, a in acc.items():
        if not a["err"]:
            continue
        err, unc, surf = (np.concatenate(a[k]) for k in ("err", "unc", "surf"))
        sp, orc = sparsification(err, unc), sparsification(err, err)
        curves[name] = (sp, orc)
        keep = unc <= np.quantile(unc, 0.9)
        res[name] = {
            "pixels": int(len(err)),
            "absrel_all": round(float(err.mean()), 4),
            "absrel_at_90pct_coverage": round(float(err[keep].mean()), 4),
            "absrel_at_90pct_oracle": round(float(np.sort(err)[: int(0.9 * len(err))].mean()), 4),
            "ause": round(ause(err, unc), 4),
            "spearman_unc_err": round(float(_spearman(unc, err)), 3),
            "median_uncertainty": {
                s: round(float(np.median(unc[surf == k])), 4)
                for s, k in (("other", 0), ("mirror", 1), ("glass", 2))
                if (surf == k).any()
            },
            "share_flagged_top10pct": {
                s: round(float(np.mean(~keep[surf == k])), 3)
                for s, k in (("other", 0), ("mirror", 1), ("glass", 2))
                if (surf == k).any()
            },
        }
        print(name, res[name], flush=True)
    (DOCS / "w6_uncertainty.json").write_text(json.dumps(res, indent=1))
    figure(curves)


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def figure(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(curves), figsize=(5.5 * len(curves), 4), squeeze=False)
    titles = {
        "disagreement": "DA-V2 Large, uncertainty = disagreement of 5 models",
        "ensemble": "Marigold LCM, uncertainty = 10-seed spread",
    }
    for ax, (name, (sp, orc)) in zip(axes[0], curves.items()):
        ax.plot(FRACTIONS * 100, sp, label="remove most uncertain first")
        ax.plot(FRACTIONS * 100, orc, "--", label="oracle (remove worst first)")
        ax.set_xlabel("% of pixels removed")
        ax.set_ylabel("AbsRel of the rest")
        ax.set_title(titles[name], fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "w6_sparsification.png", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds")
    ap.add_argument("--marigold")
    args, _ = ap.parse_known_args()
    analyze(args.preds, args.marigold) if args.preds else kaggle_marigold()

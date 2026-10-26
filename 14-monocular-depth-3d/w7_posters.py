"""W7 · Illusions at scale: the poster test grown to ~1,000 composites with controlled cue ablations.

A flat picture is pasted on a flat wall (NYU labeled images and iBims-1, wall planes fitted to the sensor / laser
depth). The true depth inside the picture is the wall plane; a model that "sees into" the picture is fooled.
2 x 2 x 2 conditions per wall:
  frame        white frame around the picture, or none
  perspective  warped with the wall's real perspective, or pasted as an upright rectangle (mismatched)
  content      a real photo (COCO), or the same photo phase-scrambled: identical colours and texture statistics,
               no objects, no perspective, i.e. no depth cues
Illusion score = mean |relative deviation from the wall plane| inside the picture, minus the same region in the
original photo (aligned on everything except the picture, least squares in each model's space).
Marigold runs with a single ensemble member here (cost). Runs on a Kaggle T4; analysis: python w7_posters.py --report.

Usage (Kaggle): python kaggle_push_v2.py --name depth-w7-posters --script w7_posters.py --commit <sha>
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

JOB = globals().get("JOB") or json.loads(os.environ.get("JOB", "{}"))
DOCS = Path(__file__).resolve().parent / "docs" if "__file__" in globals() else Path(".")
CONDITIONS = list(itertools.product(["frame", "noframe"], ["matched", "upright"], ["photo", "scrambled"]))
COCO_IDS = [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353, 1425, 1490, 1503, 1532, 1584]


def phase_scramble(img, rng):
    """Same amplitude spectrum per channel, random phase (shared across channels): keeps colour and texture
    statistics, destroys objects and perspective."""
    phase = np.angle(np.fft.fft2(rng.random(img.shape[:2])))
    out = np.empty(img.shape, np.float64)
    for c in range(img.shape[2]):
        f = np.fft.fft2(img[..., c].astype(np.float64))
        out[..., c] = np.real(np.fft.ifft2(np.abs(f) * np.exp(1j * (np.angle(f) + phase))))
    return np.clip(out, 0, 255).astype(np.uint8)


def upright_quad(quad):
    """Axis-aligned rectangle with the same centre and area as `quad` (ignores the wall's perspective)."""
    q = np.asarray(quad, np.float64)
    x, y = q[:, 0], q[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    w = np.sqrt(area * 4 / 3)
    h = area / w
    cx, cy = q.mean(axis=0)
    return np.float32(
        [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]
    )


def paste(rgb, quad, picture, frame):
    """Warp `picture` into `quad`, with a white frame or none. Returns image and picture mask."""
    import cv2

    from metrics import crop_to_aspect

    picture = crop_to_aspect(picture)
    b = picture.shape[0] // 12 if frame else 0
    framed = cv2.copyMakeBorder(picture, b, b, b, b, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    fh, fw = framed.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), np.float32(quad))
    warped = cv2.warpPerspective(framed, H, (rgb.shape[1], rgb.shape[0]))
    mask = cv2.warpPerspective(np.ones((fh, fw), np.uint8), H, (rgb.shape[1], rgb.shape[0])) > 0
    out = rgb.copy()
    out[mask] = warped[mask]
    return out, mask


def plane_deviation(pred, gt, picture, plane, kind, ds, proto):
    """Align on valid pixels outside the picture, then (mean |rel dev|, mean signed rel dev) from the plane inside."""
    import protocol as P

    m = P.valid_mask(gt, ds) & ~picture
    method = "lsq_disparity" if kind == "disparity" else "lsq_depth"
    d = P.align(pred, gt, m, method, ds["max_depth"], ds["min_depth"])
    rel = (d[picture] - plane[picture]) / plane[picture]
    return float(np.mean(np.abs(rel))), float(np.mean(rel))


def walls(limit):
    """Yield (source id, rgb, gt depth, wall mask, plane depth map, intrinsics) for flat walls."""
    import io
    import zipfile

    from scipy.io import loadmat

    import datasets_v2 as D
    from metrics import backproject, eval_mask, fit_plane, plane_depth
    from nyu import K, NYU
    from poster_test import wall_plane

    vv, uu = np.mgrid[0:480, 0:640]
    data, n = NYU(), 0
    wall_id = data.class_id("wall")
    for i in range(len(data.f["images"])):
        lab, gt = data.labels(i), data.depth(i)
        wall = (lab == wall_id) & eval_mask(gt)
        if wall.mean() < 0.15:
            continue
        nrm, d, resid = wall_plane(wall, gt, K)
        if resid < 0.02:
            yield f"nyu_{i:04d}", data.rgb(i), gt.astype(np.float32), wall, plane_depth(uu, vv, nrm, d, K), K, "nyu"
            n += 1
            if n >= limit:
                return
    path = Path(os.environ["CV_DATA"]) / "depth" / "ibims" / "ibims1_core_mat.zip"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-s", "-o", str(path), D.IBIMS_FTP], check=True)
    with zipfile.ZipFile(path) as z:
        for name in sorted(x for x in z.namelist() if x.endswith(".mat")):
            dd = loadmat(io.BytesIO(z.read(name)))["data"]
            gt, wall = dd["depth"][0, 0].astype(np.float32), dd["mask_wall"][0, 0] > 0
            if wall.mean() < 0.15:
                continue
            C = dd["calib"][0, 0]
            Ki = (C[0, 0], C[1, 1], C[0, 2], C[1, 2])
            v, u = np.nonzero(wall & (gt > 0))
            pts = backproject(u, v, gt[v, u], Ki)
            nrm, d = fit_plane(pts)
            if np.median(np.abs(pts @ nrm + d)) < 0.02:
                yield Path(name).stem, dd["rgb"][0, 0], gt, wall, plane_depth(uu, vv, nrm, d, Ki), Ki, "ibims"


def run():
    import cv2

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==4.53.0", "diffusers==0.34.0", "accelerate",
         "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"],
        check=True,
    )  # fmt: skip
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
    import protocol as P
    from poster_test import poster_quad

    root = Path("/tmp/cv-data/depth/nyu")
    root.mkdir(parents=True, exist_ok=True)
    for url, fn in [(D.NYU_URL, "nyu_depth_v2_labeled.mat"), (D.NYU_SPLIT_URL, "splits.mat")]:
        if not (root / fn).exists():
            subprocess.run(["curl", "-sL", "-o", str(root / fn), url], check=True)
    pics = []
    for k in COCO_IDS:
        try:
            b = subprocess.run(
                ["curl", "-sf", f"http://images.cocodataset.org/val2017/{k:012d}.jpg"], capture_output=True, check=True
            ).stdout
            pics.append(cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        except subprocess.CalledProcessError:
            pass
    rng = np.random.default_rng(7)
    proto = P.load_protocol()
    ds_cfg = {"min_depth": 1e-3, "max_depth": 10.0, "crop": "none"}
    stage = []  # (wall id, source, images {condition: (img, mask)}, original, gt, plane)
    for k, (wid, rgb, gt, wall, plane, Kw, source) in enumerate(walls(JOB.get("walls", 125))):
        quad = poster_quad(wall, gt, Kw)
        if quad is None:
            continue
        pic = pics[k % len(pics)]
        variants = {}
        for frame, persp, content in CONDITIONS:
            q = quad if persp == "matched" else upright_quad(quad)
            p = pic if content == "photo" else phase_scramble(pic, rng)
            img, mask = paste(rgb, q, p, frame == "frame")
            if (mask & ~wall).mean() > 0.002:  # upright rectangle spilling off the wall: skip this condition
                continue
            variants["/".join((frame, persp, content))] = (img, mask)
        stage.append((wid, source, variants, rgb, gt, plane))
    print("walls", len(stage), "composites", sum(len(s[2]) for s in stage), flush=True)
    M.MARIGOLD_ENSEMBLE = 1
    out = {"commit": JOB["commit"], "conditions": ["/".join(c) for c in CONDITIONS], "rows": []}
    for name in JOB.get("models") or list(M.MODELS):
        predict, t0 = M.load(name, "cuda"), time.time()
        for wid, source, variants, rgb, gt, plane in stage:
            base = np.asarray(predict(rgb), np.float32)
            for cond, (img, mask) in variants.items():
                before, _ = plane_deviation(base, gt, mask, plane, M.kind(name), ds_cfg, proto)
                after, signed = plane_deviation(
                    np.asarray(predict(img), np.float32), gt, mask, plane, M.kind(name), ds_cfg, proto
                )
                out["rows"].append(
                    {
                        "model": name,
                        "wall": wid,
                        "source": source,
                        "condition": cond,
                        "before": before,
                        "after": after,
                        "signed": signed,
                    }
                )
        print(name, f"{time.time() - t0:.0f}s", flush=True)
        Path("/kaggle/working/w7_posters.json").write_text(json.dumps(out))


def report(path):
    """Illusion score per model x condition (bootstrap CI over walls) and paired Wilcoxon tests per factor."""
    from scipy.stats import wilcoxon

    d = json.loads(Path(path).read_text())
    rows = d["rows"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    res = {"walls": len({r["wall"] for r in rows}), "per_model": {}}
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x)
        m = x[rng.integers(0, len(x), (1000, len(x)))].mean(axis=1)
        return [
            round(float(x.mean()), 4),
            round(float(np.quantile(m, 0.025)), 4),
            round(float(np.quantile(m, 0.975)), 4),
        ]

    for model in models:
        r = [x for x in rows if x["model"] == model]
        score = {}
        for cond in d["conditions"]:
            s = [x["after"] - x["before"] for x in r if x["condition"] == cond]
            if s:
                score[cond] = {
                    "illusion_score": ci(s),
                    "share_worse": round(float(np.mean(np.array(s) > 0)), 3),
                    "n": len(s),
                }
        tests = {}
        for i, (factor, lv) in enumerate(
            [
                ("frame", ("frame", "noframe")),
                ("perspective", ("matched", "upright")),
                ("content", ("photo", "scrambled")),
            ]
        ):
            by = {}
            for x in r:
                parts = x["condition"].split("/")
                key = (x["wall"], tuple(p for j, p in enumerate(parts) if j != i))
                by.setdefault(key, {})[parts[i]] = x["after"] - x["before"]
            paired = [(v[lv[0]], v[lv[1]]) for v in by.values() if len(v) == 2]
            if len(paired) > 10:
                a, b = map(np.array, zip(*paired))
                tests[factor] = {
                    f"{lv[0]}_minus_{lv[1]}": ci(a - b),
                    "wilcoxon_p": float(wilcoxon(a, b).pvalue),
                    "pairs": len(paired),
                }
        res["per_model"][model] = {"conditions": score, "factor_tests": tests}
    (DOCS / "w7_posters.json").write_text(json.dumps(res, indent=1))
    figure(res)
    return res


def figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(res["per_model"])
    conds = list(next(iter(res["per_model"].values()))["conditions"])
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(models) + 2))
    mat = np.array(
        [
            [res["per_model"][m]["conditions"].get(c, {}).get("illusion_score", [np.nan])[0] * 100 for c in conds]
            for m in models
        ]
    )
    lim = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(conds)), [c.replace("/", "\n") for c in conds], fontsize=8)
    ax.set_yticks(range(len(models)), models)
    ax.set_title(
        f"Poster illusion score (percentage points of depth deviation added by the picture), {res['walls']} walls"
    )
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(DOCS / "w7_posters.png", dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report")
    args, _ = ap.parse_known_args()
    report(args.report) if args.report else run()

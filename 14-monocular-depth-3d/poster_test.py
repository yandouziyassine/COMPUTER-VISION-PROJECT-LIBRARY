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


def orient_quad(quad):
    """Corners ordered top-left, top-right, bottom-right, bottom-left in the image. The plane normal from
    fit_plane has an arbitrary sign, which used to flip or rotate the pasted picture (fixed 2026-10-26)."""
    q = np.asarray(quad, np.float32)
    s, d = q.sum(axis=1), q[:, 0] - q[:, 1]
    return np.float32([q[s.argmin()], q[d.argmax()], q[s.argmax()], q[d.argmin()]])


def crop_to_aspect(picture, aspect=4 / 3):
    """Centre crop to a landscape aspect ratio so a portrait photo is not squashed into the poster."""
    h, w = picture.shape[:2]
    if w / h > aspect:
        nw = int(h * aspect)
        return picture[:, (w - nw) // 2 : (w - nw) // 2 + nw]
    nh = int(w / aspect)
    return picture[(h - nh) // 2 : (h - nh) // 2 + nh]


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
    for _ in range(60):  # shrink until all corners are well inside the mask
        corners = [
            centre + sx * half * right + sy * half * 0.75 * upw for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
        quad = np.array([project(c) for c in corners], np.float32)
        if np.all(np.linalg.norm(quad - [ux, vy], axis=1) <= fill * r):
            return orient_quad(quad)
        half *= 0.92
    return None


def composite(rgb, quad, picture):
    """Warp `picture` into `quad` with a white frame; returns the new image and the poster mask."""
    picture = crop_to_aspect(picture)
    h = picture.shape[0]
    b = h // 12
    framed = cv2.copyMakeBorder(picture, b, b, b, b, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    fh, fw = framed.shape[:2]
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [fw, 0], [fw, fh], [0, fh]]), quad)
    warped = cv2.warpPerspective(framed, H, (rgb.shape[1], rgb.shape[0]))
    mask = cv2.warpPerspective(np.ones((fh, fw), np.uint8), H, (rgb.shape[1], rgb.shape[0])) > 0
    out = rgb.copy()
    out[mask] = warped[mask]
    return out, mask


def deviation(pred_disp, gt, poster, n, d):
    """Align on everything *except* the poster, then measure relative deviation from the wall plane inside it.
    Returns (mean absolute relative deviation, mean signed deviation: > 0 = seen behind the wall)."""
    mask = eval_mask(gt) & ~poster
    depth = align_disparity(pred_disp, gt, mask)
    v, u = np.nonzero(poster)
    plane = plane_depth(u, v, n, d, K)
    rel = (depth[v, u] - plane) / plane
    return float(np.mean(np.abs(rel))), float(np.mean(rel))


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
        res[name] = {
            "images": len(cands),
            "deviation_before": round(float(np.mean(before)), 4),
            "deviation_with_poster": round(float(np.mean(after)), 4),
            "illusion_score": round(float(np.mean(np.array(after) - np.array(before))), 4),
            "share_worse_with_poster": round(float(np.mean(np.array(after) > np.array(before))), 3),
            "mean_signed_deviation": round(float(np.mean(signed)), 4),
        }
        print(name, res[name], flush=True)
    results_path = DOCS / "results.json"
    results = json.loads(results_path.read_text())
    results["poster_test"] = res
    results_path.write_text(json.dumps(results, indent=1))
    examples.sort(key=lambda e: -e[0])
    np.save(DOCS / "_poster_best.npy", examples[0][1])  # used by see_vs_infer.py
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

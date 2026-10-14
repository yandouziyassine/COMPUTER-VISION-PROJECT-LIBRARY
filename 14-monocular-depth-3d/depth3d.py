"""Single photo → depth map → 3D point cloud → new camera views ("3D photo").

Depth Anything V2 (Small) predicts relative depth from one RGB image. Every pixel is
back-projected through a pinhole camera into a colored 3D point cloud (saved as .ply: open it in
MeshLab, CloudCompare, Blender, or `python view.py file.ply`). To show the 3D structure in a
README-friendly way, the cloud is re-rendered from slightly rotated cameras with a z-buffer,
which produces the parallax "3D photo" effect.

Usage:
    python depth3d.py photo1.jpg photo2.jpg ...
    python depth3d.py --demo            # runs on a few COCO sample images

Note: monocular depth is *relative* (up to an unknown scale/shift). The geometry has the right
shape, not true metric size.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import pipeline  # noqa: E402

OUT = output_dir("depth")
DOCS = Path(__file__).resolve().parent / "docs"
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEMO_IMAGES = ["000000000139.jpg", "000000000632.jpg", "000000000724.jpg", "000000000785.jpg"]  # COCO val2017


def load_estimator():
    return pipeline("depth-estimation", model=MODEL_ID, device="cpu")


def predict_depth(estimator, image):
    """Returns relative inverse depth (disparity) at image size; larger = closer."""
    with torch.inference_mode():
        out = estimator(image)
    disp = out["predicted_depth"].squeeze().numpy()
    return np.array(Image.fromarray(disp).resize(image.size, Image.BILINEAR))


def backproject(image, disparity, fov_deg=60.0):
    """Pinhole back-projection. Returns camera-space points (X right, Y down, Z forward),
    RGB colors in [0, 1], and the focal length in pixels."""
    w, h = image.size
    d = (disparity - disparity.min()) / (disparity.max() - disparity.min() + 1e-8)
    z = 1.0 / (0.15 + 0.85 * d)  # disparity → depth in [1, ~6.7]: keeps the background finite
    f = 0.5 * w / np.tan(np.radians(fov_deg) / 2)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([(u - w / 2) * z / f, (v - h / 2) * z / f, z], axis=-1).reshape(-1, 3)
    cols = np.asarray(image.convert("RGB")).reshape(-1, 3) / 255.0
    return pts, cols, f


def to_point_cloud(pts, cols, max_points=200_000):
    """Open3D cloud in the usual 3D-viewer convention (Y up, camera looking down -Z)."""
    keep = pts[:, 2] < np.percentile(pts[:, 2], 98)  # drop the far-background tail
    pts, cols = pts[keep], cols[keep]
    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts * [1, -1, -1]))
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


def render_view(pts, cols, f, size, yaw_deg, pitch_deg=0.0):
    """Re-render the cloud from a camera rotated around the scene center (z-buffered splats)."""
    w, h = size
    pivot = np.array([0.0, 0.0, np.median(pts[:, 2])])
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    p = (pts - pivot) @ (rx @ ry).T + pivot
    ok = p[:, 2] > 0.1
    p, c = p[ok], cols[ok]
    u = np.round(f * p[:, 0] / p[:, 2] + w / 2).astype(int)
    v = np.round(f * p[:, 1] / p[:, 2] + h / 2).astype(int)
    img = np.ones((h, w, 3)) * 0.12  # dark gray = disoccluded (no data)
    order = np.argsort(-p[:, 2])  # far → near, so near points win
    for du in (0, 1):  # 2x2 splats close small gaps
        for dv in (0, 1):
            uu, vv = u[order] + du, v[order] + dv
            inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[inside], uu[inside]] = c[order][inside]
    return (img * 255).astype(np.uint8)


def save_views(pts, cols, f, size, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, yaw in zip(axes, (-15, 0, 15)):
        ax.imshow(render_view(pts, cols, f, size, yaw))
        ax.set_title(f"camera rotated {yaw:+d}°", fontsize=10)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def save_sweep_gif(pts, cols, f, size, path, width=400, frames=24, amplitude=12):
    angles = amplitude * np.sin(np.linspace(0, 2 * np.pi, frames, endpoint=False))
    w, h = size
    out_size = (width, int(h * width / w))
    imgs = [
        Image.fromarray(render_view(pts, cols, f, size, a, pitch_deg=a / 3)).resize(out_size).quantize(colors=128)
        for a in angles
    ]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=90, loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fov", type=float, default=60.0, help="assumed horizontal field of view (degrees)")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)

    images = args.images
    if args.demo:
        coco = data_dir("clip-search") / "val2017"
        images = [coco / n for n in DEMO_IMAGES if (coco / n).exists()]
    if not images:
        ap.error("give image paths or --demo")

    estimator = load_estimator()
    timings = []
    fig, axes = plt.subplots(2, len(images), figsize=(3.4 * len(images), 5.4), squeeze=False)
    for j, p in enumerate(images):
        img = Image.open(p).convert("RGB")
        t0 = time.time()
        disp = predict_depth(estimator, img)
        timings.append(time.time() - t0)
        pts, cols, f = backproject(img, disp, args.fov)
        pcd = to_point_cloud(pts, cols)
        o3d.io.write_point_cloud(str(OUT / f"{p.stem}.ply"), pcd)
        save_views(pts, cols, f, img.size, DOCS / f"{p.stem}_views.jpg", f"{p.name}: new viewpoints from one photo")
        if j == 0:
            save_sweep_gif(pts, cols, f, img.size, DOCS / "demo.gif")
        axes[0, j].imshow(img)
        axes[0, j].axis("off")
        axes[1, j].imshow(disp, cmap="inferno")
        axes[1, j].axis("off")
        print(
            f"{p.name}: {img.size[0]}x{img.size[1]} | depth {timings[-1] * 1000:.0f} ms | {len(pcd.points)} points",
            flush=True,
        )
    axes[0, 0].set_title("input", loc="left")
    axes[1, 0].set_title("predicted depth (bright = near)", loc="left")
    fig.tight_layout()
    fig.savefig(DOCS / "depth_maps.jpg", dpi=80)
    plt.close(fig)
    # The first image includes model warm-up, so it's excluded from the average when possible.
    stats = {
        "model": MODEL_ID,
        "images": len(images),
        "mean_ms_per_image_cpu": round(1000 * float(np.mean(timings[1:] or timings)), 1),
    }
    (DOCS / "timing.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()

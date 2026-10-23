"""Part C figure: what you see / what the computer receives / what it infers / the truth / its 3D.

Rows: the NYU test image the model gets most right, the image with the most mirror pixels, the poster
composite with the strongest illusion, and the Ames room photo.

Usage:
    python see_vs_infer.py          # after benchmark.py, poster_test.py and illusions.py
"""

import matplotlib
import numpy as np
from PIL import Image

import models as M
from benchmark import DOCS, OUT
from depth3d import backproject as bp_render
from depth3d import render_view, save_sweep_gif
from illusions import DIR as ILLUSIONS
from metrics import EIGEN_CROP, align_disparity
from nyu import NYU

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def pixel_panel(ax, rgb, y, x, size=8):
    """The numbers a computer actually receives: an 8x8 block of brightness values."""
    gray = np.asarray(Image.fromarray(rgb).convert("L"))[y : y + size, x : x + size]
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    for (r, c), val in np.ndenumerate(gray):
        ax.text(c, r, str(val), ha="center", va="center", fontsize=6, color="red" if val > 127 else "yellow")


def main():
    data = NYU()
    idx = data.test_indices()
    absrel = np.load(OUT / "per_image_absrel_da2-small.npy")
    mirror = data.class_id("mirror")
    best = idx[int(np.argmin(absrel))]
    most_mirror = max(idx, key=lambda i: int((data.labels(i) == mirror).sum()))
    est = M.load("da2-small")
    crop = EIGEN_CROP  # NYU frames have a white border outside this region
    rows = [
        ("A scene the model gets right", data.rgb(best)[crop], data.depth(best)[crop]),
        ("A mirror: the room inside it is not real", data.rgb(most_mirror)[crop], data.depth(most_mirror)[crop]),
        ("A flat poster on a wall (Part A test)", np.load(DOCS / "_poster_best.npy")[crop], None),
        ("The Ames room illusion", np.asarray(Image.open(ILLUSIONS / "ames_room.jpg").convert("RGB")), None),
    ]
    titles = [
        "What you see",
        "What the computer receives",
        "What it infers (bright = near)",
        "The truth (Kinect) / error",
        "Its 3D, seen from the side",
    ]
    fig, axes = plt.subplots(len(rows), 5, figsize=(19, 3.6 * len(rows)))
    for r, (label, rgb, gt) in enumerate(rows):
        disp = M.predict(est, rgb)
        h, w = rgb.shape[:2]
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_ylabel(label, fontsize=10)
        pixel_panel(axes[r, 1], rgb, h // 2, w // 2)
        axes[r, 2].imshow(disp, cmap="inferno")
        if gt is not None:
            mask = (gt > 1e-3) & (gt < 10)
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

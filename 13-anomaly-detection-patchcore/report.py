"""Chart + Markdown table of docs/results.json (run after run.py): our image / pixel AUROC per MVTec AD
category next to the PatchCore paper's PatchCore-10 % numbers. Charts only: no dataset images.

Usage:
    python report.py            # writes docs/results_chart.png and prints the README table
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOCS = Path(__file__).resolve().parent / "docs"


def main():
    res = json.loads((DOCS / "results.json").read_text())
    cats = [c for c in res if c != "mean"]
    ours = np.array([[res[c]["image_auroc"], res[c]["pixel_auroc"]] for c in cats]) * 100
    paper = np.array([[res[c]["paper_image_auroc"], res[c]["paper_pixel_auroc"]] for c in cats]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 0.36 * len(cats) + 1.8), sharey=True)
    y = np.arange(len(cats))
    for k, (ax, title) in enumerate(zip(axes, ["Image AUROC (detection)", "Pixel AUROC (localization)"])):
        ax.barh(y + 0.2, ours[:, k], height=0.4, color="#4f81bd", label="this implementation")
        ax.barh(y - 0.2, paper[:, k], height=0.4, color="#bfbfbf", label="paper (PatchCore-10 %)")
        for i, v in enumerate(ours[:, k]):
            ax.text(v, i + 0.2, f" {v:.1f}", va="center", fontsize=7)
        lo = min(ours[:, k].min(), paper[:, k].min())
        ax.set_xlim(np.floor(lo) - 1, 100.8)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [f"{c} ({res[c]['type'][0]})" for c in cats])
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower left", fontsize=8)
    if "mean" in res:
        m = res["mean"]
        fig.suptitle(
            f"PatchCore from scratch on MVTec AD (WideResNet-50, 224 px, 10 % coreset): mean image AUROC "
            f"{m['image_auroc'] * 100:.1f} (paper {m['paper_image_auroc'] * 100:.1f}), pixel AUROC "
            f"{m['pixel_auroc'] * 100:.1f} (paper {m['paper_pixel_auroc'] * 100:.1f})   (o) object, (t) texture",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(DOCS / "results_chart.png", dpi=110)
    plt.close(fig)

    print("| Category | Image AUROC | Pixel AUROC | Paper (image / pixel) | Train / test (defective) | Fit (CPU) |")
    print("|---|---|---|---|---|---|")
    for c in cats:
        r = res[c]
        print(
            f"| {c} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | "
            f"{r['paper_image_auroc']:.3f} / {r['paper_pixel_auroc']:.3f} | "
            f"{r['n_train']} / {r['n_test']} ({r['n_test_defective']}) | {r['fit_seconds'] / 60:.1f} min |"
        )
    if "mean" in res:
        m = res["mean"]
        print(
            f"| **mean** | **{m['image_auroc']:.3f}** | **{m['pixel_auroc']:.3f}** | "
            f"{m['paper_image_auroc']:.3f} / {m['paper_pixel_auroc']:.3f} | | |"
        )


if __name__ == "__main__":
    main()

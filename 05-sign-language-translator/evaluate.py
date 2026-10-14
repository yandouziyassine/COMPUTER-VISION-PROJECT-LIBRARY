"""Evaluate the sign-letter models on the official test CSV.

1. Accuracy / macro F1: CNN with augmentation, same CNN without, and a pixel logistic-regression baseline.
2. Leakage audit: how close is each test image to its nearest training image? The dataset was made by
   augmenting a few hundred photos, so near-duplicates across the split are expected. Accuracy is reported
   separately for near-duplicate and genuinely new test images.
3. Robustness: accuracy as the test images are rotated, shifted, scaled, darkened/brightened or blurred,
   the kind of variation a real webcam adds.

Usage:
    python evaluate.py          # ~5 min on CPU
"""

import json
import sys
import time

import cv2
import matplotlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

from asl import DOCS, LETTERS, load_model, load_split, predict_proba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

NEAR, FAR = 0.95, 0.90   # correlation with the nearest training image


def metrics(pred, y):
    return {"accuracy": round(float((pred == y).mean()), 4), "macro_f1": round(float(f1_score(y, pred, average="macro")), 4)}


def nearest_train_correlation(train, test, chunk=512):
    """Pearson correlation between each test image and its most similar training image (pixel space)."""
    def unit(a):
        v = a.reshape(len(a), -1).astype(np.float32)
        v -= v.mean(1, keepdims=True)
        return torch.from_numpy(v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-6))

    tr, te = unit(train), unit(test)
    best, idx = [], []
    for i in range(0, len(te), chunk):
        sims = te[i:i + chunk] @ tr.T
        v, j = sims.max(1)
        best.append(v)
        idx.append(j)
    return torch.cat(best).numpy(), torch.cat(idx).numpy()


def warp(images, angle=0.0, shift=0.0, scale=1.0):
    """Rotate (degrees), shift right/down (pixels) and scale every image; borders repeat the edge pixels."""
    m = cv2.getRotationMatrix2D((13.5, 13.5), angle, scale)
    m[:, 2] += shift
    return np.stack([cv2.warpAffine(im, m, (28, 28), borderMode=cv2.BORDER_REPLICATE) for im in images])


PERTURBATIONS = {
    "rotation (°)": ([0, 5, 10, 15, 20, 25, 30], lambda x, v: warp(x, angle=v)),
    "shift (px)": ([0, 1, 2, 3, 4, 5, 6], lambda x, v: warp(x, shift=v)),
    "scale (×)": ([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], lambda x, v: warp(x, scale=v)),
    "brightness (×)": ([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                       lambda x, v: np.clip(x.astype(np.float32) * v, 0, 255).astype(np.uint8)),
    "blur (σ px)": ([0, 0.5, 1.0, 1.5, 2.0],
                    lambda x, v: x if v == 0 else np.stack([cv2.GaussianBlur(im, (0, 0), v) for im in x])),
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # labels contain ≥, °, σ; the Windows console default can't print them
    torch.set_num_threads(4)
    tr_x, tr_y = load_split("train")
    te_x, te_y = load_split("test")
    models = {"cnn_aug": load_model("cnn_aug"), "cnn_noaug": load_model("cnn_noaug")}
    results = {"n_train": len(tr_y), "n_test": len(te_y), "classes": len(LETTERS)}

    # 1. Headline accuracy
    preds = {k: predict_proba(m, te_x).argmax(1) for k, m in models.items()}
    t0 = time.time()
    logreg = LogisticRegression(max_iter=1000).fit(tr_x.reshape(len(tr_x), -1) / 255.0, tr_y)
    preds["pixel_logreg"] = logreg.predict(te_x.reshape(len(te_x), -1) / 255.0)
    print(f"logreg fitted in {time.time() - t0:.0f}s", flush=True)
    results["test"] = {k: metrics(p, te_y) for k, p in preds.items()}
    t0 = time.time()
    for _ in range(3):
        predict_proba(models["cnn_aug"], te_x[:1000])
    results["cnn_ms_per_image_batched"] = round((time.time() - t0) / 3000 * 1000, 3)
    print(results["test"], flush=True)

    cm = confusion_matrix(te_y, preds["cnn_aug"], labels=range(len(LETTERS)))
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = sorted(((off[i, j], LETTERS[i], LETTERS[j]) for i in range(len(LETTERS)) for j in range(len(LETTERS))),
                   reverse=True)[:6]
    results["top_confusions_cnn_aug"] = [{"true": a, "pred": b, "count": int(c)} for c, a, b in pairs if c > 0]

    # 2. Leakage audit
    corr, nn_idx = nearest_train_correlation(tr_x, te_x)
    bins = {"near-duplicate (r ≥ 0.95)": corr >= NEAR, "similar (0.90–0.95)": (corr >= FAR) & (corr < NEAR),
            "new (r < 0.90)": corr < FAR}
    results["leakage"] = {"median_nearest_r": round(float(np.median(corr)), 4),
                          "bins": {b: {"share": round(float(m.mean()), 4),
                                       **{k: round(float((p[m] == te_y[m]).mean()), 4) if m.any() else None
                                          for k, p in preds.items()}}
                                   for b, m in bins.items()}}
    print(json.dumps(results["leakage"], indent=1, ensure_ascii=False), flush=True)

    # 3. Robustness
    results["robustness"] = {}
    for name, (levels, fn) in PERTURBATIONS.items():
        results["robustness"][name] = {"levels": levels}
        for k in ("cnn_aug", "cnn_noaug"):
            results["robustness"][name][k] = [round(float((predict_proba(models[k], fn(te_x, v)).argmax(1) == te_y).mean()), 4)
                                              for v in levels]
        print(name, results["robustness"][name], flush=True)

    DOCS.mkdir(exist_ok=True)
    (DOCS / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    plot_confusion(cm)
    plot_robustness(results["robustness"])
    plot_leakage(corr, te_x, tr_x, nn_idx)
    plot_examples(te_x, te_y, preds["cnn_aug"], predict_proba(models["cnn_aug"], te_x).max(1))


def plot_confusion(cm):
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(LETTERS)), LETTERS)
    ax.set_yticks(range(len(LETTERS)), LETTERS)
    for i in range(len(LETTERS)):
        for j in range(len(LETTERS)):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="crimson")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CNN (with augmentation), test set: off-diagonal counts in red")
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def plot_robustness(rob):
    fig, axes = plt.subplots(1, len(rob), figsize=(4 * len(rob), 3.4), sharey=True)
    for ax, (name, r) in zip(axes, rob.items()):
        ax.plot(r["levels"], r["cnn_aug"], "o-", label="trained with augmentation")
        ax.plot(r["levels"], r["cnn_noaug"], "s--", label="no augmentation")
        ax.set_title(name)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "robustness.png", dpi=120)
    plt.close(fig)


def plot_leakage(corr, te_x, tr_x, nn_idx):
    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_subplot(1, 3, 1)
    ax.hist(corr, bins=60, color="#555")
    ax.axvline(NEAR, color="crimson", ls="--")
    ax.set_xlabel("correlation with nearest training image")
    ax.set_ylabel("test images")
    ax.set_title("How new is each test image?")
    picks = np.argsort(-corr)[[0, 500, 2000]].tolist() + np.argsort(corr)[[0, 50]].tolist()
    for k, i in enumerate(picks):
        for row, (img, label) in enumerate([(te_x[i], f"test r={corr[i]:.2f}"), (tr_x[nn_idx[i]], "nearest train")]):
            a = fig.add_subplot(2, 9, 4 + k + row * 9)
            a.imshow(img, cmap="gray")
            a.set_title(label, fontsize=7)
            a.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "leakage.png", dpi=120)
    plt.close(fig)


def plot_examples(x, y, pred, conf):
    wrong = np.flatnonzero(pred != y)[:6]
    right = np.flatnonzero(pred == y)[:12]
    fig, axes = plt.subplots(2, 9, figsize=(12, 3.2))
    for ax, i in zip(axes.flat, np.concatenate([right, wrong])):
        ax.imshow(x[i], cmap="gray")
        ok = pred[i] == y[i]
        ax.set_title(f"{LETTERS[pred[i]]} ({conf[i]:.0%})" + ("" if ok else f" ≠ {LETTERS[y[i]]}"),
                     fontsize=8, color="black" if ok else "crimson")
        ax.axis("off")
    fig.suptitle("Test images: prediction (confidence); red = error, true letter after ≠", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "examples.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()

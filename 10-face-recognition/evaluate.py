"""Face verification on LFW with the standard 10-fold protocol.

For each fold, the similarity threshold is chosen on the other 9 folds (maximizing accuracy) and
applied to the held-out fold; the score is the mean accuracy over the 10 folds. Also: ROC AUC,
TAR at FAR = 1 % / 0.1 %, FP32 vs INT8 model, and the hardest mistakes.

Usage:
    python evaluate.py
"""

import json
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from faces import DATA, DOCS, FaceEmbedder, cosine

LFW = DATA / "lfw_home"


def load_pairs():
    lines = (LFW / "pairs.txt").read_text().strip().splitlines()[1:]
    pairs = []
    for ln in lines:
        p = ln.split("\t")
        if len(p) == 3:
            pairs.append(((p[0], int(p[1])), (p[0], int(p[2])), 1))
        else:
            pairs.append(((p[0], int(p[1])), (p[2], int(p[3])), 0))
    return pairs  # 10 folds × (300 same + 300 different), in file order


def img_path(name, n):
    return LFW / "lfw" / name / f"{name}_{n:04d}.jpg"


def embed_all(embedder, keys):
    embs, fails, t = {}, 0, []
    for k in keys:
        img = cv2.imread(str(img_path(*k)))
        t0 = time.time()
        e = embedder.embed(img)
        t.append(time.time() - t0)
        if e is None:  # no face found: fall back to the whole (already face-centered) image
            fails += 1
            e = embedder.recognizer.feature(cv2.resize(img[70:180, 70:180], (112, 112))).ravel()
            e = e / np.linalg.norm(e)
        embs[k] = e
    return embs, fails, 1000 * float(np.mean(t))


def ten_fold_accuracy(scores, labels):
    folds = np.arange(len(scores)) // 600
    accs, thrs = [], []
    candidates = np.linspace(-0.2, 1.0, 601)
    for f in range(10):
        tr, te = folds != f, folds == f
        best = max(candidates, key=lambda t: ((scores[tr] > t) == labels[tr]).mean())
        accs.append(((scores[te] > best) == labels[te]).mean())
        thrs.append(best)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(thrs))


def tar_at_far(scores, labels, far):
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(far, fpr, tpr))


def main():
    DOCS.mkdir(exist_ok=True)
    pairs = load_pairs()
    labels = np.array([p[2] for p in pairs])
    keys = sorted({p[0] for p in pairs} | {p[1] for p in pairs})
    res = {"pairs": len(pairs), "unique_images": len(keys)}
    curves = {}
    for precision in ("fp32", "int8"):
        embs, fails, ms = embed_all(FaceEmbedder(precision), keys)
        scores = np.array([cosine(embs[a], embs[b]) for a, b, _ in pairs])
        acc, std, thr = ten_fold_accuracy(scores, labels)
        res[precision] = {
            "accuracy_10fold": round(acc, 4),
            "std": round(std, 4),
            "mean_threshold": round(thr, 3),
            "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
            "TAR@FAR=1%": round(tar_at_far(scores, labels, 0.01), 4),
            "TAR@FAR=0.1%": round(tar_at_far(scores, labels, 0.001), 4),
            "no_face_detected": fails,
            "ms_per_image_detect+embed": round(ms, 1),
        }
        curves[precision] = (scores, thr)
        print(precision, res[precision], flush=True)
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    # Score distributions + hardest errors (FP32)
    scores, thr = curves["fp32"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.hist(scores[labels == 1], bins=60, alpha=0.6, label="same person")
    ax.hist(scores[labels == 0], bins=60, alpha=0.6, label="different people")
    ax.axvline(thr, color="k", ls="--", label=f"threshold {thr:.2f}")
    ax.set_xlabel("cosine similarity")
    ax.legend()
    ax.set_title("LFW: 3,000 genuine vs 3,000 impostor pairs")
    fig.tight_layout()
    fig.savefig(DOCS / "score_distribution.png", dpi=100)
    plt.close(fig)

    false_rej = np.where((labels == 1) & (scores <= thr))[0]
    false_acc = np.where((labels == 0) & (scores > thr))[0]
    worst = list(false_rej[np.argsort(scores[false_rej])][:3]) + list(false_acc[np.argsort(-scores[false_acc])][:3])
    fig, axes = plt.subplots(2, len(worst), figsize=(2.3 * len(worst), 5))
    for j, i in enumerate(worst):
        a, b, y = pairs[i]
        for r, k in enumerate((a, b)):
            axes[r, j].imshow(cv2.cvtColor(cv2.imread(str(img_path(*k)))[40:210, 50:200], cv2.COLOR_BGR2RGB))
            axes[r, j].axis("off")
        axes[0, j].set_title(
            ("same, rejected" if y else "different, accepted") + f"\nscore {scores[i]:.2f}", fontsize=8
        )
    fig.suptitle("Hardest mistakes (left: same person rejected · right: different people accepted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "hardest_errors.jpg", dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    main()

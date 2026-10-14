"""Test-set evaluation: fine-tuned ResNet-18 vs zero-shot CLIP (no training at all).

Outputs (docs/): results.json, confusion_matrix.png, examples.jpg
"""

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open_clip  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix  # noqa: E402

from eurosat import CLASSES, DOCS, load_split, load_trained, predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "16-text-to-image-search"))
from clip_search import MODEL, OUT as CLIP_OUT, PRETRAINED  # noqa: E402

# Readable class names for the text prompts
PROMPT_NAMES = [
    "annual crop land",
    "forest",
    "herbaceous vegetation",
    "highway or road",
    "industrial buildings",
    "pasture land",
    "permanent crop land",
    "residential buildings",
    "river",
    "sea or lake",
]


def clip_zero_shot(images):
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, pretrained=PRETRAINED, cache_dir=str(CLIP_OUT / "weights")
    )
    tok = open_clip.get_tokenizer(MODEL)
    with torch.no_grad():
        t = model.encode_text(tok([f"a satellite photo of {n}." for n in PROMPT_NAMES]))
        t = t / t.norm(dim=-1, keepdim=True)
        preds = []
        for i in range(0, len(images), 128):
            x = torch.stack([preprocess(im) for im in images[i : i + 128]])
            f = model.encode_image(x)
            preds.append(((f / f.norm(dim=-1, keepdim=True)) @ t.T).argmax(1).numpy())
    return np.concatenate(preds)


def main():
    DOCS.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    images, labels = load_split("test")
    y = np.array(labels)

    t0 = time.time()
    pred = predict(load_trained(), images)
    ms = (time.time() - t0) / len(y) * 1000
    t0 = time.time()
    zs = clip_zero_shot(images)
    ms_clip = (time.time() - t0) / len(y) * 1000

    report = classification_report(y, pred, target_names=CLASSES, digits=3, output_dict=True)
    res = {
        "n_test": len(y),
        "resnet18_finetuned": {
            "accuracy": round(float((pred == y).mean()), 4),
            "macro_f1": round(report["macro avg"]["f1-score"], 4),
            "ms_per_tile": round(ms, 2),
        },
        "clip_zero_shot": {"accuracy": round(float((zs == y).mean()), 4), "ms_per_tile": round(ms_clip, 2)},
        "per_class_f1": {c: round(report[c]["f1-score"], 3) for c in CLASSES},
        "clip_per_class_acc": {c: round(float((zs[y == k] == k).mean()), 3) for k, c in enumerate(CLASSES)},
    }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", colorbar=False, xticks_rotation=45
    )
    ax.set_title(f"EuroSAT test (n={len(y)}) — fine-tuned ResNet-18, acc {res['resnet18_finetuned']['accuracy']:.1%}")
    fig.tight_layout()
    fig.savefig(DOCS / "confusion_matrix.png", dpi=90)
    plt.close(fig)

    # One correct example per class + the first 5 errors
    rng = np.random.default_rng(0)
    picks = [int(rng.choice(np.where((y == k) & (pred == k))[0])) for k in range(len(CLASSES))]
    errors = list(np.where(pred != y)[0][:5])
    fig, axes = plt.subplots(3, 5, figsize=(11, 7.4))
    for ax, i in zip(axes.ravel(), picks + errors):
        ax.imshow(images[i])
        ax.axis("off")
        ok = pred[i] == y[i]
        ax.set_title(
            CLASSES[y[i]] if ok else f"true {CLASSES[y[i]]}\npred {CLASSES[pred[i]]}",
            fontsize=8,
            color="black" if ok else "red",
        )
    fig.suptitle("Top two rows: one correct tile per class · bottom row: mistakes (red)")
    fig.tight_layout()
    fig.savefig(DOCS / "examples.jpg", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()

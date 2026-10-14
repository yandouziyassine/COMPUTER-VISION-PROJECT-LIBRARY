"""Evaluate the trained model on the held-out test set and produce the README figures.

Outputs (in docs/):
    confusion_matrix.png, training_curves.png, gradcam_examples.png, results.json
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, f1_score

from solar import CLASSES, DOCS_DIR, IMG_SIZE, OUT_DIR, eval_tf, load_rgb, load_trained, make_split


def predict_all(model, items):
    preds, probs = [], []
    with torch.no_grad():
        for p, _ in items:
            out = model(eval_tf(load_rgb(p)).unsqueeze(0)).softmax(1)[0]
            probs.append(out.numpy())
            preds.append(int(out.argmax()))
    return np.array(preds), np.array(probs)


def plot_curves():
    hist = json.load(open(OUT_DIR / "history.json"))
    ep = [h["epoch"] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    a1.plot(ep, [h["train_loss"] for h in hist], label="train")
    a1.plot(ep, [h["val_loss"] for h in hist], label="val")
    a1.set_title("Loss")
    a1.set_xlabel("epoch")
    a1.legend()
    a2.plot(ep, [h["val_acc"] for h in hist], color="tab:green")
    a2.set_title("Validation accuracy")
    a2.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "training_curves.png", dpi=110)
    plt.close(fig)


def gradcam_grid(model, items, preds):
    """One correctly classified test image per class, with the Grad-CAM heatmap."""
    cam = GradCAM(model=model, target_layers=[model.conv_head])
    picks = []
    for c in range(len(CLASSES)):
        for i, (p, y) in enumerate(items):
            if y == c and preds[i] == c:
                picks.append((p, y))
                break
    fig, axes = plt.subplots(2, len(picks), figsize=(3 * len(picks), 6))
    for j, (p, y) in enumerate(picks):
        img = load_rgb(p)
        x = eval_tf(img).unsqueeze(0)
        heat = cam(input_tensor=x, targets=None)[0]  # None → explain the predicted class
        rgb = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255
        axes[0, j].imshow(rgb)
        axes[0, j].set_title(CLASSES[y], fontsize=10)
        axes[1, j].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
        axes[0, j].axis("off")
        axes[1, j].axis("off")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle("What the model looks at (bottom: Grad-CAM)")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "gradcam_examples.png", dpi=90)
    plt.close(fig)


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    model = load_trained()
    test = make_split()["test"]
    y_true = np.array([y for _, y in test])
    preds, _ = predict_all(model, test)

    acc = float((preds == y_true).mean())
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    report = classification_report(y_true, preds, target_names=CLASSES, digits=3, output_dict=True)
    print(classification_report(y_true, preds, target_names=CLASSES, digits=3))
    print(f"test accuracy {acc:.3f} | macro F1 {macro_f1:.3f} | n={len(test)}")

    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    ax.set_title(f"Test set confusion matrix (acc {acc:.1%})")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    plot_curves()
    gradcam_grid(model, test, preds)
    with open(DOCS_DIR / "results.json", "w") as f:
        json.dump({"test_accuracy": acc, "macro_f1": macro_f1, "n_test": len(test), "per_class": report}, f, indent=1)


if __name__ == "__main__":
    main()

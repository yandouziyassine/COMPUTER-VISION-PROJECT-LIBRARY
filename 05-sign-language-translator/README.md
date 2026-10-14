# 05 · Sign Language Translator (ASL fingerspelling)

Reads the 24 static letters of the American Sign Language alphabet from a hand image, and types words from a
webcam: MediaPipe finds the hand, a small CNN trained from scratch reads the letter. The headline accuracy is
**99.9 %**, and the interesting part of this project is showing why that number says much less than it seems:
the test set is too easy, and simple camera changes break the model.

![test examples](docs/examples.png)

## Results

7,172 official test images, 24 letters (J and Z are left out because they need motion).

| Model | Test accuracy | Macro F1 |
|---|---|---|
| Logistic regression on raw pixels (baseline) | 69.7 % | 0.673 |
| CNN from scratch, no augmentation | **100.0 %** | 1.000 |
| CNN from scratch, with augmentation (used by the app) | 99.9 % | 0.999 |

The CNN has ~0.4 M parameters and trains in ~8 minutes on a laptop CPU.

### Why 100 % is suspicious: the test set is not new

According to the dataset card, all 34,627 images were generated from only **1,704 original photos**, each
expanded into 50+ variations (resampling filters, pixelation, ±15 % brightness, 3° rotation), then split into
train and test. I measured how close each test image is to its most similar training image (pixel correlation):

![leakage audit](docs/leakage.png)

| Test images, by similarity to the nearest training image | Share | Pixel baseline | CNN |
|---|---|---|---|
| Near-duplicate (r ≥ 0.95) | 15 % | 98.9 % | 100 % |
| Similar (0.90–0.95) | 18 % | 91.6 % | 100 % |
| Not close in pixels (r < 0.90) | 67 % | 57.4 % | 99.9 % |

A third of the test set is a near copy of training data: even a linear model on raw pixels scores ~99 % there.
The CNN stays perfect on the rest too, and the figure shows why: the *least* similar test images (r ≈ 0.5, right)
are still the **same photo** as their nearest training image, just shifted and cropped. Pixel correlation
underestimates the overlap. **The test set cannot tell us how the model does on a new person or a new room.**

### What a webcam does to it: robustness

Same test images, perturbed the way a real camera would (borders filled by repeating the edge):

![robustness](docs/robustness.png)

| Perturbation | No augmentation | With augmentation |
|---|---|---|
| Rotation 20° / 30° | 75.1 % / 38.3 % | **96.5 % / 71.7 %** |
| Shift 4 px / 6 px (of 28) | 61.6 % / 26.4 % | **92.4 % / 53.2 %** |
| Scale 0.8× / 1.2× | 91.8 % / 96.2 % | **96.4 % / 99.8 %** |
| Brightness 0.4× / 1.6× | 89.1 % / 93.4 % | **99.9 % / 96.7 %** |
| Blur σ 1.5 px / 2 px | **68.1 % / 35.6 %** | 45.0 % / 23.0 % |

- **Augmentation is what makes the model usable**: identical clean accuracy (99.9 % vs 100 %), but 2× more
  accurate on a 30° tilt or a 6-pixel shift. The clean test set alone would have hidden this completely.
- **Augmentation only protects against what it includes.** It had no blur, and the augmented model is *worse*
  under blur than the plain one: it learned to rely on fine edges that survive rotation but not defocus. A
  webcam image is often slightly blurred, so the next training run should add blur and noise.

## How it works

1. **Data** ([`asl.py`](asl.py)): 28×28 grayscale crops, labels 0–25 mapped to 24 classes (no J, no Z).
2. **Model**: three conv blocks (two 3×3 convs + batch norm + ReLU, then max-pool: 28 → 14 → 7 → 3), dropout,
   two dense layers. Trained from scratch, AdamW + one-cycle schedule, label smoothing 0.05, 8 epochs.
3. **Augmentation** ([`train.py --aug`](train.py)): per-image random rotation ±12°, shift ±10 %, scale 0.9–1.1,
   brightness and contrast ±30 %.
4. **Evaluation** ([`evaluate.py`](evaluate.py)): accuracy and macro F1 against a pixel logistic-regression
   baseline, the leakage audit, robustness curves, confusion matrix ([`docs/confusion_matrix.png`](docs/confusion_matrix.png)).
5. **Webcam app** ([`app.py`](app.py)): MediaPipe Hand Landmarker gives the hand's landmarks; their bounding box
   (+25 % margin) is cropped, converted to the dataset's format (grayscale, padded to a square with the
   background tone, 28×28) and classified. A letter is typed when it stays the top prediction (≥ 80 %) for 12
   frames; no hand for 1.5 s types a space.

## Data

- **Sign Language MNIST** ([Kaggle `datamunge/sign-language-mnist`](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)),
  **CC0**: 27,455 train / 7,172 test images, 28×28 grayscale.
- **Location:** `../../cv-data/asl/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 05-sign-language-translator
kaggle datasets download datamunge/sign-language-mnist -p ../../cv-data/asl --unzip
python train.py --aug             # ~8 min on CPU → cnn_aug.pt
python train.py                   # the no-augmentation ablation → cnn_noaug.pt
python evaluate.py                # ~15 min: metrics, leakage audit, robustness, figures
python app.py                     # webcam translator (Esc to quit)
```

## What I learned / what didn't work

- **A perfect score was a warning, not a result.** The dataset card explains it: 1,704 photos stretched into
  34,627 images. Measuring robustness told me more about real-world use than the test set did.
- **Validation was useless for model selection** (100 % from epoch 3): it is drawn from the same training CSV.
  A split by original photo is impossible because the dataset does not record which photo each image came from.
- **Augmentation is a specification.** Every transformation I included made the model robust to it; the one I
  left out (blur) got worse.
- **The webcam app is not measured.** There is no labeled webcam data here; its accuracy on a new signer is
  unknown and probably far below the test numbers. That is the honest status of this demo.

## Next steps

- Add blur, noise and JPEG compression to the augmentation; re-measure the robustness curves.
- Record a small labeled set of webcam images from new signers to measure real accuracy.
- Classify MediaPipe hand **landmarks** instead of pixels: invariant to background, lighting and skin tone, and
  J/Z become possible with a sequence model over landmark trajectories.

## Tech stack

Python · PyTorch · torchvision · scikit-learn · OpenCV · MediaPipe · Matplotlib

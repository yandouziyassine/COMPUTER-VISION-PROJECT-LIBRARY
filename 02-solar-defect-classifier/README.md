# 02 · Solar Panel Defect Classifier

Classifies photos of solar panels into 6 conditions (bird droppings, clean, dusty, electrical
damage, physical damage, snow-covered) by fine-tuning a pretrained **EfficientNet-B0**, and
shows *why* it decided with **Grad-CAM** heatmaps. Trained entirely on a laptop CPU.

> Example images not shown: the dataset licence (licence not recorded, Kaggle) does not allow redistribution; see the dataset page.

## Results

Held-out test set: 129 images the model never saw during training or model selection.

| Metric | Value |
|---|---|
| Test accuracy | **82.9 %** |
| Macro F1 (all classes weighted equally) | **0.830** |
| Training time | ~15 min (15 epochs × ~60 s, Intel Core Ultra 7 258V CPU) |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Bird-drop | 0.839 | 0.897 | 0.867 | 29 |
| Clean | 0.722 | 0.897 | 0.800 | 29 |
| Dusty | 0.818 | 0.643 | 0.720 | 28 |
| Electrical-damage | 1.000 | 1.000 | 1.000 | 15 |
| Physical-Damage | 0.857 | 0.600 | 0.706 | 10 |
| Snow-Covered | 0.889 | 0.889 | 0.889 | 18 |

<p float="left">
  <img src="docs/confusion_matrix.png" width="49%" />
  <img src="docs/training_curves.png" width="49%" />
</p>

## How it works

1. **Data split**: stratified 70 / 15 / 15 train / validation / test split with a fixed seed,
   saved to CSV so training and evaluation always use the same images.
2. **Pretrained backbone**: EfficientNet-B0 (ImageNet weights, via `timm`) with a new 6-class head.
   Small (5.3 M parameters), so full fine-tuning is practical on a CPU.
3. **Augmentation**: random resized crops, horizontal/vertical flips, ±15° rotation and color
   jitter. Panels look valid from any orientation, so flips are free extra data.
4. **Class imbalance**: Physical-Damage has 49 training images against 135 for Clean. The loss
   uses inverse-frequency class weights (plus label smoothing 0.1) so rare classes aren't ignored.
5. **Training**: AdamW with a one-cycle learning-rate schedule. The checkpoint with the best
   validation accuracy is kept.
6. **Explainability**: Grad-CAM on the last convolutional layer highlights the regions that
   drove each prediction.

Images are decoded once and cached in RAM: the dataset lives on a USB key, and random reads
from it would otherwise dominate training time.

## Data

- **Source:** "Faulty Solar Panel" image dataset (Kaggle), 6 classes of web-collected photos.
- **Size used:** 869 images (Bird-drop 191, Clean 193, Dusty 190, Electrical-damage 103,
  Physical-Damage 69, Snow-Covered 123), split 611 / 129 / 129 (train / validation / test).
  The archive holds 885 images, but some filenames differ only by upper/lower case and overwrite
  each other when extracted on Windows.
- **Location:** `../../cv-data/solar/Faulty_solar_panel/<class>/` (not committed).

## Run it

```bash
# from the repo root, with the shared environment (see setup_env.ps1)
.venv\Scripts\activate
cd 02-solar-defect-classifier
python train.py --epochs 15     # ~15 min on CPU → ../../cv-data/solar/outputs/best.pt
python evaluate.py              # test metrics + figures in docs/
python app.py                   # web demo: upload a photo → prediction + Grad-CAM
```

## What I learned / what didn't work

- **Shortcut learning is visible.** For the "Dusty" example above, Grad-CAM lights up the
  *roof in the foreground*, not the dust on the panels. The web-collected dusty photos often
  share a rooftop context, and the model partly learned that instead of the dust itself.
  This is the main risk with small scraped datasets, and exactly what Grad-CAM is for.
- **Dusty vs Clean is the hardest pair** (Dusty recall 0.64). Light dust is subtle at
  224×224, and some "clean" panels are photographed in dusty environments.
- **Physical-Damage** has the fewest images (49 for training), and its recall (0.60) suffers
  despite class weighting. Thin cracks also get weak Grad-CAM attention.
- **Validation accuracy plateaued at ~79 % for several epochs** before the learning-rate
  decay pushed it to 82.9 % (epoch 12; the test accuracy happens to be the same 82.9 %). With
  only 129 validation images, one image is ~0.8 %, so differences between epochs are noisy.

## Next steps

- Collect more Physical-Damage and Dusty images, or crop to the panel area to remove
  background shortcuts (e.g. a panel segmentation step first).
- Higher input resolution (320–384 px) for thin cracks and light dust.
- k-fold cross-validation for a more reliable accuracy estimate on this small dataset.

## Tech stack

Python · PyTorch · timm (EfficientNet-B0) · torchvision · pytorch-grad-cam · scikit-learn · Gradio

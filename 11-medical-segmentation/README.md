# 11 · Medical Image Segmentation: Polyps in Colonoscopy

Outlines **colorectal polyps** pixel by pixel in colonoscopy images with a **U-Net** (ResNet-18
encoder, ImageNet-pretrained), trained on a laptop CPU. Polyps are precursors of colorectal
cancer; automatic segmentation helps endoscopists not miss them.

![predictions](docs/predictions.jpg)

*Three typical test cases (left) and the two worst (right). Red: expert annotation. Blue: U-Net prediction.*

## Results

Held-out test set: 100 images never seen during training or checkpoint selection.

| Metric | Value |
|---|---|
| **Mean Dice** | **0.897** |
| **Mean IoU** | **0.833** |
| Median Dice | 0.937 |
| Test images with Dice > 0.8 | 87 % |
| Test images with Dice < 0.5 (failures) | 3 % |
| Inference (CPU, 256×256) | 69 ms / image |
| Training | 19 epochs × ~3.5 min on CPU; best checkpoint at epoch 14 (val Dice 0.882) |

For reference, published U-Net baselines on Kvasir-SEG report Dice of roughly 0.8–0.9 depending on
split and resolution; specialized architectures reach the low 0.9s. Different papers use
different splits, so treat this as context, not a head-to-head comparison.

![training curves](docs/training_curves.png)

## How it works

1. **Data**: 1,000 images with expert masks, resized to 256×256, split 800 / 100 / 100
   (train / validation / test) with a fixed seed.
2. **Model**: U-Net. The encoder (ResNet-18, ImageNet weights) extracts features at 5 scales, and
   the decoder upsamples back to full resolution with **skip connections** carrying fine detail
   (polyp boundaries) from the encoder. Built with `segmentation_models_pytorch`.
3. **Loss: Dice + binary cross-entropy.** Polyps often cover < 20 % of the image. Pure BCE
   rewards predicting "background" everywhere, while the Dice term directly optimizes overlap
   with the polyp. BCE keeps gradients stable early on.
4. **Augmentation**: flips and 90° rotations (applied identically to image and mask; colonoscopy
   has no canonical orientation), plus brightness/contrast jitter for endoscope lighting.
5. **Training**: AdamW, cosine learning-rate decay; the checkpoint with the best validation Dice is kept.

## Data

- **Source:** [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) (Jha et al., MMM 2020),
  1,000 polyp images with masks verified by an experienced gastroenterologist. Free for research
  and educational use; cite the paper if you use it.
- **Location:** `../../cv-data/kvasir/Kvasir-SEG/` (not committed). Direct download:
  `https://datasets.simula.no/downloads/kvasir-seg.zip` (44 MB).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 11-medical-segmentation
python train.py --epochs 20       # ~70 min on CPU → ../../cv-data/kvasir/outputs/best.pt
python evaluate.py                # test Dice/IoU + figures in docs/
```

## What I learned / what didn't work

- **Worst case (Dice 0.00) is a data artifact.** Some Kvasir images carry a black inset box (an
  endoscope position overlay) that covers part of the frame; here it hides almost the whole
  polyp. A real system would detect and mask out such overlays, or crop them before training.
- **False positives on folds** (Dice 0.48 case): bulging mucosal folds look like flat polyps.
  Clinically a false alarm is cheaper than a missed polyp, so the decision threshold should be
  tuned for recall, not Dice.
- **Validation Dice plateaued after epoch ~14** (0.87–0.88) while training loss kept falling.
  The model is starting to overfit 800 images; stronger augmentation or more data would help more
  than more epochs.
- **The training run died during the last epoch** (the background job was stopped, not a Python
  error). The best checkpoint was already saved and `history.json` was rebuilt from the log.
  Lesson: write history every epoch, not only at the end.
- **Not fully reproducible as run.** The augmentations used Python's `random` module, which wasn't
  seeded (a code review caught it; `train.py` now seeds it). Re-training will give slightly different
  numbers than the Dice 0.897 reported here.

## Next steps

- Test-time augmentation (average predictions over flips): usually +1 point of Dice for free.
- Higher resolution (352–512 px) for small, flat polyps; a stronger encoder (EfficientNet-B3).
- Evaluate generalization on another center's data (CVC-ClinicDB) instead of only the same dataset.
- Tune the threshold for sensitivity, which matters more than Dice in screening.

## Tech stack

Python · PyTorch · segmentation_models_pytorch (U-Net, ResNet-18) · NumPy · Matplotlib

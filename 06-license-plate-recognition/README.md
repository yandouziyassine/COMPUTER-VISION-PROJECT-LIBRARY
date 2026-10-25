# 06 · License Plate Recognition

Finds license plates in photos and reads them. A two-stage pipeline: **YOLO11n**, fine-tuned here
on a laptop CPU, locates plates; **RapidOCR** (via OpenVINO) reads the cropped plate; a
**format-aware correction** step fixes the classic O/0, I/1, S/5 confusions. Each stage is
measured separately.

![end to end](docs/end_to_end.jpg)

*End to end on test photos: plates found and read ("51F-244.03" → `51F24403`, "29A-517.96" →
`29A51796`). Tiny, distant plates (top left, bottom right) are found but not readable.*

## Results

### Stage 1: detection (882 test images)

| Detector | Training | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|
| **YOLO11n fine-tuned here** | 1,500 images, 12 epochs, CPU (~2.4 h) | **0.983** | 0.686 | 0.986 | 0.957 |
| Public plate detector (morsetechlab v1n) | much larger dataset, GPU | 0.993 | 0.717 | 0.994 | 0.980 |

Fine-tuning on a quarter of the training set on a CPU gets within ~1 point mAP50 of a public
model trained on far more data. Caveat: the public model may have seen images from this same
Roboflow source, so its numbers could be optimistic.

### Retrained on all the data (Kaggle GPU)

The detector above was trained on a laptop CPU with **1,500** of the 6,176 training images (12 epochs). The same
YOLO11n, same single class, same 640 px, was retrained on **all 6,176 images for 50 epochs** on a free Kaggle T4
GPU (41 min; [`kaggle_train_full.py`](kaggle_train_full.py)), selecting on the full official valid split (1,765
images), then evaluated here on the **same 882 test images** (`python evaluate.py --model full`, numbers in
[`docs/results_full.json`](docs/results_full.json)):

| | Laptop CPU, 1,500 images | Kaggle GPU, 6,176 images | Public model (morsetechlab v1n) |
|---|---|---|---|
| mAP50 / mAP50-95 | 0.983 / 0.686 | **0.990 / 0.717** | 0.993 / 0.717 |
| Precision / recall | 0.986 / 0.957 | **0.993 / 0.966** | 0.994 / 0.980 |

Four times the data closes most of the gap to the public detector: mAP50-95 (tighter boxes) now matches it,
recall is still ~1.4 points lower. The gain is small because the CPU model was already at 0.98 mAP50; the
OCR stage is unchanged (it does not depend on the detector), so reading is still the weak link.

### Stage 2: reading (300 UK plates with known registrations)

| Plates | Exact match (raw OCR) | + UK format correction | Character accuracy (corrected) |
|---|---|---|---|
| Clean plate crops | 61.3 % | **89.7 %** | 95.0 % |
| Augmented (angle, blur, lighting, in a scene) | 50.3 % | **80.3 %** | 92.4 % |

![OCR examples](docs/ocr_examples.jpg)

## How it works

1. **Detection**: YOLO11n starts from COCO weights and is fine-tuned on 1,500 plate images
   (640 px, 12 epochs). Transfer learning makes this work on a CPU: mAP50 was already 0.95 after one epoch.
2. **Crop + pad**: each detected plate is cropped with a small margin, upscaled if small, and
   **padded with a border**.
3. **OCR**: RapidOCR (PaddleOCR detection + recognition models running on OpenVINO) reads the text;
   boxes are joined left to right, non-alphanumeric characters and the "UK"/"GB" band are dropped.
4. **Format-aware correction**: current UK plates follow `LLDDLLL` (e.g. `MY69JKI`). With 7
   characters read, positions 3–4 must be digits and the rest letters, so `0→O`, `1→I`, `5→S`… are
   corrected by position. Real ANPR systems do this with each country's plate grammar.

## The bug that mattered

The first OCR run scored **0 % exact match on clean plates but 41 % on augmented ones**, which can't
be right (clean should be easier). Looking at raw OCR output showed why: the clean plates are
**tight crops with no margin**, and the text *detector* (trained on scene text) split them into
single characters (`VW7` · `75` · `G`) and picked up the "UK" band and stray symbols. Padding the
crop fixed it: **0 % → 61 %**, and the format correction took it to **90 %**.

## Data

- **Detection:** "Vehicle Registration Plates" (Augmented Startups, via Roboflow; Hugging Face mirror
  `keremberke/license-plate-object-detection`): 6,176 / 1,765 / 882 train / valid / test images,
  many of them Vietnamese plates. A random 1,500 / 300 subset was used for training / validation (CPU time);
  the full test split for evaluation.
- **Reading:** `spawn99/UK-Car-Plate-VRN-Dataset` (MIT): **synthetic** UK plates with their registration,
  clean and augmented. First 300 of the first shard.
- **Location:** `../../cv-data/plates/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 06-license-plate-recognition
python prepare.py                 # COCO JSON → YOLO subset (short file names)
python train.py --epochs 12       # ~2.4 h on CPU; --resume continues an interrupted run
python evaluate.py                # detection mAP, OCR accuracy, figures
```

The comparison model and the data (PowerShell, from this folder; `prepare.py` expects the zips extracted
into `train/`, `valid/`, `test/` folders under `../../cv-data/plates/`):

```powershell
$d = "..\..\cv-data\plates"; New-Item -ItemType Directory -Force "$d\outputs" | Out-Null
Invoke-WebRequest "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt" -OutFile "$d\outputs\pretrained_plate_v1n.pt"
foreach ($s in "train","valid","test") {
  Invoke-WebRequest "https://huggingface.co/datasets/keremberke/license-plate-object-detection/resolve/main/data/$s.zip" -OutFile "$d\$s.zip"
  New-Item -ItemType Directory -Force "$d\$s" | Out-Null; tar -m -xf "$d\$s.zip" -C "$d\$s"
}
Invoke-WebRequest "https://huggingface.co/datasets/spawn99/UK-Car-Plate-VRN-Dataset/resolve/main/data/train-00000-of-00003.parquet" -OutFile "$d\uk_plates_0.parquet"
```

## What I learned / what didn't work

- **OCR on tight crops needs padding.** Text detectors are trained on text *with surroundings*.
- **Plate grammar is worth ~30 points of exact match.** Most residual errors are O/0, I/1, S/5
  confusions, which positional rules fix for free.
- **The OCR is trained on scene text, not plates.** A recognizer fine-tuned on plate crops (or a
  small CRNN trained on the synthetic set) would likely beat the 80 % on augmented plates.
- **Distant plates fail at the reading stage, not detection** (see the demo figure). A few pixels high is too little for
  any OCR; higher-resolution input or super-resolution would be needed.
- The UK OCR test set is **synthetic**; real UK plates under real cameras would score lower.
  Detection was measured on real photos (mostly Vietnamese plates), reading on synthetic UK plates,
  so end-to-end accuracy isn't directly measured.
- OCR timings (190–350 ms per plate) were measured while another model trained on the same CPU,
  so they're pessimistic; on an idle CPU, project 03 read a whole receipt (dozens of text lines)
  with the same OCR in ~0.7 s.

## Next steps

- Fine-tune a plate-specific text recognizer on the synthetic UK set.
- Measure end to end on a dataset with both boxes and plate text (e.g. CCPD).
- Combine with project 08 (traffic tracking): read each tracked vehicle's plate once, keeping the
  most confident read across frames.

## Tech stack

Python · Ultralytics YOLO11 · RapidOCR (PaddleOCR models) · OpenVINO · OpenCV · pyarrow

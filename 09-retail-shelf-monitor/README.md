# 09 · Retail Shelf Monitor

Detects and **counts every product** on a supermarket shelf photo, where a single image can hold
**100–700+ tightly packed items**, and flags possible empty facings. YOLO11n fine-tuned on a laptop
CPU with only 219 training photos from the SKU-110K benchmark.

![examples](docs/examples.jpg)

*Green: detected products. Red: spots flagged as possibly empty by a simple rule. Many red boxes
are actually missed products, see below.*

## Results

488 test shelf photos (SKU-110K validation shard, never seen in training): **76,083 products**.

### Detection

| Metric | Value |
|---|---|
| **mAP50** | **0.838** |
| **mAP50-95** | **0.487** |
| Precision / recall | 0.842 / 0.766 |
| Inference (CPU, 960 px) | 128 ms per image |

For context, the original SKU-110K paper (Goldman et al., CVPR 2019) reports about 0.49 AP
(0.5:0.95) for its method, trained on ~8,000 images on a GPU. This model trained on **219 images on a CPU**.
Image sizes and protocol differ, so it's a sanity check, not a head-to-head comparison.

### Counting

| Metric | Value |
|---|---|
| Products per image (true) | mean 155.9, max 718 |
| Mean absolute count error | **22.1 products** |
| Mean relative count error | **13.3 %** |
| Bias (predicted − true) | −12.7: it **under-counts** |

![counting](docs/counting.png)

## How it works

1. **Data prep** ([`prepare.py`](prepare.py)): SKU-110K parquet shards (COCO `xywh` boxes) are exported to
   the YOLO layout. Photos (up to 2340×4160) are shrunk to a max side of 1,280 px.
2. **Training** ([`train.py`](train.py)): YOLO11n from COCO weights, **960 px** input (products are small),
   30 epochs (~1.6 h on CPU), and **`max_det=1000`**. YOLO's default cap of 300 detections per image would
   silently cut off a 700-product shelf.
3. **Counting**: number of detections above 0.35 confidence per image, compared with the true box count.
4. **Empty-facing rule** ([`evaluate.py`](evaluate.py)): products are grouped into shelf rows by their
   vertical position; a horizontal gap wider than 1.5× the median product width between neighbors in a
   row is flagged.

## Data

- **SKU-110K** (Goldman et al., CVPR 2019), from the parquet mirror `finedet/sku110k` (CC BY-NC 2.0,
  non-commercial). Used: training shard 8 of 9 (219 images, 31,394 boxes) and the validation shard
  (100 images for validation, 488 for test). The full dataset is ~11,700 images / 13 GB.
- **Location:** `../../cv-data/shelf/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 09-retail-shelf-monitor
python prepare.py
python train.py --epochs 30       # ~1.6 h on CPU; --resume continues an interrupted run
python evaluate.py                # mAP, counting error, examples
```

## What I learned / what didn't work

- **Dense scenes need two settings changed**: a larger input size (small objects) and a higher
  `max_det` (YOLO's default would make counting impossible on large shelves).
- **The empty-facing rule mostly finds missed products.** It flagged gaps in 401 of 488 images: with
  23 % of products undetected (recall 0.77), a missed item looks exactly like a hole. Real gap detection
  needs either much higher recall or labeled "empty facing" examples. SKU-110K has none, so the rule
  stays a demo.
- **Under-counting comes from the same source.** Missed products in crowded or hanging displays
  (left example: 138 of 224) pull the count down; more training data is the obvious fix (8 more shards available).
- **Validation and test agree** (val mAP50 0.849 at the last epoch, from the training log
  `../../cv-data/shelf/outputs/runs/yolo11n_shelf/results.csv`, vs test 0.838), so the 100-image
  validation set wasn't misleading.

## Next steps

- Train on more shards (and longer) to raise recall; then revisit gap detection.
- Label a few hundred empty facings and train a dedicated "gap" class.
- Planogram compliance: match detected positions against the expected layout.

## Tech stack

Python · Ultralytics YOLO11 · pyarrow · OpenCV · NumPy · Matplotlib

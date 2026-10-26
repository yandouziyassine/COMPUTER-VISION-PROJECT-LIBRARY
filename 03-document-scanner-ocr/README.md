# 03 · Document Scanner + Receipt OCR

Turns a phone photo of a receipt into a flat, clean scan and extracts **date** and **total** as
JSON. Page detection and perspective correction use classical OpenCV (no deep learning); text
recognition uses RapidOCR (PaddleOCR models) running through **OpenVINO**, ~0.7 s per receipt on a laptop CPU.

![results](docs/results_chart.png)

> Example photos not shown: the synthetic photos use COCO backgrounds (Flickr, mixed licences).

*Top: synthetic phone photos with the detected page (green). Bottom: scanner output. The two
table photos succeed; the snowy scene fails (white snow merges with the white paper) and in the
last one a shadow edge cuts the receipt in half. Failures are kept in the figure on purpose.*

## How it's evaluated

SROIE receipts are flat scans with ground-truth fields, so realistic **phone photos are
generated** from them: each receipt is warped with a random perspective (±12° rotation plus tilt)
onto a background, with a lighting gradient, a soft shadow, blur, noise and JPEG compression.
Generating the photos means the **true page corners are known**, so the scanner can be scored exactly.

Two settings:
- **Table**: a plain textured surface, the setup scanner apps ask users for.
- **COCO scene (stress test)**: a random COCO photo as background (snow, walls, cars, animals).

## Results

100 SROIE test receipts. Accuracy = exact match of the normalized field with the ground truth.

| | Page found (corner error < 3 %) | Date accuracy | Total accuracy |
|---|---|---|---|
| Clean scan (no photo, upper bound) | — | 86 % | 78 % |
| **Table** photo, raw OCR | — | 78 % | 58 % |
| **Table** photo, after scanner | **68 %** (median error 0.6 %) | 67 % | 59 % |
| COCO-scene photo, raw OCR | — | 75 % | 59 % |
| COCO-scene photo, after scanner | 20 % (median error 10.8 %) | 64 % | 53 % |

**When the page is found** (68 table photos), the scanner raises total accuracy from **57.4 %
to 64.7 %**. When it picks the wrong region, it hurts: the crop loses part of the receipt.

| Speed (CPU) | |
|---|---|
| Page detection + warp + lighting correction | ~60 ms |
| OCR (detection + recognition, OpenVINO) | 0.7–1.2 s (varies with other load on the CPU) |

## How it works

**Scanner** ([`scanner.py`](scanner.py))
1. **Find the paper by color**: paper is bright and colorless, so score = brightness − saturation (HSV).
   Several global thresholds plus **adaptive (local) thresholds** produce candidate masks;
   adaptive thresholds survive lighting gradients and soft shadows.
2. **Pick the best page**: for each blob, fit a quadrilateral (convex hull → polygon
   approximation) and keep the largest blob that fills its quadrilateral almost exactly. Shapes no
   page could have are rejected: touching the image border, concave, corners outside 55–125°, slivers.
3. **Fallback**: Canny edge contours if no color candidate passes.
4. **Warp + clean**: perspective transform to a straight-on rectangle, then divide by a blurred
   background estimate to remove shadows and uneven lighting.

**OCR + fields** ([`receipt.py`](receipt.py))
1. RapidOCR detects and recognizes text boxes.
2. **Deskewed line grouping**: the dominant text slope is measured from the boxes themselves, and
   boxes are grouped into lines along that slope. On a slightly rotated receipt, prices in the right
   column otherwise land on the next line.
3. **Rules**: date regexes (tolerating OCR gluing the time onto the year: `15/01/201911:05AM`) and a
   priority score for total lines: `payable` / `grand total` > `total incl. GST` > plain `total`,
   excluding subtotal, tax-only, quantity, cash and change lines.

## The debugging story (what moved the numbers)

| Change | Measured effect (same receipts before/after) |
|---|---|
| ONNX Runtime → **OpenVINO** backend for RapidOCR | OCR **5.2 s → 0.5 s** per receipt (~11×), same text |
| Edge contours → color + adaptive thresholds + shape checks + "page must not touch the image border" | 6-photo plain-background check: pages found **2/6 → 5/6**. The border rule fixed background regions framed by the photo edge being picked as "pages" |
| Date regex without a trailing word boundary + total-line priority (instead of "skip any line containing GST") | 20-receipt check on clean scans: date **70 % → 95 %**, total **45 % → 65 %** |
| Median-height + **deskewed** line grouping | 100 table photos, raw OCR: total **28 % → 52 %**; clean scans: total 77 % → 79 % |
| Amount parser handles thousands separators (`1,234.00` was read as `234.00`; found in a code review) | Clean-scan total 79 % → 78 %, table raw 52 % → 54 % (a few receipts change either way) |
| Amounts no longer glued across spaces (`3.00 6.00` was read as `3006.00`; found by a unit test) | Final numbers above: table raw 54 % → 58 %, COCO scanned 43 % → 53 %; clean scans unchanged |

## Data

- **Receipts:** [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) test split (361 receipts; first 100 used),
  CC BY 4.0, from the Hugging Face mirror `jsdnrs/ICDAR2019-SROIE`. Fields: company, date, address, total.
- **Backgrounds:** COCO val2017 images (shared with project 16) for the stress test.
- **Location:** `../../cv-data/receipts/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 03-document-scanner-ocr
python evaluate.py --n 100        # ~7 min on CPU
python app.py                     # web demo: photo → scan → JSON
```

## What I learned / what didn't work

- **Classical page detection is fragile on busy backgrounds.** It works when paper contrasts with the
  surface (68 % on a table, sub-1 % corner error when it works) and fails when the background is
  also white (snow, walls) or a strong shadow edge splits the page. A small learned segmentation
  model (e.g. a U-Net trained on synthetic photos like these) would be the next step.
- **The biggest OCR gain came from layout, not vision.** Grouping words into the right lines
  (deskewed) did more for total accuracy than the scanner itself.
- **Scanning slightly lowers date accuracy** (78 % → 67 % on table photos). Likely causes: the
  lighting-correction step thinning small digits, or resolution lost in the warp. Not yet isolated.
- **Rule-based extraction tops out around 80 %** even on clean scans. Layouts vary too much;
  a layout-aware model (LayoutLM, Donut) learns field positions instead of keywords.

## Next steps

- Learned page segmentation for busy backgrounds; ablation of the lighting-correction step for dates.
- Replace rules with a small layout-aware model for field extraction (SROIE train split: 626 receipts).
- Evaluate on real phone photos of receipts.

## Tech stack

Python · OpenCV · RapidOCR (PaddleOCR models) · OpenVINO · pandas · Gradio

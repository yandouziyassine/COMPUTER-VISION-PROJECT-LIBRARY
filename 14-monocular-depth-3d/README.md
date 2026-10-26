# 14 · Monocular Depth: Where Machine 3D Vision Fails

Computers are often said to have trouble seeing in 3D. This project measures **exactly where**. Four depth
models read one ordinary photo and guess how far away every pixel is; I compare their guesses with a depth
sensor on 654 real rooms, break the error down by material (mirrors, screens, windows, walls…), test them on
optical illusions with a known answer, and then use the best model to **measure real distances from a single
photo**, with the accuracy of that measurement reported.

## What you see vs what the computer sees

![what you see vs what the computer sees](docs/see_vs_infer.jpg)

You see a room. The computer receives only a grid of brightness numbers (second column, the 8 × 8 pixels at
the image centre): no depth, no objects. From those numbers alone, the model *infers* depth (third column),
which can be checked against a Kinect depth sensor (fourth, red = error) and turned into 3D (fifth).
Row by row:

1. **An ordinary room**: inference and sensor agree almost perfectly.
2. **A mirror**: the model and the sensor disagree, and *both* are wrong in their own way (see the failure map).
3. **A flat poster** pasted onto a wall: modern models keep it flat, older ones see into it (the poster test).
4. **The Ames room**: a trapezoid room built to look like a box. The model, like a human, sees a box.

## Results

### 1 · Benchmark: 654 NYU Depth V2 test images

Relative models predict depth up to an unknown scale and shift; they are aligned to the sensor per image
(least squares in inverse depth), the standard protocol. AbsRel = mean relative error, δ1 = share of pixels
within 25 % of the truth.

| Model | AbsRel ↓ | δ1 ↑ | CPU time / image |
|---|---|---|---|
| Depth Anything V2 **Base** (97 M) | **0.058** | **95.9 %** | 2.9 s |
| Depth Anything V2 Small (25 M) | 0.062 | 95.6 % | 1.0 s |
| DPT-Hybrid / MiDaS 3.0 (2021) | 0.098 | 90.4 % | 1.6 s |
| Depth Anything V2 Metric-Indoor (meters, no alignment) | 0.219 | 66.8 % | 1.3 s |
| ↳ same, after one scale factor per image | 0.091 | 91.0 % | |

The metric model gives distances in meters, but on NYU it places every room **~27 % too far** (a consistent
scale error, not noise). Monocular metric depth silently assumes a camera: trained on Hypersim's synthetic
camera, it reads NYU's Kinect images at the wrong scale. Part 3 shows how one known length fixes this.

### 2 · Failure map: which surfaces break 3D vision

![depth error by surface](docs/failure_by_class.png)

Error by object class, scored only on pixels the Kinect actually measured (percentages under each label):

| Surface | AbsRel (DA2-Small) | vs walls | Sensor coverage |
|---|---|---|---|
| **Mirror** | **0.316** | **7.2×** | 63 % |
| Computer monitor | 0.087 | 2.0× | 47 % |
| Door | 0.068 | 1.5× | 80 % |
| Ceiling | 0.067 | 1.5× | 92 % |
| Chair | 0.066 | 1.5× | 79 % |
| Window | 0.059 | 1.4× | 45 % |
| Wall | 0.044 | 1.0× | 98 % |
| Floor | 0.041 | 0.9× | 94 % |
| Sofa | 0.039 | 0.9× | 92 % |

- **Mirrors are the one dramatic failure**: the worst surface for all three models (7× the error of a wall for
  Depth Anything V2, 3× for DPT-Hybrid). But the "truth"
  is fooled too: the Kinect's infrared light bounces off the mirror and measures the **reflected room**, a room
  that isn't there. On 87 % of mirror images the model places the mirror **nearer** than the sensor (median
  −26 %), i.e. closer to the glass itself. Neither the sensor nor the model reports "a flat mirror on the wall".
- **The sensor fails where the model doesn't get a chance to be scored.** The Kinect measured only 16 % of
  glass pixels, 45 % of windows and about half of TVs and monitors: transparent and dark-reflective surfaces
  defeat active depth sensors as well. Glass is left out of the chart (1,471 measured pixels in 654 images).
- **Pictures and whiteboards are fine** (0.044 and 0.034: as good as a wall): a flat image hanging on a wall does
  not fool these models. The poster test confirms it with a known answer.
- **Distance and position matter less than material**: error is 0.048 at 2–4 m, 0.056 under 2 m, 0.070 beyond
  4 m, and 14 % higher in the image border band than in the centre.
- **Newer is better everywhere**: DPT-Hybrid has 1.5–2.5× the error of Depth Anything V2 on every class
  except mirrors.

### 3 · The poster test: an illusion with a known answer

![poster test](docs/poster_test.jpg)

48 test images with a large flat wall (plane fitted to the sensor depth, median residual < 2 cm). A COCO photo
is pasted onto the wall as a framed poster, warped with the wall's real perspective. The true depth inside the
poster is still exactly the wall plane; the model is aligned on everything *except* the poster, then its depth
inside the poster is compared with the plane.

| Model | Deviation from the wall, before | With the poster | Worse with the poster |
|---|---|---|---|
| DA2-Small | 2.9 % | 3.0 % | 52 % of images (chance) |
| DA2-Base | 2.6 % | 2.4 % | 40 % |
| **DPT-Hybrid** | 5.0 % | **7.2 %** (+43 %) | **73 %** |

Modern models are **not fooled by a framed picture**: the poster stays on the wall. The 2021 model is: it sees
into the photo and pulls the poster ~4 % toward the camera on average.

### 4 · Real illusions

![illusions](docs/illusions.jpg)

Depth Anything V2 Small on six optical illusions (open-licensed photos, [credits](docs/illusion_credits.md)):

| Illusion | Fooled? |
|---|---|
| Ames room (a trapezoid room built to look rectangular) | **Yes**, like humans: it reconstructs an ordinary box and puts both people at the back wall |
| Trompe-l'œil colonnade painted on a flat wall | **Yes**: a deep corridor opens in the wall |
| 3D street art on the ground (parrots, giant shoes) | **No**: the ground stays a flat plane |
| Painted film crew on a facade | Mostly no: the painted figures stay close to the facade |
| Facing mirrors ("infinite corridor") | Confused: neither the corridor nor flat glass |

The pattern: models trust **strong learned priors** (rooms are boxes, a corridor drawn in perfect perspective is
a corridor) and ignore weak painted cues on the ground. Small framed pictures read as pictures; a whole wall of
paint reads as architecture.

### 5 · Measuring the world from one photo

![measurement error](docs/measure_error.png)

25,506 random point pairs (0.3–3 m apart) on the 654 test images, true distance from the sensor, predicted
distance from the metric model with the true camera intrinsics:

| | Median error | 90 % of measurements within | 0.3–1 m | 1–2 m | 2–3 m |
|---|---|---|---|---|---|
| Metric model only | 17.2 % | 47.7 % | 20.1 % | 16.5 % | 15.5 % |
| **+ one known length in the photo** | **10.4 %** | 42.0 % | 11.1 % | 9.7 % | 10.5 % |

One reference length per photo (a door, an A4 sheet, a credit card) removes the camera scale error and cuts the
typical error from 17 % to 10 %. The long tail (10 % of measurements off by > 40 %) comes from points on
different objects at different depths, where any depth error adds up.

**Try it:** `python measure_app.py`, upload a photo, click two points, get meters with that uncertainty.

## v2 (in progress): 8 models × 5 datasets, reproducible protocol

Stage 1 of the v2 study ([spec](../docs/superpowers/specs/2026-10-24-depth-v2-explain-and-fix-design.md)):
W1 protocol, W2 four new models, W3 four new datasets. Inference ran on a Kaggle T4; every image was scored with
[`protocol.py`](protocol.py) following [`configs/protocol.yaml`](configs/protocol.yaml). Numbers are the mean over
images with a **95 % bootstrap CI** (1,000 resamples of images). Raw data: [`docs/results_v2.json`](docs/results_v2.json).

![v2 results](docs/results_v2.png)

**AbsRel [95 % CI] / δ1**, headline alignment per model kind: relative-disparity models (Depth Anything, DPT) are
aligned by least squares in disparity, Marigold by least squares in depth, metric models are **not aligned** (raw meters).

| Model | NYU (654) | KITTI Eigen (652) | ETH3D (454) | DIODE val (771) | iBims-1 (100) |
|---|---|---|---|---|---|
| DA-V2 Large *(new)* | **0.043** [0.041–0.045] / 97.9 % | **0.080** [0.078–0.082] / 94.4 % | **0.058** [0.049–0.069] / 98.3 % | 0.235 [0.211–0.261] / 76.2 % | **0.043** [0.035–0.053] / 97.9 % |
| DA-V2 Base | 0.046 [0.043–0.048] / 97.7 % | 0.085 [0.082–0.087] / 93.5 % | 0.062 [0.052–0.075] / 98.1 % | 0.239 [0.214–0.266] / 76.0 % | 0.079 [0.038–0.156] / 97.6 % |
| DA-V2 Small | 0.051 [0.049–0.054] / 97.3 % | 0.087 [0.085–0.090] / 92.9 % | 0.074 [0.060–0.090] / 97.3 % | **0.231** [0.209–0.253] / 75.4 % | 0.155 [0.042–0.373] / 96.7 % |
| Marigold LCM, 10 × 1 step *(new)* | 0.060 [0.056–0.063] / 95.8 % | 0.121 [0.118–0.125] / 85.6 % | not run (budget) | 0.282 [0.256–0.308] / 77.0 % | 0.055 [0.047–0.063] / 96.6 % |
| DPT-Hybrid | 0.091 [0.087–0.094] / 91.9 % | 0.118 [0.114–0.122] / 86.4 % | 0.093 [0.082–0.105] / 93.9 % | 0.236 [0.217–0.256] / 73.4 % | 0.101 [0.071–0.148] / 94.0 % |
| DA-V2 Metric-Indoor (meters) | 0.212 [0.202–0.223] / 67.9 % | 0.567 / 0.5 % | 0.345 / 43.6 % | 0.430 / 23.4 % | 0.164 / 78.8 % |

**Metric models outside the main table (reproduction gate failed, see below)**, raw meters / after one median scale per image:

| Model | NYU | KITTI | ETH3D | DIODE | iBims-1 |
|---|---|---|---|---|---|
| Depth Pro *(new)* | 0.092 / 0.050 | 0.145 / 0.066 | 0.365 / 0.067 | 0.382 / 0.252 | 0.166 / 0.046 |
| MoGe-2 ViT-L *(new)* | 0.090 / 0.040 | 0.203 / 0.053 | 0.107 / 0.037 | 0.313 / 0.203 | 0.139 / 0.041 |
| DA-V2 Metric-Indoor | 0.212 / 0.081 | 0.567 / 0.130 | 0.345 / 0.079 | 0.430 / 0.252 | 0.164 / 0.071 |

**Reproduction gate** (one published NYU number per new model, same alignment, tolerance ±0.003 AbsRel):

| Model | Published | Ours | Gap | Outcome |
|---|---|---|---|---|
| DA-V2 Large (DA-V2 paper, Table 2) | 0.045 | 0.0428 | −0.0022 | ✅ pass |
| Marigold LCM 10 × 1 (Marigold journal paper) | 0.058 | 0.0596 | +0.0016 | ✅ pass |
| Depth Pro, metric, no GT focal (MoGe-2 paper, Table B.4) | 0.107 | 0.0919 | −0.015 | ❌ not reproduced |
| MoGe-2, metric, no GT focal (MoGe-2 paper, Table B.4) | 0.0733 | 0.0902 | +0.017 | ❌ not reproduced |

The two metric gaps go in opposite directions, so they are not one systematic error on our side. The likely cause is
MoGe-2's own NYU preprocessing (crop, depth range and image resolution are not specified in the paper); until that
is matched, Depth Pro and MoGe-2 are reported separately and not ranked against the others.

**What the numbers say so far**
- **Scoring NYU on raw sensor depth matters.** v1 scored on the hole-filled depth (DA-V2 Small 0.062); the
  community protocol (Marigold's evaluation code) scores on the raw Kinect depth: 0.051, close to the paper's 0.053.
  The filled depth is interpolation exactly where the sensor failed, so it penalised the model for the sensor's holes.
- **DA-V2 Large is the most accurate relative model on 4 of 5 datasets**, but bigger isn't always better: on DIODE
  all relative models tie (≈ 0.23, CIs overlap) because errors there are dominated by far outdoor laser points.
- **Metric scale is still the weak point**: every metric model loses 2–5× accuracy without a per-image scale
  (e.g. Depth Pro ETH3D 0.365 → 0.067). The indoor metric model is useless outdoors (KITTI 0.567).
- **Wide CIs flag unstable cases**: DA-V2 Small and Base on iBims-1 (CI up to 0.37) come from a few long-corridor
  images where the disparity alignment pushes far pixels to huge depths; iBims-1 has no depth cap in the protocol.

**Checks.** Cached NYU predictions (float16, sha256 in a manifest) re-scored locally match the Kaggle scores to
≤ 2·10⁻⁶ AbsRel for all 8 models. Unit tests cover the YAML reader, crops, every alignment, clipping, bootstrap and
manifest ([`tests/test_14_protocol.py`](../tests/test_14_protocol.py)).

**Not done in this stage** (honest list): Marigold on ETH3D (dropped to stay within the 10 T4-hour budget; it runs
at ~7.5 s/image with a 10-member ensemble); Booster (the monocular test split's ground truth is withheld for the online
benchmark, and the 19 GB labeled archive's server did not answer); Depth Pro / MoGe-2 with the true focal length
(W5); per-material and edge metrics, uncertainty, illusions at scale, fine-tuning, human study (W4–W9).

**Reproduce:** `python kaggle_push_v2.py --name <job> --datasets nyu ibims kitti --commit <sha>` (and `diode`,
`eth3d`), download each job's `scores/`, then `python results_v2.py --runs <folder> --recheck-nyu`.

## How it works

- **Alignment** ([`metrics.py`](metrics.py)): for relative models, fit `s · prediction + t ≈ 1 / depth` by least
  squares on valid pixels (inverse depth, because that's what the networks predict), then compare in depth.
  Evaluation follows the standard NYU protocol: Eigen crop, 0.001–10 m, the official 654-image test split.
- **Per-class scoring** ([`failures.py`](failures.py)): NYU's per-pixel labels (894 classes) select surfaces; only
  *raw* sensor pixels count, because the dataset's hole-filled depth is interpolation exactly where the sensor
  fails.
- **Poster composites** ([`poster_test.py`](poster_test.py)): back-project the wall pixels with the camera
  intrinsics, fit a 3D plane (SVD), build a rectangle on that plane, project its corners (pinhole model) and
  warp the poster in with that homography, so the perspective is physically correct.
- **Measuring** ([`measure.py`](measure.py)): metric depth for both points, back-projection with the camera's
  focal length (EXIF `FocalLengthIn35mmFilm` when present, else a 60° field of view), Euclidean distance.
- **3D views** ([`depth3d.py`](depth3d.py)): back-projection to a colored point cloud and z-buffered re-rendering
  from a rotated camera (the original demo of this project).

## Data and models

| | Source | License |
|---|---|---|
| NYU Depth V2 labeled subset (1,449 RGB-D images, official split) | [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), Silberman et al., ECCV 2012 | research use; evaluated locally, never redistributed |
| Poster content | COCO val2017 (shared with project 16) | CC BY 4.0 |
| Illusion photos | Wikimedia Commons, [credits](docs/illusion_credits.md) | CC BY / BY-SA / CC0 |
| Depth Anything V2 Small | Hugging Face `depth-anything/Depth-Anything-V2-Small-hf` | Apache-2.0 |
| Depth Anything V2 Metric-Indoor Small | Hugging Face `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf` | not stated on the model card (the Small family is Apache-2.0) |
| Depth Anything V2 Base | Hugging Face `depth-anything/Depth-Anything-V2-Base-hf` | CC BY-NC 4.0 |
| DPT-Hybrid | Hugging Face `Intel/dpt-hybrid-midas` | Apache-2.0 |

Data lives in `../../cv-data/depth/` (not committed); predictions are cached there as one `.npy` per model.

## Run it

```bash
.venv\Scripts\activate                         # shared environment, from the repo root
cd 14-monocular-depth-3d
# NYU labeled subset (2.8 GB) + official split into ../../cv-data/depth/nyu/
curl -L -o ../../cv-data/depth/nyu/nyu_depth_v2_labeled.mat http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
curl -L -o ../../cv-data/depth/nyu/splits.mat http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
python benchmark.py        # ~70 min on CPU, resumable: all 4 models on 654 images, cached
python failures.py         # ~10 min: error by surface, distance, region
python measure_eval.py     # ~2 min: distance accuracy
python poster_test.py      # ~15 min: poster illusion test
python illusions.py        # downloads 6 Commons photos, gallery
python see_vs_infer.py     # the "what you see" figure
python measure_app.py      # click-to-measure web app
python depth3d.py --demo   # point clouds and 3D views of any photo
```

## What I learned

- **"Computers can't see 3D" is too broad.** On ordinary indoor surfaces a 25 M-parameter model is within ~6 %
  of a depth sensor, from one photo, in a second on a laptop CPU. The failures are specific and measurable:
  mirrors, screens, strong architectural illusions, and absolute scale.
- **Ground truth has blind spots too.** The depth sensor cannot see glass, half of the windows and screens, and
  measures a fake room inside mirrors. Scoring a model where the sensor itself fails would have measured the
  sensor, not the model; reporting coverage per class is part of the result.
- **Scale is the hardest part of "metric".** The metric model's 27 % error is almost entirely one scale factor
  per image. One known length in the scene is worth more than a bigger model.
- **Illusions need a known answer to be measured.** Real illusion photos make a striking gallery but no number;
  the synthetic poster test turns "is it fooled?" into a measurement, and showed that the 2021 model was fooled
  where 2024 models are not.

## Next steps

- A mirror detector (or a mirror-aware depth model) to flag the one surface that breaks everything.
- Outdoor evaluation (KITTI LiDAR) for the outdoor metric model.
- Correct the metric scale from EXIF focal length when the camera differs from the training camera.
- OpenVINO export (project 19) for real-time depth on the NPU.

## Tech stack

Python · PyTorch · Hugging Face Transformers (Depth Anything V2, DPT) · NumPy · h5py / SciPy (NYU) · OpenCV
(homography) · Open3D · Matplotlib · Gradio

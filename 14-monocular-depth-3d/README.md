# 14 · Human vs Machine 3D Vision: Where Each Fails and Why

People are often told that computers "can't see in 3D". This project measures **where** depth from a single
photo breaks down, **why** it breaks down, and **whether** a model can be fixed, and it compares the failures
with those of human vision. Eight depth models are tested on five datasets with laser or sensor ground truth. Their
errors are broken down by material, camera and illusion, and the same "which point is closer?" questions are
prepared for people.

**Status (2026-10-27).** Everything on the machine side is measured: models, datasets, protocol, focal sweep,
illusions at scale, uncertainty, the mirror fine-tune and the model side of the human study. The **human side is
pending**: the study page and the analysis are ready, but no human answers have been collected yet. No human
numbers below come from this project unless a row says so. Where a row describes human vision, it cites the
vision-science literature.

## The comparison in one table

| Situation | Humans (literature; our study pending) | Depth models (measured here) |
|---|---|---|
| **Absolute size and distance from one photo** | Use familiar size and eye height. Distances are compressed beyond a few metres (Loomis et al. 1992; Cutting & Vishton 1995) | Metric models score AbsRel 0.09–0.37 without help (NYU / ETH3D). Their scale follows the camera they assume: tell Depth Pro the wrong focal length and depth scales with it (×2.2 at half the focal); DA-V2 Metric has no camera input and is off by ×2.07 → ×0.60 across a 4× zoom (W5) |
| **Layout up to scale (which is nearer)** | Pictorial relief is recovered only up to an affine / bas-relief ambiguity (Koenderink et al. 1992, 2001; Belhumeur et al. 1999) | Ordinary surfaces: 90–100 % correct on ordinal pairs (W9). Relative depth within 4–6 % AbsRel of the sensor / laser on NYU and iBims-1 (W3) |
| **Mirrors** | Know a mirror is a surface, but mispredict what a mirror shows (the "Venus effect", Bertamini et al. 2003) | The typical error is **×2.4–3.0** larger on mirrors with distance, texture and lighting held fixed (W4, vs the sensor). Mirrors are the most uncertain surface: 59 % of mirror pixels fall in the top 10 % of model disagreement, vs 10 % elsewhere (W6) |
| **Glass, windows, screens** | Mostly fine. Active depth sensors fail here too (v1: the Kinect measured 16 % of glass pixels) | Error ×1.1–1.25 on glass, ×1.2–1.5 on screens (W4) |
| **Hollow face (concave mask seen as convex)** | Strongly fooled: a face is "seen" convex (Gregory 1970; Hill & Bruce 1993) | **Also fooled** on frontal hollow faces: 7/8 and 8/8 models wrong. Not fooled when the mask is seen at an angle on a slanted board (0/8 wrong) (W9) |
| **Ames room** | Fooled: the room is seen as a box, people as giants or dwarfs (Ittelson 1952) | Not fooled on ordinal pairs (8/8 right): the floor contact points give the answer away (W9). v1: the whole reconstruction is still a box |
| **Anamorphic street art (paint on flat ground)** | Fooled from the intended viewpoint | Parrots: 4/8 models fooled. Boots: 1/8 (W9) |
| **A framed photo on a wall** | See a flat picture (in real life helped by binocular and motion cues) | DA-V2 is essentially not fooled (< 0.5 percentage points (pp) of extra depth deviation). **Depth Pro and MoGe-2 are, by +3–5 pp**, and it is the photo's content that fools them (phase-scrambled pictures fool them 3× less, p < 10⁻⁴). The frame barely matters (W7) |
| **Upside down, gray, blurred** | Shading is read assuming light from above, so upside-down shading flips relief (Ramachandran 1988) | Accuracy moves by ≤ 7 points across these ablations on 76 pairs (CIs overlap) (W9) |
| **Knowing when it's wrong** | Confidence is reported, not measured here | Model disagreement flags errors (Spearman 0.33). Removing the 10 % most uncertain pixels cuts AbsRel from 0.043 to 0.035, vs 0.026 for a perfect ranking (W6) |
| **Cues a single photo lacks** | Real-world depth uses binocular disparity, motion parallax and vergence (Cutting & Vishton 1995) | None of these exist in one photo, for models **or** for people looking at a photo. That is why the human study uses the same photos |

**Fixing the one surface that breaks everything (W8).** I fine-tuned DA-V2 Small to put mirrors on their surface instead of inside the reflected room. It did **not**
halve the mirror error (the target). The best variant, LoRA with a plane target, cut it by 9 % (CI −22 % to +1 %)
while improving everything else by 5 %. Plain fine-tuning on sensor depth made mirrors 62 % *worse*: the sensor
teaches the reflection. Details in section 7.

## 1 · Setup: protocol, models, datasets (W1–W3)

Every number follows one written protocol ([`configs/protocol.yaml`](configs/protocol.yaml), code
[`protocol.py`](protocol.py), unit tests [`tests/test_14_protocol.py`](../tests/test_14_protocol.py)). It covers:
- the official crops and depth caps for each dataset;
- the alignment for each kind of model output: least squares in disparity for Depth Anything / DPT, least squares in
  depth for Marigold, and none (raw meters) for metric models, which are also reported with one median scale per
  image;
- the mean over images, with **95 % bootstrap confidence intervals** (1,000 resamples of images);
- cached predictions, with a sha256 manifest.

Inference ran on a Kaggle T4.

![v2 results](docs/results_v2.png)

**AbsRel [95 % CI] / δ1**, headline alignment per model kind:

| Model | NYU (654) | KITTI Eigen (652) | ETH3D (454) | DIODE val (771) | iBims-1 (100) |
|---|---|---|---|---|---|
| DA-V2 Large | **0.043** [0.041–0.045] / 97.9 % | **0.080** [0.078–0.082] / 94.4 % | **0.058** [0.049–0.069] / 98.3 % | 0.235 [0.211–0.261] / 76.2 % | **0.043** [0.035–0.053] / 97.9 % |
| DA-V2 Base | 0.046 [0.043–0.048] / 97.7 % | 0.085 [0.082–0.087] / 93.5 % | 0.062 [0.052–0.075] / 98.1 % | 0.239 [0.214–0.266] / 76.0 % | 0.079 [0.038–0.156] / 97.6 % |
| DA-V2 Small | 0.051 [0.049–0.054] / 97.3 % | 0.087 [0.085–0.090] / 92.9 % | 0.074 [0.060–0.090] / 97.3 % | **0.231** [0.209–0.253] / 75.4 % | 0.155 [0.042–0.373] / 96.7 % |
| Marigold LCM, 10 × 1 step | 0.060 [0.056–0.063] / 95.8 % | 0.121 [0.118–0.125] / 85.6 % | not run (budget) | 0.282 [0.256–0.308] / 77.0 % | 0.055 [0.047–0.063] / 96.6 % |
| DPT-Hybrid | 0.091 [0.087–0.094] / 91.9 % | 0.118 [0.114–0.122] / 86.4 % | 0.093 [0.082–0.105] / 93.9 % | 0.236 [0.217–0.256] / 73.4 % | 0.101 [0.071–0.148] / 94.0 % |
| DA-V2 Metric-Indoor (raw meters) | 0.212 [0.202–0.223] / 67.9 % | 0.567 / 0.5 % | 0.345 / 43.6 % | 0.430 / 23.4 % | 0.164 / 78.8 % |

Metric models, raw meters / after one median scale per image:

| Model | NYU | KITTI | ETH3D | DIODE | iBims-1 |
|---|---|---|---|---|---|
| Depth Pro | 0.092 / 0.050 | 0.145 / 0.066 | 0.365 / 0.067 | 0.382 / 0.252 | 0.166 / 0.046 |
| MoGe-2 ViT-L | 0.090 / 0.040 | 0.203 / 0.053 | 0.107 / 0.037 | 0.313 / 0.203 | 0.139 / 0.041 |

**Reproduction gate**: each new model must match one published NYU number to within ±0.003 AbsRel.

| Model | Published | Ours | Gap | Outcome |
|---|---|---|---|---|
| DA-V2 Large (DA-V2 paper, Table 2) | 0.045 | 0.0428 | −0.0022 | ✅ pass |
| Marigold LCM 10 × 1 (Marigold journal paper) | 0.058 | 0.0596 | +0.0016 | ✅ pass |
| Depth Pro, metric, no GT focal (MoGe-2 paper, Table B.4) | 0.107 | 0.0919 | −0.015 | ❌ not reproduced |
| MoGe-2, metric, no GT focal (MoGe-2 paper, Table B.4) | 0.0733 | 0.0902 | +0.017 | ❌ not reproduced |

<!--E1-->

- **NYU is now scored on the raw sensor depth**, as Marigold's public evaluation code does. v1 used the hole-filled
  depth, which is interpolation exactly where the sensor failed. That choice alone moved DA-V2 Small from 0.062 to
  0.051 (published: 0.053).
- **The cache is checked.** Cached NYU predictions re-scored locally match the Kaggle scores to ≤ 2·10⁻⁶ AbsRel.
- **Not run:** Booster (the monocular test split's ground truth is withheld, and the 19 GB labeled archive did not
  answer) and Marigold on ETH3D (Kaggle budget).

## 2 · What drives the error (W4)

![attribute factors](docs/w4_attributes.png)

A mixed model fits the log relative error of 141,553 NYU pixels per model. The fixed effects are surface, texture,
thin structure, distance and darkness; each image gets a random intercept
([`w4_attributes.py`](w4_attributes.py), [`docs/w4_attributes.json`](docs/w4_attributes.json)). The factors below
multiply the typical error, all other attributes held fixed:

| Attribute | Relative models (DA-V2 S/B/L, DPT) | Metric models (Depth Pro, MoGe-2) | Marigold |
|---|---|---|---|
| Mirror (vs the sensor, which itself sees the reflection) | ×2.1–3.0 | ×2.6–2.7 | ×2.4 |
| Screen (TV, monitor) | ×1.2–1.5 | ×1.4 | ×1.3 |
| Glass / window | ×1.2–1.25 | ×1.1–1.2 | ×1.1 |
| Beyond 4 m (vs 2–4 m) | ×1.25–1.35 | ×1.1 | ×1.0 |
| Under 2 m (vs 2–4 m) | ×0.94–0.97 | ×1.23–1.24 | ×1.3 |
| Dark (luminance < 15 %) | ×1.10–1.12 | ×1.13 | ×1.12 |
| Low texture (vs high) | ×0.91–0.94 | ×0.88 | ×0.89 |
| Thin structure (< 5 px) | no consistent effect (CIs include 1 for several models) | | |

Material matters more than anything else. Texture-free walls are *easier*, not harder: the hard pixels are
cluttered edges. Relative models lose accuracy far away, while metric and diffusion models lose it close up.

## 3 · Scale and the camera (W5)

![focal sweep](docs/w5_focal.png)

The sweep simulates a longer lens by cropping the centre and enlarging it, and a wider lens by shrinking the photo
into a black border. The scene's depths do not change. Results are on 218 NYU images
([`w5_focal.py`](w5_focal.py), [`docs/w5_focal.json`](docs/w5_focal.json)). Median predicted / true depth:

| Model, focal it uses | ×0.5 | ×0.75 | ×1 | ×1.5 | ×2 |
|---|---|---|---|---|---|
| DA-V2 Metric-Indoor (no camera model) | 2.07 | 1.48 | 1.16 | 0.79 | 0.60 |
| Depth Pro, its own estimate | 1.83 | 1.30 | 0.96 | 0.90 | 0.74 |
| Depth Pro, true focal | 1.08 | 0.99 | 0.95 | 0.99 | 0.99 |
| Depth Pro, wrong focal (the unzoomed camera's) | 2.17 | 1.32 | — | 0.66 | 0.49 |
| MoGe-2, its own estimate | 0.86 | 0.89 | 0.94 | 0.99 | 0.99 |
| MoGe-2, true focal | 0.88 | 0.92 | 0.96 | 1.00 | 1.00 |
| MoGe-2, wrong focal | 1.60 | 1.14 | — | 0.74 | 0.61 |

This explains v1's "27 % too far". A metric model without a camera model reads apparent size as distance: zoom in
2× and everything is 40 % too near. Depth Pro's own focal estimate is fooled by the black border (it estimates
1.77× the true focal length). MoGe-2's estimate is robust (within 3–6 %). Given the true focal length, both are
right to within 1–5 %, except at ×0.5. Given a wrong one, both inherit the error almost one-to-one. Humans face the
same ambiguity in a photo: without a known object size, a photo does not fix its own scale.

## 4 · Does the model know when it's wrong? (W6)

![sparsification](docs/w6_sparsification.png)

Uncertainty is measured as the spread of five aligned relative models, and the prediction scored is DA-V2 Large's
([`w6_uncertainty.py`](w6_uncertainty.py)). Removing the 10 % most uncertain pixels cuts AbsRel from 0.043 to
0.035; a perfect ranking would reach 0.026. AUSE is 0.0135 and the Spearman correlation between uncertainty and
error is 0.33. **Mirrors light up**: median uncertainty is 3× that of other surfaces, and 59 % of mirror pixels are
in the top 10 % (glass 15 %, everything else 10 %). <!--W6E-->

## 5 · Illusions at scale (W7)

![poster ablations](docs/w7_posters.png)

The setup pastes a flat picture on a flat wall with the wall's own perspective, on 121 walls (NYU + iBims-1
planes fitted to the sensor or laser), with 2 × 2 × 2 cue conditions:
- with or without a frame;
- perspective-matched or pasted upright;
- a real photo, or the same photo phase-scrambled (same colours and texture statistics, no objects and no
  perspective).

The illusion score is the extra deviation from the wall plane that the picture causes, in percentage points
([`w7_posters.py`](w7_posters.py), [`docs/w7_posters.json`](docs/w7_posters.json)). The p-values come from paired
Wilcoxon tests over walls.

| Model | Framed photo, matched perspective | Effect of real content vs scrambled | Effect of a frame |
|---|---|---|---|
| DA-V2 S / B / L | +0.3 / −0.1 / +0.1 pp | ≤ +0.4 pp | ≤ 0.2 pp |
| DPT-Hybrid | +0.7 pp | +0.1 pp (p = 0.6) | 0 |
| Marigold LCM (1 sample) | +0.6 pp; **+4.4 pp** with no frame and upright | **+1.9 pp** (p < 10⁻⁴) | the frame *reduces* it by 0.7 pp (p < 10⁻⁴) |
| Depth Pro | **+3.3 pp** | **+2.9 pp** (p < 10⁻⁴) | n.s. |
| MoGe-2 | **+5.1 pp** | **+3.0 pp** (p < 10⁻⁴) | +0.6 pp (p = 0.04) |

DA-V2 treats a picture as part of the wall. The models with the best metric geometry (Depth Pro, MoGe-2) see into
the photo, and it is the photo's objects and perspective that do it, not its colours or texture. A person standing
in the room would not be fooled, because binocular and motion cues expose a flat print. That is precisely what a
single photo removes.

*Fixed on 2026-10-26:* the v1 poster code could flip or rotate the pasted picture, because the sign of the fitted
plane normal is arbitrary. Corners are now ordered in the image, and portrait photos are centre-cropped instead of
squashed. v1's poster numbers predate this fix.

## 6 · Human vs machine study (W9)

![W9 accuracy by category](docs/w9_results.png)

**Pairs.** 76 open-licensed "which point is closer?" pairs make up
[`docs/w9_public_pairs.json`](docs/w9_public_pairs.json) (images in `docs/w9_public_img/`, credits inside):
- normal (20), textureless (16), glass (12) and poster (16) pairs on iBims-1 (CC BY 4.0), answered by the laser
  scan;
- mirror (4) and illusion (8) pairs on Wikimedia Commons photos (Public domain / CC BY / CC BY-SA), whose answers
  follow from certain geometry: a hollow mask is concave, paint on a flat floor lies on the floor, and the Ames
  room's layout is known.

Two mirror pairs were dropped after review because their geometry was not certain. A larger 300-pair NYU set
(including 50 mirror pairs with the mirror surface fitted from the frame) is used for the model side only, because
NYU frames cannot be published.

**Model side** (accuracy, 95 % CI over pairs; the model majority uses the five relative models):

| | normal | textureless | glass | poster | mirror | illusion | all |
|---|---|---|---|---|---|---|---|
| Model majority | 100 % | 100 % | 100 % | 100 % | 100 % (4 pairs) | 62 % [25–88] | 96 % [91–100] |
| Range over the 8 models | 90–100 % | 100 % | 67–100 % | 94–100 % | 100 % | 50–88 % | 87–97 % |

- **Error consistency.** Geirhos et al. 2020 measure it as Cohen's κ of trial-by-trial errors. Between models it
  has a median of κ = 0.52, which shows that models fail on the *same* pairs (κ = 0.71 on illusions). **The interesting question is whether people fail on the same pairs too.**
- **Cue ablations, model side.** Rotating 180°, removing colour or blurring changes model accuracy by ≤ 7 points
  (CIs overlap).

**Human side: pending.** The study page ([`web_study/index.html`](web_study/index.html)) is static and anonymous:
- no name, e-mail or other personal data;
- answers are downloaded as JSON or copied as text;
- half the trials flash the photo for 200 ms and then show a noise mask (fast feed-forward vision, cf. Thorpe et al.
  1996); the other half have unlimited time;
- each trial is shown original, rotated 180°, gray or blurred, in balanced random order.

[`w9_analyze.py`](w9_analyze.py) turns the answer files into:
- human accuracy by category, exposure and transform;
- Fleiss' κ inter-annotator agreement;
- human-vs-model error consistency;
- a "humans win / models win / both fail" breakdown.

Website-game sessions (12 pairs each) are analysed as their own group.

**Protocol for volunteers (10–20 people):**
1. Run `python w9_public.py --build`, then `python w9_public.py --assemble <model answers>`. This writes the bundle
   to `<cv-data>/depth/outputs/w9_public/study/` (`index.html`, `pairs.js`, `img/`), all open-licensed, so it can be
   hosted publicly.
2. Send the link. Each volunteer takes 10–15 minutes and sends back the `answers_<code>.json` file.
3. Put the files in one folder and run `python w9_analyze.py --answers <folder>`.

Sessions with fewer than 250 answers are kept out of the main analysis.

Until real answers exist, this README reports **no human accuracy from this project**, and no human data is
simulated.

**References.** Belhumeur, Kriegman & Yuille (1999), The bas-relief ambiguity, IJCV · Bertamini, Spooner & Hecht
(2003), Naive optics: predicting and perceiving reflections in mirrors, JEP:HPP · Chen, Fu, Yang & Deng (2016),
Single-image depth perception in the wild, NeurIPS · Cutting & Vishton (1995), Perceiving layout and knowing
distances, in *Perception of Space and Motion* · Geirhos, Meding & Wichmann (2020), Beyond accuracy: quantifying
trial-by-trial behaviour of CNNs and humans by measuring error consistency, NeurIPS · Gregory (1970), *The
Intelligent Eye* · Hill & Bruce (1993), Independent effects of lighting, orientation, and stereopsis on the
hollow-face illusion, Perception · Ittelson (1952), *The Ames Demonstrations in Perception* · Koenderink, van Doorn &
Kappers (1992), Surface perception in pictures, Perception & Psychophysics · Koenderink, van Doorn, Kappers & Todd
(2001), Ambiguity and the "mental eye" in pictorial relief, Perception · Loomis, Da Silva, Fujita & Fukusima (1992),
Visual space perception and visually directed action, JEP:HPP · Ramachandran (1988), Perception of shape from
shading, Nature · Thorpe, Fize & Marlot (1996), Speed of processing in the human visual system, Nature.

## 7 · Fixing mirrors (W8)

![W8](docs/w8_mirror.png)

DA-V2 Small was fine-tuned on the NYU labeled training split: 795 images, 89 of them with mirrors, oversampled 4×.
The schedule is 5 epochs with a scale-and-shift-invariant L1 loss in disparity, on a Kaggle T4, about 7 minutes per
variant ([`w8_mirror_finetune.py`](w8_mirror_finetune.py), [`docs/w8_mirror.json`](docs/w8_mirror.json)). The test
set is 41 NYU test mirrors whose frame gives a plane. Results are aligned on non-mirror pixels, and changes are
paired over images with 95 % CIs:

| Variant | Mirror AbsRel vs mirror plane | Change vs pretrained | Mirror planarity (RMS / depth) | All other pixels | Worst class change |
|---|---|---|---|---|---|
| Pretrained | 0.359 | — | 0.012 | 0.051 | — |
| Plain fine-tune (sensor depth everywhere) | 0.580 | **+0.221** [+0.106, +0.332] | 0.036 | 0.053 | +19 % (sofa) |
| Ignore mirror pixels in the loss | 0.387 | +0.028 [−0.016, +0.070] | 0.026 | 0.052 | +16 % (TV) |
| Mirror-plane target | 0.358 | −0.001 [−0.095, +0.064] | 0.019 | 0.053 | +15 % (wall) |
| **Mirror-plane target, LoRA r = 8** | **0.328** | −0.031 [−0.080, +0.004] | 0.015 | **0.048** | +20 % (TV), all other classes ≤ +3 % |

- The sensor's own failure is what a model learns: training on Kinect depth inside mirrors teaches the reflected
  room.
- Removing that signal (ignore) or replacing it with the plane only brings mirrors back to where the pretrained
  model already was. Full fine-tuning on 795 images also costs 5–15 % on walls and ceilings.
- LoRA keeps the pretrained features and is the only variant that improves both mirrors (not significantly) and
  everything else (significantly: −0.003 [−0.004, −0.002]). It still makes TVs 20 % worse, the other dark
  reflective surface.
- **Verdict:** 89 mirror images are not enough to teach "mirror = a flat surface". The target of halving the mirror
  error with ≤ 2 % regression elsewhere was **not met**. The next attempt would need many more mirror examples
  (synthetic, e.g. Hypersim, whose depth is the mirror surface) or an explicit mirror mask as input.

## Threats to validity

- **The ground truth has blind spots.** Kinect depth inside mirrors is the reflected room, and the sensor sees
  through glass. Mirror and glass truth therefore comes from a plane fitted to the frame. A hand check of 12 NYU
  mirrors found 9 clearly right, 2 suspect (mirrors cut by the image border, where the ring is partial) and 1
  uncertain.
- **The alignment choice changes the numbers.** Disparity vs depth least squares can move AbsRel by a factor of 2
  (DA-V2 Small: 0.051 vs 0.13 on NYU). Each model is scored in its own output space, as its authors do.
- **The gate failed for Depth Pro and MoGe-2**, so their numbers here are our protocol's, not the papers'.
- **Licences.** NYU, KITTI, ETH3D and Booster are research / non-commercial. They are evaluated, never displayed or
  redistributed. Every published image is CC BY / CC BY-SA / public domain ([DATA_LICENSES.md](../DATA_LICENSES.md)).
- **Small samples.** The public pair set has 4 mirror and 8 illusion pairs, so its CIs are wide. The NYU set (50
  mirror pairs) is model-only.

## Appendix: v1 results (the original four-model study)

### v1 · 1 · Benchmark: 654 NYU Depth V2 test images

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

### v1 · 2 · Failure map: which surfaces break 3D vision

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

### v1 · 3 · The poster test: an illusion with a known answer

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

### v1 · 4 · Real illusions

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

### v1 · 5 · Measuring the world from one photo

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

## How it works (v1 core)

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
| v2: Depth Anything V2 Large | `depth-anything/Depth-Anything-V2-Large-hf` | CC BY-NC 4.0 |
| v2: Depth Pro | `apple/DepthPro-hf` | Apple ML Research licence (non-commercial) |
| v2: MoGe-2 ViT-L | `Ruicheng/moge-2-vitl` | MIT |
| v2: Marigold Depth LCM v1.0 | `prs-eth/marigold-depth-lcm-v1-0` | Apache-2.0 |
| v2: KITTI Eigen test, ETH3D, DIODE val | Marigold authors' evaluation archives, streamed on Kaggle | CC BY-NC-SA 3.0 / CC BY-NC-SA 4.0 / MIT; evaluated only |
| v2: iBims-1 core | mediaTUM 1455541 (TUM) | CC BY 4.0 (displayed in the public pair set) |
| v2: mirror and illusion photos for W9 | Wikimedia Commons, credits in `docs/w9_public_pairs.json` | Public domain / CC BY / CC BY-SA |

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


**v2 commands** (Kaggle T4 for inference; every job is pushed with `kaggle_push_v2.py --script <job> --commit <sha>`):

| Workstream | Job / script | Local analysis |
|---|---|---|
| W1–W3 benchmark | `kaggle_infer_v2.py` (datasets `nyu ibims kitti`, `diode`, `eth3d`) | `python results_v2.py --runs <outputs> --recheck-nyu` |
| W4 attributes | — (uses the NYU prediction cache) | `python w4_attributes.py --preds <outputs>` |
| W5 focal sweep | `w5_focal.py` | `python w5_focal.py --report w5_focal.json` |
| W6 uncertainty | `w6_uncertainty.py` (Marigold seeds) | `python w6_uncertainty.py --preds <outputs> --marigold <folder>` |
| W7 posters | `w7_posters.py` | `python w7_posters.py --report w7_posters.json` |
| W8 mirror fine-tune | `w8_mirror_finetune.py` | results in `docs/w8_mirror.json` |
| W9 human vs machine | `w9_kaggle_models.py` (+ private Kaggle dataset of the pair images) | `python w9_public.py --build / --assemble`, `python w9_analyze.py [--answers <folder>]` |
| E1 gate check | `e1_moge_official_eval.py` (MoGe's own evaluation code) | `docs/e1_moge_eval.json` |

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

- **Collect the human answers** (10–20 volunteers, protocol in section 6) and fill the human column of the table.
- Match MoGe's data processing exactly so Depth Pro / MoGe-2 can pass the reproduction gate.
- Booster labeled set (specular / transparent materials with masks) if a reachable mirror appears.
- OpenVINO export (project 19) for real-time depth on the NPU.

## Tech stack

Python · PyTorch · Hugging Face Transformers (Depth Anything V2, DPT, Depth Pro) · Diffusers (Marigold) · MoGe ·
statsmodels (mixed models) · SciPy (Wilcoxon) · Kaggle T4 · NumPy · h5py / SciPy (NYU) · OpenCV
(homography) · Open3D · Matplotlib · Gradio

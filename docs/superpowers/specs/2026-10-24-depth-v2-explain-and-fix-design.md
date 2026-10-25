# Project 14 v2: where machine 3D vision fails, why, and can it be fixed

Date: 2026-10-24 · Status: scope B chosen by the owner; written spec awaiting review

## Goal

Turn project 14 from a single-dataset, four-model description into a study a professional CV engineer respects:
**what fails** (measured by material and visual attribute, across several datasets and current models),
**why** (camera/scale assumptions and missing visual cues, tested with controlled experiments),
and **can it be fixed** (a mirror-aware fine-tune with ablations and measured side effects).

## What stays

Everything already published stays valid: NYU benchmark (4 models), failure map, poster test, illusion gallery,
metric measurement, see-vs-infer figure. v2 extends it; no existing number is replaced without a note.

## Workstreams

### W1 · Reproducible protocol (first, everything else depends on it)
- `configs/protocol.yaml`: dataset crops (Eigen for NYU/KITTI), depth caps, evaluation resolution, alignment
  (least squares in disparity for relative models; none and median-scale for metric models).
- Cached predictions per model × dataset with a manifest (file hashes, model revision).
- Bootstrap 95 % confidence intervals (1,000 resamples over images) on every reported metric.
- Gate: for each new model, reproduce one published number within ±0.003 AbsRel (NYU or KITTI) before using it.
  If it can't be reproduced, report the gap instead of the model's results.

### W2 · More models (8 total)
Existing: DA-V2 Small, Base, DPT-Hybrid, DA-V2 Metric-Indoor. Added: **DA-V2 Large, Depth Pro (Apple), MoGe-2,
Marigold (LCM, few steps)**. Inference on the Kaggle T4; evaluation on CPU.

### W3 · More datasets (5 total)
NYU (existing), **iBims-1** (planarity, edge and boundary metrics), **KITTI Eigen test** (outdoor LiDAR),
**ETH3D** and **DIODE val** (indoor + outdoor laser scans), **Booster test split** (specular/transparent surfaces
with per-material masks). Disk budget ≤ 30 GB on `C:\Users\yando\Code\cv-data\depth`.

### W4 · Explain: what drives the error
- **Measured attributes per pixel**: specular/transparent (Booster masks, NYU labels), texture (local gradient
  energy), thin structures (morphological width < 5 px), distance bins, luminance/under-exposure.
- **Mixed-effects regression** of log absolute relative error on those attributes with the image as random effect:
  "transparency multiplies error by k, holding distance and texture fixed", with CIs.
- **iBims-1 edge and planarity metrics** per model (ε_PE, ε_DBE completeness/accuracy, directed near/far errors).

### W5 · Explain: scale and the camera
- Focal sweep: crop/resize to simulate 0.5×–2× focal length; metric models run with no focal, true focal, and
  wrong focal. Plot metric-scale error against focal error per model. Explains v1's "27 % too far".

### W6 · Explain: does the model know when it's wrong?
- Uncertainty from Marigold seed ensembles and cross-model disagreement; sparsification curves (AUSE);
  AbsRel at 90 % coverage; is uncertainty higher on mirrors/glass?

### W7 · Illusions at scale
- Poster test grown to ~1,000 composites over NYU, iBims-1 and ETH3D walls.
- Cue ablations: frame vs no frame, perspective-matched vs mismatched, depth-rich vs flat poster content.
- Paired Wilcoxon tests + bootstrap CIs on the illusion score per model.

### W8 · Fix: mirror-aware fine-tuning
- DA-V2 Small fine-tuned on NYU train + mirror masks (NYU labels; MSD/PMD masks if their licence allows).
- Inside a mirror, the target becomes the **mirror plane** fitted from the frame boundary, not the reflected room.
- Ablations: (a) plain fine-tune, (b) ignore-mask loss, (c) plane-target loss, (d) LoRA vs full fine-tune.
- Report mirror AbsRel and planarity, and the change on every other class. Target: mirror error halved with at most
  2 % regression elsewhere. Only mirrors with a visible boundary are used for plane fitting.

### W9 · Human vs model (from scope C, cheap, the most memorable figure)
- ~300 point pairs ("which point is closer?") over normal, mirror, glass and illusion images; a static web form,
  10–20 volunteers, anonymous, no personal data collected; compare human accuracy, model accuracy and
  inter-annotator agreement.

## Deliverables

- Code: protocol + evaluation package, one script per workstream, `make`-style reproduce commands per figure.
- README rewritten as a short study: question → setup → results (tables with CIs) → explanations → fix →
  threats to validity (sensor ground-truth bias, alignment choice, dataset licences) → limits.
- Website: project 14 page updated with the new figures (sent to the website session).
- Unit tests for metrics, attribute extraction, plane fitting and the focal sweep geometry.

## Compute and order

About 25 Kaggle T4 hours over ~3 weeks: W1 → W2/W3 (inference) → W4–W7 (CPU analysis) → W8 (fine-tune) → W9.
Checkpoint every training epoch (12 h session limit).

## Licences

Non-commercial models/datasets (DA-V2 Base/Large, Booster, KITTI, ETH3D, UniDepth if used) are acceptable for this
non-commercial portfolio and are listed in DATA_LICENSES.md. Each licence and download size marked "to verify" in
the research is checked before download; anything requiring a signed agreement or not allowing this use is dropped.

## Out of scope

Video depth / temporal flicker, Blender trompe-l'œil renders, transparent-object fine-tuning, a technical-report
PDF (scope C items other than W9).

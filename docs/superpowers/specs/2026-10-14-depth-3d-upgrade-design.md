# Project 14 upgrade: where machine 3D vision fails, and measuring the world from one photo

Date: 2026-10-14 · Status: approved by the owner on 2026-10-14, with one addition (Part C below)

## Goal

Turn project 14 from a demo (★) into an in-depth project (★★★) with two measured parts, chosen by the owner:

- **A · Where machine 3D vision fails.** Quantify, on ground-truth depth, *where* monocular depth models go wrong,
  and test them on visual illusions with a known answer.
- **B · Measure the real world from one photo.** Click two points on a photo, get the distance in meters;
  report how accurate that really is.

The existing point-cloud / novel-view code stays as the visual hook.

## Data and models

| What | Source | License / use |
|---|---|---|
| Ground-truth depth + per-pixel object labels | NYU Depth V2 labeled subset: 1,449 indoor RGB-D images (Kinect), 894 object classes | research use; evaluated locally, never redistributed |
| Relative depth models | Depth Anything V2 Small and Base | Apache-2.0 (Small), CC BY-NC 4.0 (Base) |
| Classic baseline | DPT-Hybrid (MiDaS 3.0) | MIT |
| Metric depth (part B) | Depth Anything V2 Metric-Indoor Small | per model card |
| Illusion photos (qualitative figure only) | ~8 Wikimedia Commons images (Ames room, 3D street art, mirrors, trompe-l'œil) | CC licenses, attributed |

Evaluation uses the standard NYU test split (654 images). Everything runs on the laptop CPU; three relative models
× 654 images is roughly one hour.

## Part A · Where it fails

1. **Benchmark harness** (`benchmark.py`): for each model and image, align the relative prediction to ground truth
   with a per-image least-squares **scale and shift** (in disparity space, as in the MiDaS protocol), then compute
   **AbsRel** and **δ < 1.25** on valid pixels. Results per model in `docs/results.json`.
2. **Failure map by material and object** (`failures.py`): use NYU's per-pixel labels to report the error on
   specific classes: *mirror, window, glass, television/monitor, picture/poster* versus *wall, floor, ceiling,
   table*. Hypothesis to test: reflective, transparent and "pictures of things" surfaces have much higher error.
   Also error versus distance (0–2 m, 2–4 m, 4 m+) and versus image region (border vs centre).
3. **The poster test** (synthetic illusion with a known answer): take NYU images with a large flat wall
   (plane fit on ground truth), paste a COCO photo onto the wall as a framed poster with the correct
   perspective (homography). The true depth stays a **plane**. Measure how far the predicted depth inside the
   poster deviates from that plane, compared with the same wall before pasting. A model that "sees into" the
   poster has a high **illusion score**. About 100 composites.
4. **Illusion gallery** (qualitative): predicted depth for the curated illusion photos, side by side with what a
   human perceives, to show the failure modes the numbers describe.

## Part B · Measuring from one photo

1. **Metric depth + focal length**: Depth Anything V2 Metric-Indoor gives depth in meters; the focal length comes
   from EXIF when present, otherwise from an assumed field of view (reported as a source of error).
2. **Distance between two pixels**: back-project both points (pinhole model) and take the 3D distance.
3. **Accuracy on ground truth** (`measure_eval.py`): on the NYU test split, sample random pixel pairs (0.3–3 m
   apart) and compare the predicted distance with the distance computed from true depth: median and 90th
   percentile relative error, by distance range.
4. **One known object fixes the scale**: re-run step 3 after rescaling the depth so one reference measurement is
   exact (simulating "put a credit card / A4 sheet in the photo"). Report how much the error drops.
5. **Measuring app** (`measure_app.py`, Gradio): upload a photo, click two points, get meters, with an
   uncertainty band taken from step 3. Optional: a reference length to calibrate.

## Part C · What you see vs what the computer "sees" (owner request)

A visual section, first in the README, that makes the gap between human and machine 3D perception obvious at a
glance. For each example, one row of panels:

1. **What you see**: the photo.
2. **What the computer sees**: the raw numbers it receives (a zoomed pixel grid with values: no depth, no
   objects, only brightness).
3. **What it infers**: the predicted depth map (near = bright).
4. **The truth** (NYU examples only): the measured depth from the Kinect sensor, plus an error map.
5. **In 3D**: the point cloud re-rendered from a side view, where errors become visible as bent walls or
   posters bulging out of the wall.

Examples are chosen to tell the story: one where the model is right, one mirror or window (failure), one poster
composite from Part A (the illusion), one real illusion photo. Built by `see_vs_infer.py`; the same panels feed
the website card and a short GIF.

## Deliverables

- Code: `benchmark.py`, `failures.py`, `poster_test.py`, `see_vs_infer.py`, `measure.py`, `measure_eval.py`, `measure_app.py`,
  existing `depth3d.py` / `view.py` unchanged.
- Figures: error-by-class bar chart, poster-test examples and illusion-score distribution, illusion gallery,
  measurement-error curve.
- README rewritten around the two questions; `docs/results.json`; unit tests for alignment, metrics,
  back-projection distance and the poster homography; THEORY.md chapter 45 updated with the findings.
- Website card and CV library row updated (depth ★ → ★★★).

## Testing

Unit tests (synthetic arrays, no model download): scale-shift alignment recovers a known scale/shift, AbsRel
and δ match hand-computed values, distance between back-projected points matches a known 3D distance, the pasted
poster stays inside the wall mask.

## Out of scope

Training or fine-tuning a depth model; outdoor metric evaluation (KITTI); mesh reconstruction and inpainting;
real-time NPU depth (possible later, via project 19's tooling).

## Risks

- NYU download is 2.8 GB in one file (fits FAT32's 4 GB limit). If the mirror is unavailable, fall back to the
  Hugging Face copy of the test split (no object labels: the per-class analysis would then use a smaller labeled
  subset).
- Depth Anything V2 Base is CC BY-NC: fine for this non-commercial portfolio, noted in DATA_LICENSES.md.

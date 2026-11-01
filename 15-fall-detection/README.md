# 15 · Fall Detection from Video

Detects when a person **falls** in a video and raises an alarm, while **not** alarming when
they sit, crouch or lie down on purpose. Each frame is reduced to a body skeleton (YOLO11n-pose),
the skeleton to posture and motion features, and a classifier decides whether a fall is happening.

> Example images not shown: the dataset licence (UR Fall Detection, research use only) does not allow redistribution; see the dataset page.

## Results

UR Fall Detection dataset, front camera: **30 falls and 40 daily activities**, 16 of which include
lying down on purpose (the hard negatives). Scored per video: did a fall raise an alarm
(sensitivity), and did a non-fall stay quiet (specificity)?

| Detector | Sensitivity (falls caught) | Specificity (no false alarm) | F1 | Median alarm latency |
|---|---|---|---|---|
| Hand-written rules | 80.0 % (24/30) | 95.0 % (38/40) | 0.857 | 0.95 s |
| Learned (gradient-boosted trees, 5-fold CV), threshold fixed at 0.5 | 100 % (30/30) | 80.0 % (32/40) | 0.882 | 0.60 s |
| **Learned, threshold chosen by nested CV** | **96.7 %** (29/30) | **95.0 %** (38/40) | **0.951** | 0.70 s |

Same split for all three rows (5 folds grouped by video, shuffling seed 0); F1 counts a fall as the positive class.

### Choosing the alarm threshold without peeking (nested CV)

The first version of this project reported the threshold sweep below and noted that 0.9 looked best,
but 0.9 was picked *after* seeing the test results. The threshold is now chosen honestly: inside each
training fold, a second grouped 4-fold CV produces validation probabilities for the training videos
only, the threshold with the best video-level F1 on them is kept (median of the tied values, grid
0.30-0.95), and only then is it applied to the held-out fold. To see how much the numbers depend on
the particular split, the whole outer CV is repeated with 5 shuffling seeds:

| Protocol (gradient-boosted trees) | Sensitivity | Specificity | F1 |
|---|---|---|---|
| Fixed threshold 0.5 (original protocol) | 99.3 % (96.7-100) | 83.5 % (80.0-85.0) | 0.898 (0.882-0.909) |
| **Nested CV threshold** (honest) | **93.3 %** (90.0-96.7) | **96.5 %** (95.0-97.5) | **0.943** (0.931-0.951) |
| Best threshold picked on the test results (optimistic, reference only) | 95.3 % (90.0-100) | 97.0 % (95.0-100) | 0.956 (0.933-0.983) |

*Mean over 5 outer-CV seeds, (min-max) in brackets. 30 falls, 40 daily activities per seed.*

- The nested thresholds land between 0.65 and 0.95 (most often 0.9-0.95): the inner folds agree that
  0.5 is too eager. The honest F1 (0.943) is only 0.013 below the peeked one (0.956), so the earlier
  "~97 % caught, ~0 false alarms at 0.9" was optimistic but not by much.
- The trade-off is real: compared with 0.5, the nested threshold removes about 5 of the 6-8 false
  alarms per run but misses 1-3 of the 30 falls (fall-29 and fall-25 are missed most often), and the
  alarm comes 0.1-0.2 s later.
- `evaluate.py` applies the same rule to the final model: a grouped 5-fold CV over all 70 videos
  picks **0.90**, which is saved in `fall_model.joblib` and used by `detect.py`.

The operating curve on the seed-0 out-of-fold probabilities, for reference (descriptive, not a selection):

| Alarm threshold | Sensitivity | Specificity | F1 | Median latency |
|---|---|---|---|---|
| 0.5 | 100 % | 80.0 % | 0.882 | 0.6 s |
| 0.7 | 96.7 % | 87.5 % | 0.906 | 0.7 s |
| 0.8 | 96.7 % | 92.5 % | 0.935 | 0.7 s |
| 0.9 | 96.7 % | 100 % | 0.983 | 0.7 s |

![probability traces](docs/probability_traces.png)

*A fall (red) drives the probability to ~1.0 and keeps it there. Lying down on purpose (blue) spikes
briefly but doesn't stay above the threshold for 10 consecutive frames (0.33 s), so no alarm.*

## How it works

1. **Pose**: YOLO11n-pose finds the largest person in each frame and their 17 body keypoints.
2. **Features** ([`fall.py`](fall.py)), per frame:
   - *Posture*: torso angle from vertical (shoulders vs hips), bounding-box aspect ratio, head/hip height.
   - *Dynamics*: how fast the head and box top moved down, how fast the torso tilted, and the
     maximum of those over the last 1.5 s; current height relative to the last 2 s.
3. **Rule baseline**: alarm if the body is horizontal (torso > 55° or box wider than tall)
   AND a fast descent happened in the last 1.5 s, for **5** consecutive frames (the learned model
   below uses 10, so the comparison isn't perfectly like-for-like: the rule is given the faster trigger).
4. **Learned model**: gradient-boosted trees predict, per frame, "a fall is happening or just
   happened". Training target = the fall plus the 2 s after it began; every frame of a daily
   activity is negative, *including deliberate lying down*.
5. **Alarm**: probability above the threshold for **10 consecutive frames** (0.33 s). This filters
   one-frame spikes. The threshold is chosen by an inner cross-validation on training videos only
   ([`protocol.py`](protocol.py): alarm rule, video-level scoring and threshold choice, shared by all models).
6. **Evaluation**: 5-fold cross-validation **grouped by video**, so a video's frames are never split
   between training and testing (that would leak and inflate the scores), repeated with 5 shuffling seeds.

## Data

- **UR Fall Detection Dataset** (Kwolek & Kepski, 2014), University of Rzeszów:
  [fenix.ur.edu.pl/~mkepski/ds/uf.html](http://fenix.ur.edu.pl/~mkepski/ds/uf.html). Research use; cite the paper.
  Videos are depth | RGB side by side (640×240); only the RGB half is used. Per-frame labels:
  not lying / falling / lying.
- **Location:** `../../cv-data/fall/` (not committed): the two label files `urfall-cam0-falls.csv` and
  `urfall-cam0-adls.csv`, and the 70 `*-cam0.mp4` videos in `videos/` (94 MB, needed only until the
  skeletons are cached in `../../cv-data/fall/outputs/poses/`; all later runs read the cache).
- **Reproduced after a data loss (2026-10-31):** the original videos and cached skeletons were lost with a
  USB key. Re-downloading the 70 videos and re-extracting the skeletons with the same scripts gave
  **exactly** the published numbers (same 6 missed falls for the rules, same 8 false-alarm videos and the
  same threshold sweep for the learned model).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 15-fall-detection
python evaluate.py                                  # poses (first run 10 min to 1 h on CPU), nested CV x5 seeds, figures, final model
python evaluate.py --seeds 0                        # only the original split (faster)
python detect.py my_video.mp4                       # any video → annotated copy + alarm time
python detect.py 0                                  # live webcam (q to quit)
```

## What I learned / what didn't work

- **Rules are conservative, learning is eager.** The fixed thresholds miss 6 falls (fall-12, 13,
  16, 18, 22, 24) whose descent or final posture stays under the hand-picked limits, while the learned
  model catches every fall but over-fires on 8 daily activities at threshold 0.5. The sweep shows
  those false alarms peak between 0.5 and 0.9, so a higher threshold removes them.
- **Peeking at the test set to pick a threshold cost little here, but it did cost something.** Chosen
  honestly (nested CV), the threshold gives F1 0.943 instead of 0.956, and sensitivity drops from ~97 %
  to ~93 %: with 30 falls, one video is 3.3 points, so every threshold decision is coarse.
- **Dynamics are what separate a fall from lying down.** Posture alone can't: 16 of the 40 daily
  activities end horizontal. The features that look back 1.5 s carry the distinction.
- **30 + 40 videos is small**, recorded indoors by a few actors performing *simulated* falls. Real falls by elderly people are slower and often partial; this needs validation on
  real-world data before anyone relies on it.
- **The webcam mode** scores a rolling 3-second window. It runs on CPU but wasn't tested on live
  falls (for obvious reasons).

## Next steps

- Add the second camera (cam1) and depth: falls seen from the side are easier to judge.
- A temporal model on keypoint sequences (1D-CNN or GRU) instead of hand-crafted window features.

## Tech stack

Python · Ultralytics YOLO11-pose · scikit-learn (HistGradientBoosting, StratifiedGroupKFold) · OpenCV · NumPy

## Run with Docker

CPU-only image (`python:3.10-slim` + CPU PyTorch + Ultralytics) for `detect.py` on a video file. The
`cv-data` folder is mounted at `/data`; it must already contain `fall/outputs/fall_model.joblib` (from
`python evaluate.py`). The pose model is downloaded into the same folder on first use. Commands from this folder:

```bash
docker build -f Dockerfile -t cv-fall ..                # build context = repo root (needs common/)
docker run --rm -v "$(pwd)/../../cv-data:/data" cv-fall \
  /data/fall/videos/fall-01-cam0.mp4 --ur-fall --out /data/fall/outputs/detected_docker.mp4
```

Any other video works too: put it under `../../cv-data/` and pass its `/data/...` path (drop `--ur-fall`
for normal videos). Webcam mode isn't supported in the container. On Windows PowerShell, use `${PWD}`
instead of `$(pwd)`.

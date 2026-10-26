# 15 · Fall Detection from Video

Detects when a person **falls** in a video and raises an alarm, while **not** alarming when
they sit, crouch or lie down on purpose. Each frame is reduced to a body skeleton (YOLO11n-pose),
the skeleton to posture and motion features, and a classifier decides whether a fall is happening.

> Example images not shown: the dataset licence (UR Fall Detection, research use only) does not allow redistribution; see the dataset page.

*Left: a real fall, alarm raised. Right: lying down on purpose, no alarm. The bar at the bottom
is the fall probability. Both are test-time (out-of-fold) predictions.*

## Results

UR Fall Detection dataset, front camera: **30 falls and 40 daily activities**, 16 of which include
lying down on purpose (the hard negatives). Scored per video: did a fall raise an alarm
(sensitivity), and did a non-fall stay quiet (specificity)?

| Detector | Sensitivity (falls caught) | Specificity (no false alarm) | Accuracy | Median alarm latency |
|---|---|---|---|---|
| Hand-written rules | 80.0 % (24/30) | 95.0 % (38/40) | 88.6 % | 0.95 s |
| **Learned** (gradient-boosted trees, 5-fold CV) | **100 %** (30/30) | 80.0 % (32/40) | 88.6 % | **0.60 s** |

The alarm threshold of the learned model sets the trade-off (out-of-fold probabilities):

| Alarm threshold | Sensitivity | Specificity | Median latency |
|---|---|---|---|
| 0.5 | 100 % | 80.0 % | 0.6 s |
| 0.7 | 96.7 % | 87.5 % | 0.7 s |
| 0.8 | 96.7 % | 92.5 % | 0.7 s |
| 0.9 | 96.7 % | 100 % | 0.7 s |

At 0.9 the learned model misses 1 of 30 falls and raises no false alarm. That threshold was
picked after seeing these results, so it's optimistic: a real deployment would choose it on
separate data. The honest headline is the curve: **~97 % of falls caught within ~0.7 s, with false
alarms adjustable from 20 % down to ~0 %.**

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
   one-frame spikes.
6. **Evaluation**: 5-fold cross-validation **grouped by video**, so a video's frames are never split
   between training and testing (that would leak and inflate the scores).

## Data

- **UR Fall Detection Dataset** (Kwolek & Kepski, 2014), University of Rzeszów:
  [fenix.ur.edu.pl/~mkepski/ds/uf.html](http://fenix.ur.edu.pl/~mkepski/ds/uf.html). Research use; cite the paper.
  Videos are depth | RGB side by side (640×240); only the RGB half is used. Per-frame labels:
  not lying / falling / lying.
- **Location:** `../../cv-data/fall/` (not committed). Skeletons are cached in `../../cv-data/fall/outputs/poses/`.

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 15-fall-detection
python evaluate.py                                  # poses (first run ~10 min), CV, figures, final model
python detect.py my_video.mp4                       # any video → annotated copy + alarm time
python detect.py 0                                  # live webcam (q to quit)
```

## What I learned / what didn't work

- **Rules are conservative, learning is eager.** The fixed thresholds miss 6 falls (fall-12, 13,
  16, 18, 22, 24) whose descent or final posture stays under the hand-picked limits, while the learned
  model catches every fall but over-fires on 8 daily activities at threshold 0.5. The sweep shows
  those false alarms peak between 0.5 and 0.9, so a higher threshold removes them.
- **Dynamics are what separate a fall from lying down.** Posture alone can't: 16 of the 40 daily
  activities end horizontal. The features that look back 1.5 s carry the distinction.
- **30 + 40 videos is small**, recorded indoors by a few actors performing *simulated* falls. Real falls by elderly people are slower and often partial; this needs validation on
  real-world data before anyone relies on it.
- **The webcam mode** scores a rolling 3-second window. It runs on CPU but wasn't tested on live
  falls (for obvious reasons).

## Next steps

- Choose the threshold on a separate validation split (nested CV) and report that.
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

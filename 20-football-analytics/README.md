# 20 · Football Analytics (capstone)

Detects **players and the ball** in broadcast football frames, splits players into **two teams by jersey
color without any team labels**, and decides **which team has the ball**. A YOLO11n detector fine-tuned
on a laptop CPU, plus unsupervised color clustering and simple geometry: the pieces of a match-analytics
pipeline, each measured where ground truth exists.

> Example images not shown: the dataset licence (broadcast footage, copyright of the broadcasters) does not allow redistribution; see the dataset page.

*Blue / orange: teams found by jersey color (no labels used). Grey: referee, goalkeeper or unsure.
Yellow circle: ball. Title: team in possession (nearest player to the ball).*

## Results

Test split: 130 broadcast frames (1280×720).

### Detection

| Class | AP50 | AP50-95 |
|---|---|---|
| **player** | **0.980** | 0.709 |
| ball | 0.438 | 0.214 |
| mean | 0.709 | 0.461 |

Inference: 60 ms per frame on CPU (960 px input). Training: 859 frames, 15 epochs, ~2.5 h on CPU.

**Players are essentially solved; the ball is the hard part**: it's a few pixels wide, often blurred
or hidden by players, and appears in the training data ~17× less often than players (864 vs 15,096 boxes).

### Team assignment (unsupervised)

No team labels exist in the dataset, so this is measured indirectly:

| Metric | Value |
|---|---|
| Frames where two teams were found | 121 / 130 |
| Mean silhouette score of the two color clusters | **0.80** (range −1 to 1; > 0.7 = clearly separated) |
| Players set aside as "other" (referee, keeper, unsure) | 18.5 % |

### Ball possession

A team is assigned possession when the ball is detected and a player's feet are within 1.5 player
heights of it: **41 of 130 frames**. The bottleneck is ball detection, not the possession rule.

### Tactical map: from the broadcast view to the pitch

*Tactical-map figure (broadcast frames + top-down view) not shown: the frames are broadcast footage owned by the broadcasters.*

A second model finds **32 pitch landmarks** (corners, penalty and goal boxes, halfway line, centre circle) in
each frame; any 4 or more give a **homography** from the image to the pitch, and each player's feet are projected
onto a top-down map. Held-out test frames (24, never used for training or tuning):

| Metric | Value |
|---|---|
| Frames where a homography could be built | **24 / 24** |
| Landmark error (1920 px wide frames) | median **21 px**, 90th percentile 50 px |
| Position error on the pitch | median **2.5 m**, 90th percentile 6.2 m |
| Same measurement using the *human-labeled* landmarks (floor) | median 0.8 m, 90th percentile 1.8 m |

The position error is measured by building the homography from the *predicted* landmarks, projecting the
*labeled* landmark pixels with it, and comparing with where those landmarks really are on the pitch. The settings
(keep landmarks with confidence ≥ 0.3, RANSAC tolerance 1.5 m) were chosen on the 30 validation frames; picking
them on the test frames would have reported a slightly better, dishonest number.

## v2 · Stage S4 statistics engine

v2 turns this project into a match-analytics system in six stages (design:
[`docs/superpowers/specs/2026-10-24-football-v2-match-analytics-design.md`](../docs/superpowers/specs/2026-10-24-football-v2-match-analytics-design.md)).
Stage S4 is the **statistics engine** ([`stats/`](stats/)): pure Python on pitch coordinates, independent of the
tracking provider. Input = per-frame player positions in metres + team per player + ball position. It is built and
validated *before* the video pipeline (S1–S3) exists, on public data with ground truth, so that we know in advance
which statistics our tracking errors will break.

| Statistic | Method ([`stats/`](stats/)) |
|---|---|
| Distance, speed, sprints | Savitzky–Golay derivative (1 s window, NumPy only), speed capped at 12 m/s, sprint = > 7 m/s for ≥ 1 s |
| Possession % | nearest player within a control radius; the team changes only after 5 control frames in a row (hysteresis) |
| Passes, pass network | control by A → (ball in flight) → control by team-mate B; control runs shorter than N frames ignored; `networkx` graph |
| Formation | average outfield positions in the attacking direction, standardised, Hungarian match to 4-4-2, 4-3-3, 4-2-3-1, 3-5-2, 5-3-2 |
| Compactness, defensive line | convex-hull area of the 10 outfield players; depth of the deepest outfield player |
| PPDA-style pressing | opponent passes in their own 60 % per possession regain in that zone (also computed from event data, for comparison) |

### Validation on Metrica Sports tracking (2 matches, 25 Hz, with event data)

**Pass detection** against the Metrica PASS events (start within 1 s, same passer *and* receiver). Control radius
and minimum control time were tuned on game 1 only; game 2 is held out.

| Match | Precision | Recall | **F1** |
|---|---|---|---|
| Game 1 (tuning) | 0.946 | 0.905 | 0.925 |
| **Game 2 (held out)** | **0.959** | **0.900** | **0.929** (team-level only: 0.937) |

Target was F1 ≥ 0.7 on clean tracks: met. Tuned setting: 0.5 m control radius, 2 frames of control (the edge of
the searched grid; Metrica's ball positions sit very close to the player's feet, which broadcast tracking won't).

Clean full-match values (sanity check): 8,980–9,550 m per player per 90 min, possession 52.8 % / 52.5 % for the
home sides, 764 / 905 passes detected vs 799 / 964 in the events. PPDA-style pressing from tracking vs from events,
per team and half: 3.9 vs 4.1, 4.1 vs 3.7, 7.2 vs 5.6, 8.7 vs 7.9, 5.7 vs 5.9, 6.6 vs 7.5, 8.9 vs 5.2, 5.2 vs 8.0:
same scale, but single halves can be off by up to 4 because a half has only 11–31 regains. Formations found per half
are 4-2-3-1, 3-5-2, 4-4-2, 4-3-3, 5-3-2 (Metrica gives no formation labels, so these are not scored). A few
Metrica frames contain glitches (single-frame jumps of 100+ m/s); the 12 m/s cap is what keeps them out of the
distances.

### What our tracking errors will do to the statistics (noise-injection ablation)

![noise ablation](docs/stats_noise_ablation.png)

*Each 10-minute window of the two matches (20 windows, 2 noise draws each) is corrupted, then every statistic is
compared with the same window's clean value. Bars: mean bias; whiskers: 95 % bootstrap CI over windows.*

The errors injected are the ones measured in v1: **position error median 2.5 m / p90 6.2 m** (log-normal-scaled
2-D Gaussian fitted to those two numbers), changing every frame or smoothly (1 s), either **independent per object**
(pessimistic) or **shared by everything in the frame** (one camera model, + 0.5 m independent; closer to a
homography error); plus missing frames (10 %, 30 %, in ~1 s gaps) and ID swaps between neighbouring players
(1 or 5 per minute).

| Statistic | Camera-level noise only | 10 % missing | 1 ID swap / min | **Combined** (all three) |
|---|---|---|---|---|
| Distance per 90 min | **+83 %** [78, 88] | +0.0 % | +0.4 % | **+81 %** [77, 85] |
| Sprints | +2,400 % | −7 % | 0 % | +1,900 % |
| Mean top speed | +4.6 m/s | 0.0 | +1.6 m/s | +4.6 m/s |
| Possession (home) | −2.8 pp [−5.2, −0.5] | 0.0 pp | −1.3 pp | **+0.2 pp** [−3.2, 3.9] |
| Pass F1 (detector re-tuned for noise) | 0.68 | 0.62 | 0.40 | **0.31** [0.27, 0.34] |
| Formation same as clean | 100 % | 96 % | 49 % | 39 % |
| Compactness | +0.2 % | −6 % | +4 % | −3 % |
| Defensive line | −0.1 m | +0.2 m | −1.4 m | −0.8 m |
| PPDA-style | −63 % | −22 % | −68 % | −84 % |

What this says, before any video is processed:

- **Robust:** possession % (within ±3 pp in every condition), compactness and defensive-line height (a few % /
  about 1 m), and formation under position noise alone (91–100 % agreement).
- **Broken by v1-level position error:** distance, speed and sprints. A 2.5 m error that moves within a second
  *is* 2–3 m/s of fake running. The sweep in `results_stats.json` (`camera_noise_sweep`) gives S3 its target:
  with the error varying over **10 s** instead of 1 s, the distance bias drops from +87 % to **+2.8 %** (2.5 m
  error) and to +0.9 % at 1.25 m; at 3 s it is +20 % / +5.8 %. So S3's homography smoothing matters more than
  the raw accuracy.
- **Broken by identity errors:** formation (a swap moves a player's average position), passes with the right
  passer/receiver, PPDA (needs regains *and* passes). ID swaps also accumulate: over a full match (no re-ID
  correction), pass F1 under the combined errors falls to **0.09** (0.32 at team level). S1–S2 re-identification
  is a precondition for player-level pass statistics.
- Missing frames mostly cost *recall* (passes −28 % at 10 % missing), not bias in the per-player physical stats,
  because gaps under 1 s are interpolated.

![pass network](docs/stats_pass_network.png)

*Left: pass network of the home team (game 1, first half) from clean tracks; arrows = ≥ 3 passes, node size = passes
made, node = average position while in possession. Right: the same half after injecting the combined errors: ID
swaps scramble average positions and most links disappear.*

### Expected goals (xG) on StatsBomb open data

<img src="https://raw.githubusercontent.com/statsbomb/open-data/master/img/SB%20-%20Icon%20Lockup%20-%20Colour%20positive.png" alt="StatsBomb" height="40">

**15,308 shots** from 597 matches (14 men's competition-seasons, 2018–2024, incl. World Cups 2018/2022, Euros
2020/2024, Bundesliga 2023/24, La Liga, Ligue 1, MLS, ISL, AFCON, Copa América). Split **by match**: 3,080 test
shots from 120 matches never seen in training. Features: distance and angle to goal, body part / header, shot type
(open play, free kick, penalty, corner), play pattern, first-time, under pressure, and from the shot's freeze frame
(available for 97 % of shots): defenders in the shooting cone, goalkeeper in the cone, goalkeeper distance, nearest
defender distance.

| Model (held-out test) | Log-loss ↓ | Brier ↓ | ROC AUC |
|---|---|---|---|
| Base rate (11.7 % goals) | 0.354 | 0.101 | 0.500 |
| **Logistic regression** | **0.271** | **0.0773** | 0.819 |
| Gradient boosting | 0.273 | 0.0778 | 0.817 |
| StatsBomb's own xG | 0.270 | 0.0773 | 0.823 |

5-fold grouped cross-validation: logistic regression 0.278 ± 0.007 vs StatsBomb 0.276. Without penalties: 0.263
vs 0.259. Total xG on the test shots: 364 (ours) vs 357 (StatsBomb) for 350 goals.

![xG calibration](docs/stats_xg_calibration.png)

A simple, well-calibrated logistic regression on geometry + freeze-frame features lands within 0.001–0.002
log-loss of StatsBomb's commercial model; gradient boosting doesn't beat it on 12 k training shots. StatsBomb's
model sees things we don't (e.g. the ball's height), and it may have been trained on some of these matches, so the
comparison flatters it if anything.

### Run stage S4

```bash
cd 20-football-analytics
python -m stats.download_data      # Metrica games 1–2 (~160 MB) + StatsBomb shots (a few minutes, keeps only shots.csv)
python -m stats.validate_metrica   # pass F1, clean stats, noise ablation, figures (~6 min on CPU)
python -m stats.xg                 # xG models + calibration figure (~30 s)
```

All numbers are in [`docs/results_stats.json`](docs/results_stats.json). Unit tests: `tests/test_20_stats.py`
(speed/sprints, possession hysteresis, passes on synthetic plays, formation matching, the noise model).

**Remaining v2 stages:** S1 tracking (SoccerNet-Tracking HOTA), S2 team/jersey identity, S3 calibration
smoothing + ball (→ GS-HOTA), S5 action spotting + replays, S6 OpenVINO, demo video and match report.

## How it works

1. **Detection** ([`train.py`](train.py)): YOLO11n from COCO weights, fine-tuned at **960 px** (instead of
   640) so the ball stays more than a couple of pixels wide.
2. **Jersey color** ([`analytics.py`](analytics.py)): for each player, the shirt area (upper-middle of the
   box) is converted to Lab color; **grass-green pixels are masked out** so the pitch doesn't dominate.
3. **Teams**: k-means with 2 clusters on those colors. Players far from both cluster centers (> 2.5× the
   median distance) are labeled "other": typically the referee and goalkeepers, who wear different colors.
4. **Possession**: the player whose feet (bottom-center of the box) are closest to the ball, if close
   enough, gives the possessing team.
5. **Pitch landmarks** ([`train_pitch.py`](train_pitch.py)): YOLO11n-**pose** fine-tuned to output one "pitch"
   object with 32 keypoints (60 epochs, ~1 h on CPU; mosaic augmentation off, since four stitched pitches break
   the geometry). Horizontal flips are allowed because the layout has a known left/right mirror map.
6. **Homography + map** ([`pitch.py`](pitch.py)): `cv2.findHomography` with RANSAC from the confident landmarks
   to their real coordinates (a 120 m × 70 m template), then players' feet and the ball are projected and drawn.

## Data

- **Football players & ball detection** (Roboflow, Hugging Face mirror `keremberke/football-object-detection`,
  CC BY 4.0): 859 / 243 / 130 train / valid / test broadcast frames, classes `football` and `player`.
- **Football pitch keypoints** (Kaggle `danielmachniak/football-pitch-keypoints-detection`, MIT; Bundesliga
  broadcast frames): 222 / 30 / 24 train / valid / test, 32 landmarks per frame. The landmark numbering was
  checked by drawing the labels on training frames and against the dataset's left/right flip map.
- **Metrica Sports sample data** (github.com/metrica-sports/sample-data, games 1–2, tracking + events): no formal
  licence; the repo asks users to use it responsibly and to acknowledge Metrica Sports in public work. Used for
  stage S4 validation, not redistributed.
- **StatsBomb open data** (github.com/statsbomb/open-data), under the StatsBomb Public Data User Agreement
  (research/analysis use, no redistribution, StatsBomb attribution on published analysis). Only the shots are kept.
- **Location:** `../../cv-data/football/` (not committed).

## Run it

```bash
.venv\Scripts\activate            # shared environment, from the repo root
cd 20-football-analytics
python prepare.py                 # zips → YOLO layout
python train.py --epochs 15       # ~2.5 h on CPU; --resume continues an interrupted run
python evaluate.py                # detection mAP, team clustering, possession, figure
kaggle datasets download danielmachniak/football-pitch-keypoints-detection -p ../../cv-data/football/pitch2 --unzip
python train_pitch.py             # ~1 h on CPU
python evaluate_pitch.py          # landmark + pitch-position error, tactical map figure
```

## What I learned / what didn't work

- **Small objects decide the score.** Player AP50 0.98 vs ball 0.44. Higher resolution (1280 px), a
  ball-specific detector, or tracking the ball between frames (it moves smoothly) are the standard fixes.
- **Unsupervised team split works well on clear kits** (silhouette 0.80), but team "A" and "B" are
  assigned **per frame**: the same team can be A in one frame and B in the next. A real system fits the
  team colors once per match, or tracks players across frames.
- **A moving camera needs a learned ruler.** Project 08 calibrates a fixed camera once by hand; a broadcast
  camera pans and zooms, so the pitch landmarks have to be found in every frame. 222 labeled frames were enough
  for a 2.5 m median position error.
- **Errors grow away from the visible landmarks.** The homography is exact where landmarks are and
  extrapolates elsewhere; the 90th-percentile error (6.2 m) comes from points far from any detected landmark,
  e.g. the far touchline when the camera shows one penalty area. Even perfect labels leave 0.8 m (lens
  distortion, labeling precision, and real pitches being smaller than the 120 × 70 m template).
- **Single frames, not video.** Possession and team stats over time need tracking (project 08's ByteTrack)
  on match footage; this dataset is a set of independent frames.

## Next steps

- Smooth the homography over time on match video (it changes slowly), and report distances run per player.
- Track players and ball across frames (ByteTrack) for possession percentages and passes.
- Fix team colors per match (fit on the first frames) so team identities stay stable.

## Tech stack

Python · Ultralytics YOLO11 · OpenCV (Lab color, masking) · scikit-learn (k-means, silhouette, xG models) · NumPy · pandas · SciPy (Hungarian) · networkx · Matplotlib

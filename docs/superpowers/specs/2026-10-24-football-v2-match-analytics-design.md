# Project 20 v2: match statistics from broadcast video

Date: 2026-10-24 · Status: scope C (ambitious) chosen by the owner; written spec awaiting review

## Goal

Turn project 20 (single-frame detection, colour team split, possession, tactical map with 2.5 m median error) into
a full, measured **match-analytics system**: follow every player through a broadcast clip, know their team, jersey
number and pitch position, track the ball, compute match statistics, spot events, and output a demo video and an
auto-generated match report. Every component is scored on a public benchmark or validated against real data.

## Headline benchmark

**SoccerNet Game State Reconstruction (GSR)**, metric GS-HOTA (pitch position within 5 m + role/team/jersey).
Reference: official baseline 23.4; 2024 entries 34–43; winner 63.8. Target: **≥ 35** on the test split, tuned only
on validation and reported once.

## Stages (each one ships on its own, with its own numbers)

### S1 · Tracking foundation
- Detector: YOLO11 retrained on SoccerNet-GSR boxes at 1280 px on Kaggle T4 (players, goalkeepers, referees,
  ball as its own class); tiled ball detector (SAHI-style) for small balls.
- Trackers: ByteTrack vs BoT-SORT with camera-motion compensation, with and without appearance re-ID.
- Scored on SoccerNet-Tracking: HOTA / DetA / AssA / IDF1, with ground-truth and with our own detections.
- Ablation table: tracker × re-ID × resolution.

### S2 · Identity: team, role, jersey number
- Team: SigLIP crop embeddings → UMAP → k-means(2), fitted per clip, majority vote per track; compared with v1's
  Lab colour per frame. Team accuracy measured on GSR labels.
- Jersey number: torso crop from pose → scene-text recognition (PARSeq or similar), confidence-weighted vote per
  track. Accuracy on GSR jersey labels.

### S3 · Pitch and ball over time
- Calibration: v1 landmark model + homography smoothing (One-Euro/Kalman on parameters) and keypoint tracking
  between frames with reset on shot cuts. Frame-to-frame jitter (m) and GS-HOTA with/without smoothing.
- Ball: detection → gated constant-velocity Kalman filter → interpolation over gaps ≤ 10 frames. Metric: share of
  frames with the ball within 2 m of ground truth. Target ≥ 60 %.
- Full GSR submission through the official `sn-gamestate` / TrackLab format → **GS-HOTA**.

### S4 · Statistics engine (on pitch coordinates)
- Distance, speed (Savitzky–Golay, capped 12 m/s), sprints (> 7 m/s for ≥ 1 s), possession % with hysteresis,
  passes (possession change between team-mates with ball in flight) and pass networks, formation (Hungarian match
  to template formations), compactness (convex hull), defensive-line height, PPDA-style pressing.
- **Validation without matching video:** Metrica Sports sample tracking (and SkillCorner open data if its licence
  allows): inject our measured errors (position noise, ID swaps, missing frames) into clean tracks and measure the
  bias of each statistic; pass-detection F1 against Metrica event files (target ≥ 0.7).
- **xG model** trained on StatsBomb open data (distance, angle, body part, 360 defenders when available); log-loss
  and Brier score compared with StatsBomb's own xG.

### S5 · Events and broadcast structure
- Action spotting (goal, shot, corner, …) on SoccerNet's **precomputed features** (no video download), compared with
  the NetVLAD++ baseline average-mAP. Detected shots get an xG value from S4.
- Shot-change and replay detection so replays don't corrupt the statistics.
- Re-identification across camera cuts using jersey number + team + appearance.

### S6 · Deployment and presentation
- OpenVINO INT8 export of the detector (reusing project 19's tooling); measured FPS on the laptop's CPU, iGPU and NPU.
- 2–3 minute demo video: broadcast view with IDs, team colours and numbers, live minimap and stat panels.
- Auto-generated match report (HTML + PDF): heatmaps, pass network, distance/speed table, possession timeline,
  xG timeline, detected events.
- README written as a study: benchmark numbers vs published entries, ablations, confidence intervals over clips,
  failure gallery (camera cuts, close-ups, ID swaps, ball occlusion), and the limit that broadcast video only partly
  shows the pitch.

## Data and licences

SoccerNet-GSR, SoccerNet-Tracking, SoccerNet-Calibration (research use, downloaded with the `SoccerNet` package,
fetched selectively to fit the disk), SoccerNet action-spotting features (labels and features without the video NDA;
if the features also require the NDA form, the owner fills it in himself), Metrica sample data, StatsBomb open data
(user agreement + attribution), SkillCorner open data and SportsMOT only if their licences allow this use. Every
licence and download size is verified before download and recorded in DATA_LICENSES.md; nothing is redistributed.

## Compute, disk, calendar

- Kaggle T4: about 100–130 hours over ~5 weeks of free quota (S1 detector training, S2 OCR, S3 GSR inference, S5).
- Disk: ≤ 25 GB on `C:\Users\yando\Code\cv-data\football`; heavy video stays on Kaggle where possible.
- Calendar: about 12 weeks; stages in order S1 → S6. Each stage ends with committed results and a README update.

## Risks

- GSR/TrackLab dependencies on Kaggle; mitigated by pinning versions and checkpointing (12 h session limit).
- Jersey OCR is the hardest component; if it stalls, GS-HOTA is reported with and without jersey numbers.
- Scope creep: S5 starts only after S3's GS-HOTA is submitted; S6's OpenVINO step is small and can move earlier.

## Out of scope

Live streams, multi-camera rigs, paid data (Wyscout/Opta/SkillCorner commercial feeds), full 90-minute processing
in real time.

# Football AI Coach: design

Date: 2026-11-01 · Project 20 (football analytics) · Owner-approved design (chat, 2026-11-01)

## Goal

An AI coach for football that works in two ways:

1. **Replay mode** (first): a full real match replayed on a 2D pitch in the browser, switchable to 3D at any moment
   (stadium, tactical top view, or the eyes of any player). Pause at any moment: the coach ranks the ball carrier's
   options. The viewer follows or refuses the advice, the match resumes, and the page shows what the professional did
   and how it ended.
2. **Live camera mode** (later): a phone or webcam filming a match on a TV. The browser detects the players and the
   ball, maps them onto the 2D pitch, and the same coach gives advice in real time.

It must be a serious project: models trained and evaluated on held-out matches, honest numbers, stated limits.

## Data (licence-checked)

| Dataset | What | Licence | Use |
|---|---|---|---|
| IDSSE (Bassek et al. 2025, *Scientific Data*, DOI 10.1038/s41597-025-04505-y; data DOI 10.6084/m9.figshare.28196177) | 7 full Bundesliga 2022/23 matches (1st + 2nd division), TRACAB optical tracking of all players + ball at 25 fps, synchronised official events, metadata | CC BY 4.0 (data owner DFL, published with permission) | Training and evaluation; the replay match shown on the site |
| SkillCorner open data (github.com/SkillCorner/opendata) | 10 A-League 2024/25 matches, broadcast tracking at 10 fps, dynamic events incl. passing options | MIT | Cross-league test only (broadcast tracking: partial, noisier) |
| Existing project 20 assets | YOLO11 player/ball detector, pitch keypoint model + homography (`pitch.py`) trained on research-only broadcast data | Weights are ours; training footage is not redistributable | Live camera mode only; footage never published |

Credits on every public page that shows the data: "Bundesliga tracking and event data: Bassek et al. (2025),
DFL / Sportec Solutions, CC BY 4.0". Loaded with kloppy (`sportec.load_open_tracking_data`,
`sportec.load_open_event_data`). No broadcast video is ever published.

## Architecture

```
            ┌─────────── Replay mode ───────────┐        ┌──────── Live camera mode ────────┐
            │ IDSSE match → frames (10 fps,     │        │ camera → YOLO11n (players, ball)  │
            │ 22 players + ball, metres)        │        │ → pitch keypoints → homography    │
            └───────────────┬───────────────────┘        │ → team by shirt colour → frames   │
                            │                            └───────────────┬───────────────────┘
                            ▼                                            ▼
                 ┌──────────────────────── Coach engine ─────────────────────────┐
                 │ ball carrier → options (passes, carry, shot)                  │
                 │ pitch control · pass success · value · ranking                │
                 └───────────────────────────────┬───────────────────────────────┘
                                                 ▼
                     2D pitch / 3D view · advice arrows with % · follow / refuse
```

### Units

| Unit | Where | Does | Depends on |
|---|---|---|---|
| `coach/data.py` | library, project 20 | Load IDSSE / SkillCorner with kloppy; one frame format: `t, ball (x, y), players [(id, team, x, y, vx, vy)]` in metres, home attacks left-to-right; possession and pass labels from the events | kloppy |
| `coach/pitch_control.py` | library | Spearman (2018) physics pitch control on a grid; time-to-intercept per player | numpy |
| `coach/pass_model.py` | library | Pass success probability per (passer, target): features = distance, angle, defenders near the lane, pitch control at the target, receiver speed; logistic regression + gradient boosting | scikit-learn |
| `coach/value_model.py` | library | Value of a ball position: P(shot or goal by the team in possession within 10 s), trained on IDSSE possessions (grid + location features) | scikit-learn |
| `coach/advice.py` | library | Ranks options: score = P(success) × value(target) − (1 − P(success)) × turnover cost(target); shot option uses the project's xG model | the three models above |
| `coach/export.py` | library | Exports model weights as JSON and one match as a compact binary (int16 cm, 10 fps; about 5 MB per match) for the site | |
| `coach/evaluate.py` | library | All numbers in the README (below) | |
| `src/lib/coach/*.ts` | site | TypeScript port of pitch control, pass model, value model and ranking (from the exported JSON); tested for parity against Python outputs on fixed frames | |
| `src/components/FootballReplay.astro` | site | 2D canvas pitch, timeline, speed, pause → advice, follow / refuse, score tally | coach lib |
| `src/components/Football3D` | site | three.js view: stadium, tactical, player's eyes (click a player); lazy-loaded on first switch | three |
| `src/pages/demos/football/` | site | Demo page + credits | |
| `coach/live/` + site live page | both, phase 4 | Browser CV pipeline: ONNX Runtime Web YOLO11n, pitch keypoint model, homography, shirt-colour teams, feed to the engine | project 20 detector + pitch model |

The engine is pure functions of a frame, so the same code serves replay and live camera.

## Replay experience

- Full match, 10 fps, play / pause / speed (1×, 2×, 4×), timeline with goals and shots marked.
- **Pause** (button or Space): if a player has the ball, arrows to the top options with "pass to #8 · 74 % arrives ·
  +0.03 danger" labels; the best option highlighted. Other options listed below.
- **Follow / Refuse**: the viewer clicks the advised option or picks another. On resume the page shows what the pro did
  and the outcome over the next 10 s (kept, lost, shot, goal). Running tally: viewer vs coach vs pros, by expected
  value of the chosen option.
- **3D toggle at any moment** (same timestamp): stadium camera, tactical top view, or a player's eyes (click a player;
  camera at head height looking where he moves). Players are simple low-poly figures with team colours and numbers.
- Works on a phone in 2D; 3D needs WebGL and is loaded only when asked for.

## Live camera experience (phase 4)

- Camera permission → live overlay on the video: detected players with team colours, the 2D mini-map, and the coach's
  current advice (refreshed about every 0.5 s), each with a confidence label.
- Stated limits on screen: only players inside the TV frame are known (often a third of the pitch), so pass advice is
  limited to visible teammates; glare, camera cuts and replays reset the tracking; the coach is silent when the pitch
  mapping is not confident.
- Runs on the device; nothing is uploaded (same promise as the painter demo).

## Evaluation (README numbers, all on held-out matches)

- **Pass success model**: leave-one-match-out over the 7 IDSSE matches: log-loss, Brier, AUC, reliability curve;
  baselines = distance-only logistic and pitch-control-at-target only. Cross-league: trained on IDSSE, tested on
  SkillCorner passes.
- **Value model**: leave-one-match-out log-loss / Brier vs the base rate and vs a location-only grid (xT-style).
- **Advice sanity**: how often the pro's actual choice is in the coach's top-1 / top-3, and the success rate of passes
  the coach ranked high vs low (the coach should not simply copy pros; the README says so).
- **Live camera** (phase 4): detection and homography accuracy on our local test clips (numbers only, no footage),
  advice latency and frame rate on a laptop and a phone.
- Python ↔ TypeScript parity test: same frames, same option ranking, probabilities within 1e-6.

## Phases (each ships on its own)

1. **Engine + models** in the library with the evaluation above → README section "v3 · AI coach", charts only.
2. **Replay 2D** on the site: one IDSSE match, advice, follow / refuse, tally. Demo card + project 20 "Live demo" link.
3. **3D view** toggle.
4. **Live camera mode**.

## Out of scope

- Publishing any broadcast footage or frames, or non-permissive dataset images.
- Betting, player ratings or scouting claims.
- Off-ball runs and pressing advice (possible later from the same pitch-control model).

## Risks

- Seven matches is small for the value model → keep it simple (grid + few features), report uncertainty across folds.
- IDSSE event ↔ tracking sync quality is unknown until loaded → phase 1 starts with a sync audit (pass start frame vs
  ball movement).
- Browser performance of pitch control at 10 fps → compute only when paused (replay) or at 2 Hz (live).
- Live mode depends on the broadcast camera; if pitch mapping is unreliable on TV footage filmed by a phone, phase 4
  reports that honestly and keeps the overlay detection-only.

# Coach benchmark: our pass values vs expected threat (xT) and VAEP

Project 20, coach insight C (design: section C of
[`docs/superpowers/specs/2026-11-02-coach-insight-design.md`](../../docs/superpowers/specs/2026-11-02-coach-insight-design.md)).
Code: [`coach/xt.py`](../coach/xt.py), [`coach/vaep.py`](../coach/vaep.py), [`coach/benchmark.py`](../coach/benchmark.py).
Numbers: [`coach_benchmark_results.json`](coach_benchmark_results.json). Tests: `tests/test_20_benchmark.py` (14,
synthetic data).

**Question.** Our coach values each pass with three small models (pass completion, position value, shot) on one
tracking frame. Two well-known open methods value the same actions from event data alone. Do they agree, and which
one tracks what actually happened next?

**Short answer.** On 3,464 held-out Bundesliga passes, the coach's position value and xT rank completed passes in a
similar way (Spearman 0.66). VAEP agrees with neither (0.25 or less; with the coach's risk-aware score: −0.02). The
value *after* the pass predicts the next 15 s best for xT (Spearman 0.28 on completed passes) and the coach (0.26), then VAEP (0.20).
On completed passes xT is ahead of the coach by 0.017 [0.006, 0.028], and both are ahead of VAEP. Every method is
weak: the best explains little of what happens next. With 7 matches and 19 goals, VAEP is data-starved. Its
"concede" model never beat a constant. The coach is the only one that is shipped and runs in the browser on a live
frame.

## Data

7 full Bundesliga 2022/23 matches with tracking and official events (Bassek et al. 2025, DFL / Sportec Solutions,
CC BY 4.0). Every model is fitted **without the match it scores** (leave-one-match-out; asserted in the code).

**Action table** (`vaep.actions`, SPADL-like: type, result, start / end x-y with the acting team attacking +x, team,
player, time). The IDSSE events contain **passes and shots only**: no duels, tackles, interceptions, clearances or
dribbles. So the table holds:

| Action type | Count (7 matches) | How it is built |
|---|---|---|
| pass / cross | 5,241 / 140 | event start; a completed pass ends where the receiver stands when the ball reaches him (tracking, within 6 s); otherwise, and for a failed pass, at the start of the next action |
| set pieces (throw-in, free kick, corner, goal kick, kick-off) | 305 / 168 / 57 / 119 / 29 | as passes |
| carry | 1,756 | from the tracking: after a completed pass, the receiver's next action 3–60 m from where he received the ball, within 10 s (the way socceraction adds dribbles) |
| shot (open play / free kick / penalty) | 164 / 4 / 3 | ends at the goal centre; result = goal (19 goals) |

**Passes compared.** The 3,464 passes the match report scores (synced open-play passes, 66 % of 5,241), 2,934
of them completed. For each one, the four values below, the net xG of the passing team in the next 15 s (provider
xG of all shots, for minus against) and whether it scored within 15 s (28 passes) or within the next 10 actions
(VAEP's own label).

## Methods

| Value of a pass | Built from | Knows the result? |
|---|---|---|
| **coach_score**: p · V(target) − (1 − p) · V_opp(loss point) − V(now) | coach: pass model, value model, one tracking frame | no (ex ante) |
| **coach_dv**: V(target) − V(start) | coach value model (ridge on goal geometry) | no |
| **xt**: xT(end) − xT(start) if completed, −xT(start) if lost | xT grid | yes (ex post) |
| **vaep**: ΔP(score) − ΔP(concede) | VAEP classifiers on the last 3 actions | yes |

These are *changes*. The outcome (net xG in the next 15 s) is a *level*: it is high after any pass played near the
goal. So the calibration also scores like-for-like **values after the pass**: `coach_after` = coach_score + V(now)
(the coach's expected net xG after the decision), `xt_after` = xT where the ball ended (0 if lost), `vaep_after` =
P(score) − P(concede) after the pass.

**xT** (Singh 2019), [`coach/xt.py`](../coach/xt.py). 16 × 12 grid. Per cell: shot and move probabilities, the
scoring probability of a shot, and the move transition matrix (completed moves; failed moves count in the
denominator). Value iteration from 0 until the largest change is < 1e-6 (40–44 iterations per fold). Moves are
passes of every kind and carries; penalties are out. **Deviation:** Singh fitted a season. Here 6 matches give ~140
non-penalty shots, so the goal ratio is 0 or 1 in most cells. The scoring probability is the mean provider xG of the
cell's shots, shrunk towards the overall mean with 2 pseudo-shots (`goal_source="goals"` gives Singh's version).
Fitted on the 6 other matches for each held-out match. The grid rises from 0.006 in the own half to 0.31 in the
central cell at the goal (grid fitted on all 7 matches).

**VAEP** (Decroos et al. 2019), [`coach/vaep.py`](../coach/vaep.py). Game state = the action and the two before it
(type one-hot, result, start / end, movement, distance and angle to goal, all seen from the acting team, so an
opponent's action is rotated 180°), plus same-team flags, time gaps, the gap between consecutive actions, the score
and the time. 80 features. Labels: the acting team scores / concedes within the next 10 actions of the half. Value =
(P_score(aᵢ) − P_score(aᵢ₋₁)) − (P_concede(aᵢ) − P_concede(aᵢ₋₁)). The previous pair is swapped when aᵢ₋₁ was by
the other team, and is 0 at the start of a half or after a goal.

The coach's default booster (300 trees, depth 3) **overfit**: held-out log-loss 0.110 vs 0.094 for a constant on
"scores", and 0.048 vs 0.030 on "concedes". So each classifier is now chosen per held-out match, by an inner
leave-one-match-out over the 6 training matches only. The candidates are boosters with 300 / 50 / 30 trees (leaves
of ≥ 20 / 200 / 400 actions) and the base rate. Chosen: "scores" = 30 trees in 5 folds, 50 trees in 1, base rate
in 1. "Concedes" = **base rate in all 7** (38 positive labels: nothing to learn). Held-out "scores": log-loss 0.0931
vs 0.0943 for a constant, AUC 0.59. So VAEP here is in effect an offensive model plus the swap term.

## Results (held-out, 95 % intervals from resampling whole matches, 1,000 resamples)

**Agreement between methods** (Spearman on the same passes):

| Pair | All passes (n = 3,464) | Completed passes |
|---|---|---|
| coach_score vs coach_dv | 0.58 [0.55, 0.61] | 0.67 [0.64, 0.69] |
| coach_dv vs xT | 0.31 [0.27, 0.36] | **0.66** [0.64, 0.69] |
| coach_score vs xT | 0.31 [0.29, 0.33] | 0.37 [0.35, 0.39] |
| coach_dv vs VAEP | 0.23 [0.17, 0.29] | 0.25 [0.16, 0.34] |
| xT vs VAEP | 0.16 [0.08, 0.24] | 0.25 [0.17, 0.31] |
| coach_score vs VAEP | −0.02 [−0.06, 0.03] | −0.02 [−0.07, 0.04] |

The coach's value model and xT are both "how good is this place" maps, and on completed passes they mostly agree.
On all passes they agree less, because xT charges a lost pass −xT(start) and the coach's V(target) − V(start) does
not see the loss. The coach's score adds the risk (completion probability) and the shot option. It shares only a
third of its ranking with xT, and nothing with VAEP.

**Calibration against what happened** (Spearman with the net xG in the next 15 s; AUC for "the passing team
scores within 15 s", 28 passes):

| Value | Spearman, all | Spearman, completed | AUC goal ≤ 15 s, all |
|---|---|---|---|
| coach_score | −0.07 [−0.10, −0.05] | −0.08 [−0.12, −0.05] | 0.44 [0.27, 0.55] |
| coach_dv | 0.02 [−0.00, 0.05] | 0.05 [0.02, 0.07] | 0.56 [0.50, 0.60] |
| xT | 0.09 [0.07, 0.11] | 0.08 [0.06, 0.10] | 0.62 [0.48, 0.67] |
| VAEP | 0.06 [0.02, 0.09] | 0.07 [0.03, 0.12] | 0.57 [0.49, 0.72] |
| **coach_after** (score + V(now)) | 0.24 [0.22, 0.26] | 0.26 [0.23, 0.30] | 0.74 [0.71, 0.80] |
| **xt_after** (xT where the ball ended) | **0.25** [0.22, 0.28] | **0.28** [0.26, 0.31] | **0.77** [0.71, 0.85] |
| **vaep_after** (P_score − P_concede) | 0.18 [0.13, 0.23] | 0.20 [0.14, 0.25] | 0.66 [0.54, 0.79] |

Paired differences (same passes, same resamples):

| Difference in Spearman with net xG | All passes | Completed passes |
|---|---|---|
| coach_after − xt_after | −0.013 [−0.041, +0.012] | −0.017 [−0.028, −0.006] |
| coach_after − vaep_after | +0.055 [+0.009, +0.111] | +0.068 [+0.024, +0.122] |
| xt_after − vaep_after | +0.068 [+0.027, +0.115] | +0.086 [+0.044, +0.140] |
| coach_dv − xt | −0.069 [−0.107, −0.038] | −0.036 [−0.073, −0.009] |

<img src="coach_benchmark.png" alt="Left: Spearman agreement between the four pass values on completed passes, with 95 % intervals. Middle: mean net xG in the next 15 s by decile of each pass value. Right: the same by decile of each method's value after the pass" width="100%">

- **As changes, all four values say little about the next 15 s** (|Spearman| ≤ 0.09). The coach's score is even
  slightly negative. It subtracts V(now), so passes played from dangerous places score low, and those are exactly
  the passes followed by shots. Consistency check: both coach numbers (−0.069 as a change, 0.237 as a level) are
  identical to the earlier held-out advice evaluation (`docs/coach_results.json`).
- **As levels, xT and the coach are close, and both beat VAEP.** On all passes xT and the coach cannot be told apart
  (−0.013 [−0.041, +0.012]). On completed passes xT is slightly ahead (0.017 [0.006, 0.028]). Both beat VAEP by
  0.05–0.09. For all three, the top decile of the after-value is followed by 0.025–0.028 net xG, the bottom half by
  ≤ 0.004. The coach's after-value is in net-xG units, and its slope of 1.15 [0.79, 1.67] says its scale is roughly
  right.
- **VAEP is hurt most by the data size.** It is built to learn from seasons of events (the paper used thousands of
  matches). Here it has 19 goals, no defensive actions, and a constant "concede" model.

**The 10 passes where the methods disagree most** (percentile ranks of coach_score, xT and VAEP over the 3,464
passes; positions in metres, the passing team attacks +x, goal at x = 52.5; `pct` = percentile of the value):

| Match, min | Pass (players) | From → to | Done | p | coach_score | xT | VAEP | pct coach / xT / VAEP | Net xG 15 s |
|---|---|---|---|---|---|---|---|---|---|
| J03WOH 43' | Kastenmeier → de Wijs | (−17, 24) → (30, −27) | no | 0.57 | +0.0087 | −0.007 | −0.082 | 1.00 / 0.16 / 0.00 | 0 |
| J03WR9 7' | Hennings → Oberdorf | (46, 7) → (49, −5) | yes | 0.35 | −0.0170 | −0.088 | +0.094 | 0.00 / 0.00 / 1.00 | +0.235 |
| J03WOH 2' | Albers → Breitkreuz | (50, 11) → (47, 3) | yes | 0.41 | −0.0144 | +0.170 | +0.018 | 0.01 / 1.00 / 0.86 | +0.090 |
| J03WR9 91' | Karbownik → Kownacki | (36, 15) → (49, −16) | yes | 0.33 | −0.0140 | −0.015 | +0.097 | 0.01 / 0.06 / 1.00 | 0 |
| J03WQQ 57' | Kastenmeier → Peterson | (−41, 2) → (25, −32) | yes | 0.50 | −0.0136 | +0.014 | +0.082 | 0.01 / 0.96 / 1.00 | 0 |
| J03WPY 69' | Fernandes Neto → Oberdorf | (34, 0) → (40, 6) | yes | 0.42 | −0.0147 | +0.025 | +0.065 | 0.00 / 0.99 / 0.99 | +0.101 |
| J03WPY 77' | Kastenmeier → Karbownik | (−43, 3) → (22, −14) | no | 0.61 | −0.0109 | −0.005 | +0.092 | 0.01 / 0.19 / 1.00 | +0.039 |
| J03WOH 80' | Makridis → Faber | (51, 16) → (28, −11) | yes | 0.95 | −0.0140 | +0.028 | +0.019 | 0.01 / 0.99 / 0.87 | +0.017 |
| J03WN1 75' | Riemann → Antwi-Adjei | (−36, 1) → (31, 20) | yes | 0.25 | −0.0168 | +0.020 | +0.036 | 0.00 / 0.98 / 0.96 | 0 |
| J03WPY 49' | Castrop → Daferner | (45, −19) → (45, −9) | yes | 0.40 | −0.0097 | +0.089 | +0.032 | 0.02 / 1.00 / 0.95 | +0.074 |

Patterns:
- **Ex ante vs ex post (5 of 10).** The coach scores the decision before the ball is played. xT and VAEP score
  what happened. Risky passes that came off get a low coach score (p 0.25–0.5) and a high xT or VAEP value. Examples
  are goalkeepers' long balls (Kastenmeier, Riemann) and passes into the box. The reverse happens too: Kastenmeier's
  long diagonal at 43' was the coach's favourite kind of pass (top 0.2 %), but it was lost, and VAEP ranks it
  last.
- **Passes from inside or at the edge of the box (4 of 10).** Albers, Makridis, Castrop, Fernandes Neto. The coach
  rates these below "now" because its value of the current spot is already high and it prefers a shot. xT rewards
  moving the ball into the central cells in front of goal. All 4 were followed by positive net xG (0.02–0.10).
  Here the coach's "shoot instead" opinion looks too strong.
- **Sparse xT cells.** Hennings' completed pass to the edge of the six-yard box gets xT −0.088, because that cell
  had few shots in the 6 training matches. VAEP ranks it at the top, and 0.235 net xG followed. With one season,
  Singh's grid would be smoother.
- With the overfit default booster (first run, not kept), VAEP gave values up to +0.81 to a goalkeeper's lost long
  ball after an opponent's chance (the swap term credits ending a chance the booster had overrated). These extremes
  disappeared once the classifiers were chosen by inner cross-validation.

## Cost

| | Coach | xT | VAEP |
|---|---|---|---|
| What it needs | one tracking frame (22 players + ball, velocities) | start / end location and the result | the last 3 actions (type, result, start / end), times, score: an event stream |
| Model size | 69 numbers, 2.0 kB JSON (3 linear models) | 192 numbers, 1.9 kB JSON | 296 tree nodes ("scores") + a constant ("concedes"); 100 kB pickled (mostly sklearn binning tables) |
| CPU time (Python, this laptop) | 9.9 ms per pass, all ~15 options with pitch control | < 0.001 ms per action (table lookup) | 0.002 ms per action (vectorised) |
| In the browser | **yes, shipped** (the replay's coach, and the live mode to come) | trivially portable, not shipped | not shipped: needs the action history and a tree runtime or a JSON export of the trees |

Only the coach values options the player did *not* take. It scores every pass, carry and shot available on the
frame, which is what "pause and ask the coach" needs. xT and VAEP value only the action that was played, and only
after the fact. That is fine for rating players, but they cannot give advice on a frozen frame.

## Limits

- 7 matches, 3,464 passes, 28 passes followed by a goal within 15 s. The goal AUCs have wide intervals.
- IDSSE only. The SkillCorner A-League cache holds model tables, not action sequences or tracking. Running C there
  would mean streaming the 20 matches again (~1.8 GB, now possible on the D: key) and using the xG proxy for labels.
- Our xT uses shrunk provider xG per cell instead of the goal ratio. Failed moves get −xT(start), a convention: Singh
  values only successful moves.
- Our VAEP has no defensive actions (none in the events), a constant "concede" model, and 19 goals to learn from.
  It is a fair re-implementation of the method on this data, not a reproduction of the paper's numbers.
- The coach's value model explains ~2 % of the variance of the next 15 s (v3 README). All "calibration" here is
  weak in absolute terms.
- Pass ends come from the tracking (reception point) or the next event. Carries are inferred and succeed by
  construction.

## References

- Singh, K. (2019). *Introducing Expected Threat (xT)*. Blog post, karun.in/blog/expected-threat.html.
- Decroos, T., Bransen, L., Van Haaren, J., Davis, J. (2019). *Actions Speak Louder than Goals: Valuing Player
  Actions in Soccer*. KDD 2019. [doi:10.1145/3292500.3330758](https://doi.org/10.1145/3292500.3330758).
- Fernández, J., Bornn, L., Cervone, D. (2019). *Decomposing the Immeasurable Sport: A deep learning expected
  possession value framework for soccer*. MIT Sloan Sports Analytics Conference 2019. Background: a tracking-based
  value like the coach's.
- Bassek, M. et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific
  Data*. [doi:10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). CC BY 4.0.
- Both methods were re-implemented from the papers. No code was copied. The `socceraction` package (MIT) was not
  used. Its SPADL format and its way of adding carries inspired the action table.

# Football AI coach: match report, explanations and ghosts, benchmark (A, B, C)

Date: 2026-11-02 · Project 20 · Owner choice (chat): A then B then C. Builds on
`2026-11-01-football-ai-coach-design.md` (engine, replay, 3D) and `2026-11-02-player-profiles-design.md`.

Principle for all three: everything is the **model's opinion, measured honestly**. Every number shown carries its
sample size; every claim that something is "better" is checked on held-out matches or labelled as an opinion.

## A. Match report: decision quality and missed opportunities

**What.** For every pass in a match that the sync kept (≈ 66 % of passes in IDSSE, ≈ 95 % in SkillCorner), with
models trained **without that match**:
- the coach's options at the release frame, the option the player chose (actual intended receiver), the best option;
- `gap = score(best) − score(chosen)` in expected goals (≥ 0), and the chosen option's rank;
- the outcome within 15 s (kept / lost / shot / goal, net xG).

**Outputs.**
- `report.json` per match (library `coach/report.py`): per decision (half, frame, carrier, chosen, best, gap, rank,
  outcome), per player (decisions n, share of top-1 / top-3 choices, mean gap, total gap), per team (same + by
  pitch third), highlight lists: **10 biggest missed opportunities** (largest gap where the best option's success
  probability ≥ 0.5, so the coach isn't just asking for miracles) and **10 best decisions** (chosen = best and the
  chosen score is far above the median option, i.e. the player found the rare good option).
- Site: a "Match report" section on `/demos/football/` (or `/demos/football/report/`): team summary, player table
  (sortable, n shown, no "good/bad player" labels), two highlight lists; clicking a highlight jumps the replay to
  that moment, paused, with the chosen option and the best option both drawn.

**Honesty.** The page repeats the known limits: the value model is weak (≈ 2 % of variance), the coach is bolder than
pros, so "missed opportunity" means "the model valued another option more", not "mistake". Player numbers need
n ≥ 20 decisions to be shown; below that, "too few decisions".

**Checks.** Unit tests on synthetic matches (gap ≥ 0, ranks, aggregation, thresholds); held-out-only models
(assert the match is not in the training set); report sizes ≤ 300 kB per match.

## B. Explanations and ghosts

**Explanations.** For each option, the 2–3 biggest reasons, from the exact model terms:
- pass model is logistic: contribution of each feature = `coef · (x − mean) / scale`; turn the top terms into plain
  phrases with the value: "1 defender in the lane", "receiver has 6.2 m of space", "long pass (31 m)";
- value: "+0.021 xG if it arrives; −0.004 if it is lost there".
Shown in the option list and on hover / focus of an arrow. Python and TypeScript produce the same text (parity
fixtures extended).

**Ghost runs (attack).** On pause, for each teammate of the carrier: candidate positions reachable in 1.5 s
(5 m/s, current velocity, a 7-point fan), evaluate the pass option to that position with the same models; if the
best candidate improves the option score by more than a threshold, draw a faint ghost of the teammate there with
an arrow ("a run here makes a +0.015 xG pass available"). Show at most 3 ghosts.

**Ghost defence.** For the defending team: greedy, one defender at a time (max 3), the reachable position that most
lowers the attacking team's best option score; draw a faint ghost defender ("covering here takes away −0.02 xG").

**Honesty.** Ghosts are counterfactuals from the same models: labelled "model suggestion", with no claim they are
right. An association check in the library: do real off-ball movements toward ghost positions precede higher-value
passes? Reported as association only.

**Performance.** Ghosts are computed only on pause, ≤ 50 ms in the browser on a laptop (measured; reduce the fan if
not).

## C. Benchmark against published methods

**What.** Re-implement on the same data two open methods:
- **xT** (expected threat, Singh 2019): grid of move / shot probabilities from events, value iteration;
- **VAEP** (Decroos et al. 2019): actions converted to a minimal SPADL-like table (pass, carry, shot, ...) from the
  IDSSE and SkillCorner events; P(score) and P(concede) within the next 10 actions with gradient boosting;
  action value = ΔP(score) − ΔP(concede).
Compare with our coach on held-out matches: (1) rank agreement of the value each method gives to the same actions
(Spearman), (2) calibration against what happened (net xG within 15 s / goals), (3) where they disagree most
(examples), (4) cost (runs in the browser or not).

**Output.** `docs/coach_benchmark.md` technical report in the library (methods, data, results, limits, references),
summarised in the README and linked from the website; charts only.

**Licences.** xT and VAEP are published methods; we re-implement them (no copied code). If the `socceraction`
package (MIT) is used for reference checks, credit it.

## Order and shipping

A (library → site) → B (library → TS parity → site) → C (library report → site link). Each part ships on its own;
the website changes go live after CI and a screenshot review.

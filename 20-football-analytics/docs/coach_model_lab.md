# Coach model lab (2026-11-09)

[`coach/model_lab.py`](../coach/model_lab.py), numbers in [`coach_model_lab.json`](coach_model_lab.json), tests
`tests/test_20_model_lab.py`. Leave-one-match-out over all 27 matches (7 Bundesliga / 2. Bundesliga + 20 A-League,
18,308 passes), 95 % CIs from 1000 resamples of whole matches. Gains are "current minus candidate" for losses (> 0 =
candidate better) and "candidate minus current" for AUC / Spearman. Nothing shipped was changed.

## 1. Pass model (completion probability)

New features, all computed from the same causal frame state: receiver speed towards the passer and forward speed,
intercept margin along the lane (fastest defender's arrival minus the ball's, at 15 m/s), defenders within 5 m of the
target, distance x (1 - target control), forward x pressure on the passer.

| candidate | log-loss | AUC | log-loss gain | log-loss gain, interceptor removed | end-to-end Spearman gain |
|---|---:|---:|---|---|---|
| current logistic (9 features) | 0.326 | 0.854 | | (0.368, AUC 0.790) | |
| logistic + 6 new features | 0.317 | 0.864 | +0.0091 [+0.0074, +0.0109] | +0.0126 [+0.0102, +0.0150] | **-0.0136 [-0.0178, -0.0094]** |
| logistic + receiver motion only | 0.321 | 0.860 | +0.0054 [+0.0041, +0.0068] | +0.0103 [+0.0083, +0.0123] | **-0.0137 [-0.0169, -0.0103]** |
| logistic + interactions only | 0.323 | 0.857 | +0.0033 [+0.0018, +0.0049] | +0.0015 [+0.0005, +0.0025] | |
| logistic + lane / target defenders | 0.324 | 0.857 | +0.0024 [+0.0015, +0.0033] | +0.0010 [+0.0002, +0.0017] | |
| HGB, raw | 0.310 | 0.867 | +0.0160 [+0.0125, +0.0193] | +0.0422 [+0.0373, +0.0470] | |
| HGB + isotonic (GroupKFold inner) | 0.310 | 0.868 | +0.0163 [+0.0124, +0.0200] | +0.0423 [+0.0373, +0.0474] | |
| HGB + sigmoid (GroupKFold inner) | 0.309 | 0.868 | +0.0169 [+0.0132, +0.0204] | +0.0425 [+0.0377, +0.0473] | **-0.0087 [-0.0152, -0.0022]** |

Brier gains have the same sign and CIs (JSON). Gains hold in both leagues, including the A-League, where the
intended receiver of failed passes is recorded rather than inferred (logistic + 6: Bundesliga 0.391 -> 0.380,
A-League 0.310 -> 0.302).

**Reading.** Completion is predicted better, mostly by the receiver's movement (a leak-free feature). But a better p
makes the coach score of the pro's chosen pass *less* aligned with what followed. HGB gains about 2.5x more on the
interceptor-removed table than on the standard one. That table removes a defender from failed passes only, an edit
that depends on the label and that a flexible model can learn to detect, so the HGB leak-check numbers are not
evidence. HGB is also not portable: the website runs the exact logistic / ridge formulas.

## 2. Value (V, possession) and arrival models

Context features at a point: opponents goal-side of it, and the possessing team's mean forward speed (computed from
the same causal state; for the loss point they are mirrored with the roles swapped).

| model | candidate | MSE skill (pooled / Bundesliga / A-League) | Spearman | MSE gain | Spearman gain |
|---|---|---|---:|---|---|
| value | current ridge | 4.1 % / 2.1 % / 5.1 % | 0.265 | | |
| value | HGB, basis | 4.0 % / 2.1 % / 5.0 % | 0.263 | -2e-6 [-4e-6, +1e-6] | -0.002 [-0.003, -0.000] |
| value | HGB + context | 5.9 % / 2.8 % / 7.5 % | 0.300 | +3.4e-5 [+2.0e-5, +5.1e-5] | +0.035 [+0.028, +0.043] |
| value | **ridge, zone x context** | 5.5 % / 3.0 % / 6.9 % | 0.295 | **+2.7e-5 [+1.9e-5, +3.6e-5]** | +0.030 [+0.022, +0.038] |
| arrival | current ridge | 8.9 % / 4.0 % / 10.2 % | 0.274 | | |
| arrival | HGB, basis | 7.9 % / 4.4 % / 8.8 % | 0.268 | -1.6e-5 [-3.2e-5, +5e-6] | -0.006 [-0.009, -0.003] |
| arrival | HGB + context | 9.0 % / 6.4 % / 9.6 % | 0.285 | +1e-6 [-1.7e-5, +2.6e-5] | +0.011 [+0.004, +0.019] |
| arrival | **ridge, zone x context** | 10.0 % / 5.3 % / 11.3 % | 0.289 | **+1.7e-5 [+0.7e-5, +2.7e-5]** | +0.015 [+0.007, +0.023] |

## 3. End-to-end: coach score of the pro's chosen pass vs net xG in 15 s (all 18,308 held-out passes, 27 matches)

| models (pass / value / arrival) | Spearman | Bundesliga | A-League | gain vs current |
|---|---:|---:|---:|---|
| current | 0.088 | 0.053 | 0.097 | |
| logistic + 6 / ridge / ridge | 0.075 | 0.045 | 0.082 | -0.014 [-0.018, -0.009] |
| HGB sigmoid / ridge / ridge | 0.080 | 0.049 | 0.087 | -0.009 [-0.015, -0.002] |
| logistic / ridge zone x context (both) | 0.097 | 0.110 | 0.094 | +0.008 [-0.003, +0.021] |
| logistic / HGB + context / ridge zone x context | 0.104 | 0.124 | 0.100 | +0.016 [-0.001, +0.032] |
| logistic + receiver motion / ridge zone x context (both) | 0.090 | 0.105 | 0.087 | +0.001 [-0.010, +0.014] |

The Playmaker points (100 / rank-based, from 20 refits) were not recomputed; the coach score is their input.

## Verdicts

| candidate | verdict | why |
|---|---|---|
| pass: logistic + new features (any group) | KEEP CURRENT | better log-loss, but lowers the end-to-end score (CI below zero) |
| pass: HGB raw / isotonic / sigmoid | KEEP CURRENT | leak-check gain inflated, not portable, lowers end-to-end |
| value: HGB basis / HGB + context | KEEP CURRENT | no gain / not portable |
| **value: ridge, zone x context** | **ADOPT** | MSE CI > 0, Spearman +0.030, causal features, linear |
| arrival: HGB basis / HGB + context | KEEP CURRENT | no significant MSE gain, not portable |
| **arrival: ridge, zone x context** | **ADOPT** | MSE CI > 0, Spearman +0.015, linear |

Caveat on the two ADOPTs: they are adopted on their own metric (MSE). Their end-to-end gain (+0.008) is positive in
the Bundesliga (0.053 -> 0.110) and slightly negative in the A-League (0.097 -> 0.094), and its CI includes zero.
Shipping them also means the site computes the two context features (`model_lab.value_context`) at every valued
point, and the parity fixtures and Playmaker content are rebuilt.

```bash
python -m coach.model_lab              # full lab (~50 min on 8 CPU threads; cache in cv-data/skillcorner)
python -m coach.model_lab --followup   # linear ablation + end-to-end only (~2 min), merged into the JSON
```

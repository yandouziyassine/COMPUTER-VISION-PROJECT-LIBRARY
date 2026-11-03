# Coach insight B: explanations and ghosts (library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every coach option, the exact model terms behind it and 2-3 plain-English reasons; ghost runs (where a
teammate could run to make a better pass available) and ghost defenders (where a defender could stand to take the best
option away); an honest held-out association check; Python <-> TypeScript parity fixtures extended with both.

**Architecture:** `coach/explain.py` reads the exported JSON weights (the site's format): logistic contribution
`coef * (x - mean) / scale` per feature, exact value split `score = p * gain + (1 - p) * loss`, phrases from one table
(`PHRASES`) and one rounding rule (`fmt`, half away from zero) that the TypeScript port copies (the fixture ships the
table). `coach/ghosts.py` is pure: a 7-point fan (7.5 m = 5 m/s x 1.5 s around the heading), pass scores with
advice.py's formula, greedy defence on `advice.options`. `coach/ghosts_eval.py` runs the association check
(held-out models, `report.held_out_models`) and the timing. `parity_fixtures.frame_record` gains `v_now`,
`explanations`, `ghosts`; the fixture gains `explain` and `ghost_params`; earlier fields unchanged.

**Tech Stack:** NumPy, existing coach modules, pytest. No matplotlib (numbers in a table, no chart).

Spec: `docs/superpowers/specs/2026-11-02-coach-insight-design.md`, section B. Website repo: only the fixture file
`tests/fixtures/coach-parity.json` is regenerated, never committed from here.

**Conventions:** `PY=C:/Users/yando/Code/.venv-tools/Scripts/python.exe`; tests from the repo root; scripts from
`20-football-analytics/`. Commit only own paths (`git commit -- <paths>`), never `git add -A`, stash or reset.
Push: `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q origin main`
(on rejection `git pull --rebase --autostash -q origin main` first), then `gh run list --branch main --limit 1`.

**Parameters (chosen by measurement, see Task 4):** fan of 7 points at -90..+90 deg every 30 deg, reach 7.5 m,
heading = velocity direction if >= 1 m/s else +x; the ghost player stands still at the point, nobody else reacts;
run threshold 0.005 xG, defence threshold 0.002 xG, at most 3 of each. Browser cost bound: ghost runs 70 pass
evaluations; defence at most 7 x (11 + 10 + 9) + 1 = 211 `options()` calls. Measured with the site's own TS port in
node on the 30 parity frames: about 6 ms worst case against the 50 ms budget, so the 7-point fan stays.

Common test header (`tests/test_20_explain.py`, created in Task 1):

```python
"""Project 20, AI coach: explanations and ghosts (coach insight, part B). Synthetic states only."""

import json
import math
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import advice as A  # noqa: E402
from coach import explain as EX  # noqa: E402
from coach import features as F  # noqa: E402
from coach import ghosts as GH  # noqa: E402
from coach import ghosts_eval as GE  # noqa: E402
from coach import models as ML  # noqa: E402


def weights(names, coef, mean=None, scale=None, link="logit", intercept=0.5):
    k = len(names)
    return {
        "features": list(names),
        "link": link,
        "mean": list(mean or [0.0] * k),
        "scale": list(scale or [1.0] * k),
        "coef": list(coef),
        "intercept": intercept,
    }


def state(att, deff, att_v=None, def_v=None):
    att, deff = np.array(att, float), np.array(deff, float)
    return {
        "att": att,
        "att_v": np.zeros_like(att) if att_v is None else np.array(att_v, float),
        "def": deff,
        "def_v": np.zeros_like(deff) if def_v is None else np.array(def_v, float),
        "att_idx": np.arange(len(att)),
        "def_idx": np.arange(len(att), len(att) + len(deff)),
        "ball": att[0].copy(),
        "direction": 1,
    }
```

---

### Task 1: explanations (coach/explain.py)

**Files:** Create `20-football-analytics/coach/explain.py`, `tests/test_20_explain.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_20_explain.py`)

```python
# ------------------------------------------------------------------ explanations


def test_fmt_rounds_half_away_from_zero_like_the_site():
    assert EX.fmt(2.5, 0) == "3" and EX.fmt(0.25, 1) == "0.3" and EX.fmt(6.25, 1) == "6.3"
    assert EX.fmt(31.4, 0) == "31" and EX.fmt(0.021, 3) == "0.021" and EX.fmt(0.0, 1) == "0.0"
    assert EX.fmt(-0.0004, 3) == "0.000" and EX.fmt(-1.26, 1) == "-1.3" and EX.fmt(0.05, 3) == "0.050"
    assert EX.signed(0.021) == "+0.021" and EX.signed(-0.0042) == "-0.004" and EX.signed(-0.0004) == "+0.000"


def test_contributions_are_exact_and_sum_to_the_logit():
    w = weights(F.PASS_FEATURES, np.linspace(-1, 1, 9), mean=list(range(9)), scale=[2.0] * 9, intercept=0.3)
    x = {f: float(i * 1.5) for i, f in enumerate(F.PASS_FEATURES)}
    cs = EX.contributions(w, x)
    assert [c["feature"] for c in cs] == list(F.PASS_FEATURES)
    assert cs[3]["contribution"] == pytest.approx(w["coef"][3] * (4.5 - 3) / 2.0)
    logit = w["intercept"] + sum(c["contribution"] for c in cs)
    p = ML.linear_predictor(w)(np.array([[x[f] for f in F.PASS_FEATURES]]))[0]
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(p, abs=1e-12)


def pass_weights():
    """Only receiver space (+) and lane defenders (-) matter, distance a little."""
    coef = {"receiver_space": 1.2, "lane_defenders": -0.9, "distance": -0.1}
    mean = {"receiver_space": 9.0, "lane_defenders": 0.3, "distance": 17.0}
    return weights(
        F.PASS_FEATURES,
        [coef.get(f, 0.0) for f in F.PASS_FEATURES],
        mean=[mean.get(f, 0.0) for f in F.PASS_FEATURES],
        scale=[6.0 if f == "receiver_space" else 1.0 for f in F.PASS_FEATURES],
    )


def test_pass_phrases_top_two_terms_and_value_split():
    feats = {f: 0.0 for f in F.PASS_FEATURES}
    feats.update({"receiver_space": 6.2, "lane_defenders": 1.0, "distance": 31.0})
    # contributions: space 1.2 * (6.2 - 9) / 6 = -0.56, lane -0.9 * 0.7 = -0.63, distance -0.1 * 14 = -1.4
    opt = {"kind": "pass", "target": 1, "p": 0.8, "value": 0.031, "v_lost": -0.006, "score": 0.0}
    ex = EX.explain(opt, 0.01, {"pass": pass_weights()}, feats)
    assert ex["phrases"] == [
        "long pass (31 m)",
        "1 defender in the lane",
        "+0.021 xG if it arrives, -0.004 if lost there",
    ]
    assert ex["signs"] == [-1, -1, 0] and not ex["fragile"]
    assert ex["value_terms"] == pytest.approx({"gain": 0.021, "loss": -0.004})
    feats.update({"distance": 10.0, "receiver_space": 20.0})  # short pass helps, big space helps most
    ex = EX.explain(opt, 0.01, {"pass": pass_weights()}, feats)
    assert ex["phrases"][:2] == ["receiver has 20.0 m of space", "short pass (10 m)"] and ex["signs"][:2] == [1, 1]


def test_phrase_variants_count_zero_split_and_degrees():
    w = pass_weights()
    T = EX.PHRASES["pass"]
    assert EX.feature_phrase(T, w, "lane_defenders", 0.0)[0] == "no defender in the lane"
    assert EX.feature_phrase(T, w, "lane_defenders", 3.0)[0] == "3 defenders in the lane"
    assert EX.feature_phrase(T, w, "forward", -12.4)[0] == "12 m backward"
    assert EX.feature_phrase(T, w, "target_x", -20.0)[0] == "receiver 20 m inside his own half"
    assert EX.feature_phrase(T, w, "abs_angle", math.pi / 4)[0] == "45° off straight ahead"
    assert EX.feature_phrase(T, w, "target_control", 0.874)[0] == "87 % pitch control at the receiver"
    assert EX.feature_phrase(T, w, "receiver_space", 6.25)[1] == pytest.approx(0.0, abs=1e-9)  # on a boundary


def test_shot_and_carry_explanations():
    w = {"shot": weights(F.SHOT_FEATURES, [-0.5, 0.3, -0.3], mean=[17.0, 0.4, 1.5], scale=[7.0, 0.24, 0.94])}
    sf = {"goal_dist": 30.0, "goal_angle": 0.2, "cone_defenders": 0.0}
    ex = EX.explain({"kind": "shot", "p": 0.031, "value": 1.0, "score": 0.02}, 0.011, w, sf)
    assert ex["phrases"] == [
        "shot from 30 m",
        "no defender in the shot cone",
        "+0.031 xG for the shot, +0.011 if the ball is kept here",
    ]
    carry = {"kind": "carry", "angle": 0, "p": 0.674, "value": 0.013, "v_lost": 0.004, "score": 0.0}
    ex = EX.explain(carry, 0.01, w)
    assert ex["phrases"] == ["67 % pitch control where the carry ends", "+0.003 xG if kept, -0.014 if lost there"]
    assert ex["contributions"] == [] and ex["intercept"] is None and ex["signs"] == [0, 0]


def advice_case():
    s = state([[20.0, 0.0], [45.0, 0.0], [5.0, 0.0], [30.0, 15.0]], [[20.0, 25.0], [0.0, -25.0], [44.0, 1.5]])
    w = {
        "pass": weights(F.PASS_FEATURES, [-0.05, 0, 0, -1.0, 0.1, 0, 2.0, 0, 0], intercept=0.0),
        "value": weights(F.VALUE_FEATURES, [0.002] + [0.0] * 9, link="identity", intercept=0.0),
        "shot": weights(F.SHOT_FEATURES, [-0.2, 0, 0], intercept=0.0),
    }
    return s, w, {k: ML.linear_predictor(v) for k, v in w.items()}


def test_explain_options_follow_the_advice_order_and_value_terms_rebuild_the_score():
    s, w, pred = advice_case()
    opts = A.options(s, 0, pred)
    v_now = float(A.value_at(pred, [s["att"][0]])[0])
    ex = EX.explain_options(s, 0, opts, w, v_now)
    assert [e["kind"] for e in ex] == [o["kind"] for o in opts] and {"pass", "carry", "shot"} == {
        o["kind"] for o in opts
    }
    for o, e in zip(opts, ex):
        assert 2 <= len(e["phrases"]) <= 3
        if o["kind"] == "shot":
            assert o["score"] == pytest.approx(e["value_terms"]["xg"] - e["value_terms"]["keep"])
        else:
            t = e["value_terms"]
            assert o["score"] == pytest.approx(o["p"] * t["gain"] + (1 - o["p"]) * t["loss"], abs=1e-12)
        if o["kind"] == "pass":
            z = w["pass"]["intercept"] + sum(c["contribution"] for c in e["contributions"])
            assert 1 / (1 + math.exp(-z)) == pytest.approx(o["p"], abs=1e-12)
    json.dumps(ex, allow_nan=False)
```

- [ ] **Step 2: Run, expect FAIL** (missing module / attribute): `$PY -m pytest tests/test_20_explain.py -q -k "fmt or contributions or phrase or explain"`

- [ ] **Step 3: Implement**

```python
"""Why the coach likes or dislikes an option (coach insight, part B): exact model terms turned into short phrases.

Pass and shot models are logistic regressions exported as JSON weights (models.export_linear): logit =
intercept + sum_k coef_k * (x_k - mean_k) / scale_k, so each feature's contribution coef_k * (x_k - mean_k) / scale_k
is exact (they sum with the intercept to the logit; positive = raises the success probability). The value terms are
the exact split of the score (advice.py): score = p * gain + (1 - p) * loss, with gain = V(target) - V(now) (the
change if the pass arrives) and loss = -V_opp(loss point) - V(now) (the change if it is lost there). A shot scores
xG - V(now). A carry's p is pitch control, not a fitted model: its reasons are the control and the value terms.

Phrases (2-3 per option): for a pass or a shot the two features with the largest |contribution| (ties: feature order)
rendered with PHRASES, then the value phrase; for a carry its pitch control, then the value phrase. PHRASES and fmt are the whole recipe: the website's TypeScript port copies them (the parity fixture ships the
table) and must produce the same strings.
"""

import math

import numpy as np

# How a feature value becomes text. v = fmt(|x| * scale, nd). Variant by "split":
#   "mean": "hi" if x >= mean (training mean, from the weights) else "lo";  "zero": "hi" if x >= 0 else "lo";
#   "count": n = fmt(|x|, 0): "0" if n == "0", "1" if n == "1", else "n";  "none": "text".
# Placeholder {v} in every template.
DEG = 180.0 / math.pi
PHRASES = {
    "pass": {
        "distance": {"split": "mean", "scale": 1.0, "nd": 0, "hi": "long pass ({v} m)", "lo": "short pass ({v} m)"},
        "forward": {"split": "zero", "scale": 1.0, "nd": 0, "hi": "{v} m forward", "lo": "{v} m backward"},
        "abs_angle": {"split": "none", "scale": DEG, "nd": 0, "text": "{v}° off straight ahead"},
        "lane_defenders": {
            "split": "count",
            "scale": 1.0,
            "nd": 0,
            "0": "no defender in the lane",
            "1": "1 defender in the lane",
            "n": "{v} defenders in the lane",
        },
        "receiver_space": {"split": "none", "scale": 1.0, "nd": 1, "text": "receiver has {v} m of space"},
        "passer_pressure": {"split": "none", "scale": 1.0, "nd": 1, "text": "nearest defender {v} m from the passer"},
        "target_control": {"split": "none", "scale": 100.0, "nd": 0, "text": "{v} % pitch control at the receiver"},
        "target_x": {
            "split": "zero",
            "scale": 1.0,
            "nd": 0,
            "hi": "receiver {v} m into the opponent half",
            "lo": "receiver {v} m inside his own half",
        },
        "target_goal_dist": {"split": "none", "scale": 1.0, "nd": 0, "text": "receiver {v} m from goal"},
    },
    "shot": {
        "goal_dist": {"split": "none", "scale": 1.0, "nd": 0, "text": "shot from {v} m"},
        "goal_angle": {"split": "none", "scale": DEG, "nd": 0, "text": "{v}° of goal mouth visible"},
        "cone_defenders": {
            "split": "count",
            "scale": 1.0,
            "nd": 0,
            "0": "no defender in the shot cone",
            "1": "1 defender in the shot cone",
            "n": "{v} defenders in the shot cone",
        },
    },
    # value phrases: {gain} / {loss} / {xg} / {keep} are signed(x) ("+0.021", "-0.004"), {pct} = fmt(100 p, 0)
    "value": {
        "pass": "{gain} xG if it arrives, {loss} if lost there",
        "carry_control": "{pct} % pitch control where the carry ends",
        "carry": "{gain} xG if kept, {loss} if lost there",
        "shot": "{xg} xG for the shot, {keep} if the ball is kept here",
    },
}
TOP_FEATURES = 2
FRAGILE = 1e-6  # a value this close to a rounding or variant boundary could print differently in TypeScript


def fmt(y: float, nd: int) -> str:
    """Round half away from zero to nd decimals: floor(|y| * 10^nd + 0.5), then a plain decimal string. The site
    computes Math.floor(Math.abs(y) * 10 ** nd + 0.5) (same IEEE operations), so both print the same digits; a
    minus sign only when the rounded value is not 0."""
    n = math.floor(abs(y) * 10**nd + 0.5)
    s = str(n).rjust(nd + 1, "0")
    s = s if nd == 0 else s[:-nd] + "." + s[-nd:]
    return ("-" if y < 0 and n > 0 else "") + s


def signed(y: float, nd: int = 3) -> str:
    """fmt with an explicit sign: "+0.021", "-0.004" ("+0.000" when it rounds to 0)."""
    s = fmt(y, nd)
    return s if s.startswith("-") else "+" + s


def _round_margin(y: float, nd: int) -> float:
    """How far |y| * 10^nd is from the nearest .5 rounding boundary."""
    t = abs(y) * 10**nd
    return abs(t - math.floor(t) - 0.5)


def contributions(weights: dict, x: dict) -> list:
    """[{feature, x, contribution}] in the model's feature order; contribution = coef * (x - mean) / scale."""
    return [
        {"feature": f, "x": float(x[f]), "contribution": float(c * (float(x[f]) - mu) / sd)}
        for f, mu, sd, c in zip(weights["features"], weights["mean"], weights["scale"], weights["coef"])
    ]


def feature_phrase(table: dict, weights: dict, feature: str, x: float) -> tuple:
    """(text, margin): the phrase of one feature value, and its distance to the nearest boundary that would change
    the text (rounding, or the hi / lo split)."""
    spec = table[feature]
    y = abs(x) * spec["scale"]
    v, margin = fmt(y, spec["nd"]), _round_margin(y, spec["nd"])
    split = spec["split"]
    if split == "mean":
        mu = weights["mean"][weights["features"].index(feature)]
        key, margin = ("hi" if x >= mu else "lo"), min(margin, abs(x - mu))
    elif split == "zero":
        key, margin = ("hi" if x >= 0 else "lo"), min(margin, abs(x))
    elif split == "count":
        key = v if v in ("0", "1") else "n"
    else:
        key = "text"
    return spec[key].replace("{v}", v), margin


def value_phrase(option: dict, v_now: float) -> tuple:
    """(texts, terms, margin): the value phrases of an option (a carry first states its pitch control), the exact
    value terms and the distance of the printed numbers to a rounding boundary."""
    t = PHRASES["value"]
    if option["kind"] == "shot":
        terms = {"xg": float(option["p"]), "keep": float(v_now)}
        text = t["shot"].replace("{xg}", signed(terms["xg"])).replace("{keep}", signed(terms["keep"]))
        return [text], terms, min(_round_margin(v, 3) for v in terms.values())
    terms = {"gain": float(option["value"]) - v_now, "loss": -float(option["v_lost"]) - v_now}
    kind = "carry" if option["kind"] == "carry" else "pass"
    texts = [t[kind].replace("{gain}", signed(terms["gain"])).replace("{loss}", signed(terms["loss"]))]
    margin = min(_round_margin(v, 3) for v in terms.values())
    if kind == "carry":
        pct = float(option["p"]) * 100.0
        texts.insert(0, t["carry_control"].replace("{pct}", fmt(pct, 0)))
        margin = min(margin, _round_margin(pct, 0))
    return texts, terms, margin


def explain(option: dict, v_now: float, weights: dict, features: dict | None = None) -> dict:
    """Explanation of one advice.options option. `weights`: the exported JSON weights ("pass", "shot"); `features`:
    pass_features (pass) or shot_features (shot) of the option, unused for a carry. Returns contributions (pass /
    shot, model order), intercept, value terms, phrases (pass / shot: the top TOP_FEATURES features by |contribution|
    then the value phrase; carry: its pitch control then the value phrase), signs (+1 helps / -1 hurts / 0 value phrase) and fragile (a boundary within FRAGILE: the TypeScript text
    could differ there, the parity test skips the strings)."""
    out = {"kind": option["kind"], "contributions": [], "intercept": None}
    phrases, signs, margin = [], [], math.inf
    model = {"pass": "pass", "shot": "shot"}.get(option["kind"])
    if model is not None:
        w = weights[model]
        cs = contributions(w, features)
        out["contributions"], out["intercept"] = cs, float(w["intercept"])
        mags = np.array([abs(c["contribution"]) for c in cs])
        order = sorted(range(len(cs)), key=lambda i: (-mags[i], i))
        top = order[:TOP_FEATURES]
        if len(order) > TOP_FEATURES:  # the cut between the kept and the next feature must not be a near tie
            margin = min(margin, mags[order[TOP_FEATURES - 1]] - mags[order[TOP_FEATURES]])
        for a, b in zip(top, top[1:]):
            margin = min(margin, mags[a] - mags[b])
        for i in top:
            text, mg = feature_phrase(PHRASES[model], w, cs[i]["feature"], cs[i]["x"])
            phrases.append(text)
            signs.append(1 if cs[i]["contribution"] >= 0 else -1)
            margin = min(margin, mg, mags[i])
    texts, terms, mg = value_phrase(option, v_now)
    phrases += texts
    signs += [0] * len(texts)
    margin = min(margin, mg)
    out.update({"value_terms": terms, "phrases": phrases, "signs": signs, "fragile": bool(margin < FRAGILE)})
    return out


def explain_options(s: dict, carrier: int, opts: list, weights: dict, v_now: float) -> list:
    """explain() for every option of advice.options(s, carrier, ...), in the same order."""
    from . import features as F

    rows = {int(s["att_idx"][j]): j for j in range(len(s["att"]))}
    here = s["att"][carrier]
    out = []
    for o in opts:
        if o["kind"] == "pass":
            f = F.pass_features(s, carrier, rows[o["target"]])
        elif o["kind"] == "shot":
            f = F.shot_features(here, s["def"])
        else:
            f = None
        out.append(explain(o, v_now, weights, f))
    return out
```

- [ ] **Step 4: Run, expect PASS**: same command, then the regression
  `$PY -m pytest tests/test_20_coach.py tests/test_20_report.py -q`; lint `$PY -m ruff check 20-football-analytics/coach/explain.py tests/test_20_explain.py && $PY -m ruff format --check 20-football-analytics/coach/explain.py tests/test_20_explain.py`

- [ ] **Step 5: Commit**: `git add 20-football-analytics/coach/explain.py tests/test_20_explain.py && git commit -m "20 coach: explanations (exact logistic contributions, value split, phrase table for the TS port)" -- 20-football-analytics/coach/explain.py tests/test_20_explain.py`


### Task 2: ghost runs and ghost defence (coach/ghosts.py)

**Files:** Create `20-football-analytics/coach/ghosts.py`; modify `tests/test_20_explain.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_20_explain.py`)

```python
# ------------------------------------------------------------------ ghosts


def test_fan_points_follow_the_heading_and_stay_on_the_pitch():
    pts = GH.fan([0.0, 0.0], [0.0, 0.0])  # standing still: towards +x
    assert pts.shape == (GH.FAN_N, 2) and np.allclose(np.hypot(*pts.T), GH.REACH)
    assert np.allclose(pts[3], [GH.REACH, 0.0]) and np.allclose(pts[0], [0.0, -GH.REACH])
    pts = GH.fan([0.0, 0.0], [0.0, 4.0])  # running towards +y
    assert np.allclose(pts[3], [0.0, GH.REACH])
    assert np.all(GH.fan([50.0, 33.0], [3.0, 3.0]) <= [52.5, 34.0])


def marked_case():
    """Carrier at the centre, teammate 1 tightly marked; teammate 2 far back and free; a third teammate unmarked."""
    s = state([[0.0, 0.0], [15.0, 0.0], [-20.0, 10.0]], [[16.0, 0.5], [30.0, -20.0]])
    models = {
        "pass": lambda X: 1 / (1 + np.exp(-(X[:, F.PASS_FEATURES.index("receiver_space")] - 3.0))),
        "value": lambda X: 0.1 * (X[:, 0] + 52.5) / 105,  # >= 0: losing the ball anywhere costs
        "shot": lambda X: np.zeros(len(X)),
    }
    return s, models


def test_ghost_runs_free_the_marked_teammate():
    s, models = marked_case()
    runs = GH.ghost_runs(s, 0, models)
    assert runs and runs == GH.ghost_runs(s, 0, models)  # deterministic
    g = next(r for r in runs if r["player"] == 1)
    assert g["gain"] > GH.RUN_MIN_GAIN and g["score"] - g["score_now"] == pytest.approx(g["gain"])
    assert math.dist(g["to"], s["att"][1]) == pytest.approx(GH.REACH)
    pts = GH.fan(s["att"][1], s["att_v"][1])
    v_now = float(A.value_at(models, [s["att"][0]])[0])
    all_scores = GH.pass_scores([GH.moved(s, "att", 1, p) for p in pts], 0, 1, models, v_now)
    assert g["fan_index"] == int(np.argmax(all_scores)) and g["score"] == pytest.approx(all_scores.max())
    now = next(o for o in A.options(s, 0, models) if o["kind"] == "pass" and o["target"] == 1)
    assert g["score_now"] == pytest.approx(now["score"])  # same formula as advice.options
    assert [r["gain"] for r in runs] == sorted((r["gain"] for r in runs), reverse=True) and len(runs) <= GH.MAX_GHOSTS


def test_no_ghosts_when_position_does_not_matter():
    s, _ = marked_case()
    flat = {
        "pass": lambda X: np.full(len(X), 0.7),
        "value": lambda X: np.zeros(len(X)),
        "shot": lambda X: np.zeros(len(X)),
    }
    assert GH.ghost_runs(s, 0, flat) == [] and GH.ghost_defence(s, 0, flat) == []


def test_ghost_defence_is_greedy_and_lowers_the_best_option():
    s, models = marked_case()
    s["def"][0] = [15.0, 9.0]  # teammate 1 unmarked, a defender 9 m away can close him down
    dfn = GH.ghost_defence(s, 0, models)
    assert 1 <= len(dfn) <= GH.MAX_GHOSTS and dfn == GH.ghost_defence(s, 0, models)
    assert dfn[0]["best_before"] == pytest.approx(GH.best_score(s, 0, models))
    cur = s
    for a, b in zip(dfn, dfn[1:] + [None]):
        assert a["gain"] > GH.DEF_MIN_GAIN and a["best_after"] == pytest.approx(a["best_before"] - a["gain"])
        cur = GH.moved(cur, "def", a["row"], a["to"])
        assert GH.best_score(cur, 0, models) == pytest.approx(a["best_after"])
        if b is not None:
            assert b["best_before"] == a["best_after"] and b["row"] != a["row"]
    assert len({d["row"] for d in dfn}) == len(dfn)
```

- [ ] **Step 2: Run, expect FAIL** (missing module / attribute): `$PY -m pytest tests/test_20_explain.py -q -k "fan or ghost"`

- [ ] **Step 3: Implement**

```python
"""Ghost runs and ghost defenders (coach insight, part B): counterfactual positions scored with the same models.

Pure functions of one normalised frame state (features.state; the team in possession attacks +x) and the advice
models (advice.from_fitted, or models.linear_predictor on the exported weights). Deterministic: ties keep the lower
player row and fan index (strict comparisons).

Fan: FAN_N points at distance REACH = SPEED * HORIZON_S (5 m/s for 1.5 s = 7.5 m) from the player's position, at
FAN_DEG degrees around his heading: the direction of his velocity when he moves at >= MIN_SPEED m/s, else towards the
opponent goal (+x) for an attacker and towards his own goal (+x as well, he defends it) for a defender. Points are
clipped to the pitch. The moved player stands still at his ghost point (velocity 0); everyone else keeps position and
velocity (nobody reacts: a limit of the counterfactual).

Ghost runs (attack): for each teammate j of the carrier, the score of the pass to j (advice.py's formula) with j at
each fan point; his ghost is the best point when it beats the pass to his real position by more than RUN_MIN_GAIN.
At most MAX_GHOSTS, largest gain first. Ghost defence: greedy, at most MAX_GHOSTS defenders one at a time; at each step
every remaining defender at every fan point, keep the move that lowers the attacking team's best option score
(advice.options, all options) the most, when by more than DEF_MIN_GAIN; the move stays for the next steps.
"""

import math

import numpy as np

from . import advice as A
from . import features as F

SPEED, HORIZON_S = 5.0, 1.5
REACH = SPEED * HORIZON_S
FAN_N, FAN_DEG = 7, (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
MIN_SPEED = 1.0
RUN_MIN_GAIN = 0.005  # xG
DEF_MIN_GAIN = 0.002  # xG
MAX_GHOSTS = 3
TIE = 1e-9  # two candidates closer than this could swap between languages: flagged "fragile"


def params() -> dict:
    """The constants the website's port needs (written into the parity fixture)."""
    return {
        "speed": SPEED,
        "horizon_s": HORIZON_S,
        "reach_m": REACH,
        "fan_deg": list(FAN_DEG),
        "min_speed": MIN_SPEED,
        "run_min_gain": RUN_MIN_GAIN,
        "def_min_gain": DEF_MIN_GAIN,
        "max_ghosts": MAX_GHOSTS,
        "tie": TIE,
    }


def fan(pos, vel) -> np.ndarray:
    """[FAN_N, 2] points REACH metres from pos around the heading (velocity direction, or +x when slower than
    MIN_SPEED), clipped to the pitch."""
    pos, vel = np.asarray(pos, float), np.asarray(vel, float)
    speed = math.hypot(vel[0], vel[1])
    heading = math.atan2(vel[1], vel[0]) if speed >= MIN_SPEED else 0.0
    out = []
    for d in FAN_DEG:
        a = heading + d * (math.pi / 180)
        out.append([pos[0] + REACH * math.cos(a), pos[1] + REACH * math.sin(a)])
    return np.clip(np.array(out), [-52.5, -34.0], [52.5, 34.0])


def moved(s: dict, side: str, row: int, point) -> dict:
    """Copy of s with player `row` of side "att" / "def" standing still at `point`."""
    pos, vel = s[side].copy(), s[side + "_v"].copy()
    pos[row], vel[row] = point, 0.0
    return {**s, side: pos, side + "_v": vel}


def pass_scores(states: list, carrier: int, target: int, models: dict, v_now: float) -> np.ndarray:
    """advice.py's score of the pass carrier -> target in each state (one batch per model)."""
    here = states[0]["att"][carrier]
    X = np.array([[F.pass_features(s, carrier, target)[k] for k in F.PASS_FEATURES] for s in states])
    p = np.asarray(models["pass"](X), float)
    to = [s["att"][target] for s in states]
    loss = [F.lane_loss_point(s, here, s["att"][target]) for s in states]
    v = A.value_at(models, to)
    v_opp = A.value_at(models, [-pt for pt in loss])
    return p * v - (1 - p) * v_opp - v_now


def ghost_runs(s: dict, carrier: int, models: dict) -> list:
    """Ghosts of the carrier's teammates: [{row, player, from, to, score_now, score, gain, fragile}], gain > RUN_MIN_GAIN,
    largest first, at most MAX_GHOSTS."""
    v_now = float(A.value_at(models, [s["att"][carrier]])[0])
    out = []
    for j in range(len(s["att"])):
        if j == carrier:
            continue
        pts = fan(s["att"][j], s["att_v"][j])
        scores = pass_scores([s] + [moved(s, "att", j, pt) for pt in pts], carrier, j, models, v_now)
        now, cand = float(scores[0]), scores[1:]
        k = int(np.argmax(cand))  # first maximum: the lower fan index wins a tie
        gain = float(cand[k]) - now
        if gain > RUN_MIN_GAIN:
            others = np.delete(cand, k)
            margin = min(abs(gain - RUN_MIN_GAIN), float(cand[k] - others.max()) if len(others) else math.inf)
            out.append(
                {
                    "row": j,
                    "player": int(s["att_idx"][j]),
                    "from": s["att"][j].tolist(),
                    "to": pts[k].tolist(),
                    "fan_index": k,
                    "score_now": now,
                    "score": float(cand[k]),
                    "gain": gain,
                    "fragile": bool(margin < TIE),
                }
            )
    out.sort(key=lambda g: (-g["gain"], g["row"]))
    for a, b in zip(out[:MAX_GHOSTS], out[1 : MAX_GHOSTS + 1]):
        if a["gain"] - b["gain"] < TIE:
            a["fragile"] = b["fragile"] = True
    return out[:MAX_GHOSTS]


def best_score(s: dict, carrier: int, models: dict) -> float:
    return float(A.options(s, carrier, models)[0]["score"])


def ghost_defence(s: dict, carrier: int, models: dict) -> list:
    """Greedy ghost defenders: [{row, player, from, to, best_before, best_after, gain, fragile}] in the order chosen,
    gain = best_before - best_after > DEF_MIN_GAIN, at most MAX_GHOSTS."""
    out, cur = [], s
    before = best_score(cur, carrier, models)
    left = list(range(len(s["def"])))
    for _ in range(MAX_GHOSTS):
        best, second = None, -math.inf
        for d in left:
            pts = fan(cur["def"][d], cur["def_v"][d])
            for k, pt in enumerate(pts):
                after = best_score(moved(cur, "def", d, pt), carrier, models)
                gain = before - after
                if best is None or gain > best[0]:
                    if best is not None:
                        second = max(second, best[0])
                    best = (gain, d, k, pt, after)
                else:
                    second = max(second, gain)
        if best is None or best[0] <= DEF_MIN_GAIN:
            break
        gain, d, k, pt, after = best
        margin = min(gain - second, abs(gain - DEF_MIN_GAIN))
        out.append(
            {
                "row": d,
                "player": int(s["def_idx"][d]),
                "from": cur["def"][d].tolist(),
                "to": pt.tolist(),
                "fan_index": k,
                "best_before": before,
                "best_after": after,
                "gain": gain,
                "fragile": bool(margin < TIE),
            }
        )
        cur = moved(cur, "def", d, pt)
        before = after
        left.remove(d)
    return out
```

- [ ] **Step 4: Run, expect PASS**: same command, then the regression
  `$PY -m pytest tests/test_20_coach.py tests/test_20_report.py -q`; lint `$PY -m ruff check 20-football-analytics/coach/ghosts.py tests/test_20_explain.py && $PY -m ruff format --check 20-football-analytics/coach/ghosts.py tests/test_20_explain.py`

- [ ] **Step 5: Commit**: `git add 20-football-analytics/coach/ghosts.py tests/test_20_explain.py && git commit -m "20 coach: ghost runs and greedy ghost defence (7-point fan, same models)" -- 20-football-analytics/coach/ghosts.py tests/test_20_explain.py`


### Task 3: association check and timing (coach/ghosts_eval.py)

**Files:** Create `20-football-analytics/coach/ghosts_eval.py`; modify `tests/test_20_explain.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_20_explain.py`)

```python
# ------------------------------------------------------------------ association and timing


def test_progress_along_the_ghost_run():
    assert GE.progress([0, 0], [3.75, 0], [7.5, 0]) == pytest.approx(0.5)
    assert GE.progress([0, 0], [0, 5], [7.5, 0]) == pytest.approx(0.0)
    assert GE.progress([0, 0], [-7.5, 0], [7.5, 0]) == pytest.approx(-1.0)


def test_association_summary_differences_with_match_bootstrap():
    recs = []
    for m in "ABC":
        for i in range(20):
            followed = i % 2 == 0
            recs.append(
                {
                    "match": m,
                    "event": i // 2,
                    "followed": followed,
                    "received": followed and i % 4 == 0,
                    "gain": 0.008,
                    "score_release": 0.01 if followed else 0.0,
                    "change": 0.004 if followed else -0.001,
                }
            )
    out = GE.association_summary(recs)
    assert out["label"].startswith("association") and out["records"] == 60 and out["passes"] == 30
    assert out["followed"]["n"] == 30 and out["followed"]["received"] == pytest.approx(0.5)
    assert out["not_followed"]["received"] == 0.0 and out["followed"]["realised_share"] == pytest.approx(0.5)
    d = out["difference"]
    assert d["received"]["value"] == pytest.approx(0.5) and d["change"]["value"] == pytest.approx(0.005)
    assert d["score_release"]["ci95"][0] == pytest.approx(0.01) and len(d["received_score"]["ci95"]) == 2


def test_association_records_on_a_synthetic_match():
    from test_20_report import MODELS, small_match

    m = small_match()
    recs = GE.association_records(m, MODELS)
    for r in recs:  # players stand still: nobody follows a ghost
        assert r["match"] == "SYN" and not r["followed"] and r["progress"] == pytest.approx(0.0)
        assert r["change"] == pytest.approx(r["score_release"] - r["score_anchor"])


def test_timing_counts_model_rows():
    s, models = marked_case()
    out = GE.timing([(s, 0)], models)
    assert out["frames"] == 1 and out["runs_pass_rows_mean"] == 2 * (GH.FAN_N + 1)  # 2 teammates x (now + fan)
    assert out["defence_options_calls_max"] == GH.FAN_N * (2 + 1) + 1  # 2 defenders: 2 + 1 candidates over the steps
    assert out["runs_ms_mean"] >= 0 and out["ghost_runs_per_frame"] >= 1
```

- [ ] **Step 2: Run, expect FAIL** (missing module / attribute): `$PY -m pytest tests/test_20_explain.py -q -k "progress or association or timing"`

- [ ] **Step 3: Implement**

```python
"""Ghosts and explanations, checked (coach insight, part B): association on held-out passes and the cost per pause.

    python -m coach.ghosts_eval      # from 20-football-analytics/ -> docs/coach_ghosts_results.json (~15 min on CPU)

Association (not causation). For every usable pass of the 7 IDSSE matches (datasets.pass_samples), with pass and
value models trained WITHOUT that match: go back ghosts.HORIZON_S (1.5 s) before the release; if the passer had the
ball then (ball in play, his team in possession, ball within CARRIER_M of him), compute the ghost runs of his
teammates at that anchor frame. For each ghost teammate still on the pitch at the release: progress = how much of the
ghost run he actually made by the release (projection of his real displacement on the ghost vector, divided by its
length); followed = progress >= FOLLOW. Outcomes at the release: did he receive the pass, the score of the pass to him
(advice.py's formula on the real state) and its change since the anchor. Followed vs not followed, with 95 % intervals
resampling whole matches. Moving towards a spot the model likes raises the model's score of a pass there partly by
construction (the defenders did react in reality, they don't in the ghost), so "realised share" (mean change / mean
predicted gain) is the more telling number; "received" is the behavioural outcome.

Timing: ghosts per paused frame in Python on the parity frames (exported weights), with model evaluation counts.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import advice as A
from . import evaluate as EV
from . import features as F
from . import ghosts as GH

FOLLOW = 0.5
CARRIER_M = 2.0  # same rule as the site's carrier (parity_fixtures.CARRIER_M)
DOCS = Path(__file__).resolve().parents[1] / "docs"


def progress(start, end, ghost) -> float:
    """Share of the ghost run start -> ghost made by moving start -> end (projection, can be < 0 or > 1)."""
    g = np.asarray(ghost, float) - np.asarray(start, float)
    d = np.asarray(end, float) - np.asarray(start, float)
    return float(d @ g / max(float(g @ g), 1e-9))


def anchor_state(m, e: dict, back: int):
    """(state, passer row) at `back` frames before the release when the passer had the ball there, else None."""
    f = int(e["frame"]) - back
    h = m.halves[e["half"]]
    if f < 0 or not h.alive[f] or int(h.owner[f]) != e["team"] or np.isnan(h.ball[f, 0]):
        return None
    pos = h.xy[f, e["player"]]
    if np.isnan(pos[0]) or float(np.hypot(*(pos - h.ball[f, :2]))) > CARRIER_M:
        return None
    s = F.state(m, e["half"], f, e["team"])
    rows = np.flatnonzero(s["att_idx"] == e["player"])
    return (s, int(rows[0])) if len(rows) else None


def association_records(m, models: dict, pass_samples=None) -> list:
    """One record per ghost teammate per usable pass (see the module docstring)."""
    from . import datasets as D

    back = int(round(GH.HORIZON_S * F.match_fps(m)))
    out = []
    for p in pass_samples if pass_samples is not None else D.pass_samples(m):
        e, s_r, c_r = p["event"], p["state"], p["carrier"]
        found = anchor_state(m, e, back)
        if found is None:
            continue
        s_a, c_a = found
        chosen = int(s_r["att_idx"][p["target"]])
        v_now = float(A.value_at(models, [s_r["att"][c_r]])[0])
        for g in GH.ghost_runs(s_a, c_a, models):
            rows = np.flatnonzero(s_r["att_idx"] == g["player"])
            if not len(rows) or rows[0] == c_r:
                continue
            j = int(rows[0])
            score = float(GH.pass_scores([s_r], c_r, j, models, v_now)[0])
            prog = progress(g["from"], s_r["att"][j], g["to"])
            out.append(
                {
                    "match": m.match_id,
                    "event": int(p["index"]),
                    "player": g["player"],
                    "progress": prog,
                    "followed": prog >= FOLLOW,
                    "received": g["player"] == chosen,
                    "gain": g["gain"],
                    "score_anchor": g["score_now"],
                    "score_release": score,
                    "change": score - g["score_now"],
                }
            )
    return out


def _mean(rs, key):
    return float(np.mean([float(r[key]) for r in rs])) if rs else float("nan")


def _diff(rs, key):
    """mean(key | followed) - mean(key | not followed)."""
    return _mean([r for r in rs if r["followed"]], key) - _mean([r for r in rs if not r["followed"]], key)


def association_summary(recs: list) -> dict:
    """Followed vs not followed: n, P(received), mean score at the release and mean change since the anchor, the
    score of the passes they received, and the share of the predicted gain realised; differences with match-bootstrap
    95 % intervals (evaluate.bootstrap_ci)."""
    groups = {"followed": [r for r in recs if r["followed"]], "not_followed": [r for r in recs if not r["followed"]]}
    out = {
        "label": "association only, not causation",
        "records": len(recs),
        "passes": len({(r["match"], r["event"]) for r in recs}),
    }
    for name, rs in groups.items():
        rec = [r for r in rs if r["received"]]
        out[name] = {
            "n": len(rs),
            "received": _mean(rs, "received"),
            "score_release": _mean(rs, "score_release"),
            "change": _mean(rs, "change"),
            "predicted_gain": _mean(rs, "gain"),
            "realised_share": _mean(rs, "change") / _mean(rs, "gain") if rs else float("nan"),
            "received_n": len(rec),
            "received_score": _mean(rec, "score_release"),
        }
    received = lambda rs: _diff([r for r in rs if r["received"]], "score_release")
    out["difference"] = {
        k: {"value": _diff(recs, k), "ci95": EV.bootstrap_ci(recs, lambda rs, k=k: _diff(rs, k))}
        for k in ("received", "score_release", "change")
    }
    out["difference"]["received_score"] = {"value": received(recs), "ci95": EV.bootstrap_ci(recs, received)}
    return out


class Counter:
    """Wraps advice model functions and counts the rows each one evaluates."""

    def __init__(self, models: dict):
        self.rows = {k: 0 for k in models}
        self.models = {k: self._wrap(k, fn) for k, fn in models.items()}

    def _wrap(self, k, fn):
        def run(X):
            self.rows[k] += len(np.atleast_2d(X))
            return fn(X)

        return run


def timing(frames: list, models: dict) -> dict:
    """Python wall time per paused frame for ghost_runs and ghost_defence, plus the model rows and advice.options
    calls they need. `frames`: (state, carrier row) pairs."""
    res = {"frames": len(frames), "runs_ms": [], "defence_ms": [], "runs_rows": [], "defence_rows": [], "ghosts": []}
    for s, c in frames:
        cnt = Counter(models)
        t0 = time.perf_counter()
        runs = GH.ghost_runs(s, c, cnt.models)
        res["runs_ms"].append(1000 * (time.perf_counter() - t0))
        res["runs_rows"].append(dict(cnt.rows))
        cnt = Counter(models)
        t0 = time.perf_counter()
        dfn = GH.ghost_defence(s, c, cnt.models)
        res["defence_ms"].append(1000 * (time.perf_counter() - t0))
        res["defence_rows"].append(dict(cnt.rows))
        res["ghosts"].append([len(runs), len(dfn)])
    steps = lambda n_def: sum(n_def - k for k in range(GH.MAX_GHOSTS))
    n_def = [len(s["def"]) for s, _ in frames]
    return {
        "frames": len(frames),
        "runs_ms_mean": float(np.mean(res["runs_ms"])),
        "defence_ms_mean": float(np.mean(res["defence_ms"])),
        "defence_ms_max": float(np.max(res["defence_ms"])),
        "runs_pass_rows_mean": float(np.mean([r["pass"] for r in res["runs_rows"]])),
        "defence_pass_rows_mean": float(np.mean([r["pass"] for r in res["defence_rows"]])),
        "defence_options_calls_max": int(max(GH.FAN_N * steps(n) + 1 for n in n_def)),
        "ghost_runs_per_frame": float(np.mean([g[0] for g in res["ghosts"]])),
        "ghost_defenders_per_frame": float(np.mean([g[1] for g in res["ghosts"]])),
    }


def parity_frames(fx: dict) -> list:
    """(state, carrier row) of every frame of a parity fixture (coach.parity_fixtures)."""
    out = []
    for rec in fx["frames"]:
        st = rec["state"]
        s = {k: np.array(st[k], float) for k in ("att", "att_v", "def", "def_v", "ball")}
        s.update({"att_idx": np.array(st["att_idx"], int), "def_idx": np.array(st["def_idx"], int)})
        s["direction"] = st["direction"]
        out.append((s, rec["carrier_row"]))
    return out


def main():
    from . import parity_fixtures as PF
    from . import report as R
    from .match import load
    from .prepare import MATCHES, OUT

    fx = json.loads(PF.SITE_FIXTURE.read_text(encoding="utf-8"))
    tim = timing(parity_frames(fx), PF.predictors(fx["models"]))
    print("timing", tim, flush=True)
    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: R.match_tables(m) for mid, m in matches.items()}
    recs = []
    for mid, m in matches.items():
        models, _, _ = R.held_out_models(mid, tables)
        got = association_records(m, models)
        recs += got
        print(mid, len(got), sum(r["followed"] for r in got), flush=True)
    res = {
        "params": {**GH.params(), "follow": FOLLOW},
        "timing_python": tim,
        "association": association_summary(recs),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_ghosts_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["association"], indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, expect PASS**: same command, then the regression
  `$PY -m pytest tests/test_20_coach.py tests/test_20_report.py -q`; lint `$PY -m ruff check 20-football-analytics/coach/ghosts_eval.py tests/test_20_explain.py && $PY -m ruff format --check 20-football-analytics/coach/ghosts_eval.py tests/test_20_explain.py`

- [ ] **Step 5: Commit**: `git add 20-football-analytics/coach/ghosts_eval.py tests/test_20_explain.py && git commit -m "20 coach: ghost association check (held-out, match bootstrap) and timing" -- 20-football-analytics/coach/ghosts_eval.py tests/test_20_explain.py`


### Task 4: run the association check and timing

- [ ] Run `cd 20-football-analytics && $PY -m coach.ghosts_eval` (~15 min): writes `docs/coach_ghosts_results.json`.
- [ ] Browser estimate: copy the site's `src/lib/coach/*.ts` to the scratchpad (imports rewritten to `.ts`, node 24
  strips types) and time `options()` and the same ghost loops on the 30 parity frames. Nothing committed.
- [ ] Commit: `git add 20-football-analytics/docs/coach_ghosts_results.json && git commit -m "20 coach: ghost association and timing results" -- 20-football-analytics/docs/coach_ghosts_results.json`

### Task 5: parity fixture with explanations and ghosts

**Files:** Modify `20-football-analytics/coach/parity_fixtures.py` (frame_record, build, docstring); modify `tests/test_20_explain.py`

- [ ] **Step 1: Write the failing tests** (append to `tests/test_20_explain.py`)

```python
# ------------------------------------------------------------------ parity fixture


def test_parity_fixture_carries_explanations_and_ghosts(tmp_path):
    from coach import export as X
    from coach import parity_fixtures as PF
    from test_20_report import small_match

    m = small_match()
    m.halves[0].xy[:, 4] = [-4.5, -3.5]  # a guest on the right carry point: carries left and right no longer tie
    X.pack_match(m, tmp_path, fps=10)
    w = {
        "trained_without": "TEST",
        "pass": weights(F.PASS_FEATURES, [-0.05, 0, 0, -0.5, 0.1, 0, 0, 0, 0], intercept=2.0),
        "value": weights(F.VALUE_FEATURES, [0.002] + [0.0] * 9, link="identity", intercept=0.01),
        "shot": weights(F.SHOT_FEATURES, [-0.1, 0, 0]),
    }
    (tmp_path / "models.json").write_text(json.dumps(w), encoding="utf-8")
    fx = PF.build(tmp_path, per_half=2)
    assert fx["frames"] and fx["explain"]["phrases"] == EX.PHRASES and fx["ghost_params"]["fan_deg"] == list(GH.FAN_DEG)
    for rec in fx["frames"]:
        assert {"state", "pass_features", "value_features", "shot_features", "options"} <= set(rec)
        assert [e["kind"] for e in rec["explanations"]] == [o["kind"] for o in rec["options"]]
        assert set(rec["ghosts"]) == {"runs", "defence"} and np.isfinite(rec["v_now"])
    json.dumps(fx, allow_nan=False)
```

- [ ] **Step 2: Run, expect FAIL** (missing module / attribute): `$PY -m pytest tests/test_20_explain.py -q -k "parity"`

- [ ] **Step 3: Implement**

```diff
diff --git a/20-football-analytics/coach/parity_fixtures.py b/20-football-analytics/coach/parity_fixtures.py
index 7e669e3..e6b26b6 100644
--- a/20-football-analytics/coach/parity_fixtures.py
+++ b/20-football-analytics/coach/parity_fixtures.py
@@ -9,6 +9,11 @@ state the way the browser does: int16 cm -> float64 metres, 12.5 fps, velocity =
 rule and ranks the options with the library's own features.py / advice.py on the exported weights. It writes the
 state, features and ranked options of up to `per_half` frames per half to the site's tests/fixtures/coach-parity.json;
 the site's vitest asserts |diff| < 1e-6 and the same option order.
+
+Part B (explanations and ghosts) adds, per frame, "v_now", "explanations" (coach.explain, aligned with "options":
+contributions, value terms, phrases, signs, fragile) and "ghosts" ({"runs", "defence"}, coach.ghosts), and at the top
+"explain" (the phrase table and its constants) and "ghost_params"; the earlier fields are unchanged. The site compares
+numbers at the tolerance, and phrases / ghost choices exactly unless flagged fragile (a boundary within 1e-6 / 1e-9).
 """
 
 import argparse
@@ -19,7 +24,9 @@ from types import SimpleNamespace
 import numpy as np
 
 from . import advice as A
+from . import explain as EX
 from . import features as F
+from . import ghosts as GH
 from .models import linear_predictor
 from .prepare import CV_DATA, HERE
 
@@ -69,11 +76,13 @@ def frame_record(meta: dict, half: int, h, team_of, frame: int, models: dict):
     direction = meta["halves"][half]["attack"][team]
     s = F.state(m, half, frame, team, fps=meta["fps"], back_frames=BACK_SITE, direction=direction)
     row = int(np.flatnonzero(s["att_idx"] == player)[0])
-    opts = A.options(s, row, predictors(models))
+    pred = predictors(models)
+    opts = A.options(s, row, pred)
     scores = [o["score"] for o in opts]
     if any(abs(a - b) < TIE for a, b in zip(scores, scores[1:])):
         return None
     here = s["att"][row]
+    v_now = float(A.value_at(pred, [here])[0])
     return {
         "half": half,
         "frame": frame,
@@ -96,6 +105,9 @@ def frame_record(meta: dict, half: int, h, team_of, frame: int, models: dict):
         "value_features": F.value_features(here),
         "shot_features": F.shot_features(here, s["def"]),
         "options": opts,
+        "v_now": v_now,
+        "explanations": EX.explain_options(s, row, opts, models, v_now),
+        "ghosts": {"runs": GH.ghost_runs(s, row, pred), "defence": GH.ghost_defence(s, row, pred)},
     }
 
 
@@ -135,6 +147,8 @@ def build(export: Path, per_half: int = 15) -> dict:
         "tolerance": 1e-6,
         "models": models,
         "frames": frames,
+        "explain": {"phrases": EX.PHRASES, "top_features": EX.TOP_FEATURES, "fragile": EX.FRAGILE},
+        "ghost_params": GH.params(),
     }
```

- [ ] **Step 4: Run, expect PASS**: same command, then the regression
  `$PY -m pytest tests/test_20_coach.py tests/test_20_report.py -q`; lint `$PY -m ruff check 20-football-analytics/coach/parity_fixtures.py tests/test_20_explain.py && $PY -m ruff format --check 20-football-analytics/coach/parity_fixtures.py tests/test_20_explain.py`
- [ ] **Step 4b: Regenerate the site fixture** (`$PY -m coach.parity_fixtures`) and check with a script that every
  earlier key of every frame and of the top level equals the committed site file (`git show
  HEAD:tests/fixtures/coach-parity.json` in the website repo); report the new size. Do not commit in the website repo.

- [ ] **Step 5: Commit**: `git add 20-football-analytics/coach/parity_fixtures.py tests/test_20_explain.py && git commit -m "20 coach: parity fixture carries explanations and ghosts (earlier fields unchanged)" -- 20-football-analytics/coach/parity_fixtures.py tests/test_20_explain.py`


### Task 6: README v3 "Explanations and ghosts"

**Files:** Modify `20-football-analytics/README.md` (new subsection after "Match report", run command, test count).

- [ ] Write: method (contributions, value split, phrase table, fan / thresholds / greedy defence), an example
  explanation and ghost from J03WOH, the association table with match-bootstrap CIs labelled "association only",
  timing (Python per frame, node estimate, evaluation counts), limits ("model suggestion, not ground truth"; nobody
  reacts; the moved player stands still; the value model is weak).
- [ ] Commit `-- 20-football-analytics/README.md`, push, check CI, give the PROGRESS.md line to the coordinator (not
  edited here).

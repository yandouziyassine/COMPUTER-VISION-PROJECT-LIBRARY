# Coach insight A: match report (library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `report.json` for one match (every synced pass: chosen option vs the coach's best option, gap in xG, rank,
outcome within 15 s; per player, per team and per third; two highlight lists), built with models trained WITHOUT that
match, exported for J03WOH into the site-export folder, plus held-out summary numbers in the README.

**Architecture:** The per-pass work (options at the release frame, the chosen pass, net xG within the horizon) moves
into `evaluate.pass_decisions`, shared by the held-out advice evaluation (`evaluate.advice_records`) and the report.
`export.fit_models` (factored out of `export.export_models`) fits the site's three models, so the report uses the exact
site recipe. `coach/report.py` turns decisions into rows, aggregates and highlights; `coach/export_report.py` writes
and checks `report.json` against the site's `match.json`. matplotlib is imported lazily everywhere (CI has none).

**Tech Stack:** NumPy, scikit-learn (existing `coach.models`), matplotlib (charts only, lazy), pytest.

Spec: `docs/superpowers/specs/2026-11-02-coach-insight-design.md`, section A. Website repo: NOT touched.

**Conventions (all tasks)**

- `PY=C:/Users/yando/Code/.venv-tools/Scripts/python.exe`. Tests from the repo root:
  `$PY -m pytest tests/test_20_report.py -q`; regression: `$PY -m pytest tests/test_20_coach.py -q`.
  Scripts from `20-football-analytics/`: `$PY -m coach.report`, `$PY -m coach.export_report ...`.
- Lint: `$PY -m ruff check <paths> && $PY -m ruff format --check <paths>`.
- Commit only own paths: `git add <paths> && git commit -m "..." -- <paths>`; never `git add -A`, never stash/reset.
- Frames: `frame_src` = index in the source half (25 fps IDSSE, 10 fps SkillCorner); `frame` = index at the site's
  rate = `frame_src // step`, `step = max(1, round(fps_src / 12.5))` (2 for IDSSE, exactly `export._site_event`).
  `event` = index in `m.events` = index in the site's `match.json` events (pack_match writes all events in order).
- Definitions: options = every option of `advice.options` (passes, carries, shot), sorted by score; best = first;
  `gap = max(0, score(best) - score(chosen))`; `rank` = 1 + position of the chosen pass in that order;
  `above_median = score(chosen) - median(option scores)`; third from the carrier's attack-normalised x
  (< -17.5 defensive, > 17.5 attacking); outcome = `goal` / `shot` if the passing team scores / shoots within 15 s,
  else `kept` / `lost` (pass completed or not), plus `net_xg`.
- Thresholds: `MIN_DECISIONS = 20` (`too_few`), highlights: 10 missed (largest gap, gap > 0, `p_best >= 0.5`),
  10 best (`rank == 1`, largest `above_median`).

---

### Task 1: shared per-pass decisions in evaluate.py (and lazy matplotlib)

**Files:**
- Modify: `20-football-analytics/coach/evaluate.py` (imports, `advice_records`, the three chart functions)
- Create: `tests/test_20_report.py`

- [ ] **Step 1: Write the failing test** (new file with the synthetic match used by every later task)

```python
"""Project 20, AI coach: match report (decision quality, missed opportunities). Synthetic matches only."""

import json
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import evaluate as EV  # noqa: E402
from coach import match as M  # noqa: E402


def small_match(frames=400, match_id="SYN"):
    """Two halves at 25 fps, 3 players per team standing still; home attacks +x in half 0 and -x in half 1.
    Events: a completed pass 0 -> 1 (half 0, frame 41), a home goal 1.56 s later, a completed pass 1 -> 2 (half 1)."""
    home = np.array([[-10.0, 0.0], [5.0, 8.0], [15.0, -6.0]])
    guest = np.array([[10.0, 0.0], [-5.0, -8.0], [25.0, 12.0]])
    halves = []
    for h in range(2):
        xy = np.repeat(np.concatenate([home, guest])[None], frames, axis=0).astype(np.float32)
        ball = np.zeros((frames, 3), np.float32)
        ball[:, :2] = home[0]
        if h == 1:  # sides swap at half time
            xy[..., 0] *= -1
            ball[:, 0] *= -1
        halves.append(
            M.Half(
                n=np.arange(frames, dtype=np.int32),
                t=1000.0 + h * 4000 + np.arange(frames) / 25.0,
                xy=xy,
                ball=ball,
                owner=np.zeros(frames, np.int8),
                alive=np.ones(frames, bool),
            )
        )
    m = M.Match(
        match_id,
        list("abcdef"),
        np.array([0, 0, 0, 1, 1, 1], np.int8),
        np.array([1, 2, 3, 1, 2, 3], np.int16),
        list("ABCDEF"),
        halves,
        [],
        {"home": "Home FC", "guest": "Away FC"},
    )

    def ev(kind, half, frame, **kw):
        base = {"type": kind, "half": half, "frame": frame, "t": float(m.halves[half].t[frame]), "team": 0}
        return {**base, "player": 0, "set_piece": None, "cross": False, "x": 0.0, "y": 0.0, **kw}

    m.events = [
        ev("pass", 0, 41, recipient=1, success=True),
        ev("shot", 0, 80, goal=True, xg=0.3),
        ev("pass", 1, 61, player=1, recipient=2, success=True),
    ]
    return m


MODELS = {
    "pass": lambda X_: 1 / (1 + np.exp(-(3.0 - 0.1 * X_[:, 0]))),  # completion falls with distance
    "value": lambda X_: 0.1 * X_[:, 0] / 52.5,  # closer to the goal is better
    "shot": lambda X_: np.full(len(X_), 0.05),
}


def test_pass_decisions_choose_the_recorded_receiver_and_net_xg():
    ds = list(EV.pass_decisions(small_match(), MODELS))
    assert [d["chosen"]["target"] for d in ds] == [1, 2] and [d["sample"]["index"] for d in ds] == [0, 2]
    assert ds[0]["outcome"] == pytest.approx(0.3) and ds[1]["outcome"] == 0.0  # the goal 1.56 s after the first pass
    assert ds[0]["options"] == sorted(ds[0]["options"], key=lambda o: -o["score"])
    assert {o["kind"] for o in ds[0]["options"]} == {"pass", "carry"} and ds[0]["t"] == pytest.approx(1000 + 41 / 25)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: FAIL, `AttributeError: module 'coach.evaluate' has no attribute 'pass_decisions'`.

- [ ] **Step 3: Implement**

In `coach/evaluate.py` replace the top-level matplotlib block by plain imports and a lazy helper:

```python
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from . import advice as A
from . import datasets as D
from . import features as F
from . import models as ML
from .match import load
from .prepare import MATCHES, OUT
```

```python
def _pyplot():
    """matplotlib only for the charts (the report and the tests import this module without it)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt
```

and add `plt = _pyplot()` as the first line of `reliability_chart`, `decile_chart` and `bucket_chart`. Add the shared
generator and rewrite `advice_records` on top of it (same records as before):

```python
def pass_decisions(m, models, horizon=D.HORIZON):
    """Per usable pass (datasets.pass_samples): every option of the carrier sorted by coach score (advice.options), the
    pass option the player chose, the release time on the tracking clock and the passing team's net xG within
    `horizon` s. Shared by the held-out advice evaluation and the match report (coach.report)."""
    shots = D.shots_by_half(m)
    for p in D.pass_samples(m):
        e, s = p["event"], p["state"]
        target = int(s["att_idx"][p["target"]])
        opts = A.options(s, p["carrier"], models)
        chosen = next(o for o in opts if o["kind"] == "pass" and o["target"] == target)
        t = D.when(m, e)
        yield {
            "sample": p,
            "options": opts,
            "chosen": chosen,
            "t": t,
            "outcome": D.net_xg(shots[e["half"]], t, e["team"], horizon),
        }


def advice_records(m, models, xt_model, horizon, w_fail) -> list:
    """One record per usable pass (completed or not): the coach's view of the option the pro chose, and what came next."""
    out = []
    for d in pass_decisions(m, models, horizon):
        p, chosen = d["sample"], d["chosen"]
        e, s, c = p["event"], p["state"], p["carrier"]
        passes = [o for o in d["options"] if o["kind"] == "pass"]  # sorted by coach score
        here = s["att"][c]
        dist = lambda o, here=here: float(np.hypot(o["to"][0] - here[0], o["to"][1] - here[1]))
        xt = float(
            xt_model.predict(np.array([[F.value_features(np.array(chosen["to"]))[k] for k in F.VALUE_FEATURES]]))[0]
        )
        out.append(
            {
                "match": m.match_id,
                "success": int(e["success"]),
                "weight": 1.0 if e["success"] else w_fail,
                "score": chosen["score"],
                "v_now": float(A.value_at(models, [here])[0]),
                "p": chosen["p"],
                "value": chosen["value"],
                "xt": xt,
                "outcome": d["outcome"],
                "rank_coach": 1 + passes.index(chosen),
                "rank_p": 1 + sum(o["p"] > chosen["p"] for o in passes),
                "rank_value": 1 + sum(o["value"] > chosen["value"] for o in passes),
                "rank_near": 1 + sum(dist(o) < dist(chosen) for o in passes),
                "options": len(passes),
            }
        )
    return out
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_20_report.py tests/test_20_coach.py -q`
Expected: all pass.

- [ ] **Step 5: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/evaluate.py tests/test_20_report.py && $PY -m ruff format --check 20-football-analytics/coach/evaluate.py tests/test_20_report.py
git add tests/test_20_report.py && git commit -m "20 coach: shared per-pass decisions (evaluate.pass_decisions), matplotlib imported lazily" -- 20-football-analytics/coach/evaluate.py tests/test_20_report.py
```

### Task 2: fit the site's three models in one place (export.fit_models)

**Files:**
- Modify: `20-football-analytics/coach/export.py` (`export_models`)

- [ ] **Step 1: Implement** (pure refactor; behaviour pinned by `test_export_models_*` in `tests/test_20_coach.py`)

```python
def training_tables(train) -> dict:
    """Stacked pass / value / shot (penalties out) tables of the training matches."""
    return {
        "pass": D.stack([D.pass_table(m) for m in train]),
        "value": D.stack([D.value_table(m) for m in train]),
        "shot": D.drop_penalties(D.stack([D.shot_table(m) for m in train])),
    }


def fit_models(tables: dict, extra_shots=None) -> dict:
    """The website's three models (fitted sklearn objects) from training tables; `extra_shots` (X, y, match: open-play
    shots of another league) is added to the shot model's rows only."""
    p, v, s = tables["pass"], tables["value"], tables["shot"]
    if extra_shots is not None and len(extra_shots["y"]):
        s = {"X": np.vstack([s["X"], extra_shots["X"]]), "y": np.concatenate([s["y"], extra_shots["y"]])}
    return {
        "pass": ML.fit_logistic(p["X"], p["y"], p["weight"]),
        "value": ML.fit_ridge(v["X"], v["y"]),
        "shot": ML.fit_logistic(s["X"], s["y"]),
    }
```

`export_models` then reads:

```python
    train = [m for m in matches if m.match_id != exclude]
    tables = training_tables(train)
    fitted = fit_models(tables, extra_shots)
    n_extra = 0 if extra_shots is None else len(extra_shots["y"])
    weights = {
        "trained_without": exclude,
        "trained_on": [m.match_id for m in train],
        "shot_trained_on": {
            "idsse": [m.match_id for m in train],
            "skillcorner_matches": len(set(extra_shots["match"].tolist())) if n_extra else 0,
            "shots": len(tables["shot"]["y"]) + n_extra,
        },
        "pass": ML.export_logistic(fitted["pass"], D.F.PASS_FEATURES),
        "value": {
            **ML.export_ridge(fitted["value"], D.F.VALUE_FEATURES),
            ...unchanged...
        },
        "shot": ML.export_logistic(fitted["shot"], D.F.SHOT_FEATURES),
        ...unchanged...
```

- [ ] **Step 2: Run the regression tests**

Run: `$PY -m pytest tests/test_20_coach.py -q -k "export or pack"`
Expected: PASS (shots 120 alone, 140 pooled; pass/value weights unchanged by the extra shots).

- [ ] **Step 3: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/export.py && $PY -m ruff format --check 20-football-analytics/coach/export.py
git commit -m "20 coach: export.fit_models / training_tables (the site's model recipe in one place)" -- 20-football-analytics/coach/export.py
```

### Task 3: one decision row (frames, gap, rank, best option, third, outcome)

**Files:**
- Create: `20-football-analytics/coach/report.py`
- Modify: `tests/test_20_report.py`

- [ ] **Step 1: Write the failing tests** (append; add `from coach import export as X` and `from coach import report as R`
  to the imports)

```python
def test_site_step_matches_the_export_frame_indexing():
    assert R.site_step(25.0) == 2 and R.site_step(10.0) == 1 and R.site_step(25.0, site_fps=25.0) == 1
    for f in (0, 1, 40, 41, 12345):
        assert f // R.site_step(25.0) == X._site_event({"frame": f}, 2)["frame"]


def opt(kind, score, p, **kw):
    return {"kind": kind, "score": score, "p": p, **kw}


def record(opts, chosen_target, frame=41, half=0, success=True, net=0.01, carrier_x=0.0, t=1001.64):
    """An evaluate.pass_decisions record built by hand."""
    opts = sorted(opts, key=lambda o: -o["score"])
    chosen = next(o for o in opts if o["kind"] == "pass" and o["target"] == chosen_target)
    e = {"half": half, "frame": frame, "team": 0, "player": 7, "success": success}
    s = {"att": np.array([[carrier_x, 0.0]])}
    sample = {"index": 3, "event": e, "state": s, "carrier": 0}
    return {"sample": sample, "options": opts, "chosen": chosen, "t": t, "outcome": net}


NO_SHOTS = [np.zeros((0, 3)), np.zeros((0, 3))]


def test_decision_gap_rank_best_and_frames():
    opts = [
        opt("pass", 0.02, 0.6, target=1),
        opt("carry", 0.01, 0.9, angle=30),
        opt("pass", -0.005, 0.95, target=2),
        opt("shot", -0.03, 0.04),
    ]
    row = R.decision_row(record(opts, chosen_target=2), step=2, shots=NO_SHOTS)
    assert row["gap"] == pytest.approx(0.025) and row["rank"] == 3 and row["options"] == 4
    assert row["best"] == {"kind": "pass", "target": 1} and row["p_best"] == 0.6 and row["p_chosen"] == 0.95
    assert (row["frame_src"], row["frame"], row["carrier"], row["target"], row["event"]) == (41, 20, 7, 2, 3)
    assert row["above_median"] == pytest.approx(-0.005 - 0.0025)  # median of 0.02, 0.01, -0.005, -0.03
    assert row["third"] == "middle" and row["outcome"] == "kept" and row["net_xg"] == 0.01


def test_decision_chosen_best_has_zero_gap_and_a_carry_is_keyed_by_angle():
    row = R.decision_row(record([opt("pass", 0.03, 0.7, target=4), opt("carry", 0.01, 0.9, angle=-60)], 4, carrier_x=30.0), 2, NO_SHOTS)
    assert row["gap"] == 0 and row["rank"] == 1 and row["third"] == "attacking"
    row = R.decision_row(record([opt("pass", 0.0, 0.7, target=4), opt("carry", 0.01, 0.9, angle=-60)], 4, carrier_x=-30.0), 2, NO_SHOTS)
    assert row["best"] == {"kind": "carry", "angle": -60} and row["gap"] == pytest.approx(0.01)
    assert row["third"] == "defensive"


def test_gap_is_never_negative_and_rank_counts_better_options():
    rng = np.random.default_rng(0)
    for _ in range(200):
        k = int(rng.integers(2, 12))
        opts = [opt("pass", float(rng.normal(0, 0.02)), float(rng.random()), target=j) for j in range(k)]
        tgt = int(rng.integers(0, k))
        row = R.decision_row(record(opts, tgt), 2, NO_SHOTS)
        assert row["gap"] >= 0 and row["rank"] == 1 + sum(o["score"] > opts[tgt]["score"] for o in opts)


def test_outcome_goal_shot_kept_lost_within_the_horizon():
    shots = np.array([[105.0, 0, 0], [110.0, 1, 1], [120.0, 0, 1]])  # time, team, goal
    assert R.outcome(shots, 100.0, 0, True) == "shot"  # own shot at +5 s; own goal at +20 s is too late
    assert R.outcome(shots, 106.0, 0, False) == "goal"  # own goal at +14 s
    assert R.outcome(shots, 107.0, 0, True, horizon=10.0) == "kept"  # only the opponent's goal in (107, 117]
    assert R.outcome(shots, 121.0, 0, True) == "kept" and R.outcome(shots, 121.0, 0, False) == "lost"
    assert R.outcome(shots, 100.0, 1, False) == "goal" and R.outcome(np.zeros((0, 3)), 0.0, 0, True) == "kept"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: FAIL at import, `cannot import name 'report' from 'coach'`.

- [ ] **Step 3: Implement** `coach/report.py` (first part)

```python
"""Match report (coach insight, part A): decision quality and missed opportunities in one match.

    python -m coach.report      # from 20-football-analytics/: held-out summary over the 7 IDSSE matches
                                # -> docs/coach_report_results.json + docs/coach_report_gaps.png

For every pass the sync kept (datasets.pass_samples), with models trained WITHOUT the match (asserted in build): the
coach's options at the release frame (advice.options: passes, carries, shot), the pass the player chose, the best
option, gap = score(best) - score(chosen) >= 0 in expected goals, the chosen option's rank among all options, and what
happened within HORIZON s: goal / shot by the passing team, else kept / lost (the pass completed or not), with the net
xG. The per-pass work is evaluate.pass_decisions, shared with the held-out advice evaluation.

Frames: "frame_src" indexes the source half (25 fps IDSSE, 10 fps SkillCorner); "frame" is the index at the site's
rate, frame_src // step with step = round(fps_src / 12.5): 2 for IDSSE (the 12.5 fps of export.pack_match), 1 for
SkillCorner. "event" is the index in m.events, which is also the index in the site's match.json events.

"Missed opportunity" means the model valued another option more, not a mistake (see CAVEATS). Player and team numbers
carry n and too_few (n < MIN_DECISIONS): below it the site shows "too few decisions" instead of the numbers.
"""

import json
from pathlib import Path

import numpy as np

from . import advice as A
from . import datasets as D
from . import evaluate as EV
from . import export as X
from . import features as F

SITE_FPS = 12.5
HORIZON = D.HORIZON
MIN_DECISIONS = 20
MIN_BEST_P = 0.5
N_HIGHLIGHTS = 10
THIRD_M = 52.5 / 3  # pitch thirds at x = -17.5 / +17.5 m, attack-normalised (the passing team attacks +x)
THIRDS = ("defensive", "middle", "attacking")
DOCS = Path(__file__).resolve().parents[1] / "docs"


def site_step(fps_src: float, site_fps: float = SITE_FPS) -> int:
    """Whole subsampling step from the source frame rate to the site's (25 -> 2, as export.pack_match; 10 -> 1)."""
    return max(1, int(round(fps_src / site_fps)))


def third(x: float) -> str:
    return "defensive" if x < -THIRD_M else "attacking" if x > THIRD_M else "middle"


def option_key(o: dict) -> dict:
    """What identifies an option: its kind, plus the target player (pass) or the direction in degrees (carry)."""
    key = {"kind": o["kind"]}
    if o["kind"] == "pass":
        key["target"] = int(o["target"])
    elif o["kind"] == "carry":
        key["angle"] = int(o["angle"])
    return key


def shots_with_goals(m) -> list:
    """Per half: array (time on the tracking clock, team, goal 0/1) of every shot with a team, sorted by time."""
    out = []
    for hi in range(len(m.halves)):
        rows = sorted(
            (D.when(m, e), e["team"], float(bool(e.get("goal"))))
            for e in m.events
            if e["type"] == "shot" and e.get("half") == hi and e["team"] in (0, 1)
        )
        out.append(np.array(rows, float).reshape(-1, 3))
    return out


def outcome(shots, t: float, team: int, success: bool, horizon: float = HORIZON) -> str:
    """goal / shot when the passing team scores / shoots in (t, t + horizon], else kept / lost (pass completed or not).
    `shots`: one half of shots_with_goals."""
    if len(shots):
        mine = (shots[:, 0] > t) & (shots[:, 0] <= t + horizon) & (shots[:, 1] == team)
        if (shots[mine, 2] > 0).any():
            return "goal"
        if mine.any():
            return "shot"
    return "kept" if success else "lost"


def _r(x, nd=4):
    return round(float(x), nd)


def decision_row(d: dict, step: int, shots: list, horizon: float = HORIZON) -> dict:
    """One decision from an evaluate.pass_decisions record (options sorted by coach score, best first)."""
    p, opts, chosen = d["sample"], d["options"], d["chosen"]
    e, s = p["event"], p["state"]
    best = opts[0]
    scores = np.array([o["score"] for o in opts], float)
    return {
        "event": int(p["index"]),
        "half": int(e["half"]),
        "frame": int(e["frame"]) // step,
        "frame_src": int(e["frame"]),
        "team": int(e["team"]),
        "carrier": int(e["player"]),
        "target": int(chosen["target"]),
        "best": option_key(best),
        "gap": _r(max(0.0, best["score"] - chosen["score"]), 5),
        "rank": 1 + next(i for i, o in enumerate(opts) if o is chosen),
        "options": len(opts),
        "p_chosen": _r(chosen["p"], 3),
        "p_best": _r(best["p"], 3),
        "score_chosen": _r(chosen["score"], 5),
        "above_median": _r(chosen["score"] - float(np.median(scores)), 5),
        "third": third(float(s["att"][p["carrier"]][0])),
        "outcome": outcome(shots[e["half"]], d["t"], e["team"], bool(e["success"]), horizon),
        "net_xg": _r(d["outcome"], 4),
    }
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/report.py tests/test_20_report.py && $PY -m ruff format --check 20-football-analytics/coach/report.py tests/test_20_report.py
git add 20-football-analytics/coach/report.py && git commit -m "20 coach: match report decision rows (gap, rank, best option, site frames, outcome)" -- 20-football-analytics/coach/report.py tests/test_20_report.py
```

### Task 4: aggregation, highlights, held-out models and the full report

**Files:**
- Modify: `20-football-analytics/coach/report.py`
- Modify: `tests/test_20_report.py` (add `from coach import datasets as D`)

- [ ] **Step 1: Write the failing tests**

```python
def row(carrier=0, team=0, rank=1, gap=0.0, third="middle", p_best=0.8, above=0.01, kind="pass"):
    return {"carrier": carrier, "team": team, "rank": rank, "gap": gap, "third": third, "p_best": p_best,
            "above_median": above, "best": {"kind": kind}}


def test_summarise_shares_gaps_and_too_few():
    rows = [row(rank=1), row(rank=2, gap=0.01), row(rank=3, gap=0.02), row(rank=5, gap=0.03)]
    assert R.summarise(rows) == {"n": 4, "too_few": True, "top1": 0.25, "top3": 0.75, "mean_gap": 0.015, "total_gap": 0.06}
    many = R.summarise(rows * 5)
    assert many["n"] == 20 and many["too_few"] is False and many["total_gap"] == pytest.approx(0.3)
    assert R.summarise([])["n"] == 0 and R.summarise([])["top1"] is None and R.summarise([])["too_few"] is True


def test_players_and_teams_aggregate_with_thirds():
    m = small_match()
    rows = [row(carrier=0)] * 25 + [row(carrier=1, rank=4, gap=0.02, third="attacking")] * 5
    rows += [row(carrier=3, team=1, third="defensive")] * 2
    players = R.by_player(m, rows)
    assert set(players) == {"0", "1", "3"} and players["0"]["n"] == 25 and players["0"]["too_few"] is False
    assert players["1"]["too_few"] is True and players["1"]["total_gap"] == pytest.approx(0.1)
    assert (players["0"]["name"], players["1"]["shirt"], players["3"]["team"]) == ("A", 2, 1)
    teams = R.by_team(m, rows)
    assert teams["0"]["n"] == 30 and teams["0"]["name"] == "Home FC" and teams["1"]["n"] == 2
    assert teams["0"]["by_third"]["attacking"]["n"] == 5 and teams["0"]["by_third"]["defensive"]["n"] == 0
    assert teams["0"]["top1"] == pytest.approx(25 / 30, abs=1e-3)


def test_highlights_filter_on_best_p_and_rank_one():
    rows = [
        row(rank=2, gap=0.05, p_best=0.3),  # the coach asks for a miracle: not a missed opportunity
        row(rank=2, gap=0.04, p_best=0.6),
        row(rank=1, gap=0.0, above=0.02),
        row(rank=3, gap=0.01, p_best=0.5),
        row(rank=1, above=0.05),
        row(rank=2, gap=0.03, p_best=0.9, above=0.09),  # far above the median but not the coach's best
    ]
    h = R.highlights(rows)
    assert [e["decision"] for e in h["missed"]] == [1, 5, 3] and h["missed"][0]["gap"] == 0.04
    assert [e["decision"] for e in h["best"]] == [4, 2]
    big = [row(rank=2, gap=0.001 * i, p_best=0.9) for i in range(1, 30)]
    assert len(R.highlights(big)["missed"]) == 10 and R.highlights(big)["missed"][0]["gap"] == pytest.approx(0.029)


def test_build_report_on_a_synthetic_match():
    rep = R.build(small_match(), MODELS, trained_on=["OTHER"])
    assert rep["trained_without"] == "SYN" and rep["fps"] == 12.5 and rep["fps_src"] == 25.0
    d0, d1 = rep["decisions"]
    assert (d0["half"], d0["frame_src"], d0["frame"], d0["event"], d0["outcome"]) == (0, 41, 20, 0, "goal")
    assert (d1["half"], d1["frame"], d1["carrier"], d1["target"], d1["outcome"]) == (1, 30, 1, 2, "kept")
    assert all(d["gap"] >= 0 and 1 <= d["rank"] <= d["options"] for d in rep["decisions"])
    assert rep["players"]["0"]["n"] == 1 and rep["players"]["0"]["too_few"] is True
    assert rep["summary"]["n"] == 2 and rep["summary"]["open_play_passes"] == 2 and len(rep["caveats"]) >= 4
    assert set(rep["highlights"]) == {"missed", "best"} and len(json.dumps(rep)) < 300_000


def test_build_refuses_models_trained_on_the_match():
    with pytest.raises(AssertionError):
        R.build(small_match(), MODELS, trained_on=["A", "SYN"])


def fake_tables(mid, rng):
    nP, nV, nS = len(R.F.PASS_FEATURES), len(R.F.VALUE_FEATURES), len(R.F.SHOT_FEATURES)
    Xp, Xs = rng.normal(size=(80, nP)), rng.normal(size=(30, nS))
    ids = lambda n: np.array([mid] * n)
    return {
        "pass": {"X": Xp, "y": (Xp[:, 0] > 0).astype(int), "match": ids(80), "weight": np.ones(80)},
        "value": {"X": rng.normal(size=(80, nV)), "y": rng.normal(size=80), "match": ids(80)},
        "shot": {"X": Xs, "y": (Xs[:, 0] > 0).astype(int), "match": ids(30), "penalty": np.zeros(30, bool)},
    }


def test_held_out_models_never_see_the_target_match():
    rng = np.random.default_rng(0)
    tables = {mid: fake_tables(mid, rng) for mid in ("A", "B", "C")}
    Xe = rng.normal(size=(40, len(R.F.SHOT_FEATURES)))
    extra = {"X": Xe, "y": (Xe[:, 1] > 0).astype(int), "match": np.array(["B"] * 10 + ["s1"] * 30)}
    models, fitted, trained_on = R.held_out_models("B", tables, extra)
    assert trained_on == ["A", "C", "s1"] and set(models) == {"pass", "value", "shot"}
    ref = X.fit_models(
        {k: D.stack([tables["A"][k], tables["C"][k]]) for k in ("pass", "value", "shot")},
        {k: v[10:] for k, v in extra.items()},
    )
    for k in ("pass", "shot"):
        Xk = tables["B"][k]["X"]
        assert np.allclose(fitted[k].predict_proba(Xk), ref[k].predict_proba(Xk))
    assert np.allclose(models["value"](tables["B"]["value"]["X"]), ref["value"].predict(tables["B"]["value"]["X"]))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: FAIL, `AttributeError: module 'coach.report' has no attribute 'summarise'` (and the others).

- [ ] **Step 3: Implement** (append to `coach/report.py`)

```python
CAVEATS = [
    "Everything here is the model's opinion. A missed opportunity means the model valued another option more, "
    "not that the player made a mistake.",
    "The value model is weak: on held-out matches it explains about 2 % of the variance of the net xG in the next "
    "15 s, little more than 'closer to the goal is better'.",
    "The coach is bolder than professional players: its top-ranked options are riskier, and pros complete more of the "
    "passes it ranks lower.",
    "Only passes the event sync placed on the tracking are included (about two thirds of open-play passes in the "
    "Bundesliga data); the intended receiver of a failed pass is inferred from the ball's path.",
    f"Player and team numbers need at least {MIN_DECISIONS} decisions; below that they are flagged too_few.",
]
FIELDS = {
    "event": "index in the match's events (= match.json events)",
    "half": "0 or 1",
    "frame": "frame index in the half at the site's rate (fps)",
    "frame_src": "frame index in the half at the source rate (fps_src)",
    "team": "0 home, 1 guest",
    "carrier": "player index (match.json players) of the passer",
    "target": "player index of the receiver the passer chose (given or inferred)",
    "best": "the coach's best option: kind (pass / carry / shot), target player (pass) or angle in degrees (carry, "
    "relative to the attacking direction)",
    "gap": "score(best) - score(chosen), expected goals, >= 0",
    "rank": "rank of the chosen pass among all options by coach score (1 = the coach's best)",
    "options": "number of options",
    "p_chosen": "success probability of the chosen pass",
    "p_best": "success probability of the best option",
    "score_chosen": "coach score of the chosen pass (expected change of the net xG)",
    "above_median": "score(chosen) - median score of the options, expected goals",
    "third": "pitch third of the passer (defensive / middle / attacking, for the passing team)",
    "outcome": "within horizon_s: goal / shot by the passing team, else kept / lost (pass completed or not)",
    "net_xg": "xG of the passing team minus xG against within horizon_s",
}


def summarise(rows) -> dict:
    """n, share of choices that were the coach's top-1 / top-3 option, mean and total gap (expected goals);
    too_few when n < MIN_DECISIONS."""
    n = len(rows)
    gaps = np.array([r["gap"] for r in rows], float)
    ranks = np.array([r["rank"] for r in rows], int)
    return {
        "n": n,
        "too_few": n < MIN_DECISIONS,
        "top1": _r(np.mean(ranks == 1), 3) if n else None,
        "top3": _r(np.mean(ranks <= 3), 3) if n else None,
        "mean_gap": _r(gaps.mean(), 5) if n else None,
        "total_gap": _r(gaps.sum(), 4),
    }


def by_player(m, rows) -> dict:
    """Per passer (key: player index as a string): name, team, shirt and summarise()."""
    out = {}
    for pid in sorted({r["carrier"] for r in rows}):
        info = {"name": m.names[pid], "team": int(m.team[pid]), "shirt": int(m.shirt[pid])}
        out[str(pid)] = {**info, **summarise([r for r in rows if r["carrier"] == pid])}
    return out


def by_team(m, rows) -> dict:
    """Per team ("0" home, "1" guest): name, summarise() and the same per pitch third of the passer."""
    out = {}
    for t in (0, 1):
        mine = [r for r in rows if r["team"] == t]
        out[str(t)] = {
            "name": m.meta.get(("home", "guest")[t], str(t)),
            **summarise(mine),
            "by_third": {k: summarise([r for r in mine if r["third"] == k]) for k in THIRDS},
        }
    return out


def highlights(rows, k: int = N_HIGHLIGHTS, min_best_p: float = MIN_BEST_P) -> dict:
    """missed: the k largest gaps among decisions whose best option succeeds with p >= min_best_p (the coach is not
    asking for a miracle); best: the k decisions where the chosen pass was the coach's best option, by how far its score
    is above the median option (the player found the rare good option). Entries point into "decisions"."""
    missed = [i for i, r in enumerate(rows) if r["gap"] > 0 and r["p_best"] >= min_best_p]
    missed = sorted(missed, key=lambda i: (-rows[i]["gap"], i))[:k]
    best = sorted((i for i, r in enumerate(rows) if r["rank"] == 1), key=lambda i: (-rows[i]["above_median"], i))[:k]
    return {
        "missed": [{"decision": i, "gap": rows[i]["gap"]} for i in missed],
        "best": [{"decision": i, "above_median": rows[i]["above_median"]} for i in best],
    }


def gap_summary(rows) -> dict:
    """summarise() plus the gap distribution, the kind of the best option and the highlight candidate counts."""
    out = summarise(rows)
    if not rows:
        return out
    gaps = np.array([r["gap"] for r in rows], float)
    kinds = [r["best"]["kind"] for r in rows]
    q = np.percentile(gaps, [50, 75, 90, 99])
    out.update(
        {
            "gap_median": _r(q[0], 5),
            "gap_p75": _r(q[1], 5),
            "gap_p90": _r(q[2], 5),
            "gap_p99": _r(q[3], 5),
            "gap_max": _r(gaps.max(), 5),
            "best_kind": {kd: _r(kinds.count(kd) / len(kinds), 3) for kd in ("pass", "carry", "shot")},
            "missed_candidates": sum(r["gap"] > 0 and r["p_best"] >= MIN_BEST_P for r in rows),
            "best_candidates": sum(r["rank"] == 1 for r in rows),
        }
    )
    return out


def build(m, models: dict, trained_on, horizon: float = HORIZON) -> dict:
    """The report of match m. `models`: advice.from_fitted-style functions; `trained_on`: every match id any of the
    models learned from (asserted not to contain m)."""
    trained_on = [str(t) for t in trained_on]
    assert str(m.match_id) not in trained_on, f"match {m.match_id} is in the training set"
    fps_src = F.match_fps(m)
    step = site_step(fps_src)
    shots = shots_with_goals(m)
    rows = [decision_row(d, step, shots, horizon) for d in EV.pass_decisions(m, models, horizon)]
    open_play = sum(
        e["type"] == "pass" and not e.get("set_piece") and not e.get("cross") and e["team"] >= 0 for e in m.events
    )
    return {
        "match_id": str(m.match_id),
        "fps": fps_src / step,
        "fps_src": fps_src,
        "horizon_s": horizon,
        "trained_without": str(m.match_id),
        "trained_on": trained_on,
        "thresholds": {
            "min_decisions": MIN_DECISIONS,
            "missed_min_best_p": MIN_BEST_P,
            "highlights": N_HIGHLIGHTS,
            "third_x_m": round(THIRD_M, 2),
        },
        "fields": FIELDS,
        "caveats": CAVEATS,
        "summary": {
            **gap_summary(rows),
            "open_play_passes": open_play,
            "usable_share": _r(len(rows) / open_play, 3) if open_play else None,
        },
        "teams": by_team(m, rows),
        "players": by_player(m, rows),
        "highlights": highlights(rows),
        "decisions": rows,
    }


def match_tables(m) -> dict:
    """The training tables of one match (pass, value at the main horizon, shot with penalties flagged)."""
    return {"pass": D.pass_table(m), "value": D.value_table(m), "shot": D.shot_table(m)}


def held_out_models(target: str, tables: dict, extra_shots=None) -> tuple:
    """(advice functions, fitted models, trained_on) for match `target` with the site's recipe (export.fit_models):
    pass and value models on the other matches of `tables` (match id -> match_tables), shot model on their
    non-penalty shots plus the `extra_shots` rows (X, y, match) of matches other than `target`."""
    others = [mid for mid in tables if mid != target]
    stacked = {k: D.stack([tables[mid][k] for mid in others]) for k in ("pass", "value", "shot")}
    stacked["shot"] = D.drop_penalties(stacked["shot"])
    trained_on = list(others)
    if extra_shots is not None:
        keep = extra_shots["match"] != target
        extra_shots = {k: v[keep] for k, v in extra_shots.items()}
        trained_on += sorted(set(extra_shots["match"].tolist()) - set(trained_on))
    fitted = X.fit_models(stacked, extra_shots)
    return A.from_fitted(fitted), fitted, trained_on
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: PASS (13 tests).

- [ ] **Step 5: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/report.py tests/test_20_report.py && $PY -m ruff format --check 20-football-analytics/coach/report.py tests/test_20_report.py
git commit -m "20 coach: match report aggregation (players, teams, thirds), highlights, held-out models" -- 20-football-analytics/coach/report.py tests/test_20_report.py
```

### Task 5: report.json for the website (export_report.py)

**Files:**
- Create: `20-football-analytics/coach/export_report.py`
- Modify: `tests/test_20_report.py` (add `from coach import export_report as XR`)

- [ ] **Step 1: Write the failing test**

```python
def test_report_frames_match_the_site_export_and_the_size_limit(tmp_path, monkeypatch):
    m = small_match()
    X.pack_match(m, tmp_path)  # the real site export: match.json with events at 12.5 fps
    site = json.loads((tmp_path / "match.json").read_text(encoding="utf-8"))
    rep = R.build(m, MODELS, trained_on=["OTHER"])
    XR.check_against_site(rep, site)
    assert [site["events"][d["event"]]["frame"] for d in rep["decisions"]] == [20, 30]
    site["events"][2]["frame"] = 31
    with pytest.raises(AssertionError):
        XR.check_against_site(rep, site)
    path = XR.write(rep, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["decisions"] == rep["decisions"]
    monkeypatch.setattr(XR, "MAX_BYTES", 100)
    with pytest.raises(AssertionError):
        XR.write(rep, tmp_path)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: FAIL at import, `cannot import name 'export_report'`.

- [ ] **Step 3: Implement**

```python
"""report.json for the website's replay match, with models trained WITHOUT that match (coach insight, part A).

    python -m coach.export_report --match J03WOH --site ../../cv-data/football-coach/site-export

Same model recipe as coach.export (pass and value models on the other Bundesliga matches, shot model also on the
A-League shots), see report.held_out_models. Checked against the site's match.json when it is there: same frame rate,
and every decision's frame is the frame of the same event in match.json. The file must stay under MAX_BYTES (compact
JSON). Kept apart from coach.export on purpose (like export_profiles).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import models as ML
from . import report as R

MAX_BYTES = 300_000


def check_against_site(report: dict, site_match: dict) -> None:
    """Same frame rate as the site's match.json, and each decision sits on the frame of its event there."""
    assert abs(float(site_match["fps"]) - report["fps"]) < 1e-9, (site_match["fps"], report["fps"])
    for d in report["decisions"]:
        e = site_match["events"][d["event"]]
        assert e["type"] == "pass" and e["half"] == d["half"] and e["frame"] == d["frame"], (d, e)
        assert 0 <= d["frame"] < site_match["halves"][d["half"]]["frames"], d


def write(report: dict, site: Path) -> Path:
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    assert size <= MAX_BYTES, f"report.json is {size} bytes (limit {MAX_BYTES})"
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    (site / "report.json").write_text(text, encoding="utf-8")
    return site / "report.json"


def main():
    from . import cross_league as CL
    from .match import load
    from .prepare import MATCHES, OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: R.match_tables(m) for mid, m in matches.items() if mid != args.match}  # the target's are never used
    models, fitted, trained_on = R.held_out_models(args.match, tables, CL.skillcorner_shots()[0])
    rep = R.build(matches[args.match], models, trained_on)
    site_match = args.site / "match.json"
    if site_match.exists():
        check_against_site(rep, json.loads(site_match.read_text(encoding="utf-8")))
        print("frames checked against", site_match)
    site_models = args.site / "models.json"
    if site_models.exists():
        w = json.loads(site_models.read_text(encoding="utf-8"))
        same = w.get("trained_without") == args.match and np.allclose(
            ML.export_logistic(fitted["pass"], R.F.PASS_FEATURES)["coef"], w["pass"]["coef"]
        )
        print("same pass model as the site's models.json:", bool(same))
    path = write(rep, args.site)
    s = rep["summary"]
    print("wrote", path, path.stat().st_size, "bytes,", s["n"], "decisions,", len(rep["players"]), "passers")
    print({k: s[k] for k in ("top1", "top3", "gap_median", "gap_p90", "missed_candidates", "best_candidates")})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_20_report.py -q`
Expected: PASS (14 tests).

- [ ] **Step 5: Run the export for J03WOH**

Run (from `20-football-analytics/`): `$PY -m coach.export_report --match J03WOH --site ../../cv-data/football-coach/site-export`
Expected: "frames checked", "same pass model as the site's models.json: True", size ≤ 300000 bytes.

- [ ] **Step 6: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/export_report.py tests/test_20_report.py && $PY -m ruff format --check 20-football-analytics/coach/export_report.py tests/test_20_report.py
git add 20-football-analytics/coach/export_report.py && git commit -m "20 coach: export report.json for the replay match, built without it (frames checked against match.json)" -- 20-football-analytics/coach/export_report.py tests/test_20_report.py
```

### Task 6: held-out summary over the 7 IDSSE matches (numbers + chart)

**Files:**
- Modify: `20-football-analytics/coach/report.py` (append `held_out_summary`, `gap_chart`, `main`)
- Create (generated): `20-football-analytics/docs/coach_report_results.json`, `20-football-analytics/docs/coach_report_gaps.png`

- [ ] **Step 1: Write the failing test**

```python
def test_held_out_summary_pools_matches():
    reps = [R.build(small_match(match_id=mid), MODELS, trained_on=["OTHER"]) for mid in ("A1", "A2")]
    s = R.held_out_summary(reps)
    assert set(s["matches"]) == {"A1", "A2"} and s["pooled"]["n"] == 4 and s["pooled"]["open_play_passes"] == 4
    assert sum(s["rank_counts"].values()) == 4 and sum(s["gap_histogram"]["counts"]) == 4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_20_report.py -q -k held_out_summary`
Expected: FAIL, no attribute `held_out_summary`.

- [ ] **Step 3: Implement** (append)

```python
GAP_BINS = (0.0, 1e-9, 0.0025, 0.005, 0.01, 0.02, 0.04, 1.0)


def held_out_summary(reports) -> dict:
    """Per match and pooled numbers of several held-out reports: gap distribution, top-1 / top-3, candidates."""
    rows = [r for rep in reports for r in rep["decisions"]]
    ranks = [min(r["rank"], 10) for r in rows]
    counts, _ = np.histogram([r["gap"] for r in rows], bins=GAP_BINS)
    outcomes = [r["outcome"] for r in rows]
    return {
        "matches": {rep["match_id"]: rep["summary"] for rep in reports},
        "trained_on": {rep["match_id"]: rep["trained_on"] for rep in reports},
        "pooled": {
            **gap_summary(rows),
            "open_play_passes": sum(rep["summary"]["open_play_passes"] for rep in reports),
            "outcome_share": {k: _r(outcomes.count(k) / len(rows), 3) for k in ("goal", "shot", "kept", "lost")}
            if rows
            else {},
        },
        "rank_counts": {("10+" if k == 10 else str(k)): ranks.count(k) for k in range(1, 11)},
        "gap_histogram": {"edges": list(GAP_BINS), "counts": counts.tolist(), "note": "first bin: gap = 0"},
    }


def gap_chart(summary: dict, path: Path) -> None:
    """Two plain panels: distribution of the gap, and the coach's rank of the pass the player chose."""
    plt = EV._pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    h = summary["gap_histogram"]
    e = h["edges"]
    labels = ["0"] + [f"{e[i] * 1000:g}–{e[i + 1] * 1000:g}" for i in range(1, len(e) - 2)] + [f">{e[-2] * 1000:g}"]
    axes[0].bar(labels, h["counts"], color="#4c78a8")
    axes[0].set(xlabel="gap to the coach's best option (milli-xG)", ylabel="passes")
    rc = summary["rank_counts"]
    axes[1].bar(list(rc), list(rc.values()), color="#4c78a8")
    axes[1].set(xlabel="coach rank of the pass the player chose (all options)", ylabel="passes")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    from . import cross_league as CL
    from .match import load
    from .prepare import MATCHES, OUT

    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: match_tables(m) for mid, m in matches.items()}
    extra = CL.skillcorner_shots()[0]
    reports = []
    for mid, m in matches.items():
        models, _, trained_on = held_out_models(mid, tables, extra)
        reports.append(build(m, models, trained_on))
        s = reports[-1]["summary"]
        print(mid, s["n"], s["top1"], s["top3"], s["gap_median"], s["gap_p90"], s["missed_candidates"], flush=True)
    res = held_out_summary(reports)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_report_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    gap_chart(res, DOCS / "coach_report_gaps.png")
    print(json.dumps(res["pooled"], indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests, then the summary**

Run: `$PY -m pytest tests/test_20_report.py -q` → PASS (15 tests).
Run (from `20-football-analytics/`, background, ~10 min): `$PY -m coach.report`
Expected: one line per match, then the pooled summary; the JSON and PNG in `docs/`.

- [ ] **Step 5: Lint and commit**

```bash
$PY -m ruff check 20-football-analytics/coach/report.py tests/test_20_report.py && $PY -m ruff format --check 20-football-analytics/coach/report.py tests/test_20_report.py
git add 20-football-analytics/docs/coach_report_results.json 20-football-analytics/docs/coach_report_gaps.png
git commit -m "20 coach: match report held-out summary over the 7 Bundesliga matches (numbers + chart)" -- 20-football-analytics/coach/report.py tests/test_20_report.py 20-football-analytics/docs/coach_report_results.json 20-football-analytics/docs/coach_report_gaps.png
```

### Task 7: README "Match report" subsection, push, CI

**Files:**
- Modify: `20-football-analytics/README.md` (new `### Match report` before `### Run the coach`; two lines in the run
  block; the test count line)

- [ ] **Step 1: Write the subsection** from `docs/coach_report_results.json` (pooled n, top-1 / top-3, gap median /
  p90 / max, share where the best option is a carry, highlight candidate counts, per-match range), the chart, the
  caveats from the spec (model opinion, weak value model, bolder than pros, sync keeps ~2/3 of passes, n ≥ 20), and the
  J03WOH export (size, schema). A-League: say the report runs on SkillCorner matches (`fps_src` 10, step 1) but the
  held-out numbers are not computed because the tracking is not cached (1.8 GB to stream again).

- [ ] **Step 2: Full test run and lint**

Run: `$PY -m pytest tests/test_20_report.py tests/test_20_coach.py tests/test_20_profiles.py -q` → all pass.
Run: `$PY -m ruff check 20-football-analytics tests && $PY -m ruff format --check 20-football-analytics tests`

- [ ] **Step 3: Commit, push, check CI**

```bash
git commit -m "20 coach: README match report section (held-out numbers, caveats)" -- 20-football-analytics/README.md
GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q origin main
# rejected: git pull --rebase --autostash -q origin main, then push again
gh run list --branch main --limit 1   # expect: completed success
```

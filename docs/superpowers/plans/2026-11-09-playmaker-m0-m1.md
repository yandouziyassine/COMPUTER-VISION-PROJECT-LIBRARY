# Playmaker M0 + M1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Playmaker content pipeline in the library (M0) and a phone-first daily game with its real API
handler running locally (M1), with no AWS resource created.

**Architecture:** The library's coach produces a versioned content release (`content-v{n}`): public moment files
without answers, private answer files, daily sets, JSON schemas. A new product repo `playmaker` holds the web game
(Vite + TypeScript, canvas 2D), the API handler (TypeScript, pure functions + a `Store` interface), and a local dev
server that serves content and runs the real handler with an in-memory store. AWS infrastructure (CDK, $20 lifetime
cap, kill switch) is a separate plan (M2).

**Tech Stack:** Python 3.13 venv `C:\Users\yando\Code\work\venv20` (numpy, scikit-learn, scipy, jsonschema); Node 24,
Vite 7, TypeScript 5, vitest 3, Playwright 1.5x, ajv 8.

**Spec:** `docs/superpowers/specs/2026-11-09-playmaker-fan-game-design.md`.

**Deviations from the spec, agreed scope for this plan:** free play in M1 links to the portfolio's full-match replay
(`https://yandouziyassine.github.io/demos/football/`); the native free-play port and the 3D toggle go to the next
plan with M2. `make` does not exist on the owner's machine: commands are Python entry points.

**Paths.** Library `L = C:\Users\yando\Code\COMPUTER-VISION-PROJECT-LIBRARY`, project `P = L\20-football-analytics`,
product repo `G = C:\Users\yando\Code\playmaker`, content output `D:\cv-data\playmaker\content-v1` (USB key; the
A-League raw store `D:\cv-data\skillcorner-raw` must be present). Python commands run from `P` with
`PY = C:/Users/yando/Code/work/venv20/Scripts/python.exe`.

---

## File structure

Library (producer):

| File | Responsibility |
|---|---|
| `P/coach/game/__init__.py` | package marker, `CONTENT_VERSION = 1`, `SCHEMA_VERSION = 1` |
| `P/coach/game/inputs.py` | per-decision model-independent inputs (features, points) and fast scoring for any model set |
| `P/coach/game/scoring.py` | `share_top`, `points` (close-call rule), pure |
| `P/coach/game/refits.py` | bootstrap refits of the four models on resampled training matches |
| `P/coach/game/quality.py` | moment quality filters, pure |
| `P/coach/game/moments.py` | one match → candidate moments (public dict + answer dict + interest features) |
| `P/coach/game/days.py` | interest score and daily-set selection, pure |
| `P/coach/game/build.py` | CLI: all matches → `content-v{n}/` with checkpoints |
| `P/coach/game/schemas/*.schema.json` | day / moment / answer JSON Schemas (shipped in the release) |
| `P/coach/rebuild_results.py` | reruns every results script and stamps `generated_by` |
| `P/coach/stamp.py` | `generated_by()` = commit + date |
| `P/requirements.lock` | pinned versions of the coach venv |
| `L/tests/test_20_game.py` | tests for everything above |

Product repo `G` (consumer):

| File | Responsibility |
|---|---|
| `G/package.json`, `G/tsconfig.json`, `G/vite.config.ts` | workspace root (npm workspaces `app`, `api`) |
| `G/api/src/types.ts` | shared types: `Answer`, `PickRequest`, `PickResult`, `Store` |
| `G/api/src/validate.ts` | request validation (uuid, ids, size) |
| `G/api/src/result.ts` | builds the pick result from an answer + crowd counts |
| `G/api/src/handler.ts` | `handlePick(req, deps)`: rate limit, first-pick-wins, counts |
| `G/api/src/memoryStore.ts` | in-memory `Store` for tests and local dev |
| `G/api/src/fsAnswers.ts` | answers loader from a content folder (local) |
| `G/api/src/dev.ts` | local HTTP server: `/content/*` static, `POST /api/pick` |
| `G/app/index.html`, `G/app/src/*.ts`, `G/app/src/styles.css` | the game |
| `G/content/` | gitignored; local copy of a content release |
| `G/tests/e2e/day.spec.ts` | Playwright: full day on a phone viewport |

---

## Part M0: library

### Task 1: Pinned dependencies, CI matplotlib, results stamp

**Files:**
- Create: `P/requirements.lock`, `P/coach/stamp.py`, `P/coach/rebuild_results.py`
- Modify: `L/tests/requirements-ci.txt`
- Test: `L/tests/test_20_game.py`

- [ ] **Step 1: Generate the lock file**

Run (from `P`): `$PY -m pip freeze --exclude-editable > requirements.lock`
Then add `jsonschema` if missing: `$PY -m pip install "jsonschema>=4.23" && $PY -m pip freeze --exclude-editable > requirements.lock`
Expected: a file with lines like `numpy==2.5.4`, `scikit-learn==1.9.1`, `scipy==1.18.1`, `jsonschema==4.x`.

- [ ] **Step 2: Add matplotlib and jsonschema to CI deps**

Append to `L/tests/requirements-ci.txt`:

```
matplotlib>=3.9
scipy>=1.13
jsonschema>=4.23
```

- [ ] **Step 3: Write the failing test for the stamp**

Create `L/tests/test_20_game.py`:

```python
import json
import sys

import numpy as np
import pytest

from conftest import project_dir

sys.path.insert(0, str(project_dir("20-football-analytics")))

from coach import stamp  # noqa: E402


def test_generated_by_has_commit_and_date():
    g = stamp.generated_by()
    assert set(g) == {"commit", "date"}
    assert len(g["commit"]) in (7, 40) or g["commit"] == "unknown"
    assert len(g["date"]) == 10
```

- [ ] **Step 4: Run it, see it fail**

Run (from `L`): `$PY -m pytest tests/test_20_game.py -q`
Expected: FAIL `ImportError: cannot import name 'stamp'`.

- [ ] **Step 5: Implement `coach/stamp.py`**

```python
"""Provenance stamp for generated results: the commit of the library and the date (UTC)."""

import datetime as dt
import subprocess
from pathlib import Path


def generated_by() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"commit": commit, "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")}
```

- [ ] **Step 6: Implement `coach/rebuild_results.py`**

```python
"""Rerun every results script of the coach, in dependency order, then stamp each results JSON with generated_by.

    python -m coach.rebuild_results            # ~1 h on CPU; the A-League steps need the raw store on the USB key
    python -m coach.rebuild_results --idsse    # only the steps that need the 7 IDSSE matches (~30 min)
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .stamp import generated_by

DOCS = Path(__file__).resolve().parents[1] / "docs"
IDSSE = [
    ["coach.evaluate"],
    ["coach.report"],
    ["coach.benchmark", "--set", "idsse"],
]
ALEAGUE = [
    ["coach.cross_league"],
    ["coach.shot_pooled"],
    ["coach.profiles_eval"],
    ["coach.benchmark", "--set", "aleague"],
    ["coach.benchmark", "--set", "pooled"],
    ["coach.benchmark", "--cutbacks"],
    ["coach.cutback"],
]


def stamp_all(stamp: dict) -> list:
    done = []
    for f in sorted(DOCS.glob("*results*.json")) + sorted(DOCS.glob("coach_benchmark_cutbacks.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data["generated_by"] = stamp
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            done.append(f.name)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idsse", action="store_true")
    args = ap.parse_args()
    steps = IDSSE if args.idsse else IDSSE + ALEAGUE
    for step in steps:
        print(">>", " ".join(step), flush=True)
        subprocess.run([sys.executable, "-m", *step], check=True, cwd=Path(__file__).resolve().parents[1])
    print("stamped", stamp_all(generated_by()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 7: Run the test, lint, commit**

Run (from `L`): `$PY -m pytest tests/test_20_game.py -q` → PASS.
Run: `C:/Users/yando/Code/.venv-tools/Scripts/ruff.exe check . && C:/Users/yando/Code/.venv-tools/Scripts/ruff.exe format --check .`

```bash
git add 20-football-analytics/requirements.lock 20-football-analytics/coach/stamp.py 20-football-analytics/coach/rebuild_results.py tests/requirements-ci.txt tests/test_20_game.py
git commit -m "20 coach: pinned deps, results provenance stamp and rebuild script"
```

### Task 2: Fast option scoring (`coach/game/inputs.py`)

The content needs every option's score under 21 model sets (main + 20 refits). `advice.options` recomputes features
and pitch control every call; here the model-independent parts are computed once and only predictions change.

**Files:**
- Create: `P/coach/game/__init__.py`, `P/coach/game/inputs.py`
- Test: `L/tests/test_20_game.py`

- [ ] **Step 1: Write the failing parity test**

Append to `L/tests/test_20_game.py`:

```python
from coach import advice as A  # noqa: E402
from coach import evaluate as EV  # noqa: E402
from coach import report as R  # noqa: E402
from coach.game import inputs as GI  # noqa: E402
from test_20_report import small_match  # noqa: E402


@pytest.fixture(scope="module")
def two_matches():
    a, b = small_match(match_id="SYA"), small_match(match_id="SYB")
    tables = {m.match_id: R.match_tables(m) for m in (a, b)}
    return a, b, tables


def test_fast_scores_equal_advice_options(two_matches):
    a, _, tables = two_matches
    models, _, _ = R.held_out_models("SYA", tables)
    n = 0
    for d in EV.pass_decisions(a, models):
        p = d["sample"]
        inp = GI.inputs(p["state"], p["carrier"])
        fast = GI.scores(inp, models)
        by_key = {GI.key(o): o["score"] for o in d["options"]}
        assert [GI.key(o) for o in GI.options(inp)] == sorted(by_key, key=GI.order)
        for o, s in zip(GI.options(inp), fast):
            assert s == pytest.approx(by_key[GI.key(o)], abs=1e-12)
        n += 1
    assert n > 0
```

- [ ] **Step 2: Run it, see it fail**

Run: `$PY -m pytest tests/test_20_game.py::test_fast_scores_equal_advice_options -q`
Expected: FAIL `ModuleNotFoundError: No module named 'coach.game'`.

- [ ] **Step 3: Implement**

`P/coach/game/__init__.py`:

```python
"""Playmaker content: the AI coach's decisions turned into a daily football game (spec 2026-11-09)."""

CONTENT_VERSION = 1
SCHEMA_VERSION = 1
```

`P/coach/game/inputs.py`:

```python
"""Model-independent inputs of one decision and fast scores for any model set (same formula as advice.options).

Option order (k = index): passes by target player index, then carries by angle (advice.CARRY_ANGLES order), then the
shot when the carrier is within advice.SHOT_RANGE of the goal."""

import numpy as np

from .. import advice as A
from .. import features as F
from .. import pitch_control as PC


def _vf(points) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(points, float))
    return np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in pts]).reshape(
        len(pts), len(F.VALUE_FEATURES)
    )


def inputs(s: dict, carrier: int) -> dict:
    here = s["att"][carrier]
    rows = sorted((j for j in range(len(s["att"])) if j != carrier), key=lambda j: int(s["att_idx"][j]))
    pass_x = np.array([[F.pass_features(s, carrier, j)[k] for k in F.PASS_FEATURES] for j in rows])
    pass_to = np.array([s["att"][j] for j in rows], float)
    pass_loss = np.array([F.lane_loss_point(s, here, s["att"][j]) for j in rows], float)
    carry_to, carry_p = [], []
    for a in A.CARRY_ANGLES:
        pt = here + A.CARRY_DIST * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        pt = np.clip(pt, [-52.5, -34], [52.5, 34])
        carry_to.append(pt)
        carry_p.append(float(PC.control(pt[None], s["att"], s["att_v"], s["def"], s["def_v"])[0]))
    carry_to = np.array(carry_to)
    sf = F.shot_features(here, s["def"])
    shot_x = np.array([[sf[k] for k in F.SHOT_FEATURES]]) if sf["goal_dist"] <= A.SHOT_RANGE else None
    return {
        "targets": [int(s["att_idx"][j]) for j in rows],
        "angles": list(A.CARRY_ANGLES),
        "pass_x": pass_x,
        "pass_to": pass_to,
        "carry_to": carry_to,
        "carry_p": np.array(carry_p),
        "vf_here": _vf([here]),
        "vf_pass_to": _vf(pass_to),
        "vf_carry_to": _vf(carry_to),
        "vf_pass_loss": _vf(-pass_loss),
        "vf_carry_loss": _vf(-carry_to),
        "shot_x": shot_x,
    }


def key(o: dict) -> tuple:
    """Identity of an option across model sets: ("pass", player) / ("carry", angle) / ("shot",)."""
    if o["kind"] == "pass":
        return ("pass", int(o["target"]))
    if o["kind"] == "carry":
        return ("carry", int(o["angle"]))
    return ("shot",)


def order(k: tuple) -> tuple:
    return ({"pass": 0, "carry": 1, "shot": 2}[k[0]], *k[1:])


def options(inp: dict) -> list:
    """The options in k order: {"k", "kind", "target" | "angle", "to"} (to: normalised metres)."""
    out = [{"kind": "pass", "target": t, "to": inp["pass_to"][i].tolist()} for i, t in enumerate(inp["targets"])]
    out += [{"kind": "carry", "angle": a, "to": inp["carry_to"][i].tolist()} for i, a in enumerate(inp["angles"])]
    if inp["shot_x"] is not None:
        out.append({"kind": "shot", "to": [52.5, 0.0]})
    for k, o in enumerate(out):
        o["k"] = k
    return out


def parts(inp: dict, models: dict) -> dict:
    """p, value, v_lost, score per option (k order) and v_now under one model set."""
    v_now = float(models["value"](inp["vf_here"])[0])
    p_pass = models["pass"](inp["pass_x"])
    v_pass = models["arrival" if "arrival" in models else "value"](inp["vf_pass_to"])
    p = np.concatenate([p_pass, inp["carry_p"]])
    vk = np.concatenate([v_pass, models["value"](inp["vf_carry_to"])])
    vo = np.concatenate([models["value"](inp["vf_pass_loss"]), models["value"](inp["vf_carry_loss"])])
    score = p * vk - (1 - p) * vo - v_now
    if inp["shot_x"] is not None:
        ps = float(models["shot"](inp["shot_x"])[0])
        p, vk, vo = np.append(p, ps), np.append(vk, 1.0), np.append(vo, 0.0)
        score = np.append(score, ps - v_now)
    return {"p": p, "value": vk, "v_lost": vo, "score": score, "v_now": v_now}


def scores(inp: dict, models: dict) -> np.ndarray:
    return parts(inp, models)["score"]
```

- [ ] **Step 4: Run the test**

Run: `$PY -m pytest tests/test_20_game.py -q` → PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/game tests/test_20_game.py
git commit -m "20 game: model-independent decision inputs and fast option scoring (parity with advice.options)"
```

### Task 3: Close-call scoring (`coach/game/scoring.py`)

**Files:**
- Create: `P/coach/game/scoring.py`
- Test: `L/tests/test_20_game.py`

- [ ] **Step 1: Write the failing tests**

```python
from coach.game import scoring as GS  # noqa: E402


def test_share_top_counts_ties_for_every_tied_option():
    refit = np.array([[0.1, 0.3, 0.2], [0.3, 0.3, 0.0], [0.0, 0.1, 0.2], [0.5, 0.1, 0.1]])
    assert GS.share_top(refit).tolist() == [0.5, 0.5, 0.25]


def test_points_full_for_main_top_and_close_calls_linear_for_the_rest():
    main = np.array([0.30, 0.25, 0.10, 0.05, -0.20])
    share = np.array([0.6, 0.25, 0.1, 0.05, 0.0])
    assert GS.points(main, share).tolist() == [100, 100, 90, 45, 0]


def test_points_single_remaining_option_gets_45():
    assert GS.points(np.array([0.3, 0.1]), np.array([1.0, 0.0])).tolist() == [100, 45]


def test_points_main_top_is_full_even_with_low_share():
    assert GS.points(np.array([0.3, 0.2, 0.1]), np.array([0.1, 0.9, 0.0])).tolist() == [100, 100, 0]
```

- [ ] **Step 2: Run, see it fail** (`ImportError`).

- [ ] **Step 3: Implement**

```python
"""Playmaker points: 100 for options 'among the best', 0-90 by rank for the others (spec section 4)."""

import numpy as np

AMONG_BEST = 0.2


def share_top(refit_scores: np.ndarray) -> np.ndarray:
    """[R, K] scores under R refits -> [K] share of refits in which each option has the highest score (ties: all)."""
    s = np.asarray(refit_scores, float)
    top = s >= s.max(axis=1, keepdims=True) - 1e-15
    return top.mean(axis=0)


def points(main: np.ndarray, share: np.ndarray, among_best: float = AMONG_BEST) -> np.ndarray:
    main, share = np.asarray(main, float), np.asarray(share, float)
    full = share >= among_best
    full[int(np.argmax(main))] = True
    out = np.zeros(len(main), int)
    out[full] = 100
    rest = [i for i in np.argsort(-main, kind="stable") if not full[i]]
    if len(rest) == 1:
        out[rest[0]] = 45
    elif rest:
        r = len(rest)
        for rank, i in enumerate(rest, start=1):
            out[i] = int(round(90 * (r - rank) / (r - 1)))
    return out
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit** `git commit -m "20 game: close-call points rule"` (add the two files).

### Task 4: Bootstrap refits (`coach/game/refits.py`)

**Files:**
- Create: `P/coach/game/refits.py`
- Test: `L/tests/test_20_game.py`

- [ ] **Step 1: Failing test**

```python
from coach.game import refits as GR  # noqa: E402


def test_refits_never_use_the_target_and_are_reproducible(two_matches):
    _, _, tables = two_matches
    tables3 = {**tables, "SYC": tables["SYB"]}
    sets1 = GR.refit_models("SYA", tables3, n=3, seed=7)
    sets2 = GR.refit_models("SYA", tables3, n=3, seed=7)
    assert len(sets1) == 3
    for (m1, used1), (m2, used2) in zip(sets1, sets2):
        assert "SYA" not in used1 and used1 == used2
        x = np.zeros((1, 9))
        assert m1["pass"](x)[0] == pytest.approx(m2["pass"](x)[0])
```

- [ ] **Step 2: Run, fail.** **Step 3: Implement**

```python
"""Bootstrap refits: the four coach models refitted on whole-match resamples of the training matches."""

import numpy as np

from .. import advice as A
from .. import datasets as D
from .. import export as X

N_REFITS = 20


def refit_models(target: str, tables: dict, n: int = N_REFITS, seed: int = 0) -> list:
    """[(advice functions, resampled match ids)] for match `target`, never trained on it."""
    others = sorted(mid for mid in tables if mid != target)
    rng = np.random.default_rng([seed, abs(hash(target)) % (2**31)])
    out = []
    for _ in range(n):
        pick = [others[i] for i in rng.integers(0, len(others), len(others))]
        keys = [k for k in ("pass", "value", "shot", "arrival") if k in tables[others[0]]]
        stacked = {k: D.stack([tables[mid][k] for mid in pick]) for k in keys}
        stacked["shot"] = D.drop_penalties(stacked["shot"])
        out.append((A.from_fitted(X.fit_models(stacked)), pick))
    return out
```

Note: `hash()` of a str is salted per process. Replace with a stable digest:

```python
import zlib
...
    rng = np.random.default_rng([seed, zlib.crc32(target.encode())])
```

(use this `zlib` version in the file; do not use `hash`).

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `"20 game: bootstrap refits of the coach models"`.

### Task 5: Quality filters (`coach/game/quality.py`)

**Files:** Create `P/coach/game/quality.py`; test in `L/tests/test_20_game.py`.

- [ ] **Step 1: Failing tests**

```python
from coach.game import quality as GQ  # noqa: E402


def frames(n=10, players=22, step=0.3):
    xy = np.zeros((n, players, 2))
    xy[:, :, 0] = np.arange(players)[None, :] * 2.0
    xy[:, :, 0] += np.arange(n)[:, None] * step
    return xy


def test_quality_accepts_a_clean_moment():
    assert GQ.problems(n_att=11, n_def=11, ball_gap=1.0, lead_up=frames(), n_options=8) == []


def test_quality_rejects_missing_players_far_ball_jumps_and_few_options():
    xy = frames()
    xy[5, 3, 0] += 4.0
    got = GQ.problems(n_att=10, n_def=11, ball_gap=3.0, lead_up=xy, n_options=2)
    assert got == ["players", "ball", "jump", "options"]


def test_quality_ignores_nan_players_in_jump_check():
    xy = frames()
    xy[2:4, 7] = np.nan
    assert GQ.problems(n_att=11, n_def=11, ball_gap=0.5, lead_up=xy, n_options=5) == []
```

- [ ] **Step 2: Fail. Step 3: Implement**

```python
"""Moment quality filters (spec section 4): a moment is shown only when the data around it is clean."""

import numpy as np

BALL_GAP_M = 2.0
JUMP_M = 1.5
MIN_OPTIONS = 3


def problems(n_att: int, n_def: int, ball_gap: float, lead_up: np.ndarray, n_options: int) -> list:
    """Names of the failed checks, in a fixed order; [] when the moment is usable. lead_up: [T, P, 2] metres."""
    out = []
    if n_att != 11 or n_def != 11:
        out.append("players")
    if not ball_gap <= BALL_GAP_M:
        out.append("ball")
    step = np.linalg.norm(np.diff(lead_up, axis=0), axis=-1)
    if np.nanmax(step, initial=0.0) > JUMP_M:
        out.append("jump")
    if n_options < MIN_OPTIONS:
        out.append("options")
    return out
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `"20 game: moment quality filters"`.

### Task 6: Schemas

**Files:** Create `P/coach/game/schemas/day.schema.json`, `moment.schema.json`, `answer.schema.json`.

- [ ] **Step 1: Write `day.schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "playmaker/day.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "content_version", "day_number", "date", "moments"],
  "properties": {
    "schema": {"const": 1},
    "content_version": {"type": "integer", "minimum": 1},
    "day_number": {"type": "integer", "minimum": 1},
    "date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
    "moments": {"type": "array", "minItems": 5, "maxItems": 5, "items": {"type": "string", "pattern": "^[A-Za-z0-9-]+$"}}
  }
}
```

- [ ] **Step 2: Write `moment.schema.json`** (public: no answer fields; `additionalProperties: false` everywhere so an
answer field cannot slip in)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "playmaker/moment.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "id", "league", "home", "away", "attacking", "minute", "fps", "carrier", "players", "lead_up", "ball", "options", "credit"],
  "properties": {
    "schema": {"const": 1},
    "id": {"type": "string", "pattern": "^[A-Za-z0-9-]+$"},
    "league": {"enum": ["Bundesliga 2", "A-League"]},
    "home": {"type": "string"},
    "away": {"type": "string"},
    "attacking": {"enum": ["home", "away"]},
    "minute": {"type": "integer", "minimum": 0, "maximum": 130},
    "fps": {"type": "number", "exclusiveMinimum": 0},
    "carrier": {"type": "integer", "minimum": 0},
    "players": {
      "type": "array", "minItems": 22, "maxItems": 22,
      "items": {
        "type": "object", "additionalProperties": false, "required": ["side", "shirt", "name"],
        "properties": {"side": {"enum": [0, 1]}, "shirt": {"type": "integer"}, "name": {"type": "string"}}
      }
    },
    "lead_up": {
      "description": "frames x players x [x_cm, y_cm], attack-normalised (attacking side plays towards +x); null = not tracked",
      "type": "array", "minItems": 2,
      "items": {"type": "array", "minItems": 22, "maxItems": 22,
        "items": {"oneOf": [{"type": "null"}, {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}]}}
    },
    "ball": {"type": "array", "minItems": 2, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}},
    "options": {
      "type": "array", "minItems": 3,
      "items": {
        "type": "object", "additionalProperties": false, "required": ["k", "kind", "to"],
        "properties": {
          "k": {"type": "integer", "minimum": 0},
          "kind": {"enum": ["pass", "carry", "shot"]},
          "player": {"type": "integer", "minimum": 0},
          "angle": {"type": "integer"},
          "to": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}}
        }
      }
    },
    "credit": {"type": "string"}
  }
}
```

- [ ] **Step 3: Write `answer.schema.json`**

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "playmaker/answer.schema.json",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema", "id", "options", "ai_pick", "pro_pick", "outcome", "models"],
  "properties": {
    "schema": {"const": 1},
    "id": {"type": "string"},
    "options": {
      "type": "array", "minItems": 3,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["k", "score", "p", "share_top", "points", "reasons"],
        "properties": {
          "k": {"type": "integer"}, "score": {"type": "number"}, "p": {"type": "number"},
          "share_top": {"type": "number", "minimum": 0, "maximum": 1},
          "points": {"type": "integer", "minimum": 0, "maximum": 100},
          "reasons": {"type": "array", "maxItems": 3, "items": {"type": "string"}}
        }
      }
    },
    "ai_pick": {"type": "integer"},
    "pro_pick": {"type": "integer"},
    "outcome": {
      "type": "object", "additionalProperties": false, "required": ["kind", "net_xg_15s"],
      "properties": {"kind": {"enum": ["goal", "shot", "kept", "lost"]}, "net_xg_15s": {"type": "number"}}
    },
    "models": {
      "type": "object", "additionalProperties": false, "required": ["content_version", "trained_without", "refits"],
      "properties": {"content_version": {"type": "integer"}, "trained_without": {"type": "string"}, "refits": {"type": "integer"}}
    }
  }
}
```

- [ ] **Step 4: Commit** `"20 game: JSON schemas for day, moment and answer files"`.

### Task 7: Moments of one match (`coach/game/moments.py`)

**Files:** Create `P/coach/game/moments.py`; test in `L/tests/test_20_game.py`.

Positions are stored attack-normalised in integer centimetres; `players` lists the 11 attackers then the 11
defenders at the release frame (side 0 = attacking); `carrier` indexes `players`; pass options carry `player` (index
in `players`).

- [ ] **Step 1: Failing tests**

```python
import jsonschema  # noqa: E402

from coach.game import moments as GM  # noqa: E402

SCHEMAS = project_dir("20-football-analytics") / "coach" / "game" / "schemas"


def schema(name):
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(two_matches):
    a, _, tables = two_matches
    tables3 = {**tables, "SYC": tables["SYB"]}
    models, fitted, trained_on = R.held_out_models("SYA", tables3)
    refits = GR.refit_models("SYA", tables3, n=4, seed=1)
    return GM.match_moments(a, (models, fitted, trained_on), refits, league="Bundesliga 2", require_full=False)


def test_moment_files_match_schemas_and_hide_answers(built):
    assert built, "the synthetic match must give at least one moment"
    for c in built:
        jsonschema.validate(c["public"], schema("moment"))
        jsonschema.validate(c["answer"], schema("answer"))
        text = json.dumps(c["public"])
        for banned in ("score", "points", "share_top", "pro_pick", "ai_pick", "outcome", "reasons", "net_xg"):
            assert banned not in text


def test_answers_are_held_out_and_consistent(built):
    for c in built:
        a = c["answer"]
        assert a["models"]["trained_without"] == "SYA"
        ks = [o["k"] for o in a["options"]]
        assert ks == [o["k"] for o in c["public"]["options"]]
        best = max(a["options"], key=lambda o: o["score"])
        assert a["ai_pick"] == best["k"] and best["points"] == 100
        assert a["pro_pick"] in ks
        assert all(0 <= o["points"] <= 100 for o in a["options"])
```

- [ ] **Step 2: Fail. Step 3: Implement `moments.py`**

```python
"""One match -> Playmaker moment candidates: public file (no answers), answer file, interest features."""

import numpy as np

from .. import datasets as D
from .. import evaluate as EV
from .. import explain as E
from .. import features as F
from .. import geometry as G
from .. import models as ML
from .. import report as R
from . import CONTENT_VERSION, SCHEMA_VERSION
from . import inputs as GI
from . import quality as GQ
from . import scoring as GS

LEAD_UP_S = 3.0
CREDITS = {
    "Bundesliga 2": "Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0",
    "A-League": "Tracking and event data: SkillCorner open data (MIT)",
}
CUTBACK_X, CUTBACK_WIDE, CUTBACK_CENTRAL = 36.0, (12.0, 25.0), 8.0


def cm(a) -> list:
    return [int(round(float(v) * 100)) for v in a]


def is_cutback(here, to) -> bool:
    return bool(
        here[0] >= CUTBACK_X
        and CUTBACK_WIDE[0] <= abs(here[1]) <= CUTBACK_WIDE[1]
        and to[0] >= CUTBACK_X
        and abs(to[1]) < CUTBACK_CENTRAL
    )


def weights_of(fitted: dict) -> dict:
    return {
        "pass": ML.export_logistic(fitted["pass"], F.PASS_FEATURES),
        "shot": ML.export_logistic(fitted["shot"], F.SHOT_FEATURES),
    }


def match_moments(m, held_out, refits, league: str, require_full: bool = True) -> list:
    """held_out = report.held_out_models(...) for m; refits = refits.refit_models(...) for m.
    require_full=False keeps moments that fail the 22-players check (synthetic test matches only)."""
    models, fitted, trained_on = held_out
    assert str(m.match_id) not in [str(t) for t in trained_on]
    weights = weights_of(fitted)
    fps = F.match_fps(m)
    step = max(1, int(round(fps / 12.5)))
    back = int(round(LEAD_UP_S * fps))
    shots = R.shots_with_goals(m)
    out = []
    for d in EV.pass_decisions(m, models):
        p, e = d["sample"], d["sample"]["event"]
        s, carrier, half, frame = p["state"], p["carrier"], e["half"], e["frame"]
        if frame - back < 0:
            continue
        inp = GI.inputs(s, carrier)
        opts = GI.options(inp)
        main = GI.parts(inp, models)
        refit = np.array([GI.scores(inp, rm) for rm, _ in refits])
        share = GS.share_top(refit)
        pts = GS.points(main["score"], share)
        h = m.halves[half]
        idx = list(s["att_idx"]) + list(s["def_idx"])
        frames_src = list(range(frame - back, frame + 1, step))
        if frames_src[-1] != frame:
            frames_src.append(frame)
        lead = G.normalise(h.xy[frames_src][:, idx], s["direction"])
        ball = G.normalise(h.ball[frames_src, :2], s["direction"])
        n_att, n_def = len(s["att_idx"]), len(s["def_idx"])
        bad = GQ.problems(n_att, n_def, float(np.linalg.norm(s["ball"] - s["att"][carrier])), lead, len(opts))
        if require_full:
            if bad:
                continue
        elif [b for b in bad if b != "players"]:
            continue
        pro_key = ("pass", int(d["chosen"]["target"]))
        keys = [GI.key(o) for o in opts]
        expl = E.explain_options(s, carrier, [_as_advice(o, main, i) for i, o in enumerate(opts)], weights, main["v_now"])
        mid = f"{m.match_id}-{half}-{frame}"
        player_of = {int(pid): i for i, pid in enumerate(idx)}
        public_opts = []
        for o in opts:
            po = {"k": o["k"], "kind": o["kind"], "to": cm(o["to"])}
            if o["kind"] == "pass":
                po["player"] = player_of[o["target"]]
            if o["kind"] == "carry":
                po["angle"] = int(o["angle"])
            public_opts.append(po)
        home = m.meta.get("home", "Home")
        away = m.meta.get("guest", "Away")
        t_start = float(h.t[0])
        public = {
            "schema": SCHEMA_VERSION,
            "id": mid,
            "league": league,
            "home": home,
            "away": away,
            "attacking": "home" if int(e["team"]) == 0 else "away",
            "minute": int((float(h.t[frame]) - t_start) // 60) + 45 * half,
            "fps": fps / step,
            "carrier": player_of[int(s["att_idx"][carrier])],
            "players": [
                {"side": 0 if i < n_att else 1, "shirt": int(m.shirt[pid]), "name": str(m.names[pid])}
                for i, pid in enumerate(idx)
            ],
            "lead_up": [[None if np.isnan(xy[0]) else cm(xy) for xy in f] for f in lead],
            "ball": [cm(b) for b in ball],
            "options": public_opts,
            "credit": CREDITS[league],
        }
        answer = {
            "schema": SCHEMA_VERSION,
            "id": mid,
            "options": [
                {
                    "k": o["k"],
                    "score": round(float(main["score"][i]), 6),
                    "p": round(float(main["p"][i]), 4),
                    "share_top": round(float(share[i]), 3),
                    "points": int(pts[i]),
                    "reasons": expl[i]["phrases"][:3],
                }
                for i, o in enumerate(opts)
            ],
            "ai_pick": int(np.argmax(main["score"])),
            "pro_pick": keys.index(pro_key),
            "outcome": {
                "kind": R.outcome(shots[half], d["t"], int(e["team"]), bool(e["success"])),
                "net_xg_15s": round(float(d["outcome"]), 4),
            },
            "models": {"content_version": CONTENT_VERSION, "trained_without": str(m.match_id), "refits": len(refits)},
        }
        here = s["att"][carrier]
        out.append(
            {
                "public": public,
                "answer": answer,
                "features": {
                    "match": str(m.match_id),
                    "x": float(here[0]),
                    "v_now": main["v_now"],
                    "iqr": float(np.subtract(*np.percentile(main["score"], [75, 25]))),
                    "all_best": bool((pts == 100).all()),
                    "obvious": bool(share.max() == 1.0 and (pts == 100).sum() == 1),
                    "cutback": is_cutback(here, d["chosen"]["to"]) and bool(e["success"]),
                    "ai_is_pro": answer["ai_pick"] == answer["pro_pick"],
                    "shot_after": answer["outcome"]["kind"] in ("shot", "goal"),
                    "difficulty": float(1 - pts.mean() / 100),
                },
            }
        )
    return out


def _as_advice(o: dict, parts: dict, i: int) -> dict:
    """advice.options-shaped option for explain.explain_options."""
    a = {"kind": o["kind"], "to": o["to"], "p": float(parts["p"][i]), "value": float(parts["value"][i])}
    a["score"] = float(parts["score"][i])
    if o["kind"] == "pass":
        a["target"] = o["target"]
        a["v_lost"] = float(parts["v_lost"][i])
    elif o["kind"] == "carry":
        a["angle"] = o["angle"]
        a["v_lost"] = float(parts["v_lost"][i])
    return a
```

The `minute` uses the half's own clock (`h.t`, seconds); the second half adds 45. `D` and `D.HORIZON` are the
outcome horizon used by `pass_decisions` (15 s).

- [ ] **Step 4: Run** `$PY -m pytest tests/test_20_game.py -q` → PASS. If `explain.value_phrase` needs keys absent
from `_as_advice`, read `coach/explain.py:122` and add exactly those keys from `parts`.

- [ ] **Step 5: Commit** `"20 game: moment and answer files of one match"`.

### Task 8: Interest score and daily sets (`coach/game/days.py`)

**Files:** Create `P/coach/game/days.py`; tests.

- [ ] **Step 1: Failing tests**

```python
from coach.game import days as GD  # noqa: E402


def feat(i, match, x, difficulty, shot=False, iqr=0.01, v=0.01):
    return {"id": f"M{i}", "match": match, "x": x, "difficulty": difficulty, "shot_after": shot, "iqr": iqr,
            "v_now": v, "all_best": False, "obvious": False, "cutback": False, "ai_is_pro": False}


def pool(n=200):
    rng = np.random.default_rng(0)
    zones = [-40.0, 0.0, 0.0, 40.0, 40.0]
    return [feat(i, f"G{i % 9}", zones[i % 5] + rng.normal(0, 3), float(rng.uniform(0, 1)), shot=(i % 7 == 0),
                 iqr=float(rng.uniform(0, 0.02)), v=float(rng.uniform(0, 0.03))) for i in range(n)]


def test_interest_drops_all_best_and_obvious():
    f = pool(10)
    f[0]["all_best"] = True
    f[1]["obvious"] = True
    kept = GD.rank_pool(f, size=10)
    assert "M0" not in [k["id"] for k in kept] and "M1" not in [k["id"] for k in kept]


def test_daily_sets_respect_zones_difficulty_matches_and_shots():
    days = GD.daily_sets(GD.rank_pool(pool(), size=200), n_days=8)
    assert len(days) == 8
    used = set()
    prev_matches = set()
    for d in days:
        assert len(d) == 5 and not used & {m["id"] for m in d}
        used |= {m["id"] for m in d}
        assert [GD.zone(m["x"]) for m in d].count("build") == 1
        assert sorted(GD.zone(m["x"]) for m in d) == ["build", "final", "final", "middle", "middle"]
        assert [m["difficulty"] for m in d] == sorted(m["difficulty"] for m in d)
        assert len({m["match"] for m in d}) == 5
        assert any(m["shot_after"] for m in d)
        assert not prev_matches & {m["match"] for m in d}
        prev_matches = {m["match"] for m in d}
```

- [ ] **Step 2: Fail. Step 3: Implement**

```python
"""Interest score and daily sets of 5 moments (spec section 4)."""

import numpy as np

THIRD_X = 52.5 / 3
ZONES = ("build", "middle", "middle", "final", "final")


def zone(x: float) -> str:
    return "build" if x < -THIRD_X else ("final" if x > THIRD_X else "middle")


def interest(f: dict, v_rank: float) -> float:
    bonus = 1.0 + 0.5 * f["cutback"] + 0.3 * (not f["ai_is_pro"]) + 0.3 * f["shot_after"]
    return f["iqr"] * (0.5 + v_rank) * bonus


def rank_pool(feats: list, size: int = 1500) -> list:
    """Drop all-best and obvious moments, sort by interest, keep the top `size`."""
    keep = [f for f in feats if not f["all_best"] and not f["obvious"]]
    if not keep:
        return []
    v = np.array([f["v_now"] for f in keep])
    ranks = v.argsort().argsort() / max(1, len(v) - 1)
    scored = sorted(zip(keep, ranks), key=lambda t: -interest(t[0], float(t[1])))
    return [f for f, _ in scored[:size]]


def daily_sets(ranked: list, n_days: int) -> list:
    """Greedy: for each day take the most interesting unused moment per zone slot, distinct matches, no match of the
    previous day, at least one followed by a shot or goal; order the 5 by difficulty."""
    used, prev, days = set(), set(), []
    for _ in range(n_days):
        day, matches = [], set()
        for slot in ZONES:
            pick = next(
                (
                    f
                    for f in ranked
                    if f["id"] not in used and zone(f["x"]) == slot and f["match"] not in matches | prev
                ),
                None,
            )
            if pick is None:
                return days
            day.append(pick)
            matches.add(pick["match"])
            used.add(pick["id"])
        if not any(f["shot_after"] for f in day):
            swap = next(
                (
                    (i, f)
                    for i, slot in enumerate(ZONES)
                    if slot == "final"
                    for f in ranked
                    if f["id"] not in used
                    and f["shot_after"]
                    and zone(f["x"]) == "final"
                    and f["match"] not in (matches - {day[i]["match"]}) | prev
                ),
                None,
            )
            if swap is None:
                return days
            i, f = swap
            used.discard(day[i]["id"])
            matches.discard(day[i]["match"])
            day[i] = f
            used.add(f["id"])
            matches.add(f["match"])
        days.append(sorted(day, key=lambda f: f["difficulty"]))
        prev = matches
    return days
```

- [ ] **Step 4: Run** → PASS. **Step 5: Commit** `"20 game: interest score and daily sets"`.

### Task 9: Build CLI (`coach/game/build.py`) and the real content release

**Files:** Create `P/coach/game/build.py`; test in `L/tests/test_20_game.py`.

- [ ] **Step 1: Failing test (writer + validation on synthetic candidates)**

```python
from coach.game import build as GB  # noqa: E402


def test_write_release_layout_and_validation(tmp_path, built):
    feats = [{**c["features"], "id": c["public"]["id"]} for c in built]
    for i, f in enumerate(feats):
        f.update(x=[-40.0, 0.0, 0.0, 40.0, 40.0][i % 5], match=f"G{i}", shot_after=True)
    cands = {c["public"]["id"]: c for c in built}
    days = [[f for f in feats[:5]]] if len(feats) >= 5 else []
    out = GB.write_release(tmp_path, cands, days, start="2026-12-01")
    for d in days:
        assert (out / "public" / "days" / "2026-12-01.json").exists()
    for mid in cands:
        assert (out / "public" / "moments" / f"{mid}.json").exists()
        assert (out / "answers" / f"{mid}.json").exists()
    assert GB.validate_release(out) == []
```

- [ ] **Step 2: Fail. Step 3: Implement**

```python
"""Build a Playmaker content release from the 27 open matches (models held out per match, 20 refits).

    python -m coach.game.build --out D:/cv-data/playmaker --start 2026-12-01 --days 60
    python -m coach.game.build --validate D:/cv-data/playmaker/content-v1

Checkpoints: one candidates file per match in <out>/cache (restartable when the USB key disconnects)."""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import jsonschema

from ..stamp import generated_by
from . import CONTENT_VERSION
from . import days as GD

SCHEMAS = Path(__file__).resolve().parent / "schemas"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def write_release(root: Path, cands: dict, days: list, start: str) -> Path:
    out = Path(root) / f"content-v{CONTENT_VERSION}"
    first = dt.date.fromisoformat(start)
    used = {f["id"] for d in days for f in d}
    for mid, c in cands.items():
        if mid in used or not days:
            _write(out / "public" / "moments" / f"{mid}.json", c["public"])
            _write(out / "answers" / f"{mid}.json", c["answer"])
    for i, d in enumerate(days):
        date = (first + dt.timedelta(days=i)).isoformat()
        _write(
            out / "public" / "days" / f"{date}.json",
            {"schema": 1, "content_version": CONTENT_VERSION, "day_number": i + 1, "date": date,
             "moments": [f["id"] for f in d]},
        )
    shutil.copytree(SCHEMAS, out / "schemas", dirs_exist_ok=True)
    _write(out / "release.json", {"content_version": CONTENT_VERSION, "start": start, "days": len(days),
                                  "moments": len(used) if days else len(cands), "generated_by": generated_by()})
    return out


def validate_release(out: Path) -> list:
    out = Path(out)
    sch = {n: json.loads((SCHEMAS / f"{n}.schema.json").read_text(encoding="utf-8")) for n in ("day", "moment", "answer")}
    errors = []
    for kind, folder in (("day", "public/days"), ("moment", "public/moments"), ("answer", "answers")):
        for f in sorted((out / folder).glob("*.json")):
            try:
                jsonschema.validate(json.loads(f.read_text(encoding="utf-8")), sch[kind])
            except jsonschema.ValidationError as e:
                errors.append(f"{f.name}: {e.message}")
    for f in (out / "public" / "days").glob("*.json"):
        for mid in json.loads(f.read_text(encoding="utf-8"))["moments"]:
            if not (out / "public" / "moments" / f"{mid}.json").exists() or not (out / "answers" / f"{mid}.json").exists():
                errors.append(f"{f.name}: missing files for {mid}")
    for f in (out / "public").rglob("*.json"):
        text = f.read_text(encoding="utf-8")
        if '"points"' in text or '"pro_pick"' in text or '"share_top"' in text:
            errors.append(f"{f.name}: answer field in a public file")
    return errors


def candidates(cache: Path, n_refits: int) -> dict:
    """All matches -> {moment id: candidate}; per-match results cached as JSON in `cache`."""
    from .. import benchmark as B
    from .. import cross_league as CL
    from .. import report as R
    from . import moments as GM
    from . import refits as GR

    matches, _, _ = B.load_set("pooled")
    tables = {mid: R.match_tables(m) for mid, m in matches.items()}
    out = {}
    for mid, m in matches.items():
        f = cache / f"{mid}.json"
        if not f.exists():
            league = "A-League" if CL.league_of(m) == "skillcorner" else "Bundesliga 2"
            cands = GM.match_moments(m, R.held_out_models(mid, tables), GR.refit_models(mid, tables, n_refits), league)
            _write(f, cands)
            print(mid, len(cands), "moments", flush=True)
        for c in json.loads(f.read_text(encoding="utf-8")):
            out[c["public"]["id"]] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("D:/cv-data/playmaker"))
    ap.add_argument("--start", default="2026-12-01")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--refits", type=int, default=20)
    ap.add_argument("--validate", type=Path)
    args = ap.parse_args()
    if args.validate:
        errs = validate_release(args.validate)
        print("\n".join(errs) or "release OK")
        raise SystemExit(1 if errs else 0)
    cands = candidates(args.out / "cache", args.refits)
    feats = [{**c["features"], "id": mid} for mid, c in cands.items()]
    days = GD.daily_sets(GD.rank_pool(feats, args.pool), args.days)
    print(len(cands), "candidates,", len(days), "days")
    out = write_release(args.out, cands, days, args.start)
    errs = validate_release(out)
    print("\n".join(errs) or f"release OK: {out}")


if __name__ == "__main__":
    main()
```

Check `CL.league_of` return values before running: `grep -n "def league_of" -A6 coach/cross_league.py`; use the
exact string it returns for SkillCorner matches.

- [ ] **Step 4: Run tests, ruff check + format, commit** `"20 game: content release builder and validator"`.

- [ ] **Step 5: Build the real release (background, ~45-90 min)**

Run (from `P`): `PYTHONUNBUFFERED=1 $PY -m coach.game.build --out D:/cv-data/playmaker --start 2026-12-01 --days 60 > D:/cv-data/logs/game_build.log 2>&1`
Expected last lines: `<N> candidates, 60 days` and `release OK: D:\cv-data\playmaker\content-v1`.
Then: `$PY -m coach.game.build --validate D:/cv-data/playmaker/content-v1` → `release OK`.
Record N, the number of days and the share of candidates removed by each quality check in the README (Task 10).

### Task 10: Library docs and push

- [ ] **Step 1:** Add a "Playmaker content (2026-11-09)" section to `P/README.md`: what the release contains, the
counts from Task 9, the points rule, the commands (`coach.game.build`, `coach.rebuild_results`), the credit lines.
- [ ] **Step 2:** `DATA_LICENSES.md` row 20: add the raw-source URLs (figshare DOI 10.6084/m9.figshare.28196177,
github.com/SkillCorner/opendata) and "Playmaker content releases contain derived positions only, with credit lines".
- [ ] **Step 3:** Full library tests + ruff, then commit and push to `main`:
`git -c credential.helper= -c "credential.helper=!gh auth git-credential" push origin main`; check the CI run is green.

---

## Part M1: product repo `playmaker`

### Task 11: Scaffold

**Files:** Create `G/package.json`, `G/tsconfig.json`, `G/.gitignore`, `G/api/package.json`, `G/app/package.json`,
`G/vite.config.ts`, `G/README.md`, `G/LICENSE` (MIT, "Yassine Yandouzi").

- [ ] **Step 1: Root files**

`G/package.json`:

```json
{
  "name": "playmaker",
  "private": true,
  "type": "module",
  "workspaces": ["api", "app"],
  "scripts": {
    "dev": "node --experimental-strip-types api/src/dev.ts",
    "dev:app": "vite",
    "build": "tsc -p tsconfig.json --noEmit && vite build",
    "test": "vitest run",
    "e2e": "playwright test",
    "content:link": "node scripts/link-content.mjs"
  },
  "devDependencies": {
    "@playwright/test": "^1.56.0",
    "@types/node": "^24.0.0",
    "ajv": "^8.17.1",
    "typescript": "^5.9.0",
    "vite": "^7.1.0",
    "vitest": "^3.2.0"
  }
}
```

`G/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022", "module": "ESNext", "moduleResolution": "Bundler", "strict": true,
    "noUncheckedIndexedAccess": true, "allowImportingTsExtensions": true, "noEmit": true,
    "lib": ["ES2022", "DOM"], "types": ["node"], "skipLibCheck": true
  },
  "include": ["api/src", "app/src", "tests"]
}
```

`G/.gitignore`: `node_modules/`, `dist/`, `content/`, `test-results/`, `playwright-report/`.

`G/vite.config.ts`:

```ts
import { defineConfig } from "vite";

export default defineConfig({
  root: "app",
  build: { outDir: "../dist", emptyOutDir: true },
  server: { port: 5173, proxy: { "/api": "http://localhost:8787", "/content": "http://localhost:8787" } },
});
```

`G/scripts/link-content.mjs` copies a release into `G/content` (gitignored):

```js
import fs from "node:fs";
import path from "node:path";
const src = process.argv[2];
if (!src || !fs.existsSync(path.join(src, "release.json"))) { console.error("usage: npm run content:link -- <content-vN folder>"); process.exit(1); }
const dest = path.resolve("content");
fs.rmSync(dest, { recursive: true, force: true });
fs.cpSync(src, dest, { recursive: true });
console.log("content ->", dest);
```

- [ ] **Step 2:** `git init -b main`, `npm install`, commit `"Scaffold Playmaker (app + api workspaces)"`.

### Task 12: API types and validation

**Files:** Create `G/api/src/types.ts`, `G/api/src/validate.ts`; test `G/api/test/validate.test.ts`.

- [ ] **Step 1: `types.ts`**

```ts
export interface AnswerOption { k: number; score: number; p: number; share_top: number; points: number; reasons: string[] }
export interface Answer {
  schema: 1; id: string; options: AnswerOption[]; ai_pick: number; pro_pick: number;
  outcome: { kind: "goal" | "shot" | "kept" | "lost"; net_xg_15s: number };
  models: { content_version: number; trained_without: string; refits: number };
}
export interface PickRequest { player: string; moment: string; option: number }
export interface PickResult {
  points: number; among_best: boolean; pro_instinct: boolean; ai_pick: number; pro_pick: number;
  outcome: Answer["outcome"]; reasons: Record<number, string[]>; crowd: Record<number, number> | null; repeat: boolean;
}
export interface StoredPick { option: number; points: number; at: string }
export interface Store {
  getPick(player: string, moment: string): Promise<StoredPick | null>;
  putPickIfAbsent(player: string, moment: string, pick: StoredPick): Promise<boolean>;
  addCount(moment: string, option: number): Promise<void>;
  counts(moment: string): Promise<Record<number, number>>;
  bumpRate(key: string): Promise<number>;
}
export interface Deps {
  store: Store;
  answer(moment: string): Promise<Answer | null>;
  momentDate(moment: string): Promise<string | null>; // the day (YYYY-MM-DD) the moment is published, null if none
  today(): string; // UTC date YYYY-MM-DD
  ipHash(ip: string): string;
}
```

- [ ] **Step 2: Failing test `validate.test.ts`**

```ts
import { describe, expect, it } from "vitest";
import { parsePick } from "../src/validate.ts";

const ok = { player: "3f1c2a9e-5b7d-4c1e-9a2b-0d4e6f8a1b2c", moment: "J03WOH-0-1200", option: 3 };

describe("parsePick", () => {
  it("accepts a valid body", () => expect(parsePick(JSON.stringify(ok))).toEqual(ok));
  it("rejects bad uuid, id, option, size and junk", () => {
    expect(parsePick(JSON.stringify({ ...ok, player: "x" }))).toBeNull();
    expect(parsePick(JSON.stringify({ ...ok, moment: "../etc" }))).toBeNull();
    expect(parsePick(JSON.stringify({ ...ok, option: -1 }))).toBeNull();
    expect(parsePick(JSON.stringify({ ...ok, option: 2.5 }))).toBeNull();
    expect(parsePick("x".repeat(1025))).toBeNull();
    expect(parsePick("{")).toBeNull();
    expect(parsePick(JSON.stringify({ ...ok, extra: 1 }))).toBeNull();
  });
});
```

- [ ] **Step 3: Implement `validate.ts`**

```ts
import type { PickRequest } from "./types.ts";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const ID = /^[A-Za-z0-9-]{1,64}$/;

export function parsePick(body: string | null | undefined): PickRequest | null {
  if (!body || body.length > 1024) return null;
  let v: unknown;
  try { v = JSON.parse(body); } catch { return null; }
  if (typeof v !== "object" || v === null || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  if (Object.keys(o).sort().join() !== "moment,option,player") return null;
  if (typeof o.player !== "string" || !UUID.test(o.player)) return null;
  if (typeof o.moment !== "string" || !ID.test(o.moment)) return null;
  if (typeof o.option !== "number" || !Number.isInteger(o.option) || o.option < 0 || o.option > 63) return null;
  return { player: o.player, moment: o.moment, option: o.option };
}
```

- [ ] **Step 4:** `npx vitest run api` → PASS. **Step 5:** commit `"API: types and request validation"`.

### Task 13: Result builder and handler

**Files:** Create `G/api/src/result.ts`, `G/api/src/handler.ts`, `G/api/src/memoryStore.ts`; tests
`G/api/test/handler.test.ts`.

- [ ] **Step 1: Failing tests**

```ts
import { describe, expect, it } from "vitest";
import { handlePick } from "../src/handler.ts";
import { MemoryStore } from "../src/memoryStore.ts";
import type { Answer, Deps } from "../src/types.ts";

const answer: Answer = {
  schema: 1, id: "M1",
  options: [
    { k: 0, score: 0.02, p: 0.8, share_top: 0.7, points: 100, reasons: ["receiver has 6.2 m of space"] },
    { k: 1, score: 0.01, p: 0.9, share_top: 0.3, points: 100, reasons: [] },
    { k: 2, score: -0.01, p: 0.5, share_top: 0, points: 0, reasons: ["2 defenders in the lane"] },
  ],
  ai_pick: 0, pro_pick: 2, outcome: { kind: "kept", net_xg_15s: 0.004 },
  models: { content_version: 1, trained_without: "X", refits: 20 },
};
const P = (n: number) => `3f1c2a9e-5b7d-4c1e-9a2b-${String(n).padStart(12, "0")}`;
const deps = (over: Partial<Deps> = {}): Deps => ({
  store: new MemoryStore(), answer: async (m) => (m === "M1" ? answer : null),
  momentDate: async (m) => (m === "M1" ? "2026-12-01" : null), today: () => "2026-12-01", ipHash: (ip) => "h" + ip, ...over,
});
const body = (player: string, option: number, moment = "M1") => JSON.stringify({ player, moment, option });

describe("handlePick", () => {
  it("scores a pick and reveals the answer", async () => {
    const r = await handlePick(body(P(1), 2), "1.1.1.1", deps());
    expect(r.status).toBe(200);
    expect(r.body).toMatchObject({ points: 0, among_best: false, pro_instinct: true, ai_pick: 0, pro_pick: 2, crowd: null, repeat: false });
  });
  it("first pick wins", async () => {
    const d = deps();
    await handlePick(body(P(1), 2), "1.1.1.1", d);
    const r = await handlePick(body(P(1), 0), "1.1.1.1", d);
    expect(r.body).toMatchObject({ points: 0, repeat: true });
    expect(await d.store.counts("M1")).toEqual({ 2: 1 });
  });
  it("shows the crowd from 20 picks", async () => {
    const d = deps();
    for (let i = 0; i < 20; i++) await handlePick(body(P(i), i % 2), `10.0.0.${i}`, d);
    const r = await handlePick(body(P(99), 0), "9.9.9.9", d);
    expect(r.body.crowd).toEqual({ 0: 11 / 21, 1: 10 / 21 });
  });
  it("rejects invalid, unknown, future and rate-limited requests", async () => {
    expect((await handlePick("{", "1.1.1.1", deps())).status).toBe(400);
    expect((await handlePick(body(P(1), 9), "1.1.1.1", deps())).status).toBe(400);
    expect((await handlePick(body(P(1), 0, "M2"), "1.1.1.1", deps())).status).toBe(404);
    expect((await handlePick(body(P(1), 0), "1.1.1.1", deps({ today: () => "2026-11-29" }))).status).toBe(404);
    const d = deps();
    for (let i = 0; i < 50; i++) await handlePick(body(P(i), 0), "7.7.7.7", d);
    expect((await handlePick(body(P(77), 0), "7.7.7.7", d)).status).toBe(429);
  });
});
```

- [ ] **Step 2: Fail. Step 3: Implement**

`memoryStore.ts`:

```ts
import type { Store, StoredPick } from "./types.ts";

export class MemoryStore implements Store {
  private picks = new Map<string, StoredPick>();
  private count = new Map<string, Record<number, number>>();
  private rate = new Map<string, number>();
  async getPick(p: string, m: string) { return this.picks.get(`${p}|${m}`) ?? null; }
  async putPickIfAbsent(p: string, m: string, pick: StoredPick) {
    const k = `${p}|${m}`;
    if (this.picks.has(k)) return false;
    this.picks.set(k, pick);
    return true;
  }
  async addCount(m: string, o: number) { const c = this.count.get(m) ?? {}; c[o] = (c[o] ?? 0) + 1; this.count.set(m, c); }
  async counts(m: string) { return { ...(this.count.get(m) ?? {}) }; }
  async bumpRate(key: string) { const n = (this.rate.get(key) ?? 0) + 1; this.rate.set(key, n); return n; }
}
```

`result.ts`:

```ts
import type { Answer, PickResult } from "./types.ts";

export const CROWD_MIN = 20;

export function buildResult(a: Answer, option: number, counts: Record<number, number>, repeat: boolean): PickResult {
  const opt = a.options.find((o) => o.k === option)!;
  const total = Object.values(counts).reduce((s, n) => s + n, 0);
  const crowd = total >= CROWD_MIN ? Object.fromEntries(Object.entries(counts).map(([k, n]) => [Number(k), n / total])) : null;
  return {
    points: opt.points, among_best: opt.points === 100, pro_instinct: option === a.pro_pick,
    ai_pick: a.ai_pick, pro_pick: a.pro_pick, outcome: a.outcome,
    reasons: Object.fromEntries(a.options.map((o) => [o.k, o.reasons])), crowd, repeat,
  };
}
```

`handler.ts`:

```ts
import { buildResult } from "./result.ts";
import type { Deps } from "./types.ts";
import { parsePick } from "./validate.ts";

export const RATE_PER_DAY = 50;
export interface Reply { status: number; body: any }

const plusDays = (d: string, n: number) => new Date(Date.parse(d + "T00:00:00Z") + n * 86400000).toISOString().slice(0, 10);

export async function handlePick(raw: string | null | undefined, ip: string, deps: Deps): Promise<Reply> {
  const req = parsePick(raw);
  if (!req) return { status: 400, body: { error: "invalid request" } };
  const [answer, date] = await Promise.all([deps.answer(req.moment), deps.momentDate(req.moment)]);
  if (!answer || !date || date > plusDays(deps.today(), 1)) return { status: 404, body: { error: "unknown moment" } };
  if (!answer.options.some((o) => o.k === req.option)) return { status: 400, body: { error: "invalid option" } };
  const prev = await deps.store.getPick(req.player, req.moment);
  if (prev) return { status: 200, body: buildResult(answer, prev.option, await deps.store.counts(req.moment), true) };
  const n = await deps.store.bumpRate(`${deps.ipHash(ip)}#${deps.today()}`);
  if (n > RATE_PER_DAY) return { status: 429, body: { error: "daily limit reached" } };
  const points = answer.options.find((o) => o.k === req.option)!.points;
  const fresh = await deps.store.putPickIfAbsent(req.player, req.moment, { option: req.option, points, at: new Date().toISOString() });
  if (fresh) await deps.store.addCount(req.moment, req.option);
  const stored = fresh ? req.option : (await deps.store.getPick(req.player, req.moment))!.option;
  return { status: 200, body: buildResult(answer, stored, await deps.store.counts(req.moment), !fresh) };
}
```

Note the "crowd from 20 picks" test: the 21st pick is counted before the shares are computed (11 vs 10 of 21).

- [ ] **Step 4:** `npx vitest run api` → PASS. **Step 5:** commit `"API: pick handler, result builder, in-memory store"`.

### Task 14: Local dev server

**Files:** Create `G/api/src/fsAnswers.ts`, `G/api/src/dev.ts`; test `G/api/test/fsAnswers.test.ts`.

- [ ] **Step 1: Failing test** (temp content folder with one day + one answer):

```ts
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { expect, it } from "vitest";
import { fsDeps } from "../src/fsAnswers.ts";

it("reads answers and the publication date of a moment", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "pm-"));
  fs.mkdirSync(path.join(root, "answers"), { recursive: true });
  fs.mkdirSync(path.join(root, "public", "days"), { recursive: true });
  fs.writeFileSync(path.join(root, "answers", "M1.json"), JSON.stringify({ id: "M1" }));
  fs.writeFileSync(path.join(root, "public", "days", "2026-12-01.json"), JSON.stringify({ moments: ["M1"] }));
  const d = fsDeps(root);
  expect((await d.answer("M1"))?.id).toBe("M1");
  expect(await d.answer("M9")).toBeNull();
  expect(await d.momentDate("M1")).toBe("2026-12-01");
  expect(await d.momentDate("M9")).toBeNull();
});
```

- [ ] **Step 2: Fail. Step 3: Implement `fsAnswers.ts`**

```ts
import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { MemoryStore } from "./memoryStore.ts";
import type { Answer, Deps } from "./types.ts";

export function fsDeps(root: string): Deps {
  const dates = new Map<string, string>();
  const dir = path.join(root, "public", "days");
  for (const f of fs.existsSync(dir) ? fs.readdirSync(dir) : []) {
    const day = JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"));
    for (const m of day.moments) dates.set(m, f.replace(/\.json$/, ""));
  }
  return {
    store: new MemoryStore(),
    async answer(m) {
      const f = path.join(root, "answers", `${m}.json`);
      return fs.existsSync(f) ? (JSON.parse(fs.readFileSync(f, "utf8")) as Answer) : null;
    },
    async momentDate(m) { return dates.get(m) ?? null; },
    today: () => process.env.PLAYMAKER_TODAY ?? new Date().toISOString().slice(0, 10),
    ipHash: (ip) => createHash("sha256").update("dev-salt" + ip).digest("hex").slice(0, 16),
  };
}
```

`dev.ts` (serves `content/public` under `/content/`, never `content/answers`; `POST /api/pick`):

```ts
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fsDeps } from "./fsAnswers.ts";
import { handlePick } from "./handler.ts";

const root = path.resolve(process.env.PLAYMAKER_CONTENT ?? "content");
const pub = path.join(root, "public");
const deps = fsDeps(root);

http.createServer(async (req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  if (req.method === "POST" && url.pathname === "/api/pick") {
    let body = "";
    for await (const chunk of req) { body += chunk; if (body.length > 2048) break; }
    const r = await handlePick(body, req.socket.remoteAddress ?? "", deps);
    res.writeHead(r.status, { "content-type": "application/json" }).end(JSON.stringify(r.body));
    return;
  }
  if (req.method === "GET" && url.pathname.startsWith("/content/")) {
    const f = path.normalize(path.join(pub, url.pathname.slice("/content/".length)));
    if (!f.startsWith(pub + path.sep) || !fs.existsSync(f)) { res.writeHead(404).end(); return; }
    const day = /days[\\/](\d{4}-\d{2}-\d{2})\.json$/.exec(f)?.[1];
    if (day && day > deps.today() && day > new Date(Date.parse(deps.today()) + 86400000).toISOString().slice(0, 10)) { res.writeHead(404).end(); return; }
    res.writeHead(200, { "content-type": "application/json" }).end(fs.readFileSync(f));
    return;
  }
  res.writeHead(404).end();
}).listen(8787, () => console.log(`dev API on http://localhost:8787 (content ${root})`));
```

- [ ] **Step 4:** tests PASS; manual check: `npm run content:link -- D:/cv-data/playmaker/content-v1`, then
`PLAYMAKER_TODAY=2026-12-01 npm run dev` and `curl -s localhost:8787/content/days/2026-12-01.json` → the day JSON;
`curl -s localhost:8787/content/../answers/x.json` → 404.
- [ ] **Step 5:** commit `"Local dev server: public content + real pick handler"`.

### Task 15: Game client modules

**Files:** Create `G/app/index.html`, `G/app/src/api.ts`, `G/app/src/storage.ts`, `G/app/src/share.ts`,
`G/app/src/pitch.ts`, `G/app/src/round.ts`, `G/app/src/main.ts`, `G/app/src/styles.css`; tests
`G/app/test/storage.test.ts`, `G/app/test/share.test.ts`, `G/app/test/pitch.test.ts`.

- [ ] **Step 1: Failing unit tests**

```ts
// app/test/share.test.ts
import { expect, it } from "vitest";
import { shareText, square } from "../src/share.ts";
it("squares and spoiler-free share text", () => {
  expect([100, 60, 50, 49, 0].map(square).join("")).toBe("🟩🟨🟨⬜⬜");
  expect(shareText(42, [100, 60, 0, 100, 100])).toBe("⚽ Playmaker #42 · 360/500\n🟩🟨⬜🟩🟩");
});
```

```ts
// app/test/storage.test.ts
import { beforeEach, expect, it } from "vitest";
import { History } from "../src/storage.ts";
const mem = () => { const m = new Map<string, string>(); return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) }; };
let h: History;
beforeEach(() => { h = new History(mem()); });
it("keeps one player id", () => { expect(h.player()).toMatch(/^[0-9a-f-]{36}$/); expect(h.player()).toBe(h.player()); });
it("streak counts consecutive finished days", () => {
  h.finish("2026-12-01", [100, 0, 0, 0, 0]); h.finish("2026-12-02", [100, 100, 0, 0, 0]);
  expect(h.streak("2026-12-02")).toBe(2);
  expect(h.streak("2026-12-04")).toBe(0);
});
it("survives a throwing storage", () => {
  const bad = { getItem: () => { throw new Error("x"); }, setItem: () => { throw new Error("x"); } };
  const b = new History(bad); expect(b.player()).toMatch(/^[0-9a-f-]{36}$/); b.finish("2026-12-01", [1, 2, 3, 4, 5]);
});
```

```ts
// app/test/pitch.test.ts
import { expect, it } from "vitest";
import { toScreen, hitOption } from "../src/pitch.ts";
it("maps metres to a portrait canvas and finds the tapped option", () => {
  const view = { w: 390, h: 600 };
  const c = toScreen([0, 0], view);
  expect(c[0]).toBeCloseTo(195); expect(c[1]).toBeCloseTo(300);
  const opts = [{ k: 0, at: toScreen([10, 0], view) }, { k: 1, at: toScreen([-10, 5], view) }];
  expect(hitOption(opts, [opts[1]!.at[0] + 5, opts[1]!.at[1]], 30)).toBe(1);
  expect(hitOption(opts, [0, 0], 30)).toBeNull();
});
```

- [ ] **Step 2: Fail. Step 3: Implement**

`share.ts`:

```ts
export const square = (p: number) => (p >= 100 ? "🟩" : p >= 50 ? "🟨" : "⬜");
export const shareText = (day: number, points: number[]) =>
  `⚽ Playmaker #${day} · ${points.reduce((a, b) => a + b, 0)}/${points.length * 100}\n${points.map(square).join("")}`;
```

`storage.ts`:

```ts
export interface KV { getItem(k: string): string | null; setItem(k: string, v: string): void }
const KEY_ID = "pm.player", KEY_DAYS = "pm.days";

export class History {
  private fallbackId: string | null = null;
  constructor(private kv: KV) {}
  private get(k: string) { try { return this.kv.getItem(k); } catch { return null; } }
  private set(k: string, v: string) { try { this.kv.setItem(k, v); } catch { /* private mode: keep in memory only */ } }
  player(): string {
    const id = this.get(KEY_ID) ?? this.fallbackId ?? crypto.randomUUID();
    this.fallbackId = id; this.set(KEY_ID, id); return id;
  }
  days(): Record<string, number[]> { try { return JSON.parse(this.get(KEY_DAYS) ?? "{}"); } catch { return {}; } }
  finish(date: string, points: number[]) { const d = this.days(); d[date] = points; this.set(KEY_DAYS, JSON.stringify(d)); }
  streak(today: string): number {
    const d = this.days(); let n = 0; let t = Date.parse(today + "T00:00:00Z");
    while (d[new Date(t).toISOString().slice(0, 10)]) { n++; t -= 86400000; }
    return n;
  }
}
```

`api.ts`:

```ts
export interface Moment {
  id: string; league: string; home: string; away: string; attacking: "home" | "away"; minute: number; fps: number;
  carrier: number; players: { side: 0 | 1; shirt: number; name: string }[];
  lead_up: ([number, number] | null)[][]; ball: [number, number][];
  options: { k: number; kind: "pass" | "carry" | "shot"; player?: number; angle?: number; to: [number, number] }[];
  credit: string;
}
export interface Day { day_number: number; date: string; moments: string[] }
export interface PickResult {
  points: number; among_best: boolean; pro_instinct: boolean; ai_pick: number; pro_pick: number;
  outcome: { kind: "goal" | "shot" | "kept" | "lost"; net_xg_15s: number };
  reasons: Record<number, string[]>; crowd: Record<number, number> | null; repeat: boolean;
}
const json = async <T>(r: Response): Promise<T> => { if (!r.ok) throw new Error(`${r.status}`); return r.json() as Promise<T>; };
export const localDate = (d = new Date()) =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
export const getDay = (date: string) => fetch(`/content/days/${date}.json`).then(json<Day>);
export const getMoment = (id: string) => fetch(`/content/moments/${id}.json`).then(json<Moment>);
export const postPick = (player: string, moment: string, option: number) =>
  fetch("/api/pick", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ player, moment, option }) }).then(json<PickResult>);
```

`pitch.ts` (portrait: the attacking direction points up the screen; pitch 105 × 68 m):

```ts
import type { Moment } from "./api.ts";
export interface View { w: number; h: number }
export type Pt = [number, number];
const L = 105, W = 68, PAD = 12;

export function toScreen(p: Pt, v: View): Pt {
  const s = Math.min((v.w - 2 * PAD) / W, (v.h - 2 * PAD) / L);
  return [v.w / 2 + p[1] * s, v.h / 2 - p[0] * s];
}
export function hitOption(opts: { k: number; at: Pt }[], tap: Pt, radius: number): number | null {
  let best: number | null = null, bestD = radius;
  for (const o of opts) { const d = Math.hypot(o.at[0] - tap[0], o.at[1] - tap[1]); if (d <= bestD) { best = o.k; bestD = d; } }
  return best;
}
const m = (cm: [number, number]): Pt => [cm[0] / 100, cm[1] / 100];

export function draw(ctx: CanvasRenderingContext2D, v: View, mo: Moment, frame: number, highlight: { ai?: number; pro?: number; mine?: number } = {}) {
  ctx.clearRect(0, 0, v.w, v.h);
  ctx.fillStyle = "#1f6f3a"; ctx.fillRect(0, 0, v.w, v.h);
  ctx.strokeStyle = "rgba(255,255,255,0.7)"; ctx.lineWidth = 1.5;
  const c = (x: number, y: number) => toScreen([x, y], v);
  const [x0, y0] = c(L / 2, -W / 2), [x1, y1] = c(-L / 2, W / 2);
  ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
  ctx.beginPath(); ctx.moveTo(...c(0, -W / 2)); ctx.lineTo(...c(0, W / 2)); ctx.stroke();
  for (const sx of [1, -1]) { const [a, b] = c(sx * (L / 2 - 16.5), -20.16), [d, e] = c(sx * L / 2, 20.16); ctx.strokeRect(Math.min(a, d), Math.min(b, e), Math.abs(d - a), Math.abs(e - b)); }
  const f = mo.lead_up[Math.min(frame, mo.lead_up.length - 1)]!;
  const carrierAt = f[mo.carrier] ? c(...m(f[mo.carrier]!)) : null;
  const arrow = (k: number | undefined, color: string, width: number) => {
    if (k === undefined || !carrierAt) return;
    const o = mo.options.find((q) => q.k === k)!; const to = c(...m(o.to));
    ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath(); ctx.moveTo(...carrierAt); ctx.lineTo(...to); ctx.stroke();
  };
  arrow(highlight.pro, "#ffd166", 3); arrow(highlight.ai, "#4cc9f0", 4); arrow(highlight.mine, "#ffffff", 2);
  f.forEach((p, i) => {
    if (!p) return; const [x, y] = c(...m(p)); const pl = mo.players[i]!;
    ctx.fillStyle = pl.side === 0 ? "#e63946" : "#f1faee"; ctx.beginPath(); ctx.arc(x, y, i === mo.carrier ? 9 : 7, 0, 2 * Math.PI); ctx.fill();
    ctx.fillStyle = pl.side === 0 ? "#fff" : "#111"; ctx.font = "9px system-ui"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(pl.shirt), x, y);
  });
  const b = mo.ball[Math.min(frame, mo.ball.length - 1)]!; const [bx, by] = c(...m(b));
  ctx.fillStyle = "#ffe066"; ctx.beginPath(); ctx.arc(bx, by, 4, 0, 2 * Math.PI); ctx.fill();
}

export function optionTargets(v: View, mo: Moment): { k: number; at: Pt }[] {
  const last = mo.lead_up[mo.lead_up.length - 1]!;
  return mo.options.filter((o) => o.kind === "pass").map((o) => ({ k: o.k, at: toScreen(m(last[o.player!] ?? o.to), v) }));
}
```

`round.ts` — the state machine for one day (DOM wiring kept here, rendering in `pitch.ts`):

```ts
import { getMoment, postPick, type Day, type Moment, type PickResult } from "./api.ts";
import { draw, hitOption, optionTargets, type View } from "./pitch.ts";

export interface RoundUI {
  canvas: HTMLCanvasElement; title: HTMLElement; actions: HTMLElement; reveal: HTMLElement; next: HTMLButtonElement;
}
const OUTCOME = { goal: "Goal within 15 s", shot: "Shot within 15 s", kept: "Possession kept", lost: "Ball lost" };

export async function playMoment(ui: RoundUI, mo: Moment, player: string, index: number): Promise<PickResult> {
  const ctx = ui.canvas.getContext("2d")!;
  const v: View = { w: ui.canvas.clientWidth, h: ui.canvas.clientHeight };
  ui.canvas.width = v.w * devicePixelRatio; ui.canvas.height = v.h * devicePixelRatio; ctx.scale(devicePixelRatio, devicePixelRatio);
  const side = mo.attacking === "home" ? mo.home : mo.away;
  ui.title.textContent = `${index + 1}/5 · ${mo.home} – ${mo.away} · ${mo.minute}' · ${side} on the ball`;
  ui.reveal.hidden = true; ui.next.hidden = true;
  for (let f = 0; f < mo.lead_up.length; f++) { draw(ctx, v, mo, f); await new Promise((r) => setTimeout(r, 1000 / mo.fps)); }
  const last = mo.lead_up.length - 1;
  draw(ctx, v, mo, last);
  const pick = await new Promise<number>((resolve) => {
    const targets = optionTargets(v, mo);
    ui.canvas.onclick = (e) => { const r = ui.canvas.getBoundingClientRect(); const k = hitOption(targets, [e.clientX - r.left, e.clientY - r.top], 28); if (k !== null) resolve(k); };
    ui.actions.innerHTML = "";
    const carry = mo.options.filter((o) => o.kind === "carry");
    const best = carry.find((o) => o.angle === 0) ?? carry[0];
    if (best) ui.actions.append(button("Carry forward", () => resolve(best.k)));
    const shot = mo.options.find((o) => o.kind === "shot");
    if (shot) ui.actions.append(button("Shoot", () => resolve(shot.k)));
  });
  ui.canvas.onclick = null; ui.actions.innerHTML = "";
  const res = await postPick(player, mo.id, pick);
  draw(ctx, v, mo, last, { ai: res.ai_pick, pro: res.pro_pick, mine: pick });
  ui.reveal.hidden = false;
  ui.reveal.innerHTML = "";
  const line = (t: string, cls = "") => { const p = document.createElement("p"); p.textContent = t; if (cls) p.className = cls; ui.reveal.append(p); };
  line(`${res.points} points${res.among_best ? " · among the best" : ""}${res.pro_instinct ? " · pro instinct" : ""}`, "points");
  line(`AI pick (blue): ${label(mo, res.ai_pick)}. ${(res.reasons[res.ai_pick] ?? []).join("; ")}`);
  line(`What the pro did (yellow): ${label(mo, res.pro_pick)} → ${OUTCOME[res.outcome.kind]}.`);
  if (res.crowd) line(`Players: ${Math.round((res.crowd[pick] ?? 0) * 100)} % made your choice.`);
  line("The AI's ranking is a model opinion, not the truth.", "fine");
  return res;
}

function label(mo: Moment, k: number): string {
  const o = mo.options.find((q) => q.k === k)!;
  if (o.kind === "pass") { const p = mo.players[o.player!]!; return `pass to #${p.shirt} ${p.name}`; }
  return o.kind === "shot" ? "shoot" : `carry (${o.angle}°)`;
}
function button(text: string, on: () => void) { const b = document.createElement("button"); b.textContent = text; b.onclick = on; return b; }

export async function playDay(ui: RoundUI, day: Day, player: string): Promise<number[]> {
  const points: number[] = [];
  for (let i = 0; i < day.moments.length; i++) {
    const mo = await getMoment(day.moments[i]!);
    const res = await playMoment(ui, mo, player, i);
    points.push(res.points);
    ui.next.hidden = false; ui.next.textContent = i < day.moments.length - 1 ? "Next moment" : "See my score";
    await new Promise<void>((r) => (ui.next.onclick = () => r()));
  }
  return points;
}
```

`main.ts`:

```ts
import { getDay, localDate } from "./api.ts";
import { playDay } from "./round.ts";
import { shareText } from "./share.ts";
import { History } from "./storage.ts";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const history = new History(localStorage);
const today = new URLSearchParams(location.search).get("date") ?? localDate();

async function start() {
  const intro = $("intro"), game = $("game"), done = $("done");
  try {
    const day = await getDay(today);
    const played = history.days()[today];
    if (played) return finish(day.day_number, played);
    $<HTMLButtonElement>("play").onclick = async () => {
      intro.hidden = true; game.hidden = false;
      const pts = await playDay({ canvas: $("pitch"), title: $("title"), actions: $("actions"), reveal: $("reveal"), next: $<HTMLButtonElement>("next") }, day, history.player());
      history.finish(today, pts); finish(day.day_number, pts);
    };
  } catch { $("intro-text").textContent = "No challenge today yet. Come back tomorrow."; }
  function finish(n: number, pts: number[]) {
    intro.hidden = true; game.hidden = true; done.hidden = false;
    const text = shareText(n, pts);
    $("score").textContent = text; $("streak").textContent = `Streak: ${history.streak(today)} day(s)`;
    $<HTMLButtonElement>("share").onclick = async () => {
      try { if (navigator.share) await navigator.share({ text }); else await navigator.clipboard.writeText(text); $("share").textContent = "Copied"; } catch { /* user cancelled */ }
    };
  }
}
start();
```

`index.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Playmaker · daily football decisions</title>
  <meta name="description" content="Five real moments a day: pick the pass, then see the AI coach's pick, the pro's choice and what happened." />
  <link rel="stylesheet" href="/src/styles.css" />
</head>
<body>
  <main>
    <section id="intro">
      <h1>Playmaker</h1>
      <p id="intro-text">Five real moments from pro matches. You have the ball: pick the pass. Then see what the AI coach and the pro chose, and what happened.</p>
      <button id="play">Play today's five</button>
      <p class="fine"><a href="https://yandouziyassine.github.io/demos/football/">Free play: full-match replay with the coach</a> · <a href="/about.html">About the AI and privacy</a></p>
    </section>
    <section id="game" hidden>
      <p id="title"></p>
      <canvas id="pitch" aria-label="Pitch with the 22 players; tap a teammate to pass"></canvas>
      <div id="actions"></div>
      <div id="reveal" hidden aria-live="polite"></div>
      <button id="next" hidden>Next moment</button>
    </section>
    <section id="done" hidden>
      <pre id="score"></pre><p id="streak"></p><button id="share">Share</button>
    </section>
  </main>
  <footer class="fine">Data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0 · SkillCorner open data (MIT). Positions drawn by Playmaker.</footer>
  <script type="module" src="/src/main.ts"></script>
</body>
</html>
```

`styles.css`:

```css
:root { color-scheme: dark; --bg: #0d1b14; --fg: #f1faee; --accent: #4cc9f0; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.45 system-ui, sans-serif; }
main { max-width: 480px; margin: 0 auto; padding: 16px; }
h1 { font-size: 2rem; margin: 0.5rem 0; }
canvas { width: 100%; height: min(70vh, 640px); display: block; border-radius: 8px; touch-action: manipulation; }
button { font: inherit; padding: 0.7rem 1.1rem; border-radius: 999px; border: 0; background: var(--accent); color: #032; margin: 0.4rem 0.4rem 0 0; min-height: 44px; }
#reveal p { margin: 0.4rem 0; } .points { font-weight: 700; font-size: 1.2rem; }
.fine { font-size: 0.8rem; opacity: 0.75; } footer { max-width: 480px; margin: 0 auto; padding: 16px; }
a { color: var(--accent); } pre { font-size: 1.3rem; }
```

`app/public/about.html`: static page with the spec's limits (the value model explains ~2 % of what happens next;
the coach is bolder than pros; close calls; held-out models; xT slightly ahead in the benchmark) and the privacy
paragraph (no account, no cookie, random id in the browser, IPs never stored, 1-day salted hash for limits), with the
two credit lines. Write it as plain HTML with the same stylesheet link.

- [ ] **Step 4:** `npx vitest run` → all PASS; `npm run build` → `dist/` written, no type errors.
- [ ] **Step 5:** commit `"Game client: daily round, 2D pitch, reveal, share, streak"`.

### Task 16: End-to-end test on a phone viewport

**Files:** Create `G/playwright.config.ts`, `G/tests/e2e/day.spec.ts`, `G/tests/fixtures/content/…` (a tiny
release: 1 day, 5 moments copied from `D:/cv-data/playmaker/content-v1` by
`node scripts/make-fixture.mjs <release> <date>`).

- [ ] **Step 1: `scripts/make-fixture.mjs`** copies `public/days/<date>.json`, its 5 moment files and 5 answer files
into `tests/fixtures/content/` (committed: derived positions, licence credit inside every moment file).

```js
import fs from "node:fs";
import path from "node:path";
const [src, date] = process.argv.slice(2);
const dst = path.resolve("tests/fixtures/content");
fs.rmSync(dst, { recursive: true, force: true });
const day = JSON.parse(fs.readFileSync(path.join(src, "public/days", `${date}.json`), "utf8"));
const copy = (rel) => { fs.mkdirSync(path.dirname(path.join(dst, rel)), { recursive: true }); fs.copyFileSync(path.join(src, rel), path.join(dst, rel)); };
copy(`public/days/${date}.json`);
for (const m of day.moments) { copy(`public/moments/${m}.json`); copy(`answers/${m}.json`); }
console.log("fixture day", date, day.moments);
```

- [ ] **Step 2: `playwright.config.ts`**

```ts
import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "tests/e2e",
  use: { ...devices["Pixel 7"], baseURL: "http://localhost:5173" },
  webServer: [
    { command: "npm run dev", port: 8787, env: { PLAYMAKER_CONTENT: "tests/fixtures/content", PLAYMAKER_TODAY: "2026-12-01" }, reuseExistingServer: false },
    { command: "npm run dev:app", port: 5173, reuseExistingServer: false },
  ],
});
```

- [ ] **Step 3: `tests/e2e/day.spec.ts`**

```ts
import { expect, test } from "@playwright/test";

test("plays a full day and shows a share card", async ({ page }) => {
  await page.goto("/?date=2026-12-01");
  await page.getByRole("button", { name: "Play today's five" }).click();
  for (let i = 0; i < 5; i++) {
    await expect(page.locator("#title")).toContainText(`${i + 1}/5`);
    const carry = page.getByRole("button", { name: "Carry forward" });
    await carry.waitFor({ timeout: 10_000 });
    await carry.click();
    await expect(page.locator("#reveal")).toContainText("points");
    await expect(page.locator("#reveal")).toContainText("model opinion");
    await page.locator("#next").click();
  }
  await expect(page.locator("#score")).toContainText("Playmaker #1");
  await expect(page.locator("#score")).toContainText("/500");
  await page.reload();
  await expect(page.locator("#done")).toBeVisible();
});

test("public content never exposes answers", async ({ request }) => {
  expect((await request.get("http://localhost:8787/content/../answers/x.json")).status()).toBe(404);
  const day = await (await request.get("http://localhost:8787/content/days/2026-12-01.json")).json();
  const mo = await (await request.get(`http://localhost:8787/content/moments/${day.moments[0]}.json`)).text();
  for (const k of ["points", "pro_pick", "share_top", "reasons"]) expect(mo).not.toContain(`"${k}"`);
});
```

- [ ] **Step 4:** `npx playwright install chromium` (once), `npm run e2e` → 2 passed.
- [ ] **Step 5:** commit `"E2E: full day on a phone viewport; no answers in public content"`.

### Task 17: Content contract test in the product repo, CI, push

**Files:** Create `G/tests/contract.test.ts`, `G/.github/workflows/ci.yml`.

- [ ] **Step 1: `contract.test.ts`** validates every fixture file against the schemas shipped with the release
(`tests/fixtures/content/../schemas` is not copied by make-fixture; copy `schemas/` too: add
`fs.cpSync(path.join(src, "schemas"), path.join(dst, "schemas"), { recursive: true })` to `make-fixture.mjs`).

```ts
import fs from "node:fs";
import path from "node:path";
import Ajv2020 from "ajv/dist/2020.js";
import { expect, it } from "vitest";

const root = "tests/fixtures/content";
const ajv = new Ajv2020({ allErrors: true });
const schema = (n: string) => ajv.compile(JSON.parse(fs.readFileSync(path.join(root, "schemas", `${n}.schema.json`), "utf8")));
const files = (dir: string) => fs.readdirSync(path.join(root, dir)).map((f) => JSON.parse(fs.readFileSync(path.join(root, dir, f), "utf8")));

it("fixture content matches the release schemas", () => {
  for (const [name, dir] of [["day", "public/days"], ["moment", "public/moments"], ["answer", "answers"]] as const) {
    const v = schema(name);
    for (const f of files(dir)) expect(v(f), JSON.stringify(v.errors)).toBe(true);
  }
});
```

- [ ] **Step 2: `ci.yml`**

```yaml
name: ci
on: [push, pull_request]
permissions: { contents: read }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 24, cache: npm }
      - run: npm ci
      - run: npx tsc -p tsconfig.json --noEmit
      - run: npx vitest run
      - run: npm run build
      - run: npx playwright install --with-deps chromium
      - run: npm run e2e
```

- [ ] **Step 3:** all tests locally; commit `"Contract test against release schemas; CI"`.
- [ ] **Step 4:** create the public GitHub repo and push:
`gh repo create yandouziyassine/playmaker --public --source . --push --description "Daily football decision game built on an AI coach (open tracking data)"`;
check the CI run is green (`gh run list --limit 1`).
- [ ] **Step 5:** screenshot the game at 390×844 in the browser pane for the owner (dev servers via
`.claude/launch.json` entries `playmaker-api` and `playmaker-app`).

---

## Not in this plan (next plan, with M2)

AWS CDK stack (S3, CloudFront with the future-day function, HTTP API, Lambda, DynamoDB, SSM salt), the $20 lifetime
cap (budget + kill switch at $15), OIDC deploy, CloudWatch alarms, the DynamoDB `Store`, native free play, the 3D
toggle, the 14-day review page. Prerequisite: a regular AWS account (the training account connected on 2026-10-24
blocks S3 and CloudFormation) and the owner's go after the credit check.

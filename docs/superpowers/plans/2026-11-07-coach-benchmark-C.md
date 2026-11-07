# Coach benchmark vs xT and VAEP (insight C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-implement expected threat (Singh 2019) and VAEP (Decroos et al. 2019) on the 7 IDSSE matches and compare
them with the coach's value of the same held-out passes (agreement, calibration, disagreements, cost), written up in
`20-football-analytics/docs/coach_benchmark.md`.

**Architecture:** `coach/vaep.py` turns a `Match` into a SPADL-like action table (dict of NumPy arrays; passes and
shots from the events, carries from the tracking) and holds VAEP (game-state features, labels, two
`HistGradientBoostingClassifier`s, value formula, leave-one-match-out). `coach/xt.py` fits xT on that table (16 x 12
grid, value iteration). `coach/benchmark.py` joins, for every pass the match report scores
(`evaluate.pass_decisions`), the coach's held-out values (`report.held_out_models`) with the xT and VAEP values of the
same action, and computes the metrics with match-bootstrap 95 % CIs (`evaluate.bootstrap_ci`).

**Tech Stack:** NumPy, SciPy, scikit-learn (already in the project and CI); matplotlib lazily (charts only).

**Scope decision (data):** IDSSE only. The SkillCorner cache (`cv-data/skillcorner/derived/*.pkl`) holds model tables,
not action sequences or tracking, and the coach's pass decisions need the tracking at each release. Re-streaming the
20 A-League matches (~1.8 GB, now allowed on D:) plus the xG-proxy labels is left as a documented next step.

**Conventions:** run everything from `20-football-analytics/` with
`PY=C:/Users/yando/Code/.venv-tools/Scripts/python.exe`. Lint: `$PY -m ruff check <paths> && $PY -m ruff format --check <paths>`.
Commit only own files: `git commit -m "..." -- <paths>` (after `git add <new paths>`). Never `git add -A`.

---

### Task 1: SPADL-like action table

**Files:**
- Create: `20-football-analytics/coach/vaep.py`
- Test: `tests/test_20_benchmark.py`

- [ ] **Step 1: failing tests** (toy match: 2 halves, 2 players a side, home attacks +x in half 0)

```python
"""Project 20, AI coach: benchmark against xT and VAEP (coach insight C). Synthetic data only."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import match as M  # noqa: E402
from coach import vaep as V  # noqa: E402


def toy_match(match_id="TOY", frames=300):
    """Home players 0, 1; guest 2, 3. Half 0: pass 0 -> 1 at frame 50 (ball reaches 1 at frame 60), failed pass by 1 at
    frame 150 from (7.5, 10), guest goal at frame 200 from (-32.5, 0). Half 1 (sides swapped): kick-off 0 -> 1."""
    home = np.array([[-20.0, 0.0], [-5.0, 10.0]])
    guest = np.array([[10.0, 0.0], [20.0, -5.0]])
    halves = []
    for h in range(2):
        xy = np.repeat(np.concatenate([home, guest])[None], frames, axis=0).astype(np.float32)
        ball = np.zeros((frames, 3), np.float32)
        ball[:, :2] = home[0]
        ball[60:, :2] = home[1]
        if h == 1:
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
        list("abcd"),
        np.array([0, 0, 1, 1], np.int8),
        np.array([1, 2, 1, 2], np.int16),
        list("ABCD"),
        halves,
        [],
        {"home": "H", "guest": "G"},
    )

    def ev(kind, half, frame, x, y, **kw):
        t = float(m.halves[half].t[frame])
        return {"type": kind, "half": half, "frame": frame, "t": t, "x": x, "y": y, "set_piece": None, **kw}

    m.events = [
        ev("pass", 0, 50, 32.5, 34.0, team=0, player=0, recipient=1, success=True, cross=False),
        ev("pass", 0, 150, 60.0, 44.0, team=0, player=1, recipient=2, success=False, cross=False),
        ev("shot", 0, 200, 20.0, 34.0, team=1, player=3, goal=True, xg=0.3),
        {**ev("pass", 1, 10, 52.5, 34.0, team=0, player=0, recipient=1, success=True, cross=False), "set_piece": "KickOff"},
    ]
    return m


def test_actions_types_order_and_carry():
    a = V.actions(toy_match())
    assert list(a["type"]) == ["pass", "carry", "pass", "shot", "kickoff"]
    assert list(a["event"]) == [0, -1, 1, 2, 3]
    assert list(a["result"]) == [1, 1, 0, 1, 1]
    assert list(a["team"]) == [0, 0, 0, 1, 0]
    assert list(a["goal"]) == [0, 0, 0, 1, 0]
    assert np.allclose(a["t"], [2.0, 4.0, 6.0, 8.0, 0.4])
    assert a["xg"][3] == pytest.approx(0.3) and np.isnan(a["xg"][0])


def test_actions_coordinates_attack_plus_x():
    a = V.actions(toy_match())
    start = np.c_[a["x0"], a["y0"]]
    end = np.c_[a["x1"], a["y1"]]
    assert np.allclose(start[0], [-20, 0]) and np.allclose(end[0], [-5, 10])  # completed pass ends at the reception
    assert np.allclose(start[1], [-5, 10]) and np.allclose(end[1], [7.5, 10])  # carry: reception -> next action
    assert np.allclose(end[2], [-32.5, 0])  # failed pass ends where the other team next played the ball
    assert np.allclose(start[3], [32.5, 0]) and np.allclose(end[3], [52.5, 0])  # guest attacks -x: rotated
    assert np.allclose(start[4], [0, 0]) and np.allclose(end[4], [-5, -10])  # half 1: home attacks -x


def test_action_type_mapping():
    assert V.action_type({"type": "shot", "set_piece": "Penalty"}) == "shot_penalty"
    assert V.action_type({"type": "shot", "set_piece": "FreeKick"}) == "shot_freekick"
    assert V.action_type({"type": "pass", "set_piece": "CornerKick", "cross": True}) == "corner"
    assert V.action_type({"type": "pass", "set_piece": None, "cross": True}) == "cross"
    assert V.action_type({"type": "pass", "set_piece": None, "cross": False}) == "pass"
```

- [ ] **Step 2:** `$PY -m pytest ../tests/test_20_benchmark.py -q` -> FAIL (`cannot import name 'vaep'`).

- [ ] **Step 3: implement** `coach/vaep.py` (table part):

```python
"""VAEP (Decroos et al., KDD 2019) re-implemented on the IDSSE events, and the SPADL-like action table xT uses too.

    python -m coach.benchmark     # runs it (coach insight, part C); this module has no main

Action table (actions(m)): one row per on-ball action, a dict of NumPy arrays ordered by half and time: match, half,
t (s since the half's first frame), team, player, type (TYPES), result (1 success / 0 fail; a shot succeeds when it
scores), x0, y0, x1, y1 (metres, centred, normalised so that the ACTING team attacks +x), event (index in m.events,
-1 for carries), xg (provider xG of a shot, else NaN), goal (1 for a shot that scored).

The IDSSE events hold passes and shots only (no duels, tackles, interceptions, clearances or dribbles), so the table
has passes (by set piece: throw_in, freekick, corner, goalkick, kickoff; else cross or pass), shots (shot,
shot_freekick, shot_penalty) and carries taken from the tracking:
- a completed pass ends where the recipient stands when the ball first comes within ARRIVE_M of him after the
  release (within MAX_FLIGHT_S); otherwise, and for a failed pass, it ends at the start of the next action of the
  half (for a failed pass: where the other team next played the ball, or the restart);
- a shot ends at the goal centre;
- carry: after a completed pass, when the recipient plays the next action of the half CARRY_MIN_M..CARRY_MAX_M away
  from where he received the ball and within CARRY_MAX_S, a carry from the reception point to that action's start
  (the way socceraction adds dribbles). Carries succeed by construction.
"""

import numpy as np

from . import datasets as D
from . import features as F
from . import geometry as G
from .sync import event_xy

TYPES = (
    "pass",
    "cross",
    "throw_in",
    "freekick",
    "corner",
    "goalkick",
    "kickoff",
    "carry",
    "shot",
    "shot_freekick",
    "shot_penalty",
)
SHOTS = ("shot", "shot_freekick", "shot_penalty")
MOVES = tuple(t for t in TYPES if t not in SHOTS)
SET_PIECES = {
    "ThrowIn": "throw_in",
    "FreeKick": "freekick",
    "CornerKick": "corner",
    "GoalKick": "goalkick",
    "KickOff": "kickoff",
}
ARRIVE_M, MAX_FLIGHT_S = 2.0, 6.0
CARRY_MIN_M, CARRY_MAX_M, CARRY_MAX_S = 3.0, 60.0, 10.0
GOAL = np.array([G.HALF_LENGTH, 0.0])


def action_type(e) -> str:
    sp = e.get("set_piece")
    if e["type"] == "shot":
        return {"Penalty": "shot_penalty", "FreeKick": "shot_freekick"}.get(sp, "shot")
    if sp in SET_PIECES:
        return SET_PIECES[sp]
    return "cross" if e.get("cross") else "pass"


def _player(v):
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _arrival(m, e):
    """Tracking position of the recipient when the ball first comes within ARRIVE_M of him after the release."""
    r = e.get("recipient")
    if e.get("frame") is None or not _player(r) or not 0 <= r < len(m.player_ids):
        return None
    h = m.halves[e["half"]]
    f = int(e["frame"])
    up = min(len(h.t), f + int(MAX_FLIGHT_S * F.match_fps(m)) + 1)
    if up <= f + 1:
        return None
    d = np.linalg.norm(h.xy[f + 1 : up, r] - h.ball[f + 1 : up, :2], axis=1)
    k = np.flatnonzero(d <= ARRIVE_M)
    return h.xy[f + 1 + int(k[0]), r].astype(float) if len(k) else None


def actions(m) -> dict:
    """The SPADL-like action table of match m (see the module docstring)."""
    dirs = [G.attack_dirs(h, m.team) for h in m.halves]
    ev = [
        (i, e)
        for i, e in enumerate(m.events)
        if e["type"] in ("pass", "shot")
        and e.get("half") is not None
        and e.get("team") in (0, 1)
        and np.isfinite(e.get("x", np.nan))
    ]
    ev.sort(key=lambda ie: (ie[1]["half"], D.when(m, ie[1])))
    base = [
        {
            "half": e["half"],
            "t": D.when(m, e) - float(m.halves[e["half"]].t[0]),
            "team": int(e["team"]),
            "player": int(e["player"]) if _player(e.get("player")) else -1,
            "type": action_type(e),
            "start": event_xy(m, e).astype(float),
            "event": i,
            "e": e,
        }
        for i, e in ev
    ]
    rows = []
    for k, a in enumerate(base):
        e = a["e"]
        nxt = base[k + 1] if k + 1 < len(base) and base[k + 1]["half"] == a["half"] else None
        if a["type"] in SHOTS:
            end, result = GOAL * dirs[a["half"]][a["team"]], int(bool(e.get("goal")))
        else:
            result = int(bool(e.get("success")))
            got = _arrival(m, e) if result else None
            end = got if got is not None else (nxt["start"] if nxt is not None else a["start"])
        rows.append({**a, "end": end, "result": result})
        if a["type"] in MOVES and result and nxt is not None and nxt["team"] == a["team"]:
            if nxt["player"] == e.get("recipient"):
                gap = float(np.linalg.norm(nxt["start"] - end))
                if CARRY_MIN_M <= gap <= CARRY_MAX_M and nxt["t"] - a["t"] <= CARRY_MAX_S:
                    rows.append(
                        {
                            **a,
                            "t": (a["t"] + nxt["t"]) / 2,
                            "player": nxt["player"],
                            "type": "carry",
                            "start": end,
                            "end": nxt["start"],
                            "event": -1,
                            "e": None,
                            "result": 1,
                        }
                    )
    n = len(rows)
    d = np.array([dirs[r["half"]][r["team"]] for r in rows], float).reshape(n, 1)
    s = np.array([r["start"] for r in rows], float).reshape(n, 2) * d
    en = np.array([r["end"] for r in rows], float).reshape(n, 2) * d
    xg = [
        r["e"]["xg"] if r["type"] in SHOTS and r["e"].get("xg") is not None else np.nan for r in rows
    ]
    return {
        "match": np.array([str(m.match_id)] * n),
        "half": np.array([r["half"] for r in rows], int),
        "t": np.array([r["t"] for r in rows], float),
        "team": np.array([r["team"] for r in rows], int),
        "player": np.array([r["player"] for r in rows], int),
        "type": np.array([r["type"] for r in rows], dtype="U16"),
        "result": np.array([r["result"] for r in rows], int),
        "x0": s[:, 0],
        "y0": s[:, 1],
        "x1": en[:, 0],
        "y1": en[:, 1],
        "event": np.array([r["event"] for r in rows], int),
        "xg": np.array(xg, float),
        "goal": np.array([int(r["type"] in SHOTS and r["result"] == 1) for r in rows], int),
    }
```

- [ ] **Step 4:** run the tests -> 3 passed. Lint the two files.
- [ ] **Step 5: commit** `git add coach/vaep.py ../tests/test_20_benchmark.py && git commit -m "20 coach: SPADL-like action table (passes, shots, carries from tracking)" -- coach/vaep.py ../tests/test_20_benchmark.py`

### Task 2: expected threat

**Files:** Create `20-football-analytics/coach/xt.py`; Test: `tests/test_20_benchmark.py` (append).

- [ ] **Step 1: failing tests**

```python
from coach import xt as XT  # noqa: E402  (with the other imports)


def action_table(rows):
    """rows: (type, result, x0, x1, xg, goal) on the centre line y = 0."""
    n = len(rows)
    col = lambda i, dt=float: np.array([r[i] for r in rows], dt)
    return {
        "match": np.array(["T"] * n),
        "half": np.zeros(n, int),
        "t": np.arange(n, dtype=float),
        "team": np.zeros(n, int),
        "player": np.zeros(n, int),
        "type": col(0, "U16"),
        "result": col(1, int),
        "x0": col(2),
        "y0": np.zeros(n),
        "x1": col(3),
        "y1": np.zeros(n),
        "event": np.arange(n),
        "xg": col(4),
        "goal": col(5, int),
    }


def toy_grid():
    """Along the centre row: from every cell 10 completed and 2 failed passes one cell forward (none from the last
    cell); 4 shots from each of the last 4 cells, xG 0.05 .. 0.20 rising towards the goal, one goal per cell."""
    w = 105 / XT.NX
    rows = []
    for cx in range(XT.NX):
        x = -52.5 + (cx + 0.5) * w
        if cx < XT.NX - 1:
            rows += [("pass", 1, x, x + w, np.nan, 0)] * 10 + [("pass", 0, x, x + w, np.nan, 0)] * 2
        if cx >= XT.NX - 4:
            xg = 0.05 * (cx - XT.NX + 5)
            rows += [("shot", 1, x, 52.5, xg, 1)] + [("shot", 0, x, 52.5, xg, 0)] * 3
    return action_table(rows)


def test_xt_converges_to_fixed_point_and_rises_towards_goal():
    model = XT.ExpectedThreat(shrink=0.0).fit(toy_grid())
    assert model.iterations_ < XT.MAX_ITER
    xt = model.xt_
    assert np.allclose(xt, model.s_ * model.g_ + model.m_ * (model.T_ @ xt), atol=1e-5)
    row = model.grid()[:, XT.NY // 2]
    assert np.all(np.diff(row) > 0)
    assert row[-1] == pytest.approx(0.20)  # last cell: shots only, xT = mean xG there
    assert np.allclose(np.delete(model.grid(), XT.NY // 2, axis=1), 0)  # no action elsewhere


def test_xt_goal_source_and_action_values():
    a = toy_grid()
    goals = XT.ExpectedThreat(goal_source="goals", shrink=0.0).fit(a)
    assert goals.g_[XT.cell(np.array([52.0]), np.array([0.0]))[0]] == pytest.approx(0.25)
    model = XT.ExpectedThreat(shrink=0.0).fit(a)
    v = model.action_values(a)
    start = model.xt_[XT.cell(a["x0"], a["y0"])]
    end = model.xt_[XT.cell(a["x1"], a["y1"])]
    ok, failed, shot = a["result"] == 1, a["result"] == 0, a["type"] == "shot"
    assert np.allclose(v[ok & ~shot], (end - start)[ok & ~shot])
    assert np.allclose(v[failed & ~shot], -start[failed & ~shot])
    assert np.isnan(v[shot]).all()
```

- [ ] **Step 2:** run -> FAIL (`cannot import name 'xt'`).
- [ ] **Step 3: implement** `coach/xt.py`:

```python
"""Expected threat (xT, Singh 2019) re-implemented on the SPADL-like actions of coach.vaep (coach insight, part C).

Grid of NX x NY = 16 x 12 cells over the 105 x 68 m pitch, in the frame of the acting team (it attacks +x). Per cell,
from the actions starting there: s = shots / (shots + moves), m = moves / (shots + moves), g = scoring probability of
a shot there, T[c, c'] = completed moves from c that end in c' / moves from c (failed moves count in the denominator:
the ball is lost). xT = s * g + m * T @ xT, iterated from 0 until the largest change is below TOL (value iteration).
Moves = passes of every kind and carries; shots = open-play and free-kick shots (penalties out).

Deviation (documented in docs/coach_benchmark.md): Singh fitted a season of events; 6 matches have ~130 non-penalty
shots, so g from goals alone is 0 or 1 in most cells. By default g is the provider's xG of the shots in the cell,
shrunk towards the overall mean with SHRINK pseudo-shots (goal_source="xg"); goal_source="goals" uses goals.
Value of a move: xT(end) - xT(start) when it succeeds, -xT(start) when it fails (the threat held is lost).
"""

import json

import numpy as np

from . import vaep as V

NX, NY, LENGTH, WIDTH = 16, 12, 105.0, 68.0
TOL, MAX_ITER, SHRINK = 1e-6, 1000, 2.0
XT_SHOTS = ("shot", "shot_freekick")


def cell(x, y):
    """Cell index cx * NY + cy of attack-normalised centred points (clipped to the pitch)."""
    cx = np.clip(((np.asarray(x, float) + LENGTH / 2) / LENGTH * NX).astype(int), 0, NX - 1)
    cy = np.clip(((np.asarray(y, float) + WIDTH / 2) / WIDTH * NY).astype(int), 0, NY - 1)
    return cx * NY + cy


class ExpectedThreat:
    def __init__(self, goal_source: str = "xg", shrink: float = SHRINK):
        self.goal_source, self.shrink = goal_source, shrink

    def fit(self, a):
        n = NX * NY
        c0, c1 = cell(a["x0"], a["y0"]), cell(a["x1"], a["y1"])
        move = np.isin(a["type"], V.MOVES)
        shot = np.isin(a["type"], XT_SHOTS)
        n_move = np.bincount(c0[move], minlength=n).astype(float)
        n_shot = np.bincount(c0[shot], minlength=n).astype(float)
        total = n_move + n_shot
        self.s_ = np.divide(n_shot, total, out=np.zeros(n), where=total > 0)
        self.m_ = np.divide(n_move, total, out=np.zeros(n), where=total > 0)
        goals = a["goal"][shot].astype(float)
        xg = a["xg"][shot]
        target = goals if self.goal_source == "goals" else np.where(np.isfinite(xg), xg, goals)
        prior = float(target.mean()) if len(target) else 0.0
        num = np.bincount(c0[shot], weights=target, minlength=n) + self.shrink * prior
        den = n_shot + self.shrink
        self.g_ = np.divide(num, den, out=np.full(n, prior), where=den > 0)
        ok = move & (a["result"] == 1)
        T = np.zeros((n, n))
        np.add.at(T, (c0[ok], c1[ok]), 1.0)
        self.T_ = T / np.maximum(n_move, 1.0)[:, None]
        xt = np.zeros(n)
        for it in range(1, MAX_ITER + 1):
            new = self.s_ * self.g_ + self.m_ * (self.T_ @ xt)
            done = float(np.max(np.abs(new - xt))) < TOL
            xt = new
            if done:
                break
        self.xt_, self.iterations_ = xt, it
        return self

    def grid(self):
        """xT as an [NX, NY] array (x towards the attacked goal, then y)."""
        return self.xt_.reshape(NX, NY)

    def action_values(self, a):
        """xT(end) - xT(start) for completed moves, -xT(start) for failed ones, NaN for shots."""
        start = self.xt_[cell(a["x0"], a["y0"])]
        end = self.xt_[cell(a["x1"], a["y1"])]
        v = np.where(a["result"] == 1, end - start, -start)
        return np.where(np.isin(a["type"], V.MOVES), v, np.nan)

    def to_json(self) -> str:
        """What a browser needs: the grid, 6 decimals."""
        return json.dumps({"nx": NX, "ny": NY, "xt": [round(float(v), 6) for v in self.xt_]})
```

- [ ] **Step 4:** run -> 5 passed; lint.
- [ ] **Step 5: commit** `git add coach/xt.py && git commit -m "20 coach: expected threat (Singh 2019) re-implemented, value iteration" -- coach/xt.py ../tests/test_20_benchmark.py`

### Task 3: VAEP labels, features, values, LOMO

**Files:** Modify `20-football-analytics/coach/vaep.py` (append; add `from . import models as ML`); Test: append.

- [ ] **Step 1: failing tests**

```python
def test_vaep_labels_window_and_halves():
    a = V.actions(toy_match())
    scores, concedes = V.labels(a)
    assert list(scores) == [0, 0, 0, 1, 0]
    assert list(concedes) == [1, 1, 1, 0, 0]  # the kick-off of half 1 is out of the window
    scores, concedes = V.labels(a, k=2)
    assert list(concedes) == [0, 0, 1, 0, 0]


def test_vaep_values_formula():
    a = {
        "match": np.array(["A"] * 4),
        "half": np.array([0, 0, 0, 1]),
        "team": np.array([0, 0, 1, 1]),
        "goal": np.array([0, 0, 0, 0]),
    }
    ps, pc = np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.05, 0.1, 0.02, 0.03])
    v = V.values(a, ps, pc)
    # 0: start of the half; 1: same team; 2: other team (previous probabilities swapped); 3: new half
    assert np.allclose(v["value"], [0.05, 0.05, 0.2 + 0.18, 0.37])
    a["goal"] = np.array([0, 1, 0, 0])  # after a goal the previous probabilities are 0
    assert V.values(a, ps, pc)["value"][2] == pytest.approx(0.3 - 0.02)


def test_vaep_features_mirror_previous_opponent_action():
    a = V.actions(toy_match())
    X = V.features(a)
    assert X.shape == (5, len(V.FEATURES))
    f = {k: X[:, i] for i, k in enumerate(V.FEATURES)}
    assert f["a0_x0"][3] == pytest.approx(32.5)
    assert f["a1_x0"][3] == pytest.approx(-7.5)  # failed pass by the other team, seen from the shooter's side
    assert f["team_1"][3] == 0 and f["team_1"][2] == 1
    assert f["a1_x0"][0] == f["a0_x0"][0] and f["dt_1"][0] == 0  # first action: its own "previous"
    assert f["a1_x0"][4] == f["a0_x0"][4]  # a new half starts a new game state
    assert f["goals_against"][4] == 1 and f["goals_for"][3] == 0  # the score before the action


def test_vaep_lomo_held_out():
    tables = {mid: V.actions(toy_match(mid)) for mid in ("A", "B", "C")}
    out = V.lomo(tables)
    for mid, r in out.items():
        assert mid not in r["trained_on"] and len(r["trained_on"]) == 2
        assert len(r["value"]) == len(tables[mid]["type"])
        assert np.allclose(r["value"], V.values(tables[mid], r["p_score"], r["p_concede"])["value"])
```

- [ ] **Step 2:** run -> 4 FAIL (`no attribute labels`).
- [ ] **Step 3: implement** (append to `coach/vaep.py`, and extend its docstring with the VAEP paragraph):

```python
K = 10  # labels: goal within the next K actions of the half (the action included)
BLOCK = tuple(f"type_{t}" for t in TYPES) + (
    "result",
    "x0",
    "y0",
    "x1",
    "y1",
    "dx",
    "dy",
    "length",
    "start_goal_dist",
    "start_goal_angle",
    "end_goal_dist",
    "end_goal_angle",
)
FEATURES = tuple(f"a{lag}_{name}" for lag in range(3) for name in BLOCK) + (
    "team_1",
    "team_2",
    "dt_1",
    "dt_2",
    "gap_x",
    "gap_y",
    "goals_for",
    "goals_against",
    "goal_diff",
    "half",
    "t",
)


def labels(a, k: int = K) -> tuple:
    """(scores, concedes): 1 when the acting team scores / concedes within actions i .. i + k - 1 of the same half."""
    n = len(a["team"])
    scores, concedes = np.zeros(n, int), np.zeros(n, int)
    for i in range(n):
        for j in range(i, min(n, i + k)):
            if a["half"][j] != a["half"][i] or a["match"][j] != a["match"][i]:
                break
            if a["goal"][j]:
                if a["team"][j] == a["team"][i]:
                    scores[i] = 1
                else:
                    concedes[i] = 1
    return scores, concedes


def score_before(a) -> tuple:
    """(goals for, goals against) of the acting team before each action, per match."""
    n = len(a["team"])
    gf, ga, tally = np.zeros(n), np.zeros(n), {}
    for i in range(n):
        g = tally.setdefault(a["match"][i], [0, 0])
        gf[i], ga[i] = g[a["team"][i]], g[1 - a["team"][i]]
        if a["goal"][i]:
            g[a["team"][i]] += 1
    return gf, ga


def _previous(a, lag_idx, fallback):
    """Index of the action before lag_idx + 1 ... or `fallback` where it is in another half / match (or missing)."""
    ok = lag_idx >= 0
    j = np.maximum(lag_idx, 0)
    ok &= (a["half"][j] == a["half"]) & (a["match"][j] == a["match"])
    return np.where(ok, j, fallback)


def _block(a, idx, ref_team):
    """BLOCK features of the actions idx, seen from ref_team (an action by the other team is rotated by 180 deg)."""
    flip = np.where(a["team"][idx] == ref_team, 1.0, -1.0)
    x0, y0, x1, y1 = (a[k][idx] * flip for k in ("x0", "y0", "x1", "y1"))
    dx, dy = x1 - x0, y1 - y0
    d0, g0 = G.goal_geometry(np.c_[x0, y0])
    d1, g1 = G.goal_geometry(np.c_[x1, y1])
    onehot = (a["type"][idx][:, None] == np.array(TYPES)[None]).astype(float)
    cols = [a["result"][idx], x0, y0, x1, y1, dx, dy, np.hypot(dx, dy), d0, g0, d1, g1]
    return np.column_stack([onehot] + cols)


def features(a) -> np.ndarray:
    """Game state of each action: BLOCK of the action and of the two before it (from the acting team's side; at the
    start of a half the missing ones repeat the earliest available), plus team / time / gap / score features."""
    n = len(a["team"])
    i = np.arange(n)
    p1 = _previous(a, i - 1, i)
    p2 = _previous(a, i - 2, p1)
    ref = a["team"]
    flip1 = np.where(a["team"][p1] == ref, 1.0, -1.0)
    gf, ga = score_before(a)
    extra = [
        (a["team"][p1] == ref).astype(float),
        (a["team"][p2] == ref).astype(float),
        a["t"] - a["t"][p1],
        a["t"] - a["t"][p2],
        a["x0"] - a["x1"][p1] * flip1,
        a["y0"] - a["y1"][p1] * flip1,
        gf,
        ga,
        gf - ga,
        a["half"].astype(float),
        a["t"],
    ]
    return np.column_stack([_block(a, idx, ref) for idx in (i, p1, p2)] + extra).reshape(n, len(FEATURES))


def values(a, p_score, p_concede) -> dict:
    """offensive = P_score(i) - P_score(i-1), defensive = -(P_concede(i) - P_concede(i-1)), value = their sum; the
    previous pair is swapped when action i-1 was by the other team and is 0 at the start of a half or after a goal."""
    n = len(p_score)
    prev = np.arange(n) - 1
    j = np.maximum(prev, 0)
    start = (prev < 0) | (a["half"][j] != a["half"]) | (a["match"][j] != a["match"]) | (a["goal"][j] == 1)
    same = a["team"][j] == a["team"]
    ps = np.where(start, 0.0, np.where(same, p_score[j], p_concede[j]))
    pc = np.where(start, 0.0, np.where(same, p_concede[j], p_score[j]))
    off, dfn = p_score - ps, -(p_concede - pc)
    return {"offensive": off, "defensive": dfn, "value": off + dfn}


def fit(a) -> dict:
    """The two classifiers (ML.fit_boosting: depth 3, 300 trees) on an action table (several matches stacked)."""
    X = features(a)
    scores, concedes = labels(a)
    return {"score": ML.fit_boosting(X, scores), "concede": ML.fit_boosting(X, concedes)}


def predict(models, a) -> dict:
    X = features(a)
    ps = models["score"].predict_proba(X)[:, 1]
    pc = models["concede"].predict_proba(X)[:, 1]
    return {"p_score": ps, "p_concede": pc, **values(a, ps, pc)}


def lomo(tables: dict) -> dict:
    """match id -> predict() on that match with the classifiers fitted on the other matches, plus "trained_on"."""
    out = {}
    for mid in tables:
        others = [o for o in tables if o != mid]
        models = fit(D.stack([tables[o] for o in others]))
        out[mid] = {**predict(models, tables[mid]), "trained_on": others}
    return out
```

- [ ] **Step 4:** run -> 9 passed; lint.
- [ ] **Step 5: commit** `git commit -m "20 coach: VAEP (Decroos et al. 2019) re-implemented: game state, labels, values, LOMO" -- coach/vaep.py ../tests/test_20_benchmark.py`

### Task 4: benchmark metrics

**Files:** Create `20-football-analytics/coach/benchmark.py`; Test: append.

- [ ] **Step 1: failing tests**

```python
from coach import benchmark as B  # noqa: E402


def fake_rows(n_matches=4, n=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_matches):
        for i in range(n):
            base = float(rng.normal())
            rows.append(
                {
                    **{k: 0 for k in B.KEEP},
                    "match": f"M{g}",
                    "event": i,
                    "success": int(i % 3 > 0),
                    "coach_score": base,
                    "coach_dv": 2 * base,
                    "xt": -base,
                    "vaep": float(rng.normal()),
                    "net_xg": base + 0.1 * float(rng.normal()),
                    "goal_15s": int(base > 1.5),
                    "scores_10": int(base > 1.2),
                }
            )
    return rows


def test_benchmark_agreement_and_ci():
    ag = B.agreement(fake_rows(), reps=50)
    assert ag["coach_score~coach_dv"]["spearman"] == pytest.approx(1.0)
    assert ag["coach_score~coach_dv"]["ci95"] == pytest.approx([1.0, 1.0])
    assert ag["coach_score~xt"]["spearman"] == pytest.approx(-1.0)
    assert len(ag) == 6


def test_benchmark_calibration():
    cal = B.calibration(fake_rows(), reps=50)
    assert cal["coach_score"]["spearman_net_xg"] > 0.9 and cal["xt"]["spearman_net_xg"] < -0.9
    assert cal["coach_score"]["auc_goal_15s"] == pytest.approx(1.0)
    assert cal["xt"]["auc_goal_15s"] == pytest.approx(0.0)
    assert cal["coach_dv"]["slope_net_xg"] == pytest.approx(0.5, abs=0.05)
    assert len(cal["vaep"]["deciles"]) == 10
    assert B.auc(np.array([0.1, 0.2]), np.array([0, 0])) != B.auc(np.array([0.1, 0.2]), np.array([0, 0]))  # NaN


def test_benchmark_disagreements():
    rows = fake_rows()
    top = B.disagreements(rows)
    assert len(top) == B.N_DISAGREE
    gaps = [r["max_gap"] for r in top]
    assert gaps == sorted(gaps, reverse=True) and gaps[0] > 0.9
    assert top[0]["pair"] == "coach_score~xt"  # xt = -coach: the extremes disagree most
    assert set(B.pct_rank(np.array([3.0, 1.0, 2.0]))) == {1 / 6, 3 / 6, 5 / 6}
```

- [ ] **Step 2:** run -> FAIL (`cannot import name 'benchmark'`).
- [ ] **Step 3: implement** `coach/benchmark.py` (metrics part; `pass_rows`, `cost`, `chart`, `main` come in Task 5):

```python
"""Benchmark of the coach against two published methods, expected threat (xT, Singh 2019) and VAEP (Decroos et al.,
KDD 2019), on the same held-out passes (coach insight, part C).

    python -m coach.benchmark            # from 20-football-analytics/ (~15 min on CPU)
    python -m coach.benchmark --reuse    # metrics and chart again from the cached pass rows (seconds)
                                         # -> docs/coach_benchmark_results.json + docs/coach_benchmark.png

For every pass of the 7 IDSSE matches that the match report scores (datasets.pass_samples), with every model fitted
WITHOUT the match (asserted): the coach (report.held_out_models, the site's recipe), xT (coach.xt on the other
matches' actions) and VAEP (coach.vaep.lomo). Values of the same pass:
- coach_score: the coach's score of the chosen pass, p * V(target) - (1 - p) * V_opp(loss) - V(now); ex ante (it does
  not know whether the pass was completed);
- coach_dv: V(target) - V(start) with the coach's value model; ex ante, ignores the risk;
- xt: xT(end) - xT(start) for a completed pass, -xT(start) for a failed one; ex post;
- vaep: the VAEP value of the pass action; ex post (it sees the result and where the ball went).
Metrics, on all passes and on completed passes: Spearman agreement between methods; calibration against what
happened (Spearman and slope against the passing team's net xG within 15 s, AUC for "the passing team scores within
15 s" and for VAEP's own label "scores within the next 10 actions"); the passes where the methods disagree most
(percentile ranks); cost. 95 % intervals resample whole matches (evaluate.bootstrap_ci).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import advice as A
from . import datasets as D
from . import evaluate as EV
from . import features as F
from . import models as ML
from . import report as R
from . import vaep as V
from . import xt as XT

DOCS = Path(__file__).resolve().parents[1] / "docs"
METHODS = ("coach_score", "coach_dv", "xt", "vaep")
DISAGREE = ("coach_score", "xt", "vaep")
N_DISAGREE = 10
KEEP = (
    "match",
    "event",
    "half",
    "minute",
    "team",
    "carrier_name",
    "target_name",
    "success",
    "x0",
    "y0",
    "x1",
    "y1",
    "coach_p",
    "coach_score",
    "coach_dv",
    "xt",
    "vaep",
    "net_xg",
    "goal_15s",
)


def col(rows, k):
    return np.array([r[k] for r in rows], float)


def auc(score, y) -> float:
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def pct_rank(x):
    """Percentile rank in (0, 1), ties averaged."""
    return (rankdata(x) - 0.5) / len(x)


def agreement(rows, methods=METHODS, reps=EV.BOOTSTRAPS) -> dict:
    """Spearman between every pair of methods on the same passes, with a match-bootstrap 95 % interval."""
    out = {}
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            fn = lambda rs, a=a, b=b: EV.spearman(col(rs, a), col(rs, b))
            out[f"{a}~{b}"] = {"spearman": fn(rows), "ci95": EV.bootstrap_ci(rows, fn, reps)}
    return out


def calibration(rows, methods=METHODS, reps=EV.BOOTSTRAPS) -> dict:
    """Each method against what happened: Spearman and slope vs the net xG within 15 s, AUC for a goal by the passing
    team within 15 s and within the next 10 actions, and deciles of the value (mean value vs mean net xG)."""
    out = {}
    for k in methods:
        fns = {
            "spearman_net_xg": lambda rs, k=k: EV.spearman(col(rs, k), col(rs, "net_xg")),
            "slope_net_xg": lambda rs, k=k: EV.slope(col(rs, k), col(rs, "net_xg")),
            "auc_goal_15s": lambda rs, k=k: auc(col(rs, k), col(rs, "goal_15s")),
            "auc_scores_10": lambda rs, k=k: auc(col(rs, k), col(rs, "scores_10")),
        }
        out[k] = {}
        for name, fn in fns.items():
            out[k][name] = fn(rows)
            out[k][f"{name}_ci95"] = EV.bootstrap_ci(rows, fn, reps)
        out[k]["deciles"] = ML.calibration(col(rows, "net_xg"), col(rows, k))
    return out


def disagreements(rows, methods=DISAGREE, k=N_DISAGREE) -> list:
    """The k passes with the largest gap between the percentile ranks two methods give them (KEEP fields, the
    percentile of each method, the gap and the pair)."""
    pct = {m: pct_rank(col(rows, m)) for m in methods}
    gap, pair = np.zeros(len(rows)), [""] * len(rows)
    for i, a in enumerate(methods):
        for b in methods[i + 1 :]:
            d = np.abs(pct[a] - pct[b])
            for j in np.flatnonzero(d > gap):
                pair[j] = f"{a}~{b}"
            gap = np.maximum(gap, d)
    order = np.argsort(-gap, kind="stable")[:k]
    return [
        {
            **{kk: rows[j][kk] for kk in KEEP},
            **{f"pct_{m}": round(float(pct[m][j]), 3) for m in methods},
            "max_gap": round(float(gap[j]), 3),
            "pair": pair[j],
        }
        for j in order
    ]
```

- [ ] **Step 4:** run -> 12 passed; lint.
- [ ] **Step 5: commit** `git add coach/benchmark.py && git commit -m "20 coach: benchmark metrics (agreement, calibration, disagreements) with match bootstrap" -- coach/benchmark.py ../tests/test_20_benchmark.py`

### Task 5: pass rows, cost, chart, main; run on the 7 matches

**Files:** Modify `20-football-analytics/coach/benchmark.py` (append). Outputs: `docs/coach_benchmark_results.json`,
`docs/coach_benchmark.png`; cache `cv-data/football-coach/benchmark_rows.json` (outside the repo).

- [ ] **Step 1: append**

```python
def pass_rows(m, models, xt_model, vaep_pred, acts, horizon=D.HORIZON) -> list:
    """One row per pass of evaluate.pass_decisions: the four values, the outcome and where / who (KEEP)."""
    row_of = {int(ev): k for k, ev in enumerate(acts["event"]) if ev >= 0}
    xt_val = xt_model.action_values(acts)
    scores, _ = V.labels(acts)
    shots = R.shots_with_goals(m)
    out = []
    for d in EV.pass_decisions(m, models, horizon):
        p, ch = d["sample"], d["chosen"]
        e, s = p["event"], p["state"]
        k = row_of[int(p["index"])]
        assert acts["type"][k] in V.MOVES and acts["team"][k] == e["team"]
        v_now = float(A.value_at(models, [s["att"][p["carrier"]]])[0])
        t = d["t"] - float(m.halves[e["half"]].t[0])
        out.append(
            {
                "match": str(m.match_id),
                "event": int(p["index"]),
                "half": int(e["half"]),
                "minute": round(45 * e["half"] + t / 60, 2),
                "team": int(e["team"]),
                "carrier_name": m.names[e["player"]],
                "target_name": m.names[ch["target"]],
                "success": int(bool(e["success"])),
                **{c: round(float(acts[c][k]), 1) for c in ("x0", "y0", "x1", "y1")},
                "coach_p": round(ch["p"], 3),
                "coach_score": ch["score"],
                "coach_dv": ch["value"] - v_now,
                "xt": float(xt_val[k]),
                "vaep": float(vaep_pred["value"][k]),
                "vaep_offensive": float(vaep_pred["offensive"][k]),
                "net_xg": d["outcome"],
                "goal_15s": int(R.outcome(shots[e["half"]], d["t"], e["team"], bool(e["success"]), horizon) == "goal"),
                "scores_10": int(scores[k]),
            }
        )
    return out


def tree_nodes(model) -> int:
    """Nodes of a fitted HistGradientBoosting model (sklearn's internal predictors)."""
    return int(sum(len(p.nodes) for it in getattr(model, "_predictors", []) for p in it))


def cost(fitted, xt_model, vaep_models, timing) -> dict:
    """Size of what a browser would need, and measured CPU time per action (this laptop, one core, Python)."""
    coach_w = {
        "pass": ML.export_logistic(fitted["pass"], F.PASS_FEATURES),
        "value": ML.export_ridge(fitted["value"], F.VALUE_FEATURES),
        "shot": ML.export_logistic(fitted["shot"], F.SHOT_FEATURES),
    }
    nodes = {k: tree_nodes(v) for k, v in vaep_models.items()}
    return {
        "coach": {
            "parameters": sum(3 * len(w["coef"]) + 1 for w in coach_w.values()),
            "json_bytes": len(json.dumps(coach_w)),
            "inputs": "one tracking frame (22 players + ball, velocities)",
            "ms_per_pass": timing["coach_ms"],
            "browser": "yes, shipped (the replay's coach)",
        },
        "xt": {
            "parameters": XT.NX * XT.NY,
            "json_bytes": len(xt_model.to_json()),
            "inputs": "start and end location (and the result)",
            "ms_per_action": timing["xt_ms"],
            "browser": "yes in principle: a 16 x 12 lookup table (not shipped)",
        },
        "vaep": {
            "parameters": "two gradient-boosted ensembles of 300 depth-3 trees",
            "tree_nodes": nodes,
            "pickle_bytes": len(pickle.dumps(vaep_models)),
            "inputs": "the current and two previous actions (type, result, start / end), times, score",
            "ms_per_action": timing["vaep_ms"],
            "browser": "not shipped: needs the event stream and a tree-ensemble runtime (or a JSON export of "
            f"{sum(nodes.values())} nodes)",
        },
    }


def chart(res, path) -> None:
    """Two plain panels: pairwise Spearman (completed passes, 95 % CI), and net xG by decile of each method."""
    plt = EV._pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ag = res["completed"]["agreement"]
    names = list(ag)
    vals = [ag[k]["spearman"] for k in names]
    ci = [ag[k]["ci95"] if ag[k]["ci95"][0] is not None else [v, v] for k, v in zip(names, vals)]
    err = [[v - c[0] for v, c in zip(vals, ci)], [c[1] - v for v, c in zip(vals, ci)]]
    axes[0].barh(range(len(names)), vals, xerr=err, color="#4c78a8")
    axes[0].set_yticks(range(len(names)), [n.replace("~", " vs ") for n in names], fontsize=8)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].set(xlabel="Spearman, completed passes (95 % CI, match bootstrap)")
    for k in METHODS:
        dec = res["all"]["calibration"][k]["deciles"]
        axes[1].plot(range(1, len(dec) + 1), [c["observed"] for c in dec], "o-", label=k)
    axes[1].set(xlabel="decile of the method's pass value (all passes)", ylabel="mean net xG within 15 s")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarise(rows, reps=EV.BOOTSTRAPS) -> dict:
    comp = [r for r in rows if r["success"]]
    out = {"passes": len(rows), "completed": len(comp), "goals_15s": int(sum(r["goal_15s"] for r in rows))}
    for name, rs in (("all", rows), ("completed", comp)):
        out[name] = {"agreement": agreement(rs, reps=reps), "calibration": calibration(rs, reps=reps)}
    out["disagreements"] = disagreements(rows)
    return out


def main():
    from . import cross_league as CL
    from .match import load
    from .prepare import MATCHES, OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true", help="metrics from the cached pass rows")
    args = ap.parse_args()
    cache = OUT / "benchmark_rows.json"
    matches = {mid: load(OUT / mid) for mid in MATCHES}
    acts = {mid: V.actions(m) for mid, m in matches.items()}
    vaep = V.lomo(acts)
    extra_info = {}
    if args.reuse and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        rows, timing, fitted_last = saved["rows"], saved["timing"], None
    else:
        tables = {mid: R.match_tables(m) for mid, m in matches.items()}
        extra = CL.skillcorner_shots()[0]
        rows, coach_s, iters = [], 0.0, {}
        for mid, m in matches.items():
            models, fitted_last, trained_on = R.held_out_models(mid, tables, extra)
            others = [o for o in matches if o != mid]
            assert mid not in trained_on and mid not in vaep[mid]["trained_on"] and mid not in others
            xt_model = XT.ExpectedThreat().fit(D.stack([acts[o] for o in others]))
            iters[mid] = xt_model.iterations_
            t0 = time.perf_counter()
            new = pass_rows(m, models, xt_model, vaep[mid], acts[mid])
            coach_s += time.perf_counter() - t0
            rows += new
            print(mid, len(new), "passes", flush=True)
        timing = {"coach_ms": round(1000 * coach_s / len(rows), 2), "xt_iterations": iters}
        cache.write_text(json.dumps({"rows": rows, "timing": timing}), encoding="utf-8")
    # cost: models fitted on all 7 matches (used for size and timing only, never for the held-out numbers)
    all_acts = D.stack(list(acts.values()))
    xt_all = XT.ExpectedThreat().fit(all_acts)
    t0 = time.perf_counter()
    xt_all.action_values(all_acts)
    timing["xt_ms"] = round(1000 * (time.perf_counter() - t0) / len(all_acts["type"]), 5)
    vaep_all = V.fit(all_acts)
    t0 = time.perf_counter()
    V.predict(vaep_all, all_acts)
    timing["vaep_ms"] = round(1000 * (time.perf_counter() - t0) / len(all_acts["type"]), 4)
    if fitted_last is None:
        tables = {mid: R.match_tables(m) for mid, m in matches.items()}
        fitted_last = R.held_out_models(MATCHES[-1], tables, CL.skillcorner_shots()[0])[1]
    scores = np.concatenate([V.labels(acts[mid])[0] for mid in acts])
    concedes = np.concatenate([V.labels(acts[mid])[1] for mid in acts])
    extra_info["vaep_heldout"] = {
        "scores": ML.metrics(scores, np.concatenate([vaep[mid]["p_score"] for mid in acts])),
        "concedes": ML.metrics(concedes, np.concatenate([vaep[mid]["p_concede"] for mid in acts])),
    }
    types, counts = np.unique(all_acts["type"], return_counts=True)
    res = {
        "matches": list(matches),
        "horizon_s": D.HORIZON,
        "vaep_k": V.K,
        "xt_grid": [XT.NX, XT.NY],
        "actions": {str(t): int(c) for t, c in zip(types, counts)},
        **summarise(rows),
        **extra_info,
        "xt_iterations": timing.get("xt_iterations"),
        "cost": cost(fitted_last, xt_all, vaep_all, timing),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_benchmark_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    chart(res, DOCS / "coach_benchmark.png")
    for name in ("all", "completed"):
        print(name, {k: round(v["spearman"], 3) for k, v in res[name]["agreement"].items()})
        print(name, {k: round(v["spearman_net_xg"], 3) for k, v in res[name]["calibration"].items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 2:** tests still pass; lint.
- [ ] **Step 3: run** (> 10 min, CPU lock): `mkdir C:/Users/yando/Code/cv-data/.cpu.lock` (wait in one background loop if it
  exists), then in the background `$PY -m coach.benchmark > ../../cv-data/football-coach/benchmark.log 2>&1`;
  `rmdir` the lock when it ends. Expected: 7 lines `<match> <n> passes`, then two agreement and two calibration lines.
  Sanity: about 4,500 passes, every value finite, xT iterations < 1000, `docs/coach_benchmark_results.json` < 200 kB.
- [ ] **Step 4: commit** `git add coach/benchmark.py docs/coach_benchmark_results.json docs/coach_benchmark.png && git commit -m "20 coach: benchmark vs xT and VAEP on held-out passes (results + chart)" -- coach/benchmark.py docs/coach_benchmark_results.json docs/coach_benchmark.png`

### Task 6: report, README, push

**Files:** Create `20-football-analytics/docs/coach_benchmark.md`; Modify `20-football-analytics/README.md` (new
subsection "Benchmark vs xT and VAEP" after "Explanations and ghosts", a line in "Run the coach", the results file and
test count in the list under it).

- [ ] **Step 1:** write `docs/coach_benchmark.md` from `docs/coach_benchmark_results.json`: method (the three
  methods, their inputs, ex ante vs ex post), data (7 IDSSE matches, action counts, carries from tracking, no duels),
  results tables (agreement all / completed with CIs; calibration with CIs; cost), the 10 disagreements as a table
  plus 3-4 sentences on the patterns, limits (7 matches, xT g from xG, failed-pass convention, VAEP without
  defensive actions, coach weak value model, IDSSE only), references (Singh 2019 blog; Decroos et al. KDD 2019;
  Fernández et al. 2019 EPV as background; socceraction not used, no code copied). Embed `coach_benchmark.png`.
- [ ] **Step 2:** README subsection: 1 paragraph + headline table (Spearman vs net xG with CIs per method, agreement
  coach~xT and coach~VAEP) + link to the report.
- [ ] **Step 3:** full test file + lint of every touched path; `git commit -- <paths>`; push with
  `GIT_TERMINAL_PROMPT=0 git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q origin main`
  (on rejection `git pull --rebase --autostash -q origin main` then push); `gh run list --branch main --limit 1`
  must show success.

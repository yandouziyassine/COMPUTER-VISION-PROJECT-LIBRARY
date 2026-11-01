# Football AI Coach · Phase 1 (engine + models + evaluation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-Python coach engine for project 20 that, for any tracking frame, ranks the ball carrier's options (passes, carries, shot) with calibrated probabilities, trained and evaluated leave-one-match-out on the 7 IDSSE Bundesliga matches, and exported for the website.

**Architecture:** New package `20-football-analytics/coach/`. A streaming parser turns the DFL XML (on the USB key) into one compact `.npz` + `.json` per match in `cv-data/football-coach/`. Everything after that works on an in-memory `Match` (per-half NumPy arrays, metres, centred pitch) and is pure functions of a frame: pitch control (closed-form, Spearman-style), a pass-success model, a ball-position value model and a small shot model, combined by `advice.py`. `evaluate.py` produces every README number; `export.py` writes JSON weights and one match for the site (phase 2).

**Tech Stack:** Python 3.13 (`C:\Users\yando\Code\.venv-tools`), NumPy, scikit-learn, matplotlib, `xml.etree.ElementTree.iterparse`, pytest, ruff 0.17.0.

**Spec:** `docs/superpowers/specs/2026-11-01-football-ai-coach-design.md`

---

## Facts about the data (checked on the real files, 2026-11-01)

- Raw files: `D:\football-coach\idsse-raw\` (USB key, FAT32; it can disconnect: the converter is restartable per match).
  Per match id `J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9`:
  `DFL_02_01_matchinformation_DFL-COM-00000?_DFL-MAT-<id>.xml`, `DFL_03_02_events_raw_..._<id>.xml`,
  `DFL_04_03_positions_raw_observed_..._<id>.xml` (350–420 MB).
- Match information: `<Team TeamId=".." Role="home|guest" TeamName=".." PlayerShirtMainColor="#..">`, nested
  `<Player PersonId=".." ShirtNumber=".." Shortname=".." PlayingPosition=".." Starting="true|false">`;
  `<Environment PitchX="105.00" PitchY="68.00">`; `<General MatchId=".." HomeTeamName=".." GuestTeamName="..">`.
- Positions: `<FrameSet GameSection="firstHalf|secondHalf" TeamId="DFL-CLU-..|BALL" PersonId="..">` then one
  `<Frame N="10000" T="2022-08-26T16:32:09.720+00:00" X="-0.59" Y="8.72" S=".." .../>` per frame, 25 fps, metres,
  origin at the centre spot (X along the length, ±52.5; Y ±34). Ball frames add `Z`, `BallPossession="1|2"`
  (1 = home, 2 = guest) and `BallStatus="1|0"` (in play). One FrameSet per person per half; substitutes have shorter sets.
- Events: `<Event EventTime="2022-08-26T18:32:13.812+02:00" X-Position=".." Y-Position="..">` (0..105 × 0..68).
  Passes: `<Play Player=".." Recipient=".." Team=".." Evaluation="successfullyCompleted|unsuccessful"><Pass/>` or
  `<Cross/>` (Recipient is the *intended* receiver even when unsuccessful). Shots: `<ShotAtGoal Team=".." ...>` with a
  child `SuccessfulShot` (goal) / `SavedShot` / `ShotWide` / `BlockedShot` / `OtherShot`, and often
  `CalculatedFrame=".."` on the Event. About 760 Plays and 25–35 shots per match.
- Times: positions are UTC (`+00:00`), events `+02:00`: always parse with `datetime.fromisoformat` (offset-aware).
- Event ↔ tracking axes are **not assumed**: Task 4 measures which of `(X−52.5, Y−34)` or `(X−52.5, 34−Y)` matches the
  ball, and the converter stores the chosen convention.

## File structure

```
20-football-analytics/coach/
  __init__.py          package docstring only
  match.py             Match / Half dataclasses, save/load (.npz + .json), frame lookup by time, velocities
  dfl.py               streaming DFL XML parser → Match (+ events list)
  prepare.py           CLI: raw dir → cv-data/football-coach/<id>.npz/.json, restartable; sync audit
  sync.py              event time → frame, axis convention, residual distances
  geometry.py          attacking direction, normalisation (team attacks +x), goal distance/angle
  pitch_control.py     time-to-intercept + closed-form control probability on points / grid
  features.py          pass features, value features, shot features (pure functions of a frame)
  datasets.py          build pass / value / shot tables from a Match (labels from events)
  models.py            fit + leave-one-match-out evaluation (logistic, gradient boosting, baselines), metrics
  advice.py            rank options for a frame using fitted models
  evaluate.py          CLI: all numbers + charts → docs/coach_results.json, docs/coach_*.png
  export.py            CLI: JSON weights + one match (int16 cm, 10 fps) for the website
tests/test_20_coach.py synthetic tests for every pure function
```

Conventions (copy the existing project 20 style): module docstring with a usage line, `HERE`/`CV_DATA` paths like
`stats/xg.py`, 120-char lines, double quotes, `ruff check` and `ruff format --check` clean. Tests import with
`sys.path.insert(0, str(project_dir("20")))` then `from coach import ...` (same as `tests/test_20_stats.py`).

Commands assume the repo root `C:\Users\yando\Code\COMPUTER-VISION-PROJECT-LIBRARY` and the venv python
`PY=/c/Users/yando/Code/.venv-tools/Scripts/python.exe` (Git Bash).

---

### Task 1: `Match` container and storage

**Files:**
- Create: `20-football-analytics/coach/__init__.py`, `20-football-analytics/coach/match.py`
- Test: `tests/test_20_coach.py`

- [ ] **Step 1: Write the failing test**

```python
"""Project 20, AI coach: synthetic tests (storage, parsing, sync, pitch control, features, models, advice)."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import match as M  # noqa: E402


def tiny_match(frames=50):
    """Two halves, 2 players per team, ball moving along x; home attacks +x in half 0."""
    halves = []
    for h in range(2):
        t = 1000.0 + h * 4000 + np.arange(frames) / 25.0
        xy = np.zeros((frames, 4, 2), np.float32)
        xy[:, 0] = [-10, 0]; xy[:, 1] = [5, 5]; xy[:, 2] = [10, 0]; xy[:, 3] = [-5, -5]
        ball = np.zeros((frames, 3), np.float32)
        ball[:, 0] = np.linspace(-10, 10, frames)
        halves.append(M.Half(n=np.arange(frames, dtype=np.int32) + 10000, t=t, xy=xy, ball=ball,
                             owner=np.zeros(frames, np.int8), alive=np.ones(frames, bool)))
    return M.Match(match_id="TEST", player_ids=["a", "b", "c", "d"], team=np.array([0, 0, 1, 1], np.int8),
                   shirt=np.array([1, 2, 1, 2], np.int16), names=["A", "B", "C", "D"], halves=halves,
                   events=[{"type": "pass", "half": 0, "t": 1000.4, "team": 0, "player": 0, "recipient": 1,
                            "success": True, "x": 42.5, "y": 34.0}])


def test_save_load_roundtrip(tmp_path):
    m = tiny_match()
    M.save(m, tmp_path / "TEST")
    back = M.load(tmp_path / "TEST")
    assert back.player_ids == m.player_ids and back.events == m.events
    assert np.allclose(back.halves[1].xy, m.halves[1].xy) and np.array_equal(back.team, m.team)


def test_frame_at_time_and_velocity():
    h = tiny_match().halves[0]
    assert M.frame_at(h, 1000.0) == 0 and M.frame_at(h, 1000.41) == 10 and M.frame_at(h, 9e9) == len(h.t) - 1
    v = M.velocity(h.ball[:, :2], 25.0)
    assert np.allclose(v[5:-5, 0], 20 / (49 / 25.0), atol=1e-4)  # 20 m in 49 frames
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'coach'`

- [ ] **Step 3: Write minimal implementation**

`20-football-analytics/coach/__init__.py`:

```python
"""AI coach for project 20: rank the ball carrier's options (passes, carries, shot) on any tracking frame.

Data: IDSSE Bundesliga tracking + events (Bassek et al. 2025, CC BY 4.0). Pipeline: dfl.py → match.py storage →
pitch_control / features / models → advice.py. See docs/superpowers/specs/2026-11-01-football-ai-coach-design.md.
"""
```

`20-football-analytics/coach/match.py`:

```python
"""In-memory match: per-half NumPy arrays in metres (centre spot = origin), plus the events list.

Storage: <stem>.npz (arrays) + <stem>.json (ids, names, events). Positions are NaN when a player is not on the pitch.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 25.0


@dataclass
class Half:
    n: np.ndarray  # [F] int32 frame numbers from the provider
    t: np.ndarray  # [F] float64 epoch seconds (UTC)
    xy: np.ndarray  # [F, P, 2] float32 player positions, NaN = not on the pitch
    ball: np.ndarray  # [F, 3] float32 x, y, z
    owner: np.ndarray  # [F] int8 team in possession (0 home, 1 guest, -1 unknown)
    alive: np.ndarray  # [F] bool ball in play


@dataclass
class Match:
    match_id: str
    player_ids: list
    team: np.ndarray  # [P] int8, 0 home / 1 guest
    shirt: np.ndarray  # [P] int16
    names: list
    halves: list
    events: list = field(default_factory=list)  # dicts, see dfl.parse_events
    meta: dict = field(default_factory=dict)  # team names, colours, axis convention


def save(m: Match, stem: Path) -> None:
    stem = Path(stem)
    arrays = {"team": m.team, "shirt": m.shirt}
    for i, h in enumerate(m.halves):
        for k in ("n", "t", "xy", "ball", "owner", "alive"):
            arrays[f"h{i}_{k}"] = getattr(h, k)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    info = {"match_id": m.match_id, "player_ids": m.player_ids, "names": m.names, "halves": len(m.halves),
            "events": m.events, "meta": m.meta}
    stem.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def load(stem: Path) -> Match:
    stem = Path(stem)
    info = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(stem.with_suffix(".npz"))
    halves = [Half(**{k: z[f"h{i}_{k}"] for k in ("n", "t", "xy", "ball", "owner", "alive")})
              for i in range(info["halves"])]
    return Match(info["match_id"], info["player_ids"], z["team"], z["shirt"], info["names"], halves,
                 info["events"], info.get("meta", {}))


def frame_at(h: Half, t: float) -> int:
    """Index of the frame nearest to epoch time t (clamped to the half)."""
    i = int(np.searchsorted(h.t, t))
    if i <= 0:
        return 0
    if i >= len(h.t):
        return len(h.t) - 1
    return i if h.t[i] - t < t - h.t[i - 1] else i - 1


def velocity(xy: np.ndarray, fps: float = FPS, k: int = 2) -> np.ndarray:
    """Central difference over ±k frames (one-sided at the edges); works on [F, 2] or [F, P, 2]."""
    v = np.zeros_like(xy, dtype=np.float32)
    v[k:-k] = (xy[2 * k :] - xy[: -2 * k]) * fps / (2 * k)
    v[:k] = (xy[k : 2 * k] - xy[:k]) * fps / k
    v[-k:] = (xy[-k:] - xy[-2 * k : -k]) * fps / k
    return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/__init__.py 20-football-analytics/coach/match.py tests/test_20_coach.py
git commit -m "20 coach: Match container, npz/json storage, frame lookup, velocity"
```

---

### Task 2: DFL parser (match information, positions, events)

**Files:**
- Create: `20-football-analytics/coach/dfl.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing tests (synthetic XML in the real layout)**

```python
from coach import dfl  # noqa: E402

INFO = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><MatchInformation>
<General MatchId="DFL-MAT-T" HomeTeamName="Home FC" GuestTeamName="Away FC"/>
<Environment PitchX="105.00" PitchY="68.00"/><Teams>
<Team TeamId="H" Role="home" TeamName="Home FC" PlayerShirtMainColor="#FF0000"><Players>
<Player PersonId="h1" ShirtNumber="9" Shortname="H. Nine" PlayingPosition="STZ" Starting="true"/></Players></Team>
<Team TeamId="G" Role="guest" TeamName="Away FC" PlayerShirtMainColor="#FFFFFF"><Players>
<Player PersonId="g1" ShirtNumber="4" Shortname="G. Four" PlayingPosition="IVL" Starting="true"/></Players></Team>
</Teams></MatchInformation></PutDataRequest>"""

POS = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" TeamId="H" PersonId="h1">
<Frame N="10000" T="2022-08-26T16:32:09.720+00:00" X="-1.00" Y="2.00"/>
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="-1.10" Y="2.00"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="G" PersonId="g1">
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="5.00" Y="-3.00"/></FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="ball">
<Frame N="10000" T="2022-08-26T16:32:09.720+00:00" X="0.00" Y="0.00" Z="0.10" BallPossession="1" BallStatus="1"/>
<Frame N="10001" T="2022-08-26T16:32:09.760+00:00" X="0.50" Y="0.00" Z="0.10" BallPossession="2" BallStatus="0"/>
</FrameSet></Positions></PutDataRequest>"""

EVENTS = """<?xml version="1.0" encoding="UTF-8"?><PutDataRequest>
<Event EventTime="2022-08-26T18:32:09.760+02:00" X-Position="51.50" Y-Position="36.00">
 <Play Player="h1" Recipient="h1" Team="H" Evaluation="unsuccessful"><Pass/></Play></Event>
<Event EventTime="2022-08-26T18:32:09.760+02:00" CalculatedFrame="10001" X-Position="95.00" Y-Position="30.00">
 <ShotAtGoal Team="G" TypeOfShot="rightLeg"><SuccessfulShot/></ShotAtGoal></Event>
<Event EventTime="2022-08-26T18:32:09.700+02:00" X-Position="52.50" Y-Position="34.00">
 <KickOff GameSection="firstHalf" TeamRight="G" TeamLeft="H"><Play Player="h1" Team="H" Evaluation="successfullyCompleted"><Pass/></Play></KickOff></Event>
</PutDataRequest>"""


def test_parse_match_information(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    assert info["teams"] == {"H": 0, "G": 1} and info["players"]["g1"] == {"team": 1, "shirt": 4, "name": "G. Four",
                                                                          "position": "IVL"}
    assert info["meta"]["home"] == "Home FC" and info["meta"]["colors"] == ["#FF0000", "#FFFFFF"]


def test_parse_positions_aligns_frames_and_fills_nan(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    (tmp_path / "pos.xml").write_text(POS, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    ids, halves = dfl.parse_positions(tmp_path / "pos.xml", info)
    h = halves[0]
    assert ids == ["h1", "g1"] and list(h.n) == [10000, 10001]
    assert np.isnan(h.xy[0, 1]).all() and np.allclose(h.xy[1, 1], [5, -3])
    assert list(h.owner) == [0, 1] and list(h.alive) == [True, False]
    assert abs(h.t[1] - h.t[0] - 0.04) < 1e-6


def test_parse_events_passes_shots_and_kickoff_sides(tmp_path):
    (tmp_path / "info.xml").write_text(INFO, encoding="utf-8")
    (tmp_path / "ev.xml").write_text(EVENTS, encoding="utf-8")
    info = dfl.parse_info(tmp_path / "info.xml")
    ev = dfl.parse_events(tmp_path / "ev.xml", info)
    passes = [e for e in ev if e["type"] == "pass"]
    shots = [e for e in ev if e["type"] == "shot"]
    assert len(passes) == 2 and passes[0]["set_piece"] == "KickOff" and passes[0]["recipient"] is None  # time order
    assert passes[1]["success"] is False and passes[1]["recipient"] == "h1"
    assert shots[0]["goal"] is True and shots[0]["team"] == 1 and shots[0]["frame"] == 10001
    assert passes[1]["t"] == pytest.approx(dfl.epoch("2022-08-26T16:32:09.760+00:00"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_20_coach.py -q -k parse`
Expected: FAIL with `ImportError: cannot import name 'dfl'`

- [ ] **Step 3: Write the implementation**

`20-football-analytics/coach/dfl.py`:

```python
"""Streaming parser for the DFL / IDSSE XML files (match information, raw positions, raw events).

The positions file is ~400 MB: it is read with iterparse and each element is cleared after use, so memory stays at
the size of the output arrays (~25 MB per match). Players are stored in the order they first appear.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from .match import Half

SECTIONS = {"firstHalf": 0, "secondHalf": 1}
SHOT_RESULTS = ("SuccessfulShot", "SavedShot", "ShotWide", "BlockedShot", "OtherShot")
SET_PIECES = ("KickOff", "FreeKick", "CornerKick", "ThrowIn", "GoalKick", "Penalty")


def epoch(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def parse_info(path) -> dict:
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    teams, players, colors = {}, {}, []
    for team in root.iter("Team"):
        idx = 0 if team.get("Role") == "home" else 1
        teams[team.get("TeamId")] = idx
        colors.append((idx, team.get("PlayerShirtMainColor")))
        for p in team.iter("Player"):
            players[p.get("PersonId")] = {"team": idx, "shirt": int(p.get("ShirtNumber")), "name": p.get("Shortname"),
                                          "position": p.get("PlayingPosition")}
    meta = {"match_id": general.get("MatchId"), "home": general.get("HomeTeamName"),
            "guest": general.get("GuestTeamName"), "colors": [c for _, c in sorted(colors)]}
    return {"teams": teams, "players": players, "meta": meta}


def parse_positions(path, info):
    """Return (player_ids, [Half, Half]). Frames of a half = union of all frame numbers seen in that half."""
    raw = {0: {}, 1: {}}  # half -> person -> (N list, x list, y list)
    ball = {0: {}, 1: {}}  # half -> N -> (t, x, y, z, owner, alive)
    order = []
    current = None
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            half = SECTIONS[el.get("GameSection")]
            pid = el.get("PersonId")
            is_ball = el.get("TeamId") == "BALL"
            if not is_ball and pid not in order:
                order.append(pid)
            current = (half, pid, is_ball, [], [], [])
        elif event == "end" and el.tag == "Frame":
            half, pid, is_ball, ns, xs, ys = current
            n = int(el.get("N"))
            if is_ball:
                ball[half][n] = (epoch(el.get("T")), float(el.get("X")), float(el.get("Y")), float(el.get("Z", 0)),
                                 int(el.get("BallPossession", 0)) - 1, el.get("BallStatus") == "1")
            else:
                ns.append(n); xs.append(float(el.get("X"))); ys.append(float(el.get("Y")))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            half, pid, is_ball, ns, xs, ys = current
            if not is_ball:
                raw[half][pid] = (np.array(ns), np.array(xs, np.float32), np.array(ys, np.float32))
            el.clear()
    halves = []
    for half in (0, 1):
        frames = sorted(set(ball[half]) | {n for ns, _, _ in raw[half].values() for n in ns})
        idx = {n: i for i, n in enumerate(frames)}
        F, P = len(frames), len(order)
        xy = np.full((F, P, 2), np.nan, np.float32)
        for p, pid in enumerate(order):
            if pid in raw[half]:
                ns, xs, ys = raw[half][pid]
                rows = np.array([idx[n] for n in ns])
                xy[rows, p, 0], xy[rows, p, 1] = xs, ys
        b = np.full((F, 3), np.nan, np.float32)
        t = np.full(F, np.nan)
        owner = np.full(F, -1, np.int8)
        alive = np.zeros(F, bool)
        for n, (tt, x, y, z, o, a) in ball[half].items():
            i = idx[n]
            t[i], b[i], owner[i], alive[i] = tt, (x, y, z), o, a
        n_arr = np.array(frames, np.int32)
        known = ~np.isnan(t)  # frames without a ball sample: time from the 25 fps clock
        t = np.interp(n_arr, n_arr[known], t[known]) if known.any() else t
        halves.append(Half(n=n_arr, t=t, xy=xy, ball=b, owner=owner, alive=alive))
    return order, halves


def parse_events(path, info) -> list:
    """Passes (Play with Pass/Cross child) and shots, in time order; ids kept as PersonId strings."""
    teams = info["teams"]
    out = []
    for ev in ET.parse(path).getroot().iter("Event"):
        t = epoch(ev.get("EventTime"))
        x, y = float(ev.get("X-Position", "nan")), float(ev.get("Y-Position", "nan"))
        for child in ev:
            set_piece = child.tag if child.tag in SET_PIECES else None
            play = child if child.tag == "Play" else child.find("Play")
            if play is not None and (play.find("Pass") is not None or play.find("Cross") is not None):
                out.append({"type": "pass", "t": t, "x": x, "y": y, "team": teams.get(play.get("Team"), -1),
                            "player": play.get("Player"), "recipient": play.get("Recipient"),
                            "success": play.get("Evaluation") == "successfullyCompleted",
                            "cross": play.find("Cross") is not None, "set_piece": set_piece})
            if child.tag == "ShotAtGoal":
                result = next((c.tag for c in child if c.tag in SHOT_RESULTS), None)
                frame = ev.get("CalculatedFrame")
                out.append({"type": "shot", "t": t, "x": x, "y": y, "team": teams.get(child.get("Team"), -1),
                            "result": result, "goal": result == "SuccessfulShot",
                            "frame": int(frame) if frame else None})
    out.sort(key=lambda e: e["t"])
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/dfl.py tests/test_20_coach.py
git commit -m "20 coach: streaming DFL XML parser (match info, positions, passes and shots)"
```

---

### Task 3: Geometry helpers (attacking direction, normalisation, goal geometry)

**Files:**
- Create: `20-football-analytics/coach/geometry.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from coach import geometry as G  # noqa: E402


def test_attack_direction_from_team_positions():
    m = tiny_match()  # home players at x = -10 and 5, guests at 10 and -5 → home mean -2.5 < guest mean 2.5
    assert G.attack_dirs(m.halves[0], m.team) == (1, -1)


def test_normalise_flips_when_team_attacks_left():
    pts = np.array([[10.0, 5.0]])
    assert np.allclose(G.normalise(pts, -1), [[-10.0, -5.0]]) and np.allclose(G.normalise(pts, 1), pts)


def test_goal_distance_and_angle():
    d, a = G.goal_geometry(np.array([[52.5 - 11.0, 0.0], [52.5, 10.0]]))
    assert d[0] == pytest.approx(11.0) and a[0] == pytest.approx(2 * np.arctan(3.66 / 11.0))
    assert a[1] == pytest.approx(0.0, abs=1e-9)  # on the goal line, outside the posts: no angle
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_20_coach.py -q -k "attack or normalise or goal_distance"`
Expected: FAIL with `ImportError: cannot import name 'geometry'`

- [ ] **Step 3: Write the implementation**

```python
"""Pitch geometry. Normalised frame: the team in question attacks towards +x (goal centre at (52.5, 0))."""

import numpy as np

HALF_LENGTH, HALF_WIDTH, GOAL_HALF_WIDTH = 52.5, 34.0, 3.66


def attack_dirs(half, team) -> tuple:
    """(+1/-1 for home, +1/-1 for guest): a team attacks away from the side where it stands at the half's start."""
    first = half.xy[: min(len(half.t), 250)]  # first 10 s
    home = np.nanmean(first[:, team == 0, 0])
    guest = np.nanmean(first[:, team == 1, 0])
    return (1, -1) if home < guest else (-1, 1)


def normalise(points, direction: int):
    """Rotate by 180° when the team attacks towards -x (keeps left/right consistent)."""
    return np.asarray(points, dtype=float) * (1 if direction == 1 else -1)


def goal_geometry(points):
    """Distance to the goal centre and opening angle between the posts, for normalised points [N, 2]."""
    p = np.atleast_2d(points)
    dx = HALF_LENGTH - p[:, 0]
    dist = np.hypot(dx, p[:, 1])
    a1 = np.arctan2(GOAL_HALF_WIDTH - p[:, 1], dx)
    a2 = np.arctan2(-GOAL_HALF_WIDTH - p[:, 1], dx)
    angle = np.abs(a1 - a2)
    angle = np.where(angle > np.pi, 2 * np.pi - angle, angle)
    return dist, np.where(dx <= 1e-9, np.where(np.abs(p[:, 1]) <= GOAL_HALF_WIDTH, np.pi, 0.0), angle)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/geometry.py tests/test_20_coach.py
git commit -m "20 coach: attacking direction, normalisation, goal distance and angle"
```

---

### Task 4: Event ↔ tracking sync audit

**Files:**
- Create: `20-football-analytics/coach/sync.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from coach import sync  # noqa: E402


def test_sync_picks_axis_convention_and_attaches_frames():
    m = tiny_match()
    # ball at frame 10 is at x = -10 + 20 * 10 / 49; the event says (X, Y) in 0..105 x 0..68 with Y flipped
    bx = float(m.halves[0].ball[10, 0])
    m.events = [{"type": "pass", "t": float(m.halves[0].t[10]), "x": bx + 52.5, "y": 34.0 - 3.0, "team": 0,
                 "player": "a", "recipient": "b", "success": True}]
    m.halves[0].ball[:, 1] = 3.0
    report = sync.attach_frames(m)
    assert report["axis"] == "flip_y" and report["median_dist_m"] < 0.5
    assert m.events[0]["half"] == 0 and m.events[0]["frame"] == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_20_coach.py -q -k sync`
Expected: FAIL with `ImportError: cannot import name 'sync'`

- [ ] **Step 3: Write the implementation**

```python
"""Attach each event to a half and a frame index, and measure how well event locations match the ball.

Event coordinates are 0..105 x 0..68; tracking is centred. Two conventions are tried (Y as is, or flipped) and the
one with the smaller median event-to-ball distance is kept; the residuals are the sync-quality numbers in the README.
"""

import numpy as np

from .match import frame_at

AXES = {"same_y": lambda x, y: (x - 52.5, y - 34.0), "flip_y": lambda x, y: (x - 52.5, 34.0 - y)}


def _half_of(m, t):
    for i, h in enumerate(m.halves):
        if h.t[0] - 1 <= t <= h.t[-1] + 1:
            return i
    return None


def attach_frames(m) -> dict:
    located = []
    for e in m.events:
        e["half"] = _half_of(m, e["t"])
        if e["half"] is None:
            e["frame"] = None
            continue
        h = m.halves[e["half"]]
        e["frame"] = frame_at(h, e["t"])
        if np.isfinite(e["x"]) and np.all(np.isfinite(h.ball[e["frame"], :2])):
            located.append(e)
    best = None
    for name, f in AXES.items():
        d = [np.hypot(*(np.array(f(e["x"], e["y"])) - m.halves[e["half"]].ball[e["frame"], :2])) for e in located]
        med = float(np.median(d)) if d else float("nan")
        if best is None or med < best[1]:
            best = (name, med, d)
    name, med, d = best
    m.meta["event_axis"] = name
    return {"axis": name, "median_dist_m": med, "p90_dist_m": float(np.percentile(d, 90)) if d else float("nan"),
            "events": len(m.events), "located": len(located)}


def event_xy(m, e):
    """Event location in centred tracking coordinates, using the convention chosen by attach_frames."""
    return np.array(AXES[m.meta.get("event_axis", "same_y")](e["x"], e["y"]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/sync.py tests/test_20_coach.py
git commit -m "20 coach: event to frame sync with axis-convention check"
```

---

### Task 5: `prepare.py`: convert the 7 matches (restartable) and run the sync audit

**Files:**
- Create: `20-football-analytics/coach/prepare.py`

- [ ] **Step 1: Write the converter**

```python
"""Convert the IDSSE raw XML (USB key) into one compact match per id in cv-data/football-coach/.

    python -m coach.prepare                       # from 20-football-analytics/
    python -m coach.prepare --raw D:/football-coach/idsse-raw --only J03WOH

Restartable: a match whose .npz exists is skipped (the USB key can disconnect). Writes sync_audit.json next to the
matches. Raw data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import dfl, sync
from .match import Match, save

HERE = Path(__file__).resolve().parents[1]
CV_DATA = Path(os.environ.get("CV_DATA", HERE.parents[1] / "cv-data"))
OUT = CV_DATA / "football-coach"
RAW = Path(os.environ.get("IDSSE_RAW", "D:/football-coach/idsse-raw"))
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def find(raw: Path, kind: str, mid: str) -> Path:
    hits = sorted(raw.glob(f"DFL_{kind}_*_DFL-MAT-{mid}.xml"))
    if not hits:
        raise FileNotFoundError(f"{kind} file for {mid} not in {raw}")
    return hits[0]


def convert(raw: Path, mid: str) -> dict:
    info = dfl.parse_info(find(raw, "02_01_matchinformation", mid))
    ids, halves = dfl.parse_positions(find(raw, "04_03_positions_raw_observed", mid), info)
    events = dfl.parse_events(find(raw, "03_02_events_raw", mid), info)
    pidx = {p: i for i, p in enumerate(ids)}
    for e in events:  # PersonId -> player index (None when the person has no tracking, e.g. unknown recipient)
        for k in ("player", "recipient"):
            if k in e:
                e[k] = pidx.get(e[k])
    unknown = {"team": -1, "shirt": 0, "name": "?", "position": None}  # tracked but not in the match sheet
    players = {p: info["players"].get(p, unknown) for p in ids}
    m = Match(mid, ids, np.array([players[p]["team"] for p in ids], np.int8),
              np.array([players[p]["shirt"] for p in ids], np.int16), [players[p]["name"] for p in ids], halves,
              events, dict(info["meta"]))
    report = sync.attach_frames(m)
    save(m, OUT / mid)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--only", nargs="*", default=MATCHES)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "sync_audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    for mid in args.only:
        if (OUT / f"{mid}.npz").exists() and mid in audit:
            print("have", mid)
            continue
        audit[mid] = convert(args.raw, mid)
        audit_path.write_text(json.dumps(audit, indent=2))
        print(mid, audit[mid], flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on one match and check the audit**

Run (from `20-football-analytics/`): `$PY -m coach.prepare --only J03WOH`
Expected: prints `J03WOH {'axis': ..., 'median_dist_m': ..., ...}` in 1–3 minutes; `median_dist_m` below **3 m**
(event positions are hand-placed by operators; larger means the time mapping is off: stop and investigate before
going on). Check the file sizes: `ls -la $CV_DATA/football-coach` → `.npz` of roughly 10–40 MB.

- [ ] **Step 3: Sanity-check the converted match in a REPL**

```python
from coach.match import load
m = load("C:/Users/yando/Code/cv-data/football-coach/J03WOH")
print(len(m.player_ids), [len(h.t) for h in m.halves], sum(e["type"] == "pass" for e in m.events),
      sum(e["type"] == "shot" for e in m.events), m.meta)
```

Expected: 28–36 players, ~70 000 frames per half, 700–800 passes, 20–40 shots, `event_axis` set.

- [ ] **Step 4: Convert the other six (background, ~15 min)**

Run: `$PY -m coach.prepare` → all 7 in `sync_audit.json`, every `median_dist_m` < 3 m.

- [ ] **Step 5: Commit (code only; data stays in cv-data)**

```bash
git add 20-football-analytics/coach/prepare.py
git commit -m "20 coach: convert IDSSE matches to compact npz/json, sync audit"
```

---

### Task 6: Pitch control

**Files:**
- Create: `20-football-analytics/coach/pitch_control.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from coach import pitch_control as PC  # noqa: E402


def test_time_to_reach_includes_reaction_and_current_velocity():
    t = PC.time_to_reach(np.array([[0.0, 0.0]]), np.array([[5.0, 0.0]]), np.array([[10.0, 0.0]]))
    # after 0.7 s reaction the player is at x = 3.5, then 6.5 m at 5 m/s
    assert t[0, 0] == pytest.approx(0.7 + 6.5 / 5.0)


def test_control_is_half_when_equidistant_and_high_near_attacker():
    att = np.array([[0.0, 0.0]]); dfn = np.array([[20.0, 0.0]]); zero = np.zeros((1, 2))
    pts = np.array([[10.0, 0.0], [1.0, 0.0], [19.0, 0.0]])
    c = PC.control(pts, att, zero, dfn, zero)
    assert c[0] == pytest.approx(0.5, abs=1e-6) and c[1] > 0.95 and c[2] < 0.05


def test_grid_shape():
    g, xs, ys = PC.grid(np.array([[0.0, 0.0]]), np.zeros((1, 2)), np.array([[30.0, 0.0]]), np.zeros((1, 2)), step=5)
    assert g.shape == (len(ys), len(xs)) and 0 <= g.min() <= g.max() <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_20_coach.py -q -k "time_to_reach or control_is_half or grid_shape"`
Expected: FAIL with `ImportError: cannot import name 'pitch_control'`

- [ ] **Step 3: Write the implementation**

```python
"""Closed-form pitch control (after Spearman 2018, simplified so it ports to the browser exactly).

Each player reaches a point after a reaction time (moving with his current velocity meanwhile) and then running
straight at a top speed. A team's control of a point is a logistic function of the gap between the fastest
attacker's and the fastest defender's arrival times (sigma = arrival-time uncertainty).
"""

import numpy as np

REACTION, VMAX, SIGMA = 0.7, 5.0, 0.45
SCALE = np.pi / np.sqrt(3) / SIGMA


def time_to_reach(pos, vel, points):
    """[P, 2] positions and velocities, [N, 2] points -> [P, N] arrival times; NaN players are ignored (inf)."""
    start = pos + vel * REACTION
    d = np.linalg.norm(points[None, :, :] - start[:, None, :], axis=2)
    t = REACTION + d / VMAX
    return np.where(np.isnan(t), np.inf, t)


def control(points, att_pos, att_vel, def_pos, def_vel):
    """Probability that the attacking team controls each point [N]."""
    ta = time_to_reach(att_pos, att_vel, points).min(axis=0)
    td = time_to_reach(def_pos, def_vel, points).min(axis=0)
    return 1.0 / (1.0 + np.exp(-SCALE * np.clip(td - ta, -50, 50)))


def grid(att_pos, att_vel, def_pos, def_vel, step=2.0):
    """Control over the whole pitch: (grid [ny, nx], xs, ys)."""
    xs = np.arange(-52.5 + step / 2, 52.5, step)
    ys = np.arange(-34 + step / 2, 34, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.c_[gx.ravel(), gy.ravel()]
    return control(pts, att_pos, att_vel, def_pos, def_vel).reshape(gx.shape), xs, ys
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `12 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/pitch_control.py tests/test_20_coach.py
git commit -m "20 coach: closed-form pitch control (reaction + top speed, logistic arrival gap)"
```

---

### Task 7: Features (pass, value, shot) as pure functions of a frame

**Files:**
- Create: `20-football-analytics/coach/features.py`
- Test: `tests/test_20_coach.py` (append)

A *frame state* is a dict built by `features.state(match, half_index, frame_index, team)`:
`{"att": [Pa,2], "att_v": [Pa,2], "def": [Pd,2], "def_v": [Pd,2], "att_idx": [Pa], "ball": [2]}`, all normalised so
`team` attacks +x, NaN players dropped.

- [ ] **Step 1: Write the failing tests**

```python
from coach import features as F  # noqa: E402


def test_state_normalises_and_drops_absent_players():
    m = tiny_match()
    m.halves[0].xy[:, 1] = np.nan  # player b off the pitch
    s = F.state(m, 0, 10, team=1)  # guests attack -x in half 0 → flipped
    assert s["att"].shape == (2, 2) and s["def"].shape == (1, 2)
    assert np.allclose(s["att"][0], [-10.0, 0.0]) and list(s["att_idx"]) == [2, 3]


def test_pass_features_lane_and_pressure():
    s = {"att": np.array([[0.0, 0.0], [20.0, 0.0]]), "att_v": np.zeros((2, 2)),
         "def": np.array([[10.0, 1.0], [40.0, 30.0]]), "def_v": np.zeros((2, 2)), "att_idx": np.array([5, 6]),
         "ball": np.array([0.0, 0.0])}
    f = F.pass_features(s, passer=0, target=1)
    assert f["distance"] == pytest.approx(20.0) and f["forward"] == pytest.approx(20.0)
    assert f["lane_defenders"] == 1 and f["receiver_space"] == pytest.approx(np.hypot(10, 1))
    assert 0 < f["target_control"] < 1 and list(F.PASS_FEATURES) == list(f)


def test_value_and_shot_features():
    v = F.value_features(np.array([41.5, 0.0]))
    assert v["goal_dist"] == pytest.approx(11.0) and list(F.VALUE_FEATURES) == list(v)
    s = F.shot_features(np.array([41.5, 0.0]), defenders=np.array([[45.0, 0.0], [0.0, 0.0]]))
    assert s["cone_defenders"] == 1 and list(F.SHOT_FEATURES) == list(s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_20_coach.py -q -k "state_ or pass_features or value_and_shot"`
Expected: FAIL with `ImportError: cannot import name 'features'`

- [ ] **Step 3: Write the implementation**

```python
"""Features for the three models, computed from one normalised frame state (team in question attacks +x)."""

import numpy as np

from . import geometry as G
from . import pitch_control as PC
from .match import velocity

PASS_FEATURES = ("distance", "forward", "abs_angle", "lane_defenders", "receiver_space", "passer_pressure",
                 "target_control", "target_x", "target_goal_dist")
VALUE_FEATURES = ("x", "abs_y", "goal_dist", "goal_angle", "in_box")
SHOT_FEATURES = ("goal_dist", "goal_angle", "cone_defenders")
LANE_WIDTH = 2.0


def state(m, half: int, frame: int, team: int, k: int = 2) -> dict:
    h = m.halves[half]
    direction = G.attack_dirs(h, m.team)[team]
    lo, hi = max(0, frame - k), min(len(h.t), frame + k + 1)
    vel = np.nan_to_num(velocity(h.xy[lo:hi], k=1)[frame - lo])  # a player just on: no velocity yet
    pos = h.xy[frame]
    present = ~np.isnan(pos[:, 0])
    att = present & (m.team == team)
    dfn = present & (m.team != team)
    norm = lambda a: G.normalise(a, direction)
    return {"att": norm(pos[att]), "att_v": norm(vel[att]), "def": norm(pos[dfn]), "def_v": norm(vel[dfn]),
            "att_idx": np.flatnonzero(att), "ball": norm(h.ball[frame, :2]), "direction": direction}


def _segment_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t


def pass_features(s, passer: int, target: int) -> dict:
    a, b = s["att"][passer], s["att"][target]
    d, t = _segment_dist(s["def"], a, b)
    lane = int(np.sum((d < LANE_WIDTH) & (t > 0.05) & (t < 0.95)))
    tc = PC.control(b[None], s["att"], s["att_v"], s["def"], s["def_v"])[0]
    dist = float(np.linalg.norm(b - a))
    gd, _ = G.goal_geometry(b[None])
    return {"distance": dist, "forward": float(b[0] - a[0]),
            "abs_angle": float(abs(np.arctan2(b[1] - a[1], b[0] - a[0]))), "lane_defenders": lane,
            "receiver_space": float(np.min(np.linalg.norm(s["def"] - b, axis=1))),
            "passer_pressure": float(np.min(np.linalg.norm(s["def"] - a, axis=1))), "target_control": float(tc),
            "target_x": float(b[0]), "target_goal_dist": float(gd[0])}


def value_features(ball) -> dict:
    gd, ga = G.goal_geometry(np.asarray(ball)[None])
    in_box = abs(ball[1]) < 20.16 and ball[0] > G.HALF_LENGTH - 16.5
    return {"x": float(ball[0]), "abs_y": float(abs(ball[1])), "goal_dist": float(gd[0]), "goal_angle": float(ga[0]),
            "in_box": float(in_box)}


def shot_features(ball, defenders) -> dict:
    """Defenders inside the triangle ball-post-post count as blocking the shot."""
    gd, ga = G.goal_geometry(np.asarray(ball)[None])
    posts = np.array([[G.HALF_LENGTH, G.GOAL_HALF_WIDTH], [G.HALF_LENGTH, -G.GOAL_HALF_WIDTH]])

    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    def inside(p):
        tri = [ball, posts[0], posts[1]]
        s = [cross(tri[(i + 1) % 3] - tri[i], p - tri[i]) for i in range(3)]
        return all(v >= 0 for v in s) or all(v <= 0 for v in s)

    return {"goal_dist": float(gd[0]), "goal_angle": float(ga[0]),
            "cone_defenders": int(sum(inside(p) for p in np.atleast_2d(defenders)))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `15 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/features.py tests/test_20_coach.py
git commit -m "20 coach: frame state and pass / value / shot features"
```

---

### Task 8: Datasets (labelled tables from a match)

**Files:**
- Create: `20-football-analytics/coach/datasets.py`
- Test: `tests/test_20_coach.py` (append)

Labels:
- **Pass table**: one row per open-play pass (`set_piece is None`, not a cross) with known passer and recipient both
  on the pitch at the event frame; `y = success`.
- **Value table**: one row per second of live play with a known owner (`alive and owner >= 0`); `y = 1` if that team
  takes a shot within the next 10 s of the same half.
- **Shot table**: one row per shot with a frame; `y = goal`.

- [ ] **Step 1: Write the failing test**

```python
from coach import datasets as D  # noqa: E402


def test_tables_have_labels_and_match_ids():
    m = tiny_match(frames=600)  # 24 s per half
    for h in m.halves:
        h.owner[:] = 0
    m.events = [
        {"type": "pass", "half": 0, "frame": 10, "t": float(m.halves[0].t[10]), "team": 0, "player": 0,
         "recipient": 1, "success": True, "cross": False, "set_piece": None, "x": 0, "y": 0},
        {"type": "shot", "half": 0, "frame": 300, "t": float(m.halves[0].t[300]), "team": 0, "goal": False,
         "result": "SavedShot", "x": 95.0, "y": 34.0},
    ]
    p = D.pass_table(m)
    assert p["X"].shape == (1, len(D.F.PASS_FEATURES)) and list(p["y"]) == [1] and list(p["match"]) == ["TEST"]
    v = D.value_table(m)
    # live seconds 2..11 of half 0 (t = 2 s .. 11 s) are within 10 s before the shot at 12 s
    assert v["y"].sum() == 10 and v["X"].shape[1] == len(D.F.VALUE_FEATURES)
    s = D.shot_table(m)
    assert list(s["y"]) == [0] and s["X"].shape == (1, len(D.F.SHOT_FEATURES))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_20_coach.py -q -k tables`
Expected: FAIL with `ImportError: cannot import name 'datasets'`

- [ ] **Step 3: Write the implementation**

```python
"""Labelled tables for the three models. Every row carries its match id, for leave-one-match-out evaluation."""

import numpy as np

from . import features as F
from .sync import event_xy

HORIZON = 10.0  # seconds: "does this possession lead to a shot soon?"


def _table(rows, ys, mid, names):
    X = np.array([[r[k] for k in names] for r in rows], float).reshape(len(rows), len(names))
    return {"X": X, "y": np.array(ys, int), "match": np.array([mid] * len(rows))}


def pass_table(m) -> dict:
    rows, ys = [], []
    for e in m.events:
        if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
            continue
        if e["player"] is None or e["recipient"] is None or e["player"] == e["recipient"]:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where or e["recipient"] not in where:
            continue
        rows.append(F.pass_features(s, where[e["player"]], where[e["recipient"]]))
        ys.append(int(e["success"]))
    return _table(rows, ys, m.match_id, F.PASS_FEATURES)


def value_table(m, every: float = 1.0) -> dict:
    rows, ys = [], []
    for hi, h in enumerate(m.halves):
        shots = {0: [], 1: []}
        for e in m.events:
            if e["type"] == "shot" and e.get("half") == hi and e["team"] in shots:
                shots[e["team"]].append(e["t"])
        step = max(1, int(round(every * 25)))
        for i in range(0, len(h.t), step):
            team = int(h.owner[i])
            if not h.alive[i] or team < 0 or np.isnan(h.ball[i, 0]):
                continue
            s = F.state(m, hi, i, team)
            rows.append(F.value_features(s["ball"]))
            ys.append(int(any(0 < t - h.t[i] <= HORIZON for t in shots[team])))
    return _table(rows, ys, m.match_id, F.VALUE_FEATURES)


def shot_table(m) -> dict:
    rows, ys = [], []
    for e in m.events:
        if e["type"] != "shot" or e.get("frame") is None or e["team"] < 0:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        loc = event_xy(m, e) * s["direction"]  # event location, normalised like the state
        rows.append(F.shot_features(loc, s["def"]))
        ys.append(int(e["goal"]))
    return _table(rows, ys, m.match_id, F.SHOT_FEATURES)


def stack(tables):
    return {k: np.concatenate([t[k] for t in tables]) for k in ("X", "y", "match")}
```

Note for the implementer: `tiny_match` events in this test use `"frame"`/`"half"` directly, so `pass_table` and
`shot_table` must not call `sync.attach_frames` themselves (the converter already did it).

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `16 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/datasets.py tests/test_20_coach.py
git commit -m "20 coach: labelled pass, value and shot tables"
```

---

### Task 9: Models and leave-one-match-out evaluation

**Files:**
- Create: `20-football-analytics/coach/models.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
from coach import models as ML  # noqa: E402


def synthetic_table(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(1.5 * X[:, 0] - 1.0 * X[:, 1])))
    return {"X": X, "y": (rng.random(n) < p).astype(int), "match": np.repeat(list("ABCDEF"), n // 6)}


def test_logistic_export_matches_sklearn():
    t = synthetic_table()
    model = ML.fit_logistic(t["X"], t["y"])
    w = ML.export_logistic(model, ["a", "b", "c"])
    z = (t["X"] - np.array(w["mean"])) / np.array(w["scale"]) @ np.array(w["coef"]) + w["intercept"]
    assert np.allclose(1 / (1 + np.exp(-z)), model.predict_proba(t["X"])[:, 1], atol=1e-9)


def test_leave_one_match_out_beats_base_rate():
    r = ML.lomo(synthetic_table(), {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate})
    assert r["logistic"]["auc"] > 0.75 and r["logistic"]["log_loss"] < r["base_rate"]["log_loss"]
    assert len(r["logistic"]["per_match"]) == 6 and len(r["logistic"]["reliability"]) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_20_coach.py -q -k "logistic_export or leave_one"`
Expected: FAIL with `ImportError: cannot import name 'models'`

- [ ] **Step 3: Write the implementation**

```python
"""Model fitting and leave-one-match-out (LOMO) evaluation shared by the pass, value and shot models."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def fit_boosting(X, y):
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=0).fit(X, y)


def fit_base_rate(X, y):
    return DummyClassifier(strategy="prior").fit(X, y)


def column(i):
    """Baseline that uses one feature only (e.g. pass distance), still a logistic regression."""
    return lambda X, y: _Column(i, fit_logistic(X[:, [i]], y))


class _Column:
    def __init__(self, i, model):
        self.i, self.model = i, model

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, [self.i]])


def export_logistic(model, names) -> dict:
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return {"features": list(names), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(sel.sum()),
                    "mean_pred": float(p[sel].mean()) if sel.any() else None,
                    "observed": float(y[sel].mean()) if sel.any() else None})
    return out


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "positive_rate": float(y.mean()), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None}


def lomo(table, fitters: dict) -> dict:
    """Each model is trained on all matches but one and scored on the held-out match; metrics on the pooled
    out-of-fold predictions, plus per-match metrics for the spread."""
    X, y, groups = table["X"], table["y"], table["match"]
    result = {}
    for name, fit in fitters.items():
        oof = np.zeros(len(y))
        per = {}
        for g in np.unique(groups):
            test = groups == g
            model = fit(X[~test], y[~test])
            oof[test] = model.predict_proba(X[test])[:, 1]
            per[str(g)] = metrics(y[test], oof[test])
        result[name] = {**metrics(y, oof), "per_match": per, "reliability": reliability(y, oof), "oof": oof}
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `18 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/models.py tests/test_20_coach.py
git commit -m "20 coach: logistic / boosting / baselines, leave-one-match-out metrics, logistic export"
```

---

### Task 10: Advice (rank options for one frame)

**Files:**
- Create: `20-football-analytics/coach/advice.py`
- Test: `tests/test_20_coach.py` (append)

Score of an option = `p * V(target) - (1 - p) * V_opp(target)`, where `V` is the value model for the team in
possession at the target and `V_opp` the value for the opponent if the ball is lost there (the target mirrored).
Options: a pass to every teammate on the pitch; a carry 6 m ahead in 5 directions (-60°, -30°, 0°, 30°, 60°) with
`p` = pitch control at the carry point; a shot with `p` = shot model and value 1 (a goal) when inside 35 m.

- [ ] **Step 1: Write the failing test**

```python
from coach import advice as A  # noqa: E402


def test_advice_prefers_open_forward_teammate():
    s = {"att": np.array([[20.0, 0.0], [45.0, 0.0], [5.0, 0.0]]), "att_v": np.zeros((3, 2)),
         "def": np.array([[20.0, 25.0], [0.0, -25.0]]), "def_v": np.zeros((2, 2)), "att_idx": np.array([0, 1, 2]),
         "ball": np.array([20.0, 0.0]), "direction": 1}  # carrier 32.5 m from goal: the shot option exists
    sig = lambda w: (lambda X: 1 / (1 + np.exp(-(X @ w))))
    # pass model: success falls with distance; value model: grows with x
    models = {"pass": sig(np.r_[-0.05, 0, 0, -1.0, 0.05, 0, 2.0, 0, 0]),
              "value": sig(np.r_[0.05, 0, 0, 0, 0]),
              "shot": sig(np.r_[-0.2, 0, 0])}
    opts = A.options(s, carrier=0, models=models)
    passes = [o for o in opts if o["kind"] == "pass"]
    assert passes[0]["target"] == 1 and {o["kind"] for o in opts} == {"pass", "carry", "shot"}
    assert all(0 <= o["p"] <= 1 for o in opts) and opts == sorted(opts, key=lambda o: -o["score"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_20_coach.py -q -k advice`
Expected: FAIL with `ImportError: cannot import name 'advice'`

- [ ] **Step 3: Write the implementation**

```python
"""Rank the ball carrier's options on one frame state.

`models` maps "pass" / "value" / "shot" to a function X[n, k] -> probability[n] over the feature order of
features.PASS_FEATURES / VALUE_FEATURES / SHOT_FEATURES (see from_fitted for sklearn models).
"""

import numpy as np

from . import features as F
from . import pitch_control as PC

CARRY_DIST, CARRY_ANGLES, SHOT_RANGE = 6.0, (-60, -30, 0, 30, 60), 35.0


def from_fitted(fitted: dict) -> dict:
    return {k: (lambda X, m=m: m.predict_proba(np.atleast_2d(X))[:, 1]) for k, m in fitted.items()}


def _value(models, pts):
    return models["value"](np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in pts]))


def options(s, carrier: int, models: dict) -> list:
    out = []
    here = s["att"][carrier]
    for j in range(len(s["att"])):
        if j == carrier:
            continue
        f = F.pass_features(s, carrier, j)
        p = float(models["pass"](np.array([[f[k] for k in F.PASS_FEATURES]]))[0])
        out.append({"kind": "pass", "target": int(s["att_idx"][j]), "to": s["att"][j].tolist(), "p": p})
    for a in CARRY_ANGLES:
        pt = here + CARRY_DIST * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        pt = np.clip(pt, [-52.5, -34], [52.5, 34])
        p = float(PC.control(pt[None], s["att"], s["att_v"], s["def"], s["def_v"])[0])
        out.append({"kind": "carry", "angle": a, "to": pt.tolist(), "p": p})
    to = [np.array(o["to"]) for o in out]
    v_keep = _value(models, to)
    v_lose = _value(models, [-t for t in to])  # opponent attacks the other way
    for o, vk, vl in zip(out, v_keep, v_lose):
        o["value"], o["score"] = float(vk), float(o["p"] * vk - (1 - o["p"]) * vl)
    sf = F.shot_features(here, s["def"])
    if sf["goal_dist"] <= SHOT_RANGE:
        p = float(models["shot"](np.array([[sf[k] for k in F.SHOT_FEATURES]]))[0])
        out.append({"kind": "shot", "to": [52.5, 0.0], "p": p, "value": 1.0, "score": p})
    return sorted(out, key=lambda o: -o["score"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `19 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/advice.py tests/test_20_coach.py
git commit -m "20 coach: rank passes, carries and shot by expected value"
```

---

### Task 11: `evaluate.py`: every README number, charts only

**Files:**
- Create: `20-football-analytics/coach/evaluate.py`

- [ ] **Step 1: Write the script**

```python
"""Evaluate the AI coach on the 7 IDSSE matches (leave-one-match-out) and write docs/coach_results.json + charts.

    python -m coach.evaluate          # from 20-football-analytics/, after coach.prepare (~10-20 min on CPU)

Charts only (no broadcast imagery exists in this dataset; positions are drawn by us).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import advice as A  # noqa: E402
from . import datasets as D  # noqa: E402
from . import features as F  # noqa: E402
from . import models as ML  # noqa: E402
from .match import load  # noqa: E402
from .prepare import MATCHES, OUT  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


def strip(r):
    return {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in r.items()}


def advice_hit_rate(matches, fitted) -> dict:
    """For held-out matches: is the pro's actual receiver the coach's best pass option / in its top 3?"""
    hits1 = hits3 = n = 0
    for m in matches:
        models = A.from_fitted({k: v[m.match_id] for k, v in fitted.items()})
        for e in m.events:
            if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
                continue
            s = F.state(m, e["half"], e["frame"], e["team"])
            where = {p: i for i, p in enumerate(s["att_idx"])}
            if e["player"] not in where or e["recipient"] not in where:
                continue
            ranked = [o["target"] for o in A.options(s, where[e["player"]], models) if o["kind"] == "pass"]
            n += 1
            hits1 += ranked[:1] == [e["recipient"]]
            hits3 += e["recipient"] in ranked[:3]
    return {"passes": n, "top1": hits1 / max(n, 1), "top3": hits3 / max(n, 1)}


def reliability_chart(results, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, r in results.items():
        pts = [(b["mean_pred"], b["observed"]) for b in r["reliability"] if b["n"] >= 20]
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"{name} (Brier {r['brier']:.3f})")
    ax.set(xlabel="predicted", ylabel="observed", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    matches = [load(OUT / mid) for mid in MATCHES]
    tables = {"pass": D.stack([D.pass_table(m) for m in matches]),
              "value": D.stack([D.value_table(m) for m in matches]),
              "shot": D.stack([D.shot_table(m) for m in matches])}
    i = F.PASS_FEATURES.index
    results = {
        "pass": ML.lomo(tables["pass"], {"logistic": ML.fit_logistic, "boosting": ML.fit_boosting,
                                         "distance_only": ML.column(i("distance")),
                                         "target_control_only": ML.column(i("target_control")),
                                         "base_rate": ML.fit_base_rate}),
        "value": ML.lomo(tables["value"], {"logistic": ML.fit_logistic, "boosting": ML.fit_boosting,
                                           "goal_distance_only": ML.column(F.VALUE_FEATURES.index("goal_dist")),
                                           "base_rate": ML.fit_base_rate}),
        "shot": ML.lomo(tables["shot"], {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate}),
    }
    # held-out models per match for the advice sanity check (logistic everywhere: it is what the site runs)
    fitted = {k: {} for k in tables}
    for k, t in tables.items():
        for g in np.unique(t["match"]):
            fitted[k][str(g)] = ML.fit_logistic(t["X"][t["match"] != g], t["y"][t["match"] != g])
    out = {k: strip(v) for k, v in results.items()}
    out["advice"] = advice_hit_rate(matches, fitted)
    out["sync_audit"] = json.loads((OUT / "sync_audit.json").read_text())
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_results.json").write_text(json.dumps(out, indent=2))
    reliability_chart({k: results["pass"][k] for k in ("logistic", "boosting", "distance_only")},
                      DOCS / "coach_pass_reliability.png", "Pass success, held-out matches")
    reliability_chart({k: results["value"][k] for k in ("logistic", "boosting")},
                      DOCS / "coach_value_reliability.png", "Shot within 10 s, held-out matches")
    for k, r in out.items():
        if k in ("pass", "value", "shot"):
            print(k, {n: (round(v["log_loss"], 4), round(v["brier"], 4), v["auc"] and round(v["auc"], 3))
                      for n, v in r.items()})
    print("advice", out["advice"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run (from `20-football-analytics/`): `$PY -m coach.evaluate`
Expected: three lines of metrics and one advice line. Sanity bars (stop and investigate if missed):
pass logistic AUC > distance_only AUC and > 0.70; value logistic log-loss < base_rate log-loss; advice top-3 > 0.4.

- [ ] **Step 3: Look at both charts** (`docs/coach_pass_reliability.png`, `docs/coach_value_reliability.png`):
points near the diagonal; they must contain only axes, lines and legends.

- [ ] **Step 4: Commit**

```bash
git add 20-football-analytics/coach/evaluate.py 20-football-analytics/docs/coach_results.json 20-football-analytics/docs/coach_pass_reliability.png 20-football-analytics/docs/coach_value_reliability.png
git commit -m "20 coach: leave-one-match-out evaluation, advice hit rate, reliability charts"
```

---

### Task 12: `export.py`: weights and one match for the website

**Files:**
- Create: `20-football-analytics/coach/export.py`
- Test: `tests/test_20_coach.py` (append)

- [ ] **Step 1: Write the failing test**

```python
from coach import export as X  # noqa: E402


def test_pack_match_roundtrip_in_centimetres(tmp_path):
    m = tiny_match(frames=100)
    X.pack_match(m, tmp_path, fps=10)
    meta = __import__("json").loads((tmp_path / "match.json").read_text())
    raw = np.fromfile(tmp_path / "match.bin", dtype="<i2")
    frames = sum(meta["frames"])
    assert raw.size == frames * (len(m.player_ids) + 1) * 2 and meta["fps"] == 12.5
    first = raw[: (len(m.player_ids) + 1) * 2].reshape(-1, 2) / 100.0
    assert np.allclose(first[0], m.halves[0].xy[0, 0]) and np.allclose(first[-1], m.halves[0].ball[0, :2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/test_20_coach.py -q -k pack_match`
Expected: FAIL with `ImportError: cannot import name 'export'`

- [ ] **Step 3: Write the implementation**

```python
"""Export for the website (phase 2): logistic weights as JSON and one match as int16 centimetres at 10 fps.

    python -m coach.export --match J03WOH --site ../../yandouziyassine.github.io/public/football

Frame rate: the 25 fps source is subsampled by a whole step (fps=10 -> step 2 -> 12.5 fps, written in match.json).
Binary layout (little-endian int16): for each frame, P players then the ball, each (x, y) in cm; NaN -> -32768.
match.json holds players (id index, team, shirt, name), team names/colours, frames per half, fps, owner/alive per
frame (as strings of digits), passes and shots with frame indices, and the data credit.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import datasets as D
from . import models as ML
from .match import load
from .prepare import MATCHES, OUT

NAN = -32768
CREDIT = "Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0"


def pack_match(m, out: Path, fps: int = 10):
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(25 / fps)))  # 25 fps source: fps=10 -> every 2nd frame = 12.5 fps (stored as such)
    chunks, frames, owner, alive = [], [], [], []
    for h in m.halves:
        idx = np.arange(0, len(h.t), step)
        pos = np.concatenate([h.xy[idx], h.ball[idx, None, :2]], axis=1) * 100
        chunks.append(np.where(np.isnan(pos), NAN, np.round(pos)).astype("<i2"))
        frames.append(len(idx))
        owner.append("".join(str(int(o) + 1) for o in h.owner[idx]))
        alive.append("".join("1" if a else "0" for a in h.alive[idx]))
    np.concatenate(chunks).tofile(out / "match.bin")
    events = [{**{k: e[k] for k in e if k not in ("t",)}, "frame": (e["frame"] // step) if e.get("frame") is not None
               else None} for e in m.events]
    meta = {"match_id": m.match_id, "fps": 25 / step, "frames": frames, "players": [
        {"team": int(t), "shirt": int(s), "name": n} for t, s, n in zip(m.team, m.shirt, m.names)],
        "meta": m.meta, "owner": owner, "alive": alive, "events": events, "credit": CREDIT}
    (out / "match.json").write_text(json.dumps(meta, ensure_ascii=False))


def export_models(out: Path):
    matches = [load(OUT / mid) for mid in MATCHES]
    weights = {}
    for name, build, feats in (("pass", D.pass_table, D.F.PASS_FEATURES),
                               ("value", D.value_table, D.F.VALUE_FEATURES),
                               ("shot", D.shot_table, D.F.SHOT_FEATURES)):
        t = D.stack([build(m) for m in matches])
        weights[name] = ML.export_logistic(ML.fit_logistic(t["X"], t["y"]), feats)
    weights["pitch_control"] = {"reaction": 0.7, "vmax": 5.0, "sigma": 0.45}
    (out / "models.json").write_text(json.dumps(weights, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    pack_match(load(OUT / args.match), args.site)
    export_models(args.site)
    print("wrote", sorted(p.name for p in args.site.iterdir()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `$PY -m pytest tests/test_20_coach.py -q`
Expected: `20 passed`

- [ ] **Step 5: Commit**

```bash
git add 20-football-analytics/coach/export.py tests/test_20_coach.py
git commit -m "20 coach: export logistic weights and one match (int16 cm, 10 fps) for the website"
```

---

### Task 13: README, licences, progress, lint

**Files:**
- Modify: `20-football-analytics/README.md` (new section "v3 · AI coach" after the S4 section)
- Modify: `DATA_LICENSES.md` (IDSSE row), `PROGRESS.md` (project 20 row), `20-football-analytics/requirements.txt`
  (nothing new unless matplotlib is missing: it is already listed)

- [ ] **Step 1: Write the README section** with: what the coach does (one paragraph); data + credit line
  ("Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0", DOIs); method
  (pitch control formula, the three models and their features, the advice score); a results table copied from
  `docs/coach_results.json` (pass: logistic / boosting / distance-only / control-only / base rate; value: logistic /
  boosting / distance-only / base rate; shot: logistic / base rate — log-loss, Brier, AUC, per-match spread); the sync
  audit (median and p90 event-to-ball distance per match); the advice hit rate with the caveat that the coach is not
  meant to copy pros; both reliability charts; limits (7 matches, one league, logistic models chosen for the browser,
  SkillCorner cross-league test still to do); how to run (`python -m coach.prepare`, `coach.evaluate`, `coach.export`).

- [ ] **Step 2: Add the licence row** to `DATA_LICENSES.md`:
  `| 20 (coach) | IDSSE: 7 Bundesliga 2022/23 matches, tracking + events (Bassek et al. 2025, figshare 10.6084/m9.figshare.28196177) | CC BY 4.0 (DFL data, published with permission) | Raw XML on the USB key only, never committed; derived match + weights may be published with the credit line |`

- [ ] **Step 3: Update PROGRESS.md** project 20 row with the phase-1 numbers and the commit hashes.

- [ ] **Step 4: Lint and full test run**

Run: `$PY -m ruff check . && $PY -m ruff format --check . && $PY -m pytest tests -q`
Expected: ruff clean; all tests pass (tests whose packages are missing are skipped).

- [ ] **Step 5: Commit and push**

```bash
git add 20-football-analytics/README.md DATA_LICENSES.md PROGRESS.md
git commit -m "20 coach: README v3 section (results, sync audit, limits), licence and progress rows"
git -c credential.helper= -c "credential.helper=!gh auth git-credential" push -q origin main
```

---

## Out of this plan

- SkillCorner cross-league test (spec, evaluation section): separate small plan once phase 1 numbers exist (needs a
  loader for SkillCorner's format and a size check of their tracking files).
- Phases 2–4 (site replay, 3D, live camera): separate plans.

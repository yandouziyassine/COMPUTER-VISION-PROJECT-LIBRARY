# Player profiles, phase 1 (library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-player offsets (pass skill, finishing) on the logit scale of the coach's existing models, built from open
data, tested leave-one-match-out, and exported as `profiles.json` for the website match J03WOH built without it.

**Architecture:** `coach/profiles.py` holds the format, the empirical-Bayes shrinkage, the choice of `k` (and the
scale `a`) by nested held-out log-loss, the provider interface (open data now, API later) and the SkillCorner season
descriptors. `coach/profiles_eval.py` reads the cached derived tables (no tracking needed), recovers passer / shooter
ids, and runs the held-out comparison with a match bootstrap. `coach/export_profiles.py` writes `profiles.json`.
`coach/export.py` is NOT touched (another agent works on it).

**Tech Stack:** NumPy, scikit-learn (existing `coach.models`), matplotlib imported lazily (CI has none), pytest.

Spec: `docs/superpowers/specs/2026-11-02-player-profiles-design.md`.

**Conventions (all tasks)**

- `PY=C:/Users/yando/Code/.venv-tools/Scripts/python.exe`; tests run from the repo root:
  `$PY -m pytest tests/test_20_profiles.py -q` (new file, so the other agent's edits to `tests/test_20_coach.py` never
  collide), plus `$PY -m pytest tests/test_20_coach.py -q` at the end.
- Lint: `$PY -m ruff check <paths> && $PY -m ruff format --check <paths>`.
- Commit only own paths: `git add <paths> && git commit -m "..." -- <paths>`; never `git add -A`, never stash/reset.
- Player keys: `"dfl:<PersonId>"` (IDSSE), `"skillcorner:<player_id>"`; `""` = unknown player (ignored).

## Shrinkage, in one formula

A per-player offset `b` on the logit of a model's probability `p`, Gaussian prior `b ~ N(0, 1/k)`, one Newton step
from `b = 0`:

```
raw   = Σ w (y − p) / Σ w p (1 − p)          (unshrunk)
logit = Σ w (y − p) / (Σ w p (1 − p) + k)    (shrunk towards 0 = the model's expectation)
```

`k` and the scale `a` (coach: `logit(p) += a · offset`) are picked on a grid by held-out log-loss where each match's
rows get offsets built from the other matches only, and the expectations `p` are themselves out-of-fold.

## Files

- Create `20-football-analytics/coach/profiles.py`: format, shrinkage, `choose_k`, providers, validate / save / load,
  SkillCorner descriptors.
- Create `20-football-analytics/coach/profiles_eval.py`: data (IDSSE keys from the match JSON, SkillCorner passer ids
  from the event CSVs, cached in `cv-data/skillcorner/profiles/players.json`), `oof`, `lomo_profiles`,
  `bootstrap_diff`, `summarize`, chart, `main` → `docs/coach_profiles_results.json`, `docs/coach_profiles.png`.
- Create `20-football-analytics/coach/export_profiles.py`: `build(match_id)` + CLI → `<site>/profiles.json`.
- Create `tests/test_20_profiles.py`.
- Modify `20-football-analytics/README.md`: new subsection under v3 + run lines.

---

### Task 1: shrinkage and choice of k

**Files:** Create `20-football-analytics/coach/profiles.py`, `tests/test_20_profiles.py`

- [ ] **Step 1: failing tests**

```python
"""Project 20, AI coach: player profiles (shrinkage, k choice, format, providers, held-out evaluation). Synthetic."""

import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from conftest import project_dir  # noqa: E402

sys.path.insert(0, str(project_dir("20")))
from coach import profiles as PR  # noqa: E402


def test_shrink_small_samples_stay_near_zero_large_ones_near_raw():
    small = PR.shrink((5, 2.0, 1.0), k=30.0)  # 5 passes, 2 more completions than expected
    large = PR.shrink((2000, 80.0, 400.0), k=30.0)
    assert small["n"] == 5 and abs(small["logit"]) < 0.1 and small["raw"] == pytest.approx(2.0)
    assert large["logit"] == pytest.approx(80 / 430) and large["raw"] == pytest.approx(0.2)
    assert PR.shrink((3, 1.0, 0.5), k=0.0)["logit"] == pytest.approx(2.0)


def test_offsets_apply_and_missing_players_unchanged():
    keys = np.array(["a", "a", "b", ""])
    y = np.array([1, 1, 0, 1])
    p = np.full(4, 0.5)
    table = PR.offsets(keys, y, p, k=1.0)
    assert set(table) == {"a", "b"} and table["a"]["n"] == 2
    assert table["a"]["logit"] == pytest.approx(1.0 / (0.5 + 1.0)) and table["b"]["logit"] < 0
    off = PR.offset_vector(np.array(["a", "zz"]), table)
    q = PR.apply(np.array([0.5, 0.7]), off, a=1.0)
    assert q[0] > 0.5 and q[1] == pytest.approx(0.7)


def _synthetic(effects, matches=6, rows=300, seed=0):
    rng = np.random.default_rng(seed)
    players = np.array([f"p{i}" for i in range(len(effects))])
    keys = rng.choice(players, size=matches * rows)
    base = rng.uniform(0.4, 0.95, size=len(keys))
    eff = np.array([effects[int(k[1:])] for k in keys])
    y = (rng.uniform(size=len(keys)) < PR.sigmoid(PR.logit(base) + eff)).astype(int)
    groups = np.repeat(np.arange(matches), rows)
    return groups, keys, y, base


def test_choose_k_finds_real_player_effects():
    groups, keys, y, base = _synthetic([1.0, -1.0, 0.8, -0.8, 0.0, 0.5])
    sel = PR.choose_k(groups, keys, y, base)
    assert sel["a"] > 0 and sel["log_loss"] < sel["base_log_loss"] - 0.01 and sel["coverage"] == 1.0


def test_choose_k_without_effects_gains_nothing():
    groups, keys, y, base = _synthetic([0.0] * 6, seed=1)
    sel = PR.choose_k(groups, keys, y, base)
    assert sel["log_loss"] >= sel["base_log_loss"] - 0.002
    assert sel["a"] == 0.0 or sel["k"] >= 30
```

- [ ] **Step 2:** `$PY -m pytest tests/test_20_profiles.py -q` → FAIL (`cannot import name 'profiles'`).

- [ ] **Step 3: implementation** (`coach/profiles.py`, first part)

```python
"""Player profiles for the AI coach: one provider-independent format, filled from open data now and from a live
sports-data API later. Design: docs/superpowers/specs/2026-11-02-player-profiles-design.md.

Every skill is an offset on the logit scale of an existing model (pass_skill: the pass model; finishing: the shot
xG), stored with its sample size n and the unshrunk estimate raw. The coach uses logit(p) += a * offset; a missing
field leaves the coach exactly as without profiles. Offsets with sample sizes only: no ratings, no judgement labels.

Shrinkage (empirical Bayes): one Newton step from 0 for a per-player logit offset with a Gaussian prior of precision k,

    raw   = sum w (y - p) / sum w p (1 - p)
    logit = sum w (y - p) / (sum w p (1 - p) + k)

so a player with a handful of passes stays near 0 (the model's expectation). k and the scale a are chosen by
held-out log-loss (choose_k): each match's rows get offsets built from the other matches only.
"""

import numpy as np

VERSION = 1
SKILLS = ("pass_skill", "finishing")
K_GRID = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
A_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _arrays(keys, y, p, w):
    keys = np.asarray(keys, dtype=str)
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return keys, y, p, w


def residual_sums(keys, y, p, w=None) -> dict:
    """key -> (rows, sum w (y - p), sum w p (1 - p)); rows whose key is "" (unknown player) are ignored."""
    keys, y, p, w = _arrays(keys, y, p, w)
    ok = keys != ""
    if not ok.any():
        return {}
    u, inv = np.unique(keys[ok], return_inverse=True)
    n = np.bincount(inv, minlength=len(u))
    g = np.bincount(inv, weights=(w * (y - p))[ok], minlength=len(u))
    h = np.bincount(inv, weights=(w * p * (1 - p))[ok], minlength=len(u))
    return {str(k): (int(a), float(b), float(c)) for k, a, b, c in zip(u, n, g, h)}


def shrink(sums, k: float) -> dict:
    """(rows, sum w (y - p), sum w p (1 - p)) -> {"logit": shrunk offset, "n": rows, "raw": unshrunk offset}."""
    n, g, h = sums
    return {"logit": g / (h + k) if h + k > 0 else 0.0, "n": int(n), "raw": g / h if h > 0 else None}


def offsets(keys, y, p, w=None, k: float = 30.0) -> dict:
    """Shrunk offset of every player key against the expectations p."""
    return {key: shrink(s, k) for key, s in residual_sums(keys, y, p, w).items()}


def offset_vector(keys, table: dict) -> np.ndarray:
    """Per row: the player's offset, 0 when the player has no profile."""
    return np.array([table[k]["logit"] if k in table else 0.0 for k in np.asarray(keys, dtype=str)], float)


def apply(p, off, a: float = 1.0) -> np.ndarray:
    return sigmoid(logit(p) + a * np.asarray(off, float))


def row_loss(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(y, p, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float(np.sum(w * row_loss(y, p)) / np.sum(w))


def held_out_sums(groups, keys, y, p, w=None):
    """Per row: (rows, sum w (y - p), sum w p (1 - p)) of its player over the OTHER groups (zeros when none)."""
    keys, y, p, w = _arrays(keys, y, p, w)
    groups = np.asarray(groups)
    n, g, h = (np.zeros(len(y)) for _ in range(3))
    for grp in np.unique(groups):
        test = groups == grp
        sums = residual_sums(keys[~test], y[~test], p[~test], w[~test])
        for i in np.flatnonzero(test):
            s = sums.get(keys[i])
            if s is not None:
                n[i], g[i], h[i] = s
    return n, g, h


def choose_k(groups, keys, y, p, w=None, ks=K_GRID, scales=A_GRID) -> dict:
    """(k, a) with the lowest held-out log-loss when each group (match) gets offsets built from the other groups only.
    `p` must be out-of-fold expectations. Ties keep the first in grid order (k, then a). curve_a1: log-loss per k at
    a = 1 (when 1 is in the grid)."""
    n, g, h = held_out_sums(groups, keys, y, p, w)
    lp = logit(p)
    grid = {}
    for k in ks:
        off = np.where(h + k > 0, g / np.maximum(h + k, 1e-12), 0.0)
        for a in scales:
            grid[(float(k), float(a))] = log_loss(y, sigmoid(lp + a * off), w)
    best = min(grid, key=grid.get)
    out = {
        "k": best[0],
        "a": best[1],
        "log_loss": grid[best],
        "base_log_loss": log_loss(y, p, w),
        "coverage": float(np.mean(n > 0)),
    }
    if 1.0 in scales:
        out["curve_a1"] = {str(float(k)): grid[(float(k), 1.0)] for k in ks}
    return out
```

- [ ] **Step 4:** `$PY -m pytest tests/test_20_profiles.py -q` → 4 passed.
- [ ] **Step 5:** lint, then commit `20 coach: player profile offsets with empirical-Bayes shrinkage and held-out k`.

### Task 2: format, providers, validation, SkillCorner descriptors

**Files:** Modify `20-football-analytics/coach/profiles.py`, `tests/test_20_profiles.py`

- [ ] **Step 1: failing tests** (append)

```python
class FakeApi(PR.ApiProvider):
    """Test double for a live API: canned season records, no network, no key."""

    def __init__(self, ids, records):
        super().__init__(ids, k_finishing=10.0)
        self.records, self.calls = records, []

    def fetch(self, api_id):
        self.calls.append(api_id)
        return self.records.get(api_id)


def _players():
    return [
        {"id": "dfl:A", "name": "A. One", "team": 0},
        {"id": "dfl:B", "name": "B. Two", "team": 1, "position": "STZ"},
        {"id": "dfl:C", "name": "C. Three", "team": 1},
    ]


def test_build_open_data_and_api_providers(tmp_path):
    keys = np.array(["dfl:A"] * 50 + ["dfl:B"] * 5)
    y = np.r_[np.ones(45), np.zeros(5), np.ones(5)]
    open_data = PR.OpenDataProvider({"pass_skill": (keys, y, np.full(55, 0.8), None)}, {"pass_skill": 10.0})
    api = FakeApi({"dfl:B": 77, "dfl:C": 78}, {77: {"shots": 20, "goals": 5, "xg": 3.0, "passes": 100, "passes_completed": 80}})
    prof = PR.build(_players(), [open_data, api], built_without=["J03WOH"], k={"pass": 10.0})
    assert set(prof["players"]) == {"0", "1"}  # C: mapped to the API, but no record -> no profile
    assert prof["players"]["0"]["pass_skill"]["n"] == 50 and "finishing" not in prof["players"]["0"]
    b = prof["players"]["1"]
    assert b["position"] == "STZ" and b["finishing"]["n"] == 20 and b["finishing"]["logit"] > 0
    assert b["pass_accuracy"] == {"value": 0.8, "n": 100}
    assert prof["mapping"] == {"dfl:A": 0, "dfl:B": 1} and api.calls == [77, 78]
    assert PR.validate(prof) == []
    PR.save(prof, tmp_path / "profiles.json")
    back = PR.load(tmp_path / "profiles.json")
    assert back["built_without"] == ["J03WOH"] and back["players"]["0"]["pass_skill"]["n"] == 50


def test_validate_rejects_bad_fields():
    prof = {"version": 1, "built_without": [], "mapping": {"dfl:A": 0}, "players": {"0": {"id": "dfl:A"}}}
    assert PR.validate(prof) == []
    prof["players"]["0"]["pass_skill"] = {"logit": 0.1, "n": 0, "raw": 0.1}
    prof["players"]["0"]["rating"] = {"value": 7}
    errs = PR.validate(prof)
    assert any("pass_skill" in e for e in errs) and any("rating" in e for e in errs)
    with pytest.raises(ValueError):
        PR.save(prof, "unused.json")


def test_api_provider_is_abstract():
    with pytest.raises(TypeError):
        PR.ApiProvider({})


def test_skillcorner_descriptors(tmp_path):
    (tmp_path / PR.SC_AGGREGATES["passing"]).write_text(
        "player_id,player_birthdate,performance_included_count,pass_count_dangerous_attempted_p30tip,"
        "pass_pct_completed,pass_avgxpass_attempted,pass_count_attempted_total\n"
        "211,1993-06-02,13,5.66,75.76,74.46,198.0\n",
        encoding="utf-8",
    )
    (tmp_path / PR.SC_AGGREGATES["physical"]).write_text("player_id,psv99,count_match\n211,28.83,24\n", encoding="utf-8")
    d = PR.skillcorner_descriptors(tmp_path)
    p = d["skillcorner:211"]
    assert p["dangerous_passes_p30"] == {"value": 5.66, "n_matches": 13}
    assert p["completion_vs_xpass"] == {"value": pytest.approx(0.013), "n_passes": 198}
    assert p["top_speed_kmh"] == {"value": 28.83, "n_matches": 24} and "runs_p30" not in p
    assert not any("birth" in k for k in p)
```

- [ ] **Step 2:** run → FAIL (`ApiProvider` missing).

- [ ] **Step 3: implementation** (append to `coach/profiles.py`; add `import csv`, `import json`,
  `from abc import ABC, abstractmethod`, `from pathlib import Path` at the top)

```python
JUDGEMENT = ("rating", "grade", "good", "bad", "quality", "verdict")
SC_AGGREGATES = {
    "passing": "aus1league_passingaggregates_20242025.csv",
    "runs": "aus1league_obraggregates_20242025.csv",
    "physical": "aus1league_physicalaggregates_20242025.csv",
}


class Provider(ABC):
    """Fills profile fields for players named by provider keys ("dfl:<PersonId>", "skillcorner:<id>")."""

    name = "provider"

    @abstractmethod
    def player_fields(self, keys) -> dict:
        """key -> {field: value} for the keys it knows (unknown keys are absent)."""


class OpenDataProvider(Provider):
    """Offsets from our own tracked matches: residual tables {skill: (keys, y, p, w)} against a model's out-of-fold
    expectations, built only from the matches the caller passes (the replayed match is left out), plus optional
    descriptors {key: {field: {...}}} (e.g. SkillCorner season aggregates)."""

    name = "open-data"

    def __init__(self, residuals: dict, k: dict, descriptors: dict | None = None):
        self.tables = {skill: offsets(*r, k=k[skill]) for skill, r in residuals.items()}
        self.descriptors = descriptors or {}

    def player_fields(self, keys) -> dict:
        out = {}
        for key in keys:
            f = {skill: t[key] for skill, t in self.tables.items() if key in t}
            f.update(self.descriptors.get(key, {}))
            if f:
                out[key] = f
        return out


class ApiProvider(Provider):
    """Later phase: a live sports-data API (e.g. API-Football), the owner's key in an environment variable, local use
    only (provider terms usually forbid republishing). Subclasses implement fetch(api_id) -> season record or None;
    this phase ships no client, no key handling and no network call. `ids` maps our keys to the API's player ids;
    unmapped players get nothing. A record: {"shots", "goals", "xg", "passes", "passes_completed"} (missing -> absent).
    """

    name = "api"

    def __init__(self, ids: dict, k_finishing: float = 30.0):
        self.ids, self.k_finishing = dict(ids), k_finishing

    @abstractmethod
    def fetch(self, api_id):
        """Season record of one player, or None."""

    def player_fields(self, keys) -> dict:
        out = {}
        for key in keys:
            if key in self.ids:
                f = api_fields(self.fetch(self.ids[key]) or {}, self.k_finishing)
                if f:
                    out[key] = f
        return out


def api_fields(rec: dict, k_finishing: float = 30.0) -> dict:
    """Season totals -> profile fields. finishing: goals vs xG with the same shrinkage, sum p (1 - p) approximated by
    xG (1 - xG / shots) (per-shot xG is not in season totals). pass_accuracy: a descriptor (raw completion has no model
    expectation, so it is not an offset)."""
    out = {}
    shots, goals, xg = rec.get("shots"), rec.get("goals"), rec.get("xg")
    if shots and goals is not None and xg is not None:
        out["finishing"] = shrink((int(shots), float(goals) - float(xg), float(xg) * (1 - float(xg) / shots)), k_finishing)
    if rec.get("passes") and rec.get("passes_completed") is not None:
        out["pass_accuracy"] = {"value": rec["passes_completed"] / rec["passes"], "n": int(rec["passes"])}
    return out


def build(match_players: list, providers: list, source: str | None = None, built_without=(), **extra) -> dict:
    """profiles.json content for one match. match_players: in player-index order, {"id": key, "name", "team",
    "position"?}. The first provider that fills a field wins; players with no field are left out (the coach then
    behaves exactly as without profiles). mapping: provider key -> player index."""
    filled = [p.player_fields([mp["id"] for mp in match_players]) for p in providers]
    players, mapping = {}, {}
    for idx, mp in enumerate(match_players):
        fields = {}
        for f in filled:
            for k, v in f.get(mp["id"], {}).items():
                fields.setdefault(k, v)
        if not fields:
            continue
        base = {k: mp[k] for k in ("id", "name", "team", "position") if mp.get(k) is not None}
        players[str(idx)] = {**base, **fields}
        mapping[mp["id"]] = idx
    return {
        "version": VERSION,
        "source": source or " + ".join(p.name for p in providers),
        "built_without": list(built_without),
        **extra,
        "players": players,
        "mapping": mapping,
    }


def validate(profiles: dict) -> list:
    """Format errors (empty list = valid): required keys, skills with a finite logit and n > 0, no judgement labels,
    mapping consistent with the players."""
    errs = [f"missing {k}" for k in ("version", "built_without", "players", "mapping") if k not in profiles]
    players = profiles.get("players", {})
    for pk, pl in players.items():
        for skill in SKILLS:
            s = pl.get(skill)
            if s is not None and not (
                isinstance(s.get("n"), int) and s["n"] > 0 and np.isfinite(float(s.get("logit", np.nan)))
            ):
                errs.append(f"{pk}.{skill}: needs a finite logit and n > 0")
        errs += [f"{pk}.{f}: judgement label" for f in pl if any(w in f.lower() for w in JUDGEMENT)]
    errs += [f"mapping {k} -> {v}: no such player" for k, v in profiles.get("mapping", {}).items() if str(v) not in players]
    return errs


def _round(x, nd=5):
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: _round(v, nd) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_round(v, nd) for v in x]
    return x


def save(profiles: dict, path) -> None:
    errs = validate(profiles)
    if errs:
        raise ValueError("invalid profiles: " + "; ".join(errs[:5]))
    Path(path).write_text(json.dumps(_round(profiles), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def skillcorner_descriptors(folder) -> dict:
    """SkillCorner season aggregates (A-League 2024/25, MIT) -> {"skillcorner:<id>": descriptors}. Display only: the
    season totals include every match, so they are never used in the evaluated offsets. Birth dates are not read."""
    folder, out = Path(folder), {}

    def put(r, field, value, **counts):
        counts = {k: int(v) for k, v in counts.items() if v}
        if value is not None and counts:
            out.setdefault(f"skillcorner:{int(float(r['player_id']))}", {})[field] = {"value": round(value, 4), **counts}

    if (folder / SC_AGGREGATES["passing"]).exists():
        for r in _csv(folder / SC_AGGREGATES["passing"]):
            n = _num(r.get("performance_included_count"))
            put(r, "dangerous_passes_p30", _num(r.get("pass_count_dangerous_attempted_p30tip")), n_matches=n)
            pct, xp = _num(r.get("pass_pct_completed")), _num(r.get("pass_avgxpass_attempted"))
            if pct is not None and xp is not None:
                put(r, "completion_vs_xpass", (pct - xp) / 100, n_passes=_num(r.get("pass_count_attempted_total")))
    if (folder / SC_AGGREGATES["runs"]).exists():
        for r in _csv(folder / SC_AGGREGATES["runs"]):
            put(r, "runs_p30", _num(r.get("offballrun_count_p30tip")), n_matches=_num(r.get("performance_included_count")))
    if (folder / SC_AGGREGATES["physical"]).exists():
        for r in _csv(folder / SC_AGGREGATES["physical"]):
            put(r, "top_speed_kmh", _num(r.get("psv99")), n_matches=_num(r.get("count_match")))
    return out
```

- [ ] **Step 4:** run → 8 passed. **Step 5:** lint, commit `20 coach: profile format, providers (open data, API interface), SkillCorner descriptors`.

### Task 3: held-out evaluation core

**Files:** Create `20-football-analytics/coach/profiles_eval.py`; modify `tests/test_20_profiles.py`

- [ ] **Step 1: failing tests** (append)

```python
from coach import models as ML  # noqa: E402
from coach import profiles_eval as PE  # noqa: E402


def _pass_table(effects, matches=5, rows=250, seed=0):
    rng = np.random.default_rng(seed)
    n = matches * rows
    X = rng.normal(size=(n, 2))
    keys = np.array([f"p{i}" for i in rng.integers(0, len(effects), n)])
    z = 1.0 + 1.2 * X[:, 0] + np.array([effects[int(k[1:])] for k in keys])
    y = (rng.uniform(size=n) < PR.sigmoid(z)).astype(int)
    match = np.repeat([f"M{i}" for i in range(matches)], rows)
    return {"X": X, "y": y, "match": match, "weight": np.ones(n)}, keys


def test_lomo_profiles_improves_with_real_effects_and_reports_coverage():
    table, keys = _pass_table([1.5, -1.5, 1.0, -1.0])
    r = PE.lomo_profiles(table, keys, fit=ML.fit_logistic)
    s = PE.summarize(table, r)
    assert s["coverage"] == 1.0 and set(s["folds"]) == {f"M{i}" for i in range(5)}
    assert s["with"]["log_loss"] < s["without"]["log_loss"]
    assert s["log_loss_diff"]["ci95"][1] < 0 and s["decision"] == "advice"


def test_lomo_profiles_fixed_base_and_unseen_player():
    table, keys = _pass_table([0.0, 0.0])
    keys[table["match"] == "M0"] = "only_in_M0"  # never in a training set when M0 is held out
    r = PE.lomo_profiles(table, keys, base=np.full(len(keys), 0.7))
    m0 = table["match"] == "M0"
    assert not r["has"][m0].any() and np.allclose(r["p1"][m0], 0.7)


def test_bootstrap_diff_sign():
    y = np.array([1, 0] * 50)
    match = np.repeat(["a", "b", "c", "d"], 25)
    good, bad = np.where(y == 1, 0.9, 0.1), np.full(100, 0.5)
    d = PE.bootstrap_diff(match, y, bad, good, np.ones(100))
    assert d["mean"] < 0 and d["ci95"][1] < 0


def test_check_target_mapping():
    assert PE.check_target_mapping([5, 6, 5], [0, 1, 0]) == 2
    with pytest.raises(ValueError):
        PE.check_target_mapping([5, 6, 5], [0, 1, 1])


def test_idsse_keys_follow_the_derived_rows():
    info = {
        "player_ids": ["P0", "P1"],
        "events": [
            {"type": "pass", "player": 1, "team": 0, "half": 0},
            {"type": "shot", "player": 0, "team": 0, "half": 0},
            {"type": "shot", "player": 1, "team": 1, "half": None},  # not a shot row (no half)
        ],
    }
    d = {"pass": {"event_idx": np.array([0])}, "shot": {"y": np.array([0])}}
    pk, sk = PE.idsse_keys(d, info)
    assert list(pk) == ["dfl:P1"] and list(sk) == ["dfl:P0"]
```

- [ ] **Step 2:** run → FAIL (`profiles_eval` missing).

- [ ] **Step 3: implementation** (`coach/profiles_eval.py`, core; `main` and the chart come in Task 5)

```python
"""Held-out test of the player profiles: do the offsets improve the pass and shot models on matches the profiles
were not built from?

    python -m coach.profiles_eval      # from 20-football-analytics/, after coach.cross_league (~5 min)

Leave-one-match-out. For each held-out match the base model AND the player offsets come only from the other matches;
the offsets are residuals against out-of-fold expectations of the training matches (nested leave-one-match-out),
and k / a are picked by nested held-out log-loss inside the training matches. Data: the derived tables cached by
coach.cross_league (no tracking needed); passer / shooter ids from the IDSSE match JSON and, for SkillCorner, from
the dynamic-events CSV (streamed, only the ids are cached in cv-data/skillcorner/profiles/players.json).
Writes docs/coach_profiles_results.json and docs/coach_profiles.png.
"""

import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np

from . import datasets as D
from . import models as ML
from . import profiles as PR

DOCS = Path(__file__).resolve().parents[1] / "docs"
BOOTSTRAPS = 2000
RULE = (
    "offsets change the advice only if the match-bootstrap 95 % CI of the held-out log-loss difference "
    "(with - without) lies entirely below 0; otherwise the profiles are display-only"
)


def subset(table, sel):
    return {k: v[sel] for k, v in table.items() if isinstance(v, np.ndarray) and len(v) == len(sel)}


def oof(table, fit):
    """Leave-one-match-out predictions of `fit` on every row of `table` (weights used when present)."""
    X, y, g, w = table["X"], table["y"], table["match"], table.get("weight")
    p = np.full(len(y), np.nan)
    for m in np.unique(g):
        test = g == m
        model = fit(X[~test], y[~test]) if w is None else fit(X[~test], y[~test], w[~test])
        p[test] = model.predict_proba(X[test])[:, 1]
    return p


def lomo_profiles(table, keys, fit=None, base=None, scales=PR.A_GRID) -> dict:
    """Held-out model without (p0) and with (p1) player offsets. Either `fit` (refitted per fold; the training
    expectations behind the offsets are its inner out-of-fold predictions) or `base` (fixed expectations per row, e.g.
    the provider's xG). Per row: off (offset used), has (player had a profile); per fold: k, a, inner gain."""
    y, g = table["y"], table["match"]
    w = table["weight"] if "weight" in table else np.ones(len(y))
    keys = np.asarray(keys, dtype=str)
    p0, p1, off = (np.full(len(y), np.nan) for _ in range(3))
    has = np.zeros(len(y), bool)
    folds = {}
    for m in np.unique(g):
        test, train = g == m, g != m
        if fit is not None:
            tr = subset(table, train)
            p0[test] = fit(tr["X"], tr["y"], w[train]).predict_proba(table["X"][test])[:, 1]
            p_in = oof(tr, fit)
        else:
            p0[test], p_in = base[test], base[train]
        sel = PR.choose_k(g[train], keys[train], y[train], p_in, w[train], scales=scales)
        prof = PR.offsets(keys[train], y[train], p_in, w[train], k=sel["k"])
        off[test] = PR.offset_vector(keys[test], prof)
        has[test] = [k in prof for k in keys[test]]
        p1[test] = PR.apply(p0[test], off[test], sel["a"])
        folds[str(m)] = {"k": sel["k"], "a": sel["a"], "inner_gain": sel["base_log_loss"] - sel["log_loss"]}
    return {"p0": p0, "p1": p1, "off": off, "has": has, "folds": folds, "w": w}


def brier_rows(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def bootstrap_diff(match, y, p0, p1, w, fn=PR.row_loss, reps=BOOTSTRAPS, seed=0) -> dict:
    """Weighted mean of fn(with) - fn(without) and its 95 % interval, resampling whole matches with replacement."""
    d = w * (fn(y, p1) - fn(y, p0))
    ids, inv = np.unique(match, return_inverse=True)
    sd, sw = np.bincount(inv, weights=d), np.bincount(inv, weights=w)
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(reps, len(ids)))
    boots = sd[pick].sum(axis=1) / sw[pick].sum(axis=1)
    return {"mean": float(d.sum() / w.sum()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "n": int(len(x)),
        **{k: float(v) for k, v in zip(("p5", "p25", "median", "p75", "p95"), q)},
        "mean_abs": float(np.abs(x).mean()),
        "max_abs": float(np.abs(x).max()),
    }


def summarize(table, r) -> dict:
    y, g, w = table["y"], table["match"], r["w"]
    ll = bootstrap_diff(g, y, r["p0"], r["p1"], w)
    return {
        "rows": int(len(y)),
        "without": ML.metrics(y, r["p0"], w),
        "with": ML.metrics(y, r["p1"], w),
        "log_loss_diff": ll,
        "brier_diff": bootstrap_diff(g, y, r["p0"], r["p1"], w, fn=brier_rows),
        "coverage": float(r["has"].mean()),
        "coverage_weighted": float(w[r["has"]].sum() / w.sum()),
        "held_out_offsets": quantiles(r["off"][r["has"]]),
        "folds": r["folds"],
        "decision": "advice" if ll["ci95"][1] < 0 else "display-only",
    }


# ------------------------------------------------------------------------------------------- data


def idsse_keys(d, info):
    """Passer keys of the derived pass rows (event_idx -> event -> PersonId) and shooter keys of the derived shot rows
    (cross_league.shot_rows order: shots with a half and a team)."""
    ids, ev = info["player_ids"], info["events"]
    key = lambda e: f"dfl:{ids[e['player']]}" if e.get("player") is not None else ""
    passes = [key(ev[i]) for i in d["pass"]["event_idx"]]
    shots = [key(e) for e in ev if e["type"] == "shot" and e.get("half") is not None and e["team"] in (0, 1)]
    if len(shots) != len(d["shot"]["y"]):
        raise ValueError(f"{len(shots)} shots in the events, {len(d['shot']['y'])} derived rows")
    return np.array(passes, dtype=str), np.array(shots, dtype=str)


def idsse_data() -> list:
    """The 7 IDSSE matches: derived tables (coach.cross_league cache) + keys, match info and the team of each player."""
    from . import skillcorner as SC
    from .prepare import MATCHES, OUT

    out = []
    for mid in MATCHES:
        path = SC.DERIVED / f"idsse_{mid}.pkl"
        if not path.exists():
            raise SystemExit(f"{path} missing: run python -m coach.cross_league first")
        d = pickle.loads(path.read_bytes())
        d["info"] = json.loads((OUT / f"{mid}.json").read_text(encoding="utf-8"))
        d["team"] = np.load(OUT / f"{mid}.npz")["team"]
        d["pass_keys"], d["shot_keys"] = idsse_keys(d, d["info"])
        out.append(d)
    return out


class _Same(dict):
    """pidx stand-in for skillcorner.parse_events: keeps SkillCorner player ids instead of tracking columns."""

    def get(self, key, default=None):
        return key


def sc_event_players(sheet: dict, event_rows) -> list:
    """[actor id, recipient id] of every event, in skillcorner.parse_events order (the order behind event_idx)."""
    from . import skillcorner as SC

    events, _ = SC.parse_events(event_rows, SC.parse_sheet(sheet), _Same(), (1.0, 1.0))
    return [[e["player"], e.get("recipient")] for e in events]


def check_target_mapping(recipients, target_idx) -> int:
    """Rebuilt events name receivers by SkillCorner id, the derived rows by tracking column: the event order is right
    only if the two correspond one to one. Returns the number of distinct receivers."""
    fwd, back = {}, {}
    for r, t in zip(recipients, target_idx):
        t = int(t)
        if r is None or fwd.setdefault(r, t) != t or back.setdefault(t, r) != r:
            raise ValueError(f"event order mismatch: recipient {r} vs target column {t}")
    return len(fwd)


def sc_data() -> list:
    """The 20 SkillCorner matches: derived tables + passer keys. Match sheet and dynamic events are streamed once
    (~5 MB a match); only actor / recipient ids, names, teams and roles are cached."""
    from . import skillcorner as SC

    cache = SC.OUT / "profiles" / "players.json"
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out = []
    for mid in SC.MATCHES:
        d = pickle.loads((SC.DERIVED / f"{mid}.pkl").read_bytes())
        if mid not in store:
            base = f"{SC.RAW_URL}/matches/{mid}/{mid}"
            sheet = json.loads(SC._get(f"{base}_match.json"))
            rows = list(csv.DictReader(io.StringIO(SC._get(f"{base}_dynamic_events.csv").decode("utf-8-sig"))))
            info = SC.parse_sheet(sheet)
            store[mid] = {
                "actors": sc_event_players(sheet, rows),
                "players": {str(k): {f: v[f] for f in ("team", "name", "position")} for k, v in info["players"].items()},
                "teams": [info["meta"]["home"], info["meta"]["guest"]],
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(store), encoding="utf-8")
        actors = store[mid]["actors"]
        idx = d["pass"]["event_idx"]
        check_target_mapping([actors[i][1] for i in idx], d["pass"]["target_idx"])
        d["pass_keys"] = np.array([f"skillcorner:{actors[i][0]}" if actors[i][0] is not None else "" for i in idx], dtype=str)
        d["players"], d["teams"] = store[mid]["players"], store[mid]["teams"]
        out.append(d)
    return out


def pass_rows(parts):
    return D.stack([d["pass"] for d in parts]), np.concatenate([d["pass_keys"] for d in parts])


def shot_rows(parts):
    """Placed non-penalty shots with a provider xG and features (the shots of the coach's evaluation)."""
    st, keys = D.stack([d["shot"] for d in parts]), np.concatenate([d["shot_keys"] for d in parts])
    keep = st["placed"] & ~st["penalty"] & np.isfinite(st["xg"]) & np.isfinite(st["X"]).all(axis=1)
    return subset(st, keep), keys[keep]
```

- [ ] **Step 4:** run → 13 passed. **Step 5:** lint, commit `20 coach: held-out evaluation of player profiles (nested LOMO, match bootstrap)`.

### Task 4: exporter

**Files:** Create `20-football-analytics/coach/export_profiles.py`; modify `tests/test_20_profiles.py`

- [ ] **Step 1: failing test** (append)

```python
from coach import export_profiles as EP  # noqa: E402


def test_export_profiles_built_without_the_match(tmp_path):
    rng = np.random.default_rng(3)
    parts = []
    for mid in ("M0", "M1", "M2", "M3"):
        n, ns = 200, 12
        players = ["dfl:A", "dfl:B"] if mid != "M0" else ["dfl:A", "dfl:ONLY0"]
        parts.append(
            {
                "match": mid,
                "pass": {"X": rng.normal(size=(n, 2)), "y": rng.integers(0, 2, n), "match": np.full(n, mid), "weight": np.ones(n)},
                "pass_keys": np.array(rng.choice(players, n), dtype=str),
                "shot": {
                    "X": rng.normal(size=(ns, 3)), "y": rng.integers(0, 2, ns), "match": np.full(ns, mid),
                    "xg": rng.uniform(0.05, 0.3, ns), "penalty": np.zeros(ns, bool), "placed": np.ones(ns, bool),
                },
                "shot_keys": np.array(rng.choice(players, ns), dtype=str),
                "info": {"player_ids": [p[4:] for p in players], "names": ["A", "B"], "meta": {"home": "H", "guest": "G"}},
                "team": np.array([0, 1], np.int8),
            }
        )
    prof = EP.build("M0", parts, decision={"pass": "display-only", "shot": "display-only"})
    assert prof["built_without"] == ["M0"] and "M0" not in prof["trained_on"]
    assert set(prof["players"]) == {"0"} and prof["mapping"] == {"dfl:A": 0}  # ONLY0 plays only in M0
    assert prof["a"] == {"pass": 0.0, "shot": 0.0} and prof["advice_effect"] == "display-only"
    EP.write(prof, tmp_path)
    assert PR.load(tmp_path / "profiles.json")["players"]["0"]["pass_skill"]["n"] > 0
```

- [ ] **Step 2:** run → FAIL. **Step 3: implementation**

```python
"""profiles.json for the website's replay match, built WITHOUT that match (player profiles, phase 1).

    python -m coach.export_profiles --match J03WOH --site ../../cv-data/football-coach/site-export

pass_skill: the passer's offset against the coach's pass model (logistic, the model of models.json, trained without
the match); the training passes' expectations are out-of-fold among the training matches. finishing: goals against
the provider's xG (placed non-penalty shots). k and a: nested held-out log-loss within the training matches. Whether
the offsets may change the advice comes from the held-out evaluation (docs/coach_profiles_results.json "decision",
written by coach.profiles_eval); without it: display-only, a = 0. Players are keyed by their index in the match (the
match.json players list); the DFL PersonId is kept in "id" and "mapping".
"""

import argparse
import json
from pathlib import Path

from . import models as ML
from . import profiles as PR
from . import profiles_eval as PE

RESULTS = PE.DOCS / "coach_profiles_results.json"
SOURCE = "open-data: IDSSE events of the other matches (pass model residuals; goals vs provider xG)"


def build(match_id: str, parts=None, decision=None) -> dict:
    parts = PE.idsse_data() if parts is None else parts
    target = next(d for d in parts if d["match"] == match_id)
    train = [d for d in parts if d["match"] != match_id]
    pt, pk = PE.pass_rows(train)
    p_in = PE.oof(pt, ML.fit_logistic)
    sel_pass = PR.choose_k(pt["match"], pk, pt["y"], p_in, pt["weight"])
    st, sk = PE.shot_rows(train)
    sel_shot = PR.choose_k(st["match"], sk, st["y"], st["xg"])
    provider = PR.OpenDataProvider(
        {"pass_skill": (pk, pt["y"], p_in, pt["weight"]), "finishing": (sk, st["y"], st["xg"], None)},
        {"pass_skill": sel_pass["k"], "finishing": sel_shot["k"]},
    )
    if decision is None:
        decision = json.loads(RESULTS.read_text(encoding="utf-8"))["decision"] if RESULTS.exists() else {}
    effect = {m: decision.get(m, "display-only") for m in ("pass", "shot")}
    info = target["info"]
    players = [
        {"id": f"dfl:{pid}", "name": name, "team": int(t)}
        for pid, name, t in zip(info["player_ids"], info["names"], target["team"])
    ]
    names = [info["meta"]["home"], info["meta"]["guest"]]
    teams = {
        str(t): {"name": names[t], "matches_in_profiles": sum(names[t] in (d["info"]["meta"]["home"], d["info"]["meta"]["guest"]) for d in train)}
        for t in (0, 1)
    }
    return PR.build(
        players,
        [provider],
        source=SOURCE,
        built_without=[match_id],
        trained_on=[d["match"] for d in train],
        k={"pass": sel_pass["k"], "shot": sel_shot["k"]},
        a={"pass": sel_pass["a"] if effect["pass"] == "advice" else 0.0, "shot": sel_shot["a"] if effect["shot"] == "advice" else 0.0},
        a_fitted={"pass": sel_pass["a"], "shot": sel_shot["a"]},
        advice_effect="advice" if "advice" in effect.values() else "display-only",
        advice_effect_by_model=effect,
        usage="pass: logit(p) += a.pass * players[passer].pass_skill.logit; shot: logit(xG) += a.shot * finishing.logit",
        teams=teams,
    )


def write(profiles: dict, site: Path) -> Path:
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    PR.save(profiles, site / "profiles.json")
    return site / "profiles.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    path = write(build(args.match), args.site)
    prof = PR.load(path)
    print("wrote", path, path.stat().st_size, "bytes,", len(prof["players"]), "players, k", prof["k"], "a", prof["a"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** run → 14 passed. **Step 5:** lint, commit `20 coach: export profiles.json for the replay match, built without it`.

### Task 5: run the evaluation, chart, README, export, push

**Files:** Modify `20-football-analytics/coach/profiles_eval.py` (main + chart), `20-football-analytics/README.md`

- [ ] **Step 1: `main` and the chart** (append to `profiles_eval.py`)

```python
def final_profile(table, keys, p_oof) -> tuple:
    """k / a chosen on every match (nested held-out) and the offsets of every player: the 'best k' and the
    distribution shown in the README."""
    w = table.get("weight")
    sel = PR.choose_k(table["match"], keys, table["y"], p_oof, w)
    return sel, PR.offsets(keys, table["y"], p_oof, w, k=sel["k"])


def profile_summary(sel, prof) -> dict:
    off = [v["logit"] for v in prof.values()]
    n = [v["n"] for v in prof.values()]
    return {
        "k": sel["k"],
        "a": sel["a"],
        "held_out_log_loss": sel["log_loss"],
        "base_log_loss": sel["base_log_loss"],
        "curve_a1": sel.get("curve_a1"),
        "players": len(prof),
        "offsets": quantiles(off),
        "rows_per_player": quantiles(n),
        "share_abs_offset_over_0.1": float(np.mean(np.abs(off) > 0.1)) if off else 0.0,
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def chart(finals, res, path):
    """Offsets vs sample size, held-out log-loss vs k, and the with-minus-without log-loss with its 95 % CI."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = {"pass_idsse": "pass, Bundesliga", "pass_skillcorner": "pass, A-League", "finishing_idsse": "finishing (vs provider xG)"}
    for name, (sel, prof) in finals.items():
        n = [v["n"] for v in prof.values()]
        axes[0].scatter(n, [v["logit"] for v in prof.values()], s=10, alpha=0.6, label=f"{labels[name]} (k = {sel['k']:g})")
        curve = sel["curve_a1"]
        ks = [float(k) for k in curve]
        axes[1].plot(ks, [curve[k] - sel["base_log_loss"] for k in curve], "o-", label=labels[name])
    axes[0].set(xscale="log", xlabel="rows of the player (passes / shots)", ylabel="shrunk offset (logit)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Player offsets vs sample size")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set(xlabel="k (prior strength)", ylabel="held-out log-loss change (a = 1)")
    axes[1].set_title("Choosing k (nested held-out)")
    axes[1].legend(fontsize=7)
    rows = [("pass_idsse", "pass, Bundesliga"), ("pass_skillcorner", "pass, A-League"), ("shot_provider_xg", "shot, provider xG"), ("shot_logistic", "shot, coach logistic")]
    for i, (key, label) in enumerate(rows):
        d = res[key]["log_loss_diff"]
        axes[2].errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], fmt="o", capsize=4)
    axes[2].set_yticks(range(len(rows)), [r[1] for r in rows])
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set(xlabel="log-loss with - without offsets (held-out, 95 % CI)")
    axes[2].set_title("Held-out effect of the offsets")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    idsse, sc = idsse_data(), sc_data()
    res, finals = {}, {}
    for name, parts in (("idsse", idsse), ("skillcorner", sc)):
        pt, keys = pass_rows(parts)
        r = lomo_profiles(pt, keys, fit=ML.fit_logistic)
        r1 = lomo_profiles(pt, keys, fit=ML.fit_logistic, scales=(1.0,))
        s1 = summarize(pt, r1)
        res[f"pass_{name}"] = {**summarize(pt, r), "a_fixed_1": {k: s1[k] for k in ("with", "log_loss_diff", "folds")}}
        finals[f"pass_{name}"] = final_profile(pt, keys, oof(pt, ML.fit_logistic))
        print("pass", name, res[f"pass_{name}"]["without"]["log_loss"], res[f"pass_{name}"]["with"]["log_loss"], res[f"pass_{name}"]["log_loss_diff"], flush=True)
    st, sk = shot_rows(idsse)
    res["shot_provider_xg"] = summarize(st, lomo_profiles(st, sk, base=st["xg"]))
    res["shot_logistic"] = summarize(st, lomo_profiles(st, sk, fit=ML.fit_logistic))
    finals["finishing_idsse"] = final_profile(st, sk, st["xg"])
    for k in ("shot_provider_xg", "shot_logistic"):
        print(k, res[k]["without"]["log_loss"], res[k]["with"]["log_loss"], res[k]["log_loss_diff"], flush=True)
    desc = PR.skillcorner_descriptors(Path(__file__).resolve().parents[3] / "cv-data" / "skillcorner" / "aggregates")
    sc_keys = set(np.concatenate([d["pass_keys"] for d in sc]))
    out = {
        **res,
        "profiles_all_matches": {k: profile_summary(*v) for k, v in finals.items()},
        "skillcorner_descriptors": {"players": len(desc), "tracked_passers_with_descriptors": len(sc_keys & set(desc))},
        "decision": {"pass": res["pass_idsse"]["decision"], "shot": res["shot_provider_xg"]["decision"], "rule": RULE},
        "grids": {"k": list(PR.K_GRID), "a": list(PR.A_GRID)},
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_profiles_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    chart(finals, res, DOCS / "coach_profiles.png")
    print("decision", out["decision"])


if __name__ == "__main__":
    main()
```

The descriptors folder: use `SC.OUT / "aggregates"` (import inside `main`) rather than a path computed from the file.

- [ ] **Step 2:** `cd 20-football-analytics && $PY -m coach.profiles_eval` (streams 20 × ~5 MB event CSVs once). Expected:
  results JSON + chart; `check_target_mapping` passes for all 20 matches.
- [ ] **Step 3:** `$PY -m coach.export_profiles --match J03WOH --site C:/Users/yando/Code/cv-data/football-coach/site-export`
  → `profiles.json` (a few KB).
- [ ] **Step 4:** README: subsection "### Player profiles (held-out test)" under v3 before "### Run the coach": method
  (offset formula, nested k / a, open-data provider, API interface only), results table (with vs without log-loss,
  Brier, diff + 95 % CI, coverage, best k, offset spread), chart, ship decision with the rule, limits; run lines
  `python -m coach.profiles_eval` and `python -m coach.export_profiles --match J03WOH --site <folder>`; test count.
- [ ] **Step 5:** full tests (`tests/test_20_profiles.py`, `tests/test_20_coach.py`), lint, commit own paths
  (`profiles_eval.py`, `README.md` hunks only if no foreign edits are pending in it, `docs/coach_profiles*.{json,png}`),
  push, `gh run list --branch main --limit 1` → success.

## Self-review

- Spec coverage: format + shrinkage + k by held-out log-loss (T1), providers incl. API interface + fake in tests only
  (T2), SkillCorner descriptors (T2), LOMO with profiles from training matches only, pass and shot with vs without,
  bootstrap CI, coverage, offset distribution, best k (T3, T5), exporter without the match (T4), README + decision
  rule (T5). Receiver term: not in this phase (spec: "only if it helps held-out"; left for later).
- Names used across tasks: `choose_k`, `offsets`, `offset_vector`, `apply`, `row_loss`, `lomo_profiles`,
  `summarize`, `bootstrap_diff`, `pass_rows`, `shot_rows`, `idsse_data`, `sc_data`, `build`, `write`: consistent.

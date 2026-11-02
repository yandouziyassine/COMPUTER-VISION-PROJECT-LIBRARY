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

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path

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
    """k and the scale a by held-out log-loss, each group (match) getting offsets built from the other groups only;
    `p` must be out-of-fold expectations. k minimises the held-out log-loss with the offsets used as estimated (a = 1,
    curve_a1); a is then the best scale for that k (a = 0: the offsets do not help). Ties keep the first in grid
    order."""
    n, g, h = held_out_sums(groups, keys, y, p, w)
    lp = logit(p)

    def loss(k, a):
        off = np.where(h + k > 0, g / np.maximum(h + k, 1e-12), 0.0)
        return log_loss(y, sigmoid(lp + a * off), w)

    curve = {float(k): loss(k, 1.0) for k in ks}
    k = min(curve, key=curve.get)
    by_a = {float(a): loss(k, a) for a in scales}
    a = min(by_a, key=by_a.get)
    return {
        "k": k,
        "a": a,
        "log_loss": by_a[a],
        "base_log_loss": log_loss(y, p, w),
        "coverage": float(np.mean(n > 0)),
        "curve_a1": {str(kk): v for kk, v in curve.items()},
    }


# ------------------------------------------------------------------------------------- format and providers

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
        h = float(xg) * (1 - float(xg) / shots)
        out["finishing"] = shrink((int(shots), float(goals) - float(xg), h), k_finishing)
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
            ok = s is None or (
                isinstance(s.get("n"), int) and s["n"] > 0 and np.isfinite(float(s.get("logit", np.nan)))
            )
            if not ok:
                errs.append(f"{pk}.{skill}: needs a finite logit and n > 0")
        errs += [f"{pk}.{f}: judgement label" for f in pl if any(w in f.lower() for w in JUDGEMENT)]
    mapping = profiles.get("mapping", {})
    errs += [f"mapping {k} -> {v}: no such player" for k, v in mapping.items() if str(v) not in players]
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
    text = json.dumps(_round(profiles), ensure_ascii=False, separators=(",", ":"))
    Path(path).write_text(text, encoding="utf-8")


def load(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------- SkillCorner season descriptors


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
            key = f"skillcorner:{int(float(r['player_id']))}"
            out.setdefault(key, {})[field] = {"value": round(value, 4), **counts}

    if (folder / SC_AGGREGATES["passing"]).exists():
        for r in _csv(folder / SC_AGGREGATES["passing"]):
            n = _num(r.get("performance_included_count"))
            put(r, "dangerous_passes_p30", _num(r.get("pass_count_dangerous_attempted_p30tip")), n_matches=n)
            pct, xp = _num(r.get("pass_pct_completed")), _num(r.get("pass_avgxpass_attempted"))
            if pct is not None and xp is not None:
                put(r, "completion_vs_xpass", (pct - xp) / 100, n_passes=_num(r.get("pass_count_attempted_total")))
    if (folder / SC_AGGREGATES["runs"]).exists():
        for r in _csv(folder / SC_AGGREGATES["runs"]):
            n = _num(r.get("performance_included_count"))
            put(r, "runs_p30", _num(r.get("offballrun_count_p30tip")), n_matches=n)
    if (folder / SC_AGGREGATES["physical"]).exists():
        for r in _csv(folder / SC_AGGREGATES["physical"]):
            put(r, "top_speed_kmh", _num(r.get("psv99")), n_matches=_num(r.get("count_match")))
    return out

"""Model lab: can the coach's three models be made more accurate? Leave-one-match-out over all 27 matches (pooled).

    python -m coach.model_lab              # from 20-football-analytics/ (first run builds a cache, a few minutes)
    python -m coach.model_lab --rebuild    # rebuild the cache (tables + candidate features) from the raw matches

Compares the CURRENT recipe (export.fit_models: logistic pass model on PASS_FEATURES, ridge value / arrival models on
VALUE_FEATURES) with candidates, honestly:

1. Pass model: logistic + new features computed from the same frame state (extra_pass_features), and histogram
   gradient boosting with isotonic / sigmoid calibration on group-aware inner folds. Every comparison is repeated on
   the leak-checked table (X_noint: the interceptor of a failed pass removed from the defenders, before the new
   features are computed too), because a lane-interception feature is exactly where the inferred-target leak lives.
2. Value and arrival models: ridge vs gradient boosting, and ridge with zone x context interactions (value_context:
   opponents goal-side of the ball, the possessing team's forward speed).
3. End-to-end: the coach score of the pro's chosen pass (p * arrival(target) - (1 - p) * V_opp(loss) - V(now), as in
   advice.options) on held-out matches, Spearman with net xG in 15 s, current vs candidates.

Verdict per candidate: ADOPT only if better with the match-bootstrap 95 % CI of the difference excluding zero, on
both the standard and the leak-checked table (pass) and without a leakage risk; otherwise KEEP CURRENT. Writes
docs/coach_model_lab.json. Nothing here changes the shipped models (features.py / export.py / game content).
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import datasets as D
from . import features as F
from . import models as ML
from . import pitch_control as PC

DOCS = Path(__file__).resolve().parents[1] / "docs"
BALL_SPEED = 15.0  # m/s, typical ground pass: the ball's arrival time along the lane
NEAR_TARGET_M = 5.0
LANE_STEPS = np.linspace(0.1, 1.0, 10)
EXTRA_FEATURES = (
    "receiver_to_ball",  # receiver's speed towards the passer (m/s; > 0: showing for the ball)
    "receiver_vx",  # receiver's forward speed (m/s)
    "intercept_margin",  # min over the lane of (fastest defender's arrival - ball arrival), s, clipped to [-3, 3]
    "defenders_near_target",  # defenders within 5 m of the target
    "dist_x_contested",  # interaction: distance * (1 - target_control)
    "forward_x_pressure",  # interaction: forward * exp(-passer_pressure / 3)
)
CONTEXT_FEATURES = ("def_goal_side", "team_vx")
ZONES = ("in_box", "final_third", "middle_third")
N_PASS, N_VALUE = len(F.PASS_FEATURES), len(F.VALUE_FEATURES)
BOOT = 1000


# ------------------------------------------------------------------------------------------- new feature functions


def extra_pass_features(s, passer: int, target: int, base: dict | None = None) -> dict:
    """Candidate pass features from one normalised state (same inputs as features.pass_features)."""
    base = F.pass_features(s, passer, target) if base is None else base
    a, b = s["att"][passer], s["att"][target]
    vb = s["att_v"][target]
    ab = b - a
    length = float(np.linalg.norm(ab))
    u = ab / max(length, 1e-9)
    if len(s["def"]):
        pts = a + LANE_STEPS[:, None] * ab
        t_def = PC.time_to_reach(s["def"], s["def_v"], pts).min(axis=0)
        margin = float(np.clip(np.min(t_def - LANE_STEPS * length / BALL_SPEED), -3.0, 3.0))
        near = int(np.sum(np.linalg.norm(s["def"] - b, axis=1) < NEAR_TARGET_M))
    else:
        margin, near = 3.0, 0
    return {
        "receiver_to_ball": float(-(vb @ u)),
        "receiver_vx": float(vb[0]),
        "intercept_margin": margin,
        "defenders_near_target": near,
        "dist_x_contested": base["distance"] * (1.0 - base["target_control"]),
        "forward_x_pressure": base["forward"] * float(np.exp(-base["passer_pressure"] / 3.0)),
    }


def value_context(points, opponents, team_v) -> np.ndarray:
    """[N, 2] points (the team in possession attacks +x) -> [N, 2]: opponents goal-side of the point (x greater), and
    the possessing team's mean forward speed (transition / counter-attack indicator)."""
    points = np.atleast_2d(np.asarray(points, float))
    opp = np.atleast_2d(np.asarray(opponents, float))
    goal_side = (opp[None, :, 0] > points[:, None, 0]).sum(axis=1) if opp.size else np.zeros(len(points))
    vx = float(np.mean(team_v[:, 0])) if len(team_v) else 0.0
    return np.column_stack([goal_side, np.full(len(points), vx)]).astype(float)


def value_rows(points, opponents, team_v) -> np.ndarray:
    """VALUE_FEATURES + CONTEXT_FEATURES at each point: [N, 12]."""
    points = np.atleast_2d(np.asarray(points, float))
    basis = np.array([[F.value_features(p)[k] for k in F.VALUE_FEATURES] for p in points]).reshape(-1, N_VALUE)
    return np.hstack([basis, value_context(points, opponents, team_v)])


def interactions(V) -> np.ndarray:
    """Zone x context terms on top of value_rows: [N, 12] -> [N, 18]."""
    x, in_box = V[:, F.VALUE_FEATURES.index("x")], V[:, F.VALUE_FEATURES.index("in_box")]
    zones = np.column_stack([in_box, (x > 17.5).astype(float), (np.abs(x) <= 17.5).astype(float)])
    ctx = V[:, N_VALUE : N_VALUE + 2]
    return np.hstack([V, *(zones * ctx[:, [j]] for j in range(ctx.shape[1]))])


def chosen_row(s, carrier: int, target: int) -> dict:
    """Everything the coach score of the pass carrier -> target needs, in feature form (advice.options computes the
    same score from these): pass features (base + extra), value rows at the target (arrival), at the mirrored loss
    point (the opponent in possession) and at the carrier (V(now))."""
    base = F.pass_features(s, carrier, target)
    ext = extra_pass_features(s, carrier, target, base)
    here = s["att"][carrier]
    loss = F.lane_loss_point(s, here, s["att"][target])
    return {
        "pass": np.array([base[k] for k in F.PASS_FEATURES] + [ext[k] for k in EXTRA_FEATURES], float),
        "target": value_rows(s["att"][target], s["def"], s["att_v"])[0],
        "loss": value_rows(-loss, -s["att"], -s["def_v"])[0],
        "now": value_rows(here, s["def"], s["att_v"])[0],
    }


def chosen_score(p, v_target, v_loss, v_now):
    """advice.options' score of a pass option, vectorised."""
    return p * v_target - (1 - p) * v_loss - v_now


# ---------------------------------------------------------------------------------------------------------- tables


def match_lab_tables(m) -> dict:
    """Pass table (+ extra features, standard and interceptor-removed), value table (+ context), arrival table
    (+ context at the target), shot table, and one chosen-pass row per pass for the end-to-end check."""
    pt = D.pass_table(m)
    shots = D.shots_by_half(m)
    ext, ext_noint, rows, idx, outcome = [], [], [], [], []
    for p in D.pass_samples(m):
        s, c, t, e = p["state"], p["carrier"], p["target"], p["event"]
        idx.append(p["index"])
        r = chosen_row(s, c, t)
        rows.append(r)
        ext.append(r["pass"][N_PASS:])
        s2 = s if p["interceptor"] is None else F.without_defender(s, p["interceptor"])
        e2 = extra_pass_features(s2, c, t)
        ext_noint.append([e2[k] for k in EXTRA_FEATURES])
        outcome.append(D.net_xg(shots[e["half"]], D.when(m, e), e["team"], D.HORIZON))
    assert idx == pt["event_idx"].tolist(), "pass samples out of order"
    k = len(EXTRA_FEATURES)
    pt["X_full"] = np.hstack([pt["X"], np.array(ext, float).reshape(-1, k)])
    pt["X_full_noint"] = np.hstack([pt["X_noint"], np.array(ext_noint, float).reshape(-1, k)])
    vt = D.value_table(m)
    ctx = []
    for hi, h in enumerate(m.halves):  # same loop and filters as datasets.value_table
        for i in range(0, len(h.t), max(1, int(round(F.match_fps(m))))):
            team = int(h.owner[i])
            if not h.alive[i] or team < 0 or np.isnan(h.ball[i, 0]):
                continue
            s = F.state(m, hi, i, team)
            ctx.append(value_context(s["ball"], s["def"], s["att_v"])[0])
    assert len(ctx) == len(vt["y"]), "value context out of step"
    vt["X_full"] = np.hstack([vt["X"], np.array(ctx, float).reshape(-1, 2)])
    at = D.arrival_table(m, pt)
    done = pt["y"] == 1
    at["X_full"] = np.array([r["target"] for r, d in zip(rows, done) if d], float).reshape(-1, N_VALUE + 2)
    assert np.allclose(at["X_full"][:, :N_VALUE], at["X"], atol=1e-4), "arrival rows disagree"
    dec = {
        "pass": np.array([r["pass"] for r in rows], float).reshape(-1, N_PASS + k),
        "target": np.array([r["target"] for r in rows], float).reshape(-1, N_VALUE + 2),
        "loss": np.array([r["loss"] for r in rows], float).reshape(-1, N_VALUE + 2),
        "now": np.array([r["now"] for r in rows], float).reshape(-1, N_VALUE + 2),
        "outcome": np.array(outcome, float),
        "match": pt["match"],
    }
    keep = ("X", "y", "match", "weight", "X_noint", "X_full", "X_full_noint")
    return {
        "pass": {kk: pt[kk] for kk in keep},
        "value": {kk: vt[kk] for kk in ("X", "y", "match", "X_full")},
        "arrival": {kk: at[kk] for kk in ("X", "y", "match", "X_full")},
        "shot": D.shot_table(m),
        "decisions": dec,
    }


def cache_path() -> Path:
    from .skillcorner import OUT

    return OUT / "model_lab_cache.pkl"


def build_cache(path: Path) -> dict:
    from .benchmark import load_set

    t0 = time.time()
    matches, _, _ = load_set("pooled")
    print(f"loaded {len(matches)} matches in {time.time() - t0:.0f} s", flush=True)
    tables = {}
    for mid, m in matches.items():
        tables[mid] = match_lab_tables(m)
        print(f"  {mid}: {len(tables[mid]['pass']['y'])} passes ({time.time() - t0:.0f} s)", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(tables))
    return tables


def league(mid) -> str:
    return "idsse" if str(mid).startswith("J") else "aleague"


# ---------------------------------------------------------------------------------------------------------- models


class Cols:
    """A fitted model that reads only some columns of the full matrix (optionally through a transform)."""

    def __init__(self, cols, model, transform=None):
        self.cols, self.model, self.transform = cols, model, transform

    def _x(self, X):
        X = X[:, self.cols]
        return X if self.transform is None else self.transform(X)

    def predict_proba(self, X):
        return self.model.predict_proba(self._x(X))

    def predict(self, X):
        return self.model.predict(self._x(X))


def _hgb():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=0, early_stopping=False
    )


def fit_calibrated_hgb(method):
    """HGB calibrated on group-aware inner folds (5 GroupKFold splits of the training matches, ensemble of 5)."""

    def fit(X, y, w, groups):
        splits = list(GroupKFold(n_splits=5).split(X, y, groups))
        return CalibratedClassifierCV(_hgb(), method=method, cv=splits).fit(X, y, sample_weight=w)

    return fit


PASS_ALL = list(range(N_PASS + len(EXTRA_FEATURES)))
PASS_BASE = list(range(N_PASS))
PASS_CANDIDATES = {
    "current_logistic": (PASS_BASE, lambda X, y, w, g: ML.fit_logistic(X, y, w)),
    "logistic_extra": (PASS_ALL, lambda X, y, w, g: ML.fit_logistic(X, y, w)),
    "hgb_extra_raw": (PASS_ALL, lambda X, y, w, g: _hgb().fit(X, y, sample_weight=w)),
    "hgb_extra_isotonic": (PASS_ALL, fit_calibrated_hgb("isotonic")),
    "hgb_extra_sigmoid": (PASS_ALL, fit_calibrated_hgb("sigmoid")),
}
VALUE_ALL = list(range(N_VALUE + 2))
VALUE_BASE = list(range(N_VALUE))


def _hgb_reg():
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=0, early_stopping=False
    )


VALUE_CANDIDATES = {
    "mean": (VALUE_BASE, lambda X, y: ML.fit_mean(X, y), None),
    "current_ridge": (VALUE_BASE, lambda X, y: ML.fit_ridge(X, y), None),
    "hgb_basis": (VALUE_BASE, lambda X, y: _hgb_reg().fit(X, y), None),
    "hgb_context": (VALUE_ALL, lambda X, y: _hgb_reg().fit(X, y), None),
    "ridge_zone_x_context": (VALUE_ALL, lambda X, y: ML.fit_ridge(X, y), interactions),
}


def fit_pass(name, X, y, w, g):
    cols, fit = PASS_CANDIDATES[name]
    return Cols(cols, fit(X[:, cols], y, w, g))


def fit_value(name, X, y):
    cols, fit, tf = VALUE_CANDIDATES[name]
    Xc = X[:, cols]
    return Cols(cols, fit(Xc if tf is None else tf(Xc), y), tf)


# ------------------------------------------------------------------------------------------------------ evaluation


def _ll(y, p, w):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum()


def _brier(y, p, w):
    return (w * (p - y) ** 2).sum() / w.sum()


def _auc(y, p, w):
    return ML.metrics(y, p, w)["auc"]


def boot_diff(groups, fn_new, fn_old, reps=BOOT, seed=0):
    """Point estimate and match-bootstrap 95 % CI of fn_new(idx) - fn_old(idx)."""
    ids = np.unique(groups)
    where = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    allidx = np.arange(len(groups))
    vals = []
    for _ in range(reps):
        idx = np.concatenate([where[g] for g in rng.choice(ids, len(ids))])
        vals.append(fn_new(idx) - fn_old(idx))
    return {
        "delta": float(fn_new(allidx) - fn_old(allidx)),
        "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
    }


def pass_oof(P, key, names):
    X, y, w, g = P[key], P["y"], P["weight"], P["match"]
    oof = {n: np.zeros(len(y)) for n in names}
    for mid in np.unique(g):
        test = g == mid
        for n in names:
            model = fit_pass(n, X[~test], y[~test], w[~test], g[~test])
            oof[n][test] = model.predict_proba(X[test])[:, 1]
    return oof


def pass_report(P, oof, reps) -> dict:
    y, w, g = P["y"], P["weight"], P["match"]
    out = {}
    base = oof["current_logistic"]
    for n, p in oof.items():
        per = {mid: ML.metrics(y[g == mid], p[g == mid], w[g == mid]) for mid in np.unique(g)}
        r = {**ML.metrics(y, p, w), "per_league": {}, "per_match_log_loss": {k: v["log_loss"] for k, v in per.items()}}
        for lg in ("idsse", "aleague"):
            sel = np.array([league(x) == lg for x in g])
            r["per_league"][lg] = ML.metrics(y[sel], p[sel], w[sel])
        if n != "current_logistic":
            # lower log-loss / Brier is better: delta = current - candidate (> 0 = candidate better)
            r["gain_log_loss"] = boot_diff(
                g, lambda i, b=base: _ll(y[i], b[i], w[i]), lambda i, p=p: _ll(y[i], p[i], w[i]), reps
            )
            r["gain_brier"] = boot_diff(
                g, lambda i: _brier(y[i], base[i], w[i]), lambda i, p=p: _brier(y[i], p[i], w[i]), reps
            )
            r["gain_auc"] = boot_diff(
                g, lambda i, p=p: _auc(y[i], p[i], w[i]), lambda i: _auc(y[i], base[i], w[i]), reps
            )
            r["matches_better_log_loss"] = int(
                sum(r["per_match_log_loss"][k] < per_cur for k, per_cur in out["current_logistic"]["_pm"].items())
            )
        r["_pm"] = r["per_match_log_loss"]
        out[n] = r
    for r in out.values():
        r.pop("_pm")
    return out


def reg_oof(T, names):
    X, y, g = T["X_full"], T["y"], T["match"]
    oof = {n: np.zeros(len(y)) for n in names}
    for mid in np.unique(g):
        test = g == mid
        for n in names:
            oof[n][test] = fit_value(n, X[~test], y[~test]).predict(X[test])
    return oof


def _mse(y, p):
    return float(np.mean((y - p) ** 2))


def _sp(a, b):
    return float(spearmanr(a, b)[0]) if np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")


def reg_report(T, oof, reps) -> dict:
    y, g = T["y"], T["match"]
    lg = np.array([league(x) for x in g])
    out = {}
    for n, p in oof.items():
        r = {"pooled": {}, "per_league": {}}
        for name, sel in (("pooled", np.ones(len(y), bool)), ("idsse", lg == "idsse"), ("aleague", lg == "aleague")):
            ys, ps, ms = y[sel], p[sel], oof["mean"][sel]
            d = {"n": int(sel.sum()), "mse": _mse(ys, ps), "mse_skill": 1 - _mse(ys, ps) / _mse(ys, ms)}
            d["spearman"] = _sp(ps, ys) if n != "mean" else None
            if name == "pooled":
                r["pooled"] = d
            else:
                r["per_league"][name] = d
        if n not in ("mean", "current_ridge"):
            base = oof["current_ridge"]
            r["gain_mse"] = boot_diff(g, lambda i, b=base: _mse(y[i], b[i]), lambda i, p=p: _mse(y[i], p[i]), reps)
            r["gain_spearman"] = boot_diff(g, lambda i, p=p: _sp(p[i], y[i]), lambda i, b=base: _sp(b[i], y[i]), reps)
        out[n] = r
    return out


def end_to_end(tables, configs: dict, reps) -> dict:
    """Coach score of the pro's chosen pass on held-out matches (models refitted on the 26 others per config), vs
    net xG in 15 s. configs: name -> (pass candidate, value candidate, arrival candidate)."""
    mids = list(tables)
    stack = lambda key: D.stack([tables[m][key] for m in mids])
    P, V, A = stack("pass"), stack("value"), stack("arrival")
    dec = stack("decisions")
    scores = {c: np.zeros(len(dec["outcome"])) for c in configs}
    for mid in mids:
        trp, trv, tra = P["match"] != mid, V["match"] != mid, A["match"] != mid
        te = dec["match"] == mid
        cache = {}
        for c, (pn, vn, an) in configs.items():
            if ("p", pn) not in cache:
                cache[("p", pn)] = fit_pass(pn, P["X_full"][trp], P["y"][trp], P["weight"][trp], P["match"][trp])
            if ("v", vn) not in cache:
                cache[("v", vn)] = fit_value(vn, V["X_full"][trv], V["y"][trv])
            if ("a", an) not in cache:
                cache[("a", an)] = fit_value(an, A["X_full"][tra], A["y"][tra])
            pm, vm, am = cache[("p", pn)], cache[("v", vn)], cache[("a", an)]
            p = pm.predict_proba(dec["pass"][te])[:, 1]
            scores[c][te] = chosen_score(
                p, am.predict(dec["target"][te]), vm.predict(dec["loss"][te]), vm.predict(dec["now"][te])
            )
        print(f"  end-to-end {mid}", flush=True)
    y, g = dec["outcome"], dec["match"]
    lg = np.array([league(x) for x in g])
    out = {}
    for c, sc in scores.items():
        r = {
            "spearman": _sp(sc, y),
            "per_league": {k: _sp(sc[lg == k], y[lg == k]) for k in ("idsse", "aleague")},
            "n": int(len(y)),
        }
        if c != "current":
            base = scores["current"]
            r["gain_spearman"] = boot_diff(
                g, lambda i, sc=sc: _sp(sc[i], y[i]), lambda i, b=base: _sp(b[i], y[i]), reps
            )
        out[c] = r
    return out


def verdict(gains: list, leak_ok: bool = True) -> str:
    """ADOPT only when every listed gain's CI is above zero and there is no leakage risk."""
    return "ADOPT" if leak_ok and all(gg["ci"][0] > 0 for gg in gains) else "KEEP CURRENT"


def run(tables, reps=BOOT) -> dict:
    mids = list(tables)
    P = D.stack([tables[m]["pass"] for m in mids])
    t0 = time.time()
    res = {"matches": len(mids), "passes": int(len(P["y"]))}
    names = list(PASS_CANDIDATES)
    print("pass models (standard table) ...", flush=True)
    oof = pass_oof(P, "X_full", names)
    res["pass"] = pass_report(P, oof, reps)
    print(f"pass models (interceptor removed) ... {time.time() - t0:.0f} s", flush=True)
    oof_ni = pass_oof(P, "X_full_noint", names)
    res["pass_noint"] = pass_report(P, oof_ni, reps)
    for kind in ("value", "arrival"):
        print(f"{kind} models ... {time.time() - t0:.0f} s", flush=True)
        T = D.stack([tables[m][kind] for m in mids])
        res[kind] = reg_report(T, reg_oof(T, list(VALUE_CANDIDATES)), reps)
    verdicts = {}
    for n in names[1:]:
        g1, g2 = res["pass"][n]["gain_log_loss"], res["pass_noint"][n]["gain_log_loss"]
        verdicts[f"pass:{n}"] = {
            "gain_log_loss": g1,
            "gain_log_loss_noint": g2,
            "gain_brier": res["pass"][n]["gain_brier"],
            "gain_auc": res["pass"][n]["gain_auc"],
            # lane-interception and target-crowding features read defender positions near the inferred lane: a gain
            # that vanishes once the interceptor is removed is the leak, so both tables must agree
            "verdict": verdict([g1, g2, res["pass"][n]["gain_brier"], res["pass_noint"][n]["gain_brier"]]),
        }
    for kind in ("value", "arrival"):
        for n in ("hgb_basis", "hgb_context", "ridge_zone_x_context"):
            r = res[kind][n]
            verdicts[f"{kind}:{n}"] = {
                "gain_mse": r["gain_mse"],
                "gain_spearman": r["gain_spearman"],
                "verdict": verdict([r["gain_mse"]]),
            }
    res["verdicts"] = verdicts
    # end-to-end: current, the best pass candidate (lowest leak-checked log-loss), the best value / arrival candidates
    # (lowest MSE), and all of them together
    best_pass = min(names[1:], key=lambda n: res["pass_noint"][n]["log_loss"])
    best_v = min(("hgb_basis", "hgb_context", "ridge_zone_x_context"), key=lambda n: res["value"][n]["pooled"]["mse"])
    best_a = min(("hgb_basis", "hgb_context", "ridge_zone_x_context"), key=lambda n: res["arrival"][n]["pooled"]["mse"])
    configs = {
        "current": ("current_logistic", "current_ridge", "current_ridge"),
        f"pass={best_pass}": (best_pass, "current_ridge", "current_ridge"),
        f"value={best_v},arrival={best_a}": ("current_logistic", best_v, best_a),
        "all_best": (best_pass, best_v, best_a),
    }
    print(f"end-to-end {configs} ... {time.time() - t0:.0f} s", flush=True)
    res["end_to_end"] = {"configs": {k: list(v) for k, v in configs.items()}, **end_to_end(tables, configs, reps)}
    res["seconds"] = round(time.time() - t0)
    return res


EXTRA_GROUPS = {
    "receiver_motion": ("receiver_to_ball", "receiver_vx"),  # no defender involved: leak-free
    "interactions": ("dist_x_contested", "forward_x_pressure"),  # products of existing features
    "lane_and_target_defenders": ("intercept_margin", "defenders_near_target"),  # where the inferred-target leak lives
}


def followup(tables, reps=BOOT) -> dict:
    """Logistic ablation by feature group (standard and interceptor-removed tables) and the end-to-end score of the
    linear (browser-portable) candidates. Logistic / ridge only, so it runs in minutes."""
    mids = list(tables)
    P = D.stack([tables[m]["pass"] for m in mids])
    y, w, g = P["y"], P["weight"], P["match"]
    out = {"pass_ablation": {}}
    for key in ("X_full", "X_full_noint"):
        X = P[key]
        fits = {"current_logistic": PASS_BASE, "logistic_extra": PASS_ALL}
        for name, feats in EXTRA_GROUPS.items():
            fits[f"logistic+{name}"] = PASS_BASE + [N_PASS + EXTRA_FEATURES.index(f) for f in feats]
        oof = {n: np.zeros(len(y)) for n in fits}
        for mid in np.unique(g):
            te = g == mid
            for n, cols in fits.items():
                oof[n][te] = ML.fit_logistic(X[~te][:, cols], y[~te], w[~te]).predict_proba(X[te][:, cols])[:, 1]
        base = oof["current_logistic"]
        res = {}
        for n, p in oof.items():
            r = {"log_loss": _ll(y, p, w), "auc": _auc(y, p, w)}
            if n != "current_logistic":
                r["gain_log_loss"] = boot_diff(
                    g, lambda i, b=base: _ll(y[i], b[i], w[i]), lambda i, p=p: _ll(y[i], p[i], w[i]), reps
                )
            res[n] = r
        out["pass_ablation"]["standard" if key == "X_full" else "interceptor_removed"] = res
    PASS_CANDIDATES["logistic+receiver_motion"] = (
        PASS_BASE + [N_PASS + EXTRA_FEATURES.index(f) for f in EXTRA_GROUPS["receiver_motion"]],
        lambda X, y, w, g: ML.fit_logistic(X, y, w),
    )
    configs = {
        "current": ("current_logistic", "current_ridge", "current_ridge"),
        "pass=logistic_extra": ("logistic_extra", "current_ridge", "current_ridge"),
        "pass=logistic+receiver_motion": ("logistic+receiver_motion", "current_ridge", "current_ridge"),
        "value,arrival=ridge_zone_x_context": ("current_logistic", "ridge_zone_x_context", "ridge_zone_x_context"),
        "all_linear": ("logistic+receiver_motion", "ridge_zone_x_context", "ridge_zone_x_context"),
    }
    out["end_to_end_linear"] = {
        "configs": {k: list(v) for k, v in configs.items()},
        **end_to_end(tables, configs, reps),
    }
    return out


PORTABLE = {"logistic_extra", "logistic+receiver_motion", "ridge_zone_x_context"}  # linear: the browser can run them


def final_verdicts(res) -> dict:
    """Component verdict + end-to-end + portability + leak risk -> the recommendation for the shipped coach.
    A pass candidate must improve log-loss and Brier on both tables AND not lower the end-to-end coach score (CI);
    a value / arrival candidate must lower the MSE (CI) and not lower the end-to-end score; both must be linear (the
    website runs the exact formulas). HGB pass models gain far more on the interceptor-removed table than on the
    standard one: that table removes a defender from failed passes only, a label-dependent edit a flexible model can
    detect, so their leak-checked numbers are not trustworthy (leak risk)."""
    e2e = {**res.get("end_to_end", {}), **res.get("end_to_end_linear", {})}
    worse = lambda k: k in e2e and e2e[k]["gain_spearman"]["ci"][1] < 0
    out = {}
    for key, v in res["verdicts"].items():
        kind, name = key.split(":")
        reasons = []
        if v["verdict"] != "ADOPT":
            reasons.append("component gain not significant")
        if name not in PORTABLE:
            reasons.append("not portable to the browser")
        if kind == "pass":
            if name.startswith("hgb"):
                reasons.append("leak risk (gain inflated on the interceptor-removed table)")
            if worse(f"pass={name}") or (name == "logistic_extra" and worse("pass=logistic+receiver_motion")):
                reasons.append("lowers the end-to-end coach score (CI below zero)")
        elif worse("value,arrival=ridge_zone_x_context") and name == "ridge_zone_x_context":
            reasons.append("lowers the end-to-end coach score")
        out[key] = {"verdict": "KEEP CURRENT" if reasons else "ADOPT", "reasons": reasons}
    return out


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reps", type=int, default=BOOT)
    ap.add_argument("--out", type=Path, default=DOCS / "coach_model_lab.json")
    ap.add_argument("--followup", action="store_true", help="only the linear ablation + end-to-end, merged into --out")
    args = ap.parse_args()
    path = cache_path()
    tables = build_cache(path) if args.rebuild or not path.exists() else pickle.loads(path.read_bytes())
    if args.followup:
        res = json.loads(args.out.read_text(encoding="utf-8"))
        res.update(_clean(followup(tables, args.reps)))
        res["final_verdicts"] = final_verdicts(res)
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps({k: res[k] for k in ("pass_ablation", "end_to_end_linear")}, indent=1))
        return
    res = run(tables, args.reps)
    res.update(followup(tables, args.reps))
    res["final_verdicts"] = final_verdicts(res)
    from .stamp import generated_by

    res = {
        "generated_by": generated_by(),
        "extra_pass_features": list(EXTRA_FEATURES),
        "context_features": list(CONTEXT_FEATURES),
        **res,
    }
    args.out.write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")
    for k, v in res["final_verdicts"].items():
        print(k, v["verdict"], "; ".join(v["reasons"]))
    print("end-to-end", {k: v.get("spearman") for k, v in res["end_to_end"].items() if k != "configs"})
    print("wrote", args.out)


if __name__ == "__main__":
    main()

"""The coach's cut-back blind spot: diagnosis, the fix (an arrival value for passes) and its held-out check.

    python -m coach.cutback        # from 20-football-analytics/, minutes; -> docs/coach_cutback_results.json

Needs only what the benchmark leaves behind (the derived tables of the 27 matches, cross_league.load_all, and the cached
pass rows of the IDSSE and A-League benchmarks), not the raw tracking. Everything is leave-one-match-out within a league.

Diagnosis (see the JSON): the pass model is calibrated on cut-backs and the shot option is not the cause. The value of the
target is: a cut-back is worth 4-5 times the possession value V of its target (V is trained on every second of possession,
and the box seconds are mostly circulation), so p * V(target) - V(now) is below zero for any p < 1. A ball that ARRIVES in
the box is a different state, so the fix is a second ridge on the same basis, trained on completed passes only:
datasets.arrival_table, used for V(target) of pass options (advice.options). Everything else keeps V.

Ship rule: held-out error of the arrival value not worse than V on the completed passes (match-bootstrap 95 % CI),
the pass and possession-value models unchanged, and the cut-backs' median percentile up.
`rescore_rows` re-scores a benchmark's cached rows without the tracking: the coach's score of the chosen pass changes only
through V(target), so new = old + p * (V_arrival - V_possession) at the pass target.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss

from . import benchmark as B
from . import cross_league as CL
from . import datasets as D
from . import features as F
from . import models as ML

DOCS = Path(__file__).resolve().parents[1] / "docs"
LEAGUES = ("idsse", "skillcorner")
SET_OF = {"idsse": "idsse", "skillcorner": "aleague"}
REGIONS = {
    "central box (x >= 40, |y| < 6), any origin": lambda a: (a["x1"] >= 40) & (a["y1"] < 6),
    "  from wide (x0 >= 36, |y0| >= 12)": lambda a: (a["x1"] >= 40) & (a["y1"] < 6) & (a["x0"] >= 36) & (a["y0"] >= 12),
    "other box (x >= 36, |y| < 20.16)": lambda a: (
        (a["x1"] >= 36) & (a["y1"] < 20.16) & ~((a["x1"] >= 40) & (a["y1"] < 6))
    ),
    "outside the box, x >= 20": lambda a: (a["x1"] >= 20) & ~((a["x1"] >= 36) & (a["y1"] < 20.16)),
    "own half, x < 0": lambda a: a["x1"] < 0,
}


def match_bootstrap(groups, fn, reps=500, seed=0):
    """95 % interval of fn(index array) resampling whole matches with replacement."""
    ids = np.unique(groups)
    where = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    vals = [fn(np.concatenate([where[g] for g in rng.choice(ids, len(ids))])) for _ in range(reps)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def lomo(groups, X, y, fit, mask=None, predict=None):
    """Out-of-fold predictions: the model of held-out match g is fitted on the other matches (rows in `mask`)."""
    mask = np.ones(len(y), bool) if mask is None else mask
    out = np.zeros(len(y))
    for g in np.unique(groups):
        test = groups == g
        model = fit(X[~test & mask], y[~test & mask])
        out[test] = model.predict_proba(X[test])[:, 1] if predict == "proba" else model.predict(X[test])
    return out


def rescore_rows(rows, arrival):
    """Benchmark rows with the arrival value in place of V(target). `arrival` maps (match, event) -> V_arrival of the
    chosen pass's target. The coach fields are replaced (the former ones are kept as *_v3); a row already rescored is
    rescored again from its _v3 values."""
    out = []
    for r in rows:
        score, dv, after = (r.get(f"{k}_v3", r[k]) for k in ("coach_score", "coach_dv", "coach_after"))
        v_target = r["v_now"] + dv
        gain = arrival[(r["match"], r["event"])] - v_target
        out.append(
            {
                **r,
                "coach_score_v3": score,
                "coach_dv_v3": dv,
                "coach_after_v3": after,
                "coach_score": score + r["coach_p"] * gain,
                "coach_dv": dv + gain,
                "coach_after": after + r["coach_p"] * gain,
            }
        )
    return out


def pct(x):
    return (rankdata(x) - 0.5) / len(x)


def load():
    idsse, sc = CL.load_all()
    CL.xg_proxy(idsse, sc)  # fills the A-League shots' xG
    parts = {"idsse": idsse, "skillcorner": sc}
    rows = {lg: json.loads(B.paths(SET_OF[lg])[0].read_text(encoding="utf-8"))["rows"] for lg in LEAGUES}
    return parts, rows


def league_table(parts, rows):
    """Stacked pass table of a league with the cached benchmark fields of each pass (same rows, joined on the
    match and event) and the coordinates of the pass for the regions."""
    pt = D.stack([d["pass"] for d in parts])
    by = {(r["match"], r["event"]): r for r in rows}
    got = [by[(str(m), int(e))] for m, e in zip(pt["match"], pt["event_idx"])]
    for k in ("x0", "y0", "x1", "y1", "success", "net_xg", "goal_15s", "coach_p", "coach_score", "coach_dv", "v_now"):
        pt[k] = np.array([r[k] for r in got], float)
    pt["y0"], pt["y1"] = np.abs(pt["y0"]), np.abs(pt["y1"])
    pt["key"] = [(r["match"], r["event"]) for r in got]
    pt["match"] = pt["match"].astype(str)
    return pt


def is_cutback(pt, completed=True):
    cut = (pt["x0"] >= B.CUTBACK_X) & (pt["y0"] >= B.CUTBACK_WIDE[0]) & (pt["y0"] <= B.CUTBACK_WIDE[1])
    cut &= (pt["x1"] >= B.CUTBACK_X) & (pt["y1"] < B.CUTBACK_CENTRAL)
    return cut & (pt["success"] == 1) if completed else cut


def arrival_oof(pt, B_target):
    """Held-out arrival values at every pass target (model fitted on the completed passes of the other matches)."""
    return lomo(pt["match"], B_target, pt["net_xg"], ML.fit_ridge, mask=pt["success"] == 1)


def possession_oof(vt, B_target, groups):
    """V (possession model, trained on every second of possession of the other matches) at the pass targets."""
    out = np.zeros(len(groups))
    for g in np.unique(groups):
        model = ML.fit_ridge(vt["X"][vt["match"] != g], vt["y"][vt["match"] != g])
        out[groups == g] = model.predict(B_target[groups == g])
    return out


def diagnose(pt, vt, p_oof, v_oof_seconds):
    cb, w, y = is_cutback(pt, completed=False), pt["weight"], pt["y"]
    done = is_cutback(pt)
    seconds = {
        "central box x >= 40, |y| < 6": (vt["X"][:, 0] >= 40) & (vt["X"][:, 1] < 6),
        "wide byline x >= 46, 12 <= |y| <= 20": (vt["X"][:, 0] >= 46) & (vt["X"][:, 1] >= 12) & (vt["X"][:, 1] <= 20),
    }
    p_t, v_n = pt["coach_p"], pt["v_now"]
    return {
        "pass_model_on_cutback_shaped_passes": {
            "n": int(cb.sum()),
            "n_completed": int(done.sum()),
            "mean_predicted": float(np.average(p_oof[cb], weights=w[cb])),
            "observed_completion_weighted": float(np.average(y[cb], weights=w[cb])),
            "observed_completion_raw": float(y[cb].mean()),
            "all_passes_predicted": float(np.average(p_oof, weights=w)),
            "all_passes_observed": float(np.average(y, weights=w)),
        },
        "possession_value_by_zone": {
            k: {"n": int(s.sum()), "predicted": float(v_oof_seconds[s].mean()), "observed": float(vt["y"][s].mean())}
            for k, s in seconds.items()
        },
        "completed_cutbacks": {
            "n": int(done.sum()),
            "median_p": float(np.median(p_t[done])),
            "median_v_now": float(np.median(v_n[done])),
            "median_v_target": float(np.median((pt["coach_dv"] + v_n)[done])),
            "realised_net_xg_15s": float(pt["net_xg"][done].mean()),
            "goal_15s": float(pt["goal_15s"][done].mean()),
        },
    }


def by_region(pt, v_poss, v_arr):
    out, comp = {}, pt["success"] == 1
    for name, rule in REGIONS.items():
        s = rule(pt) & comp
        out[name.strip()] = {
            "n": int(s.sum()),
            "realised_net_xg": float(pt["net_xg"][s].mean()),
            "v_possession": float(v_poss[s].mean()),
            "v_arrival": float(v_arr[s].mean()),
        }
    return out


def arrival_vs_possession(pt, v_poss, v_arr):
    """Held-out squared error of the two values of the pass target against the net xG that followed completed passes."""
    idx = np.flatnonzero(pt["success"] == 1)
    y, g = pt["net_xg"], pt["match"]
    mse = lambda v, i: float(np.mean((y[i] - v[i]) ** 2))
    diff = lambda i: mse(v_poss, i) - mse(v_arr, i)
    r2 = lambda v: float(1 - np.sum((y[idx] - v[idx]) ** 2) / np.sum((y[idx] - y[idx].mean()) ** 2))
    return {
        "n_completed": int(len(idx)),
        "mse_possession_v": mse(v_poss, idx),
        "mse_arrival": mse(v_arr, idx),
        "mse_gain": diff(idx),
        "mse_gain_ci95": match_bootstrap(g[idx], lambda s: diff(idx[s])),
        "r2_possession_v": r2(v_poss),
        "r2_arrival": r2(v_arr),
        "mean_observed": float(y[idx].mean()),
        "mean_possession_v": float(v_poss[idx].mean()),
        "mean_arrival": float(v_arr[idx].mean()),
    }


def variants_not_shipped(pt, vt, parts):
    """Other fixes tried, each on held-out data; none passes the ship rule (the pass model is calibrated on cut-backs, the
    possession value is calibrated by zone, the narrow-angle shot is within noise of the provider's xG)."""
    out = {}
    X, y, w, g = pt["X"], pt["y"], pt["weight"], pt["match"]
    tx = X[:, F.PASS_FEATURES.index("target_x")]
    vb = D.arrival_rows(X)
    ga, aty = vb[:, F.VALUE_FEATURES.index("goal_angle")], vb[:, F.VALUE_FEATURES.index("abs_y")]
    X2 = np.column_stack([X, ga, ((tx >= 40) & (aty < 6)).astype(float), ((tx >= 36) & (aty < 20.16)).astype(float)])

    def fit_w(Xa, ya, wa=None):
        return ML.fit_logistic(Xa, ya, wa)

    def pass_oof(Xm):
        o = np.zeros(len(y))
        for m in np.unique(g):
            t = g == m
            o[t] = fit_w(Xm[~t], y[~t], w[~t]).predict_proba(Xm[t])[:, 1]
        return o

    o1, o2 = pass_oof(X), pass_oof(X2)
    ll = lambda o, i: log_loss(y[i], np.clip(o[i], 1e-6, 1 - 1e-6), sample_weight=w[i])
    br = lambda o, i: brier_score_loss(y[i], o[i], sample_weight=w[i])
    allr = np.arange(len(y))
    cb = is_cutback(pt, completed=False)
    out["pass_features_plus_target_angle_central_box"] = {
        "log_loss_gain": ll(o1, allr) - ll(o2, allr),
        "log_loss_gain_ci95": match_bootstrap(g, lambda s: ll(o1, s) - ll(o2, s)),
        "brier_gain": br(o1, allr) - br(o2, allr),
        "brier_gain_ci95": match_bootstrap(g, lambda s: br(o1, s) - br(o2, s)),
        "cutback_shaped_mean_p": [float(np.average(o[cb], weights=w[cb])) for o in (o1, o2)],
        "cutback_shaped_observed": float(np.average(y[cb], weights=w[cb])),
    }
    ax, ay = vt["X"][:, 0], vt["X"][:, 1]
    Xv2 = np.column_stack([vt["X"], ((ax >= 40) & (ay < 6)).astype(float), np.exp(-ay / 6.0) * (ax >= 36)])
    gv, yv = vt["match"], vt["y"]
    v1 = lomo(gv, vt["X"], yv, ML.fit_ridge)
    v2 = lomo(gv, Xv2, yv, ML.fit_ridge)
    mse = lambda v, i: float(np.mean((yv[i] - v[i]) ** 2))
    out["possession_value_plus_central_box_terms"] = {
        "mse_gain": mse(v1, np.arange(len(yv))) - mse(v2, np.arange(len(yv))),
        "mse_gain_ci95": match_bootstrap(gv, lambda s: mse(v1, s) - mse(v2, s), reps=200),
    }
    return out


def shot_check(parts):
    """The shot option's goal probability at narrow angles against the provider / proxy xG and the goals of real shots,
    held out by match (the site's recipe: Bundesliga shots of the other matches plus every A-League shot)."""
    st = D.stack([d["shot"] for d in parts["idsse"]])
    ss = D.stack([d["shot"] for d in parts["skillcorner"]])
    ok = lambda s: ~s["penalty"] & np.isfinite(s["X"]).all(axis=1)
    st, ss = {k: v[ok(st)] for k, v in st.items()}, {k: v[ok(ss)] for k, v in ss.items()}
    oof = np.zeros(len(st["y"]))
    for m in np.unique(st["match"]):
        te = st["match"] == m
        fit = ML.fit_logistic(np.vstack([st["X"][~te], ss["X"]]), np.r_[st["y"][~te], ss["y"]])
        oof[te] = fit.predict_proba(st["X"][te])[:, 1]
    out = {}
    for lg, s, p in (("idsse", st, oof), ("skillcorner", ss, None)):
        if p is None:  # A-League: model fitted on the other leagues' matches and the other A-League matches
            p = np.zeros(len(s["y"]))
            for m in np.unique(s["match"]):
                te = s["match"] == m
                p[te] = ML.fit_logistic(np.vstack([s["X"][~te], st["X"]]), np.r_[s["y"][~te], st["y"]]).predict_proba(
                    s["X"][te]
                )[:, 1]
        rows = []
        for lo, hi in ((0.0, 0.25), (0.25, 0.4), (0.4, 0.7), (0.7, 3.5)):
            sel = (s["X"][:, 1] >= lo) & (s["X"][:, 1] < hi)
            rows.append(
                {
                    "goal_angle_rad": [lo, hi],
                    "n": int(sel.sum()),
                    "model_p": float(p[sel].mean()),
                    "provider_or_proxy_xg": float(s["xg"][sel].mean()),
                    "goal_rate": float(s["y"][sel].mean()),
                    "goals": int(s["y"][sel].sum()),
                }
            )
        out[lg] = rows
    return out


def evaluate_league(lg, parts, rows):
    pt = league_table(parts[lg], rows[lg])
    vt = CL.relabel(parts[lg])
    vt["match"] = vt["match"].astype(str)
    B_t = D.arrival_rows(pt["X"])
    # pass model held out (weighted logistic, the coach's recipe)
    p_oof = np.zeros(len(pt["y"]))
    for m in np.unique(pt["match"]):
        te = pt["match"] == m
        p_oof[te] = ML.fit_logistic(pt["X"][~te], pt["y"][~te], pt["weight"][~te]).predict_proba(pt["X"][te])[:, 1]
    v_poss = possession_oof(vt, B_t, pt["match"])
    v_arr = arrival_oof(pt, B_t)
    v_seconds = lomo(vt["match"], vt["X"], vt["y"], ML.fit_ridge)
    arrival = dict(zip(pt["key"], v_arr))
    new_rows = rescore_rows(rows[lg], arrival)
    before, after = B.cutbacks(rows[lg]), B.cutbacks(new_rows)
    return {
        "diagnosis": diagnose(pt, vt, p_oof, v_seconds),
        "by_target_region": by_region(pt, v_poss, v_arr),
        "arrival_vs_possession_value": arrival_vs_possession(pt, v_poss, v_arr),
        "cutbacks_before": before,
        "cutbacks_after": after,
        "variants_not_shipped": variants_not_shipped(pt, vt, parts),
    }, new_rows


def rescored_rows(lg):
    """The cached benchmark rows of a league ("idsse" / "skillcorner") re-scored with the held-out arrival value."""
    parts, rows = load()
    pt = league_table(parts[lg], rows[lg])
    arrival = dict(zip(pt["key"], arrival_oof(pt, D.arrival_rows(pt["X"]))))
    return rescore_rows(rows[lg], arrival)


def main():
    parts, rows = load()
    res, rescored = {}, {}
    for lg in LEAGUES:
        res[lg], rescored[lg] = evaluate_league(lg, parts, rows)
        a, b = res[lg]["cutbacks_before"], res[lg]["cutbacks_after"]
        print(
            lg,
            "cut-back median percentile (coach)",
            a["median_pct"]["coach_score"],
            "->",
            b["median_pct"]["coach_score"],
        )
        print(lg, "arrival vs possession", res[lg]["arrival_vs_possession_value"])
    res["shot_option_at_narrow_angles"] = shot_check(parts)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_cutback_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Cross-league test and combined training: IDSSE (Bundesliga, 25 fps optical) + SkillCorner (A-League, 10 fps broadcast).

    python -m coach.skillcorner       # stream the 20 A-League matches, write cv-data/skillcorner/derived/*.pkl
    python -m coach.cross_league      # all numbers -> docs/coach_skillcorner_results.json + two charts

Every model is scored on held-out matches. For a held-out match, three training sets are compared (lomo_rules):
"same" = the other matches of its league (plain leave-one-match-out), "other" = every match of the other league
(the cross-league test), "all" = every other match of both leagues (combined training), and "all+league" = the same
with a league indicator column. Models stay in their browser-portable forms (logistic / ridge on the same features).

SkillCorner has no expected goals. Value labels there use a documented proxy: a logistic model fitted on the IDSSE
shots to reproduce the provider's xG from the coach's own shot features (distance, angle, defenders in the cone);
its quality is measured on held-out IDSSE matches (vs the provider xG) and on the A-League goals.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from . import datasets as D
from . import features as F
from . import models as ML
from .sync import event_xy

DOCS = Path(__file__).resolve().parents[1] / "docs"
LEAGUES = ("idsse", "skillcorner")
RULES = ("same", "other", "all", "all+league")
PENALTY_XG_DEFAULT = 0.76


# ------------------------------------------------------------------------------------------ per match


def league_of(m) -> str:
    return "skillcorner" if str(m.meta.get("source", "")).startswith("SkillCorner") else "idsse"


def shot_rows(m) -> dict:
    """Every shot with a team: features at its frame (or, when sync dropped it, at its nominal frame), time on the
    tracking clock (for the value labels), goal, penalty flag, provider xG (NaN when missing) and placed flag."""
    X, half, t, team, goal, pen, xg, placed = [], [], [], [], [], [], [], []
    for e in m.events:
        if e["type"] != "shot" or e.get("half") is None or e["team"] not in (0, 1):
            continue
        fr = e["frame"] if e.get("frame") is not None else e.get("nominal")
        if fr is not None and np.isfinite(e["x"]):
            s = F.state(m, e["half"], fr, e["team"])
            f = F.shot_features(event_xy(m, e) * s["direction"], s["def"])
            X.append([f[k] for k in F.SHOT_FEATURES])
        else:
            X.append([np.nan] * len(F.SHOT_FEATURES))
        half.append(e["half"])
        t.append(D.when(m, e))
        team.append(e["team"])
        goal.append(int(bool(e.get("goal"))))
        pen.append(e.get("set_piece") == "Penalty")
        xg.append(e["xg"] if e.get("xg") is not None else np.nan)
        placed.append(e.get("frame") is not None)
    n = len(half)
    return {
        "X": np.array(X, float).reshape(n, len(F.SHOT_FEATURES)),
        "half": np.array(half, int),
        "t": np.array(t, float),
        "team": np.array(team, int),
        "y": np.array(goal, int),
        "goal": np.array(goal, int),
        "penalty": np.array(pen, bool),
        "xg": np.array(xg, float),
        "placed": np.array(placed, bool),
        "match": np.array([m.match_id] * n),
    }


def derive(m) -> dict:
    """Everything the evaluation needs from one match (small: no tracking arrays)."""
    pt = D.pass_table(m)
    det = getattr(m, "detected", None)
    ev = [m.events[i] for i in pt["event_idx"]]

    def seen(e, who):
        return bool(det[e["half"]][e["frame"], who]) if det is not None else True

    def count(e):
        if det is not None:
            return int(det[e["half"]][e["frame"]].sum())
        return int((~np.isnan(m.halves[e["half"]].xy[e["frame"], :, 0])).sum())

    pt["xpass"] = np.array([np.nan if e.get("xpass") is None else e["xpass"] for e in ev], float)
    pt["carrier_detected"] = np.array([seen(e, e["player"]) for e in ev], bool)
    pt["target_detected"] = np.array([seen(e, t) for e, t in zip(ev, pt["target_idx"])], bool)
    pt["detected_players"] = np.array([count(e) for e in ev], int)
    vt = D.value_table(m)
    out = {
        "league": league_of(m),
        "match": m.match_id,
        "pass": pt,
        "value": {k: vt[k] for k in ("X", "half", "t", "team", "match")},
        "shot": shot_rows(m),
        "sync": m.meta.get("sync"),
    }
    if det is not None:
        from .skillcorner import detection_report

        out["detection"] = detection_report(m)
    return out


def value_labels(value: dict, shots: dict, xg: np.ndarray, horizon: float = D.HORIZON) -> np.ndarray:
    """Net xG (team in possession minus opponent) of the shots in (t, t + horizon] of the same half, per value row.
    `value` and `shots` belong to one match."""
    y = np.zeros(len(value["t"]))
    for hi in np.unique(value["half"]):
        sv, ss = value["half"] == hi, shots["half"] == hi
        if not ss.any():
            continue
        T = value["t"][sv][:, None]
        st = shots["t"][ss][None, :]
        within = (st > T) & (st <= T + horizon)
        sign = np.where(shots["team"][ss][None, :] == value["team"][sv][:, None], 1.0, -1.0)
        y[sv] = (within * sign * xg[ss][None, :]).sum(axis=1)
    return y


def fit_soft_logistic(X, p):
    """Logistic regression on probability targets (each row twice: y = 1 with weight p, y = 0 with weight 1 - p)."""
    X = np.asarray(X, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    return ML.fit_logistic(np.vstack([X, X]), np.r_[np.ones(len(p)), np.zeros(len(p))].astype(int), np.r_[p, 1 - p])


# ----------------------------------------------------------------------------------------- evaluation


def lomo_rules(table, league, fitters: dict, rules=RULES, kind="clf", weighted=True, return_oof=False):
    """Leave-one-match-out where the training set of a held-out match depends on the rule: "same" (its league),
    "other" (the other league), "all" (both), "all+league" (both, with a league indicator column). Metrics are
    pooled per held-out league: res[rule][fitter][league] (league = the values of `league`, e.g. 0 / 1)."""
    X, y, groups = table["X"], table["y"], table["match"]
    w = table.get("weight") if (weighted and kind == "clf") else None
    league = np.asarray(league)
    res, oofs = {}, {}
    for rule in rules:
        Xr = np.column_stack([X, league]) if rule == "all+league" else X
        res[rule], oofs[rule] = {}, {}
        for name, fit in fitters.items():
            oof = np.full(len(y), np.nan)
            for g in np.unique(groups):
                test = groups == g
                lg = league[test][0]
                train = ~test & {"same": league == lg, "other": league != lg}.get(rule, np.ones(len(y), bool))
                if not train.any() or len(np.unique(y[train])) < 2 and kind == "clf":
                    continue
                model = fit(Xr[train], y[train]) if w is None else fit(Xr[train], y[train], w[train])
                oof[test] = model.predict_proba(Xr[test])[:, 1] if kind == "clf" else model.predict(Xr[test])
            per = {}
            for lg in np.unique(league):
                sel = (league == lg) & np.isfinite(oof)
                if not sel.any():
                    continue
                if kind == "clf":
                    ww = None if w is None else w[sel]
                    per[lg.item() if hasattr(lg, "item") else lg] = {
                        **ML.metrics(y[sel], oof[sel], ww),
                        "mean_pred": float(np.average(oof[sel], weights=ww)),
                    }
                else:
                    per[lg.item() if hasattr(lg, "item") else lg] = ML.reg_metrics(y[sel], oof[sel])
            res[rule][name], oofs[rule][name] = per, oof
        if kind == "reg" and "mean" in res[rule]:
            for lg, base in res[rule]["mean"].items():
                for r in res[rule].values():
                    if lg in r:
                        r[lg]["mse_skill"] = float(1 - r[lg]["rmse"] ** 2 / base["rmse"] ** 2)
    return (res, oofs) if return_oof else res


def _named(res):
    """League 0 / 1 -> names, for the JSON."""
    return {r: {f: {LEAGUES[k]: v for k, v in per.items()} for f, per in fs.items()} for r, fs in res.items()}


def _stack(parts):
    return D.stack(parts) if parts else {}


def load_all():
    from . import skillcorner as SC
    from .match import load
    from .prepare import MATCHES
    from .prepare import OUT as IDSSE_OUT

    SC.DERIVED.mkdir(parents=True, exist_ok=True)
    idsse = []
    for mid in MATCHES:  # cached: the IDSSE tables take a few minutes to build
        path = SC.DERIVED / f"idsse_{mid}.pkl"
        if not path.exists():
            path.write_bytes(pickle.dumps(derive(load(IDSSE_OUT / mid))))
        idsse.append(pickle.loads(path.read_bytes()))
    sc = [pickle.loads((SC.DERIVED / f"{mid}.pkl").read_bytes()) for mid in SC.MATCHES]
    return idsse, sc


def skillcorner_shots(derived=None, ids=None) -> tuple:
    """Open-play shots of the cached SkillCorner tables with all shot features finite, as a training table
    (X, y = goal, match) and the match ids read. Penalties are out, like in every shot model of the coach."""
    from . import skillcorner as SC

    derived = SC.DERIVED if derived is None else Path(derived)
    ids = list(SC.MATCHES if ids is None else ids)
    rows = []
    for mid in ids:
        s = pickle.loads((derived / f"{mid}.pkl").read_bytes())["shot"]
        keep = ~s["penalty"] & np.isfinite(s["X"]).all(axis=1)
        rows.append({"X": s["X"][keep], "y": s["y"][keep], "match": s["match"][keep]})
    return D.stack(rows), ids


def xg_proxy(idsse, sc) -> dict:
    """Fit the provider-xG emulator on IDSSE shots; check it on held-out IDSSE matches and on A-League goals; fill
    the SkillCorner shots' xG with it (penalties: the mean provider xG of IDSSE penalties)."""
    st = _stack([d["shot"] for d in idsse])
    ok = ~st["penalty"] & np.isfinite(st["X"]).all(axis=1) & np.isfinite(st["xg"])
    pen = st["penalty"] & np.isfinite(st["xg"])
    pen_xg = float(st["xg"][pen].mean()) if pen.any() else PENALTY_XG_DEFAULT
    oof = np.full(len(st["y"]), np.nan)
    for g in np.unique(st["match"]):
        test, train = ok & (st["match"] == g), ok & (st["match"] != g)
        oof[test] = fit_soft_logistic(st["X"][train], st["xg"][train]).predict_proba(st["X"][test])[:, 1]
    final = fit_soft_logistic(st["X"][ok], st["xg"][ok])
    out = {
        "fit_on": "IDSSE open-play shots",
        "n_idsse": int(ok.sum()),
        "penalty_xg": pen_xg,
        "weights": ML.export_logistic(final, F.SHOT_FEATURES),
        "idsse_lomo": {
            "pearson_vs_provider_xg": float(np.corrcoef(oof[ok], st["xg"][ok])[0, 1]),
            "mae_vs_provider_xg": float(np.mean(np.abs(oof[ok] - st["xg"][ok]))),
            "proxy_vs_goals": ML.metrics(st["y"][ok], oof[ok]),
            "provider_vs_goals": ML.metrics(st["y"][ok], st["xg"][ok]),
        },
    }
    fallback = float(np.mean(oof[ok]))  # a shot without a usable location: the mean open-play proxy xG
    # IDSSE shots get their held-out proxy too (for the label check): penalties keep the provider's xG
    start = 0
    for d in idsse:
        s = d["shot"]
        k = len(s["y"])
        s["xg_proxy"] = np.where(np.isfinite(oof[start : start + k]), oof[start : start + k], s["xg"])
        start += k
    for d in sc:
        s = d["shot"]
        x = np.full(len(s["y"]), fallback)
        fin = np.isfinite(s["X"]).all(axis=1)
        if fin.any():
            x[fin] = final.predict_proba(s["X"][fin])[:, 1]
        x[s["penalty"]] = pen_xg
        s["xg_proxy"] = x
        s["xg"] = x
    ss = _stack([d["shot"] for d in sc])
    op = ~ss["penalty"] & np.isfinite(ss["X"]).all(axis=1)
    out["skillcorner"] = {
        "shots": int(len(ss["y"])),
        "open_play_with_features": int(op.sum()),
        "penalties": int(ss["penalty"].sum()),
        "goals_from_shots": int(ss["y"].sum()),
        "sum_proxy_xg": float(ss["xg"].sum()),
        "proxy_vs_goals_open_play": ML.metrics(ss["y"][op], ss["xg"][op]),
    }
    return out


def relabel(parts, xg_key="xg", horizon=D.HORIZON):
    """Value table of several matches with labels from each match's shot xG (provider or proxy)."""
    rows = []
    for d in parts:
        s = d["shot"]
        xg = np.where(np.isfinite(s[xg_key]), s[xg_key], D.XG_DEFAULT)
        rows.append({**d["value"], "y": value_labels(d["value"], s, xg, horizon)})
    return D.stack(rows)


def pass_cross_detail(pi, ps) -> dict:
    """Models fitted on every IDSSE pass, scored on every SkillCorner pass: calibration, per match, subsets."""
    i = F.PASS_FEATURES.index
    out = {}
    fits = {
        "logistic": ML.fit_logistic(pi["X"], pi["y"], pi["weight"]),
        "logistic_interceptor_removed": ML.fit_logistic(pi["X_noint"], pi["y"], pi["weight"]),
        "target_control_only": ML.column(i("target_control"))(pi["X"], pi["y"], pi["weight"]),
    }
    y, w = ps["y"], ps["weight"]
    seen = ps["carrier_detected"] & ps["target_detected"]
    many = ps["detected_players"] >= np.median(ps["detected_players"])
    for name, model in fits.items():
        p = model.predict_proba(ps["X"])[:, 1]
        per = {
            str(g): ML.metrics(y[ps["match"] == g], p[ps["match"] == g], w[ps["match"] == g])
            for g in np.unique(ps["match"])
        }
        out[name] = {
            **ML.metrics(y, p, w),
            "mean_pred": float(np.average(p, weights=w)),
            "reliability": ML.reliability(y, p, w=w),
            "carrier_and_target_detected": ML.metrics(y[seen], p[seen], w[seen]),
            "many_detected": ML.metrics(y[many], p[many], w[many]),
            "few_detected": ML.metrics(y[~many], p[~many], w[~many]),
            "auc_range_per_match": [min(v["auc"] for v in per.values()), max(v["auc"] for v in per.values())],
            "per_match": per,
        }
    has = np.isfinite(ps["xpass"])
    out["skillcorner_xpass_reference"] = {
        **ML.metrics(y[has], ps["xpass"][has], w[has]),
        "rows": int(has.sum()),
        "logistic_same_rows": ML.metrics(y[has], fits["logistic"].predict_proba(ps["X"][has])[:, 1], w[has]),
    }
    out["subset_sizes"] = {
        "all": int(len(y)),
        "carrier_and_target_detected": int(seen.sum()),
        "many_detected": int(many.sum()),
    }
    out["coefficients"] = {
        "idsse": ML.export_logistic(fits["logistic"], F.PASS_FEATURES),
        "skillcorner": ML.export_logistic(ML.fit_logistic(ps["X"], ps["y"], ps["weight"]), F.PASS_FEATURES),
    }
    return out


def _pyplot():
    """matplotlib only for the charts (the tests and CI import this module without it)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def reliability_chart(curves, path, title):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (y, p, w) in curves.items():
        pts = [(b["mean_pred"], b["observed"]) for b in ML.reliability(y, p, w=w) if b["n"] >= 30]
        ax.plot(*zip(*pts), "o-", label=f"{name} (Brier {ML.metrics(y, p, w)['brier']:.3f})")
    ax.set(xlabel="predicted", ylabel="observed (true completion rate)", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def rules_chart(res_pass, res_value, path):
    """Held-out log-loss (pass) and MSE skill (value) per held-out league and training set."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = {"same": "own league", "other": "other league", "all": "both", "all+league": "both + indicator"}
    width = 0.2
    for ax, (res, model, key, ylabel) in zip(
        axes,
        (
            (res_pass, "logistic", "log_loss", "pass log-loss (lower is better)"),
            (res_value, "ridge_basis", "mse_skill", "value MSE skill vs mean (higher is better)"),
        ),
    ):
        for k, rule in enumerate(RULES):
            vals = [res[rule][model].get(lg, {}).get(key, np.nan) for lg in (0, 1)]
            ax.bar(np.arange(2) + (k - 1.5) * width, vals, width, label=labels[rule])
        ax.set_xticks(range(2), ["held-out Bundesliga", "held-out A-League"])
        ax.set_ylabel(ylabel)
        if key == "log_loss":
            ax.set_ylim(0.3, None)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    idsse, sc = load_all()
    proxy = xg_proxy(idsse, sc)
    print("xg proxy", json.dumps({k: proxy[k] for k in ("idsse_lomo", "skillcorner")}, default=float)[:1500])
    parts = idsse + sc
    league = lambda key: np.concatenate([np.full(len(d[key]["match"]), LEAGUES.index(d["league"])) for d in parts])
    # ------------------------------------------------------------------ pass model
    pt = D.stack([d["pass"] for d in parts])
    i = F.PASS_FEATURES.index
    pass_fitters = {
        "logistic": ML.fit_logistic,
        "boosting": ML.fit_boosting,
        "distance_only": ML.column(i("distance")),
        "target_control_only": ML.column(i("target_control")),
        "base_rate": ML.fit_base_rate,
    }
    lp = league("pass")
    res_pass, oof_pass = lomo_rules(pt, lp, pass_fitters, return_oof=True)
    res_noint = lomo_rules({**pt, "X": pt["X_noint"]}, lp, {"logistic": ML.fit_logistic}, ("same", "other", "all"))
    pi, ps = D.stack([d["pass"] for d in idsse]), D.stack([d["pass"] for d in sc])
    detail = pass_cross_detail(pi, ps)
    for rule in RULES:
        print(
            "pass",
            rule,
            {
                n: {LEAGUES[k]: (round(v["log_loss"], 4), v["auc"] and round(v["auc"], 3)) for k, v in r.items()}
                for n, r in res_pass[rule].items()
            },
        )
    # ------------------------------------------------------------------ value model (net xG in 15 s)
    vt = D.stack([relabel([d]) for d in parts])
    lv = league("value")
    value_fitters = {
        "ridge_basis": ML.fit_ridge,
        "goal_distance_only": ML.column_reg(F.VALUE_FEATURES.index("goal_dist")),
        "xt_grid": ML.fit_xt_grid,
        "mean": ML.fit_mean,
    }
    res_value = lomo_rules(vt, lv, value_fitters, kind="reg")
    # label check: IDSSE with proxy-xG labels (as on SkillCorner) instead of the provider's xG
    vt_proxy = relabel(idsse, xg_key="xg_proxy")
    res_value_proxy = ML.lomo_reg(vt_proxy, {"ridge_basis": ML.fit_ridge, "mean": ML.fit_mean})
    print(
        "value, IDSSE with proxy labels",
        {k: (round(v["rmse"], 5), round(v["mse_skill"], 4)) for k, v in res_value_proxy.items()},
    )
    for rule in RULES:
        print(
            "value",
            rule,
            {
                n: {
                    LEAGUES[k]: (
                        round(v["rmse"], 5),
                        round(v.get("mse_skill", 0), 4),
                        v["spearman"] and round(v["spearman"], 3),
                    )
                    for k, v in r.items()
                }
                for n, r in res_value[rule].items()
            },
        )
    # ------------------------------------------------------------------ shot model (goals; penalties out)
    st = D.stack([d["shot"] for d in parts])
    ls = league("shot")
    keep = ~st["penalty"] & np.isfinite(st["X"]).all(axis=1)
    st = {k: v[keep] for k, v in st.items() if isinstance(v, np.ndarray)}
    res_shot = lomo_rules(
        st,
        ls[keep],
        {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate},
        ("same", "other", "all"),
        weighted=False,
    )
    print(
        "shot",
        {
            r: {
                LEAGUES[k]: (round(v["log_loss"], 4), v["auc"] and round(v["auc"], 3))
                for k, v in res_shot[r]["logistic"].items()
            }
            for r in res_shot
        },
    )
    # ------------------------------------------------------------------ write
    res = {
        "data": {
            "idsse_matches": [d["match"] for d in idsse],
            "skillcorner_matches": [d["match"] for d in sc],
            "pass_rows": {lg: int((lp == k).sum()) for k, lg in enumerate(LEAGUES)},
            "value_rows": {lg: int((lv == k).sum()) for k, lg in enumerate(LEAGUES)},
            "shot_rows_open_play": {lg: int((ls[keep] == k).sum()) for k, lg in enumerate(LEAGUES)},
            "pass_completion_weighted": {
                lg: float(np.average(pt["y"][lp == k], weights=pt["weight"][lp == k])) for k, lg in enumerate(LEAGUES)
            },
        },
        "skillcorner_sync": {d["match"]: d["sync"] for d in sc},
        "skillcorner_detection": {d["match"]: d["detection"] for d in sc},
        "xg_proxy": proxy,
        "pass": _named(res_pass),
        "pass_interceptor_removed": _named(res_noint),
        "pass_cross_league_detail": detail,
        "value": _named(res_value),
        "value_idsse_proxy_labels": {
            k: {kk: v[kk] for kk in ("n", "mean", "rmse", "r2", "spearman", "mse_skill")}
            for k, v in res_value_proxy.items()
        },
        "shot": _named(res_shot),
        "rules": {
            "same": "trained on the other matches of the held-out match's league (leave-one-match-out)",
            "other": "trained on every match of the other league (cross-league)",
            "all": "trained on every other match of both leagues (combined)",
            "all+league": "combined, with a league indicator column",
        },
        "value_labels": "IDSSE: provider xG; SkillCorner: xG proxy fitted on IDSSE shots (see xg_proxy)",
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_skillcorner_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    sc_rows = lp == 1
    reliability_chart(
        {
            "trained on Bundesliga": (pt["y"][sc_rows], oof_pass["other"]["logistic"][sc_rows], pt["weight"][sc_rows]),
            "trained on A-League (LOMO)": (
                pt["y"][sc_rows],
                oof_pass["same"]["logistic"][sc_rows],
                pt["weight"][sc_rows],
            ),
            "trained on both (LOMO)": (pt["y"][sc_rows], oof_pass["all"]["logistic"][sc_rows], pt["weight"][sc_rows]),
        },
        DOCS / "coach_skillcorner_pass_reliability.png",
        "Pass completion on held-out A-League matches",
    )
    rules_chart(res_pass, res_value, DOCS / "coach_skillcorner_rules.png")


if __name__ == "__main__":
    main()

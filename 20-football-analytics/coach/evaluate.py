"""Evaluate the AI coach on the 7 IDSSE matches (leave-one-match-out) and write docs/coach_results.json + charts.

    python -m coach.evaluate          # from 20-football-analytics/, after coach.prepare (~15-25 min on CPU)

Everything is scored on held-out matches: models are refitted on the six other matches for every fold. Charts only
(no broadcast imagery exists in this dataset; positions are drawn by us).
"""

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

DOCS = Path(__file__).resolve().parents[1] / "docs"
BUCKETS = (("1", 1, 1), ("2-3", 2, 3), ("4-6", 4, 6), ("7+", 7, 99))
BOOTSTRAPS = 1000


def strip(r):
    return {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in r.items()}


def summary(r):
    keys = ("n", "mean", "rmse", "mae", "r2", "pearson", "spearman", "mse_skill")
    return {k: {kk: v[kk] for kk in keys if kk in v} for k, v in r.items()}


# ---------------------------------------------------------------- advice on the pass the pro actually played


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


def spearman(a, b):
    return float(spearmanr(a, b)[0]) if np.ptp(a) > 0 and np.ptp(b) > 0 else float("nan")


def slope(x, y):
    return float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)) if np.ptp(x) > 0 else float("nan")


def bootstrap_ci(recs, fn, reps=BOOTSTRAPS, seed=0):
    """95 % interval of fn(records) resampling whole matches with replacement."""
    by = {}
    for r in recs:
        by.setdefault(r["match"], []).append(r)
    ids = list(by)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(ids), size=len(ids))
        v = fn([r for i in pick for r in by[ids[i]]])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [None, None]


def advice_summary(recs) -> dict:
    col = lambda k, rs=recs: np.array([r[k] for r in rs], float)
    out = {"passes": len(recs), "completed": int(col("success").sum())}
    # (a) completion rate of the real passes by the coach's rank of the option chosen (true rate: failed rows reweighted)
    buckets = []
    for name, lo, hi in BUCKETS:
        sel = [r for r in recs if lo <= r["rank_coach"] <= hi]
        if sel:
            w = col("weight", sel)
            buckets.append(
                {
                    "rank": name,
                    "n": len(sel),
                    "completion": float(np.average(col("success", sel), weights=w)),
                    "completion_unweighted": float(col("success", sel).mean()),
                }
            )
    out["completion_by_rank"] = buckets
    # (b) the coach score of the pro's choice against what happened in the next horizon (net xG, same units)
    rules = {"coach": "score", "p_only": "p", "value_only": "value", "xt_grid_only": "xt"}
    out["outcome_correlation"] = {
        name: {
            "spearman": spearman(col(k), col("outcome")),
            "ci95": bootstrap_ci(recs, lambda rs, k=k: spearman(col(k, rs), col("outcome", rs))),
        }
        for name, k in rules.items()
    }
    # The score is a change (it subtracts V(now)), and the realised outcome is dominated by where on the pitch the pass
    # is played. Two like-for-like views: the expected outcome after the pass (score + V(now)) against the outcome,
    # and the score against the realised change (outcome - V(now)).
    expected = lambda rs: col("score", rs) + col("v_now", rs)
    change = lambda rs: col("outcome", rs) - col("v_now", rs)
    out["coach_expected_outcome"] = {
        "spearman": spearman(expected(recs), col("outcome")),
        "ci95": bootstrap_ci(recs, lambda rs: spearman(expected(rs), col("outcome", rs))),
        "slope": slope(expected(recs), col("outcome")),
        "slope_ci95": bootstrap_ci(recs, lambda rs: slope(expected(rs), col("outcome", rs))),
    }
    out["coach_change"] = {
        "spearman": spearman(col("score"), change(recs)),
        "ci95": bootstrap_ci(recs, lambda rs: spearman(col("score", rs), change(rs))),
    }
    out["coach_deciles"] = ML.calibration(col("outcome"), expected(recs))
    out["mean_outcome"] = float(col("outcome").mean())
    # context only: how often the pro's receiver is the coach's pick (completed passes; the pro is not the target)
    done = [r for r in recs if r["success"]]
    ref = {}
    for name, k in (
        ("coach", "rank_coach"),
        ("p_only", "rank_p"),
        ("value_only", "rank_value"),
        ("nearest", "rank_near"),
    ):
        ref[name] = {
            "top1": float(np.mean([r[k] == 1 for r in done])),
            "top3": float(np.mean([r[k] <= 3 for r in done])),
        }
    ref["random"] = {
        "top1": float(np.mean([1 / r["options"] for r in done])),
        "top3": float(np.mean([min(3, r["options"]) / r["options"] for r in done])),
    }
    out["agreement_context"] = ref
    return out


# ---------------------------------------------------------------------------------------------- charts


def _pyplot():
    """matplotlib only for the charts (the report and the tests import this module without it)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def reliability_chart(results, path, title):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, r in results.items():
        pts = [(b["mean_pred"], b["observed"]) for b in r["reliability"] if b["n"] >= 20]
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"{name} (Brier {r['brier']:.3f})")
    ax.set(xlabel="predicted", ylabel="observed (true completion rate)", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def decile_chart(curves, path, title, xlabel, ylabel):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(min(c["mean_pred"], c["observed"]) for cs in curves.values() for c in cs)
    hi = max(max(c["mean_pred"], c["observed"]) for cs in curves.values() for c in cs)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    for name, cs in curves.items():
        ax.plot([c["mean_pred"] for c in cs], [c["observed"] for c in cs], "o-", label=name)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def bucket_chart(buckets, path):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([b["rank"] for b in buckets], [b["completion"] for b in buckets], color="#4c78a8")
    for i, b in enumerate(buckets):
        ax.text(i, b["completion"] + 0.01, f"n={b['n']}", ha="center", fontsize=8)
    ax.set(xlabel="coach rank of the option the player chose", ylabel="completion rate", ylim=(0, 1.05))
    ax.set_title("Completion of the real passes by coach rank")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main


def main():
    matches = [load(OUT / mid) for mid in MATCHES]
    pts = [D.pass_table(m) for m in matches]
    pt = D.stack(pts)
    vt = D.stack([D.value_table(m, horizons=D.HORIZONS) for m in matches])
    st = D.drop_penalties(D.stack([D.shot_table(m) for m in matches]))
    i = F.PASS_FEATURES.index
    geometry = [i(k) for k in ("distance", "forward", "abs_angle", "target_x", "target_goal_dist")]
    out = {"labels": {}}
    # pass labels: how close is the table to the true open-play completion rate (per match)?
    for m in matches:
        sel = pt["match"] == m.match_id
        out["labels"][m.match_id] = {
            "open_play_passes": sum(
                e["type"] == "pass" and not e.get("set_piece") and not e.get("cross") and e["team"] >= 0
                for e in m.events
            ),
            "true_completion": D.completion_rate(m),
            "table_rows": int(sel.sum()),
            "table_completion": float(pt["y"][sel].mean()),
            "weighted_completion": float(np.average(pt["y"][sel], weights=pt["weight"][sel])),
            "failed_weight": float(pt["weight"][sel][pt["y"][sel] == 0][0]),
        }
    print(
        "labels",
        {k: (round(v["true_completion"], 3), round(v["table_completion"], 3)) for k, v in out["labels"].items()},
    )

    # ------------------------------------------------------------------ pass model (completion probability)
    pass_models = {
        "logistic": ML.fit_logistic,
        "boosting": ML.fit_boosting,
        "distance_only": ML.column(i("distance")),
        "target_control_only": ML.column(i("target_control")),
        "base_rate": ML.fit_base_rate,
    }
    res_pass = ML.lomo(pt, pass_models)
    res_pass_unw = ML.lomo(
        pt, {"logistic": ML.fit_logistic, "boosting": ML.fit_boosting, "base_rate": ML.fit_base_rate}, weighted=False
    )
    # leak checks: the interceptor stands on the lane of a failed pass by construction (X_noint removes him from the
    # defenders of failed passes: a pessimistic bound), and the geometry-only model uses no defender at all
    res_noint = ML.lomo(
        {**pt, "X": pt["X_noint"]},
        {"logistic": ML.fit_logistic, "boosting": ML.fit_boosting, "base_rate": ML.fit_base_rate},
    )
    res_geom = ML.lomo(pt, {"logistic_geometry_only": ML.column(None, geometry)})
    # ------------------------------------------------------------------ value model, three horizons
    value_models = {
        "ridge_basis": ML.fit_ridge,
        "goal_distance_only": ML.column_reg(F.VALUE_FEATURES.index("goal_dist")),
        "xt_grid": ML.fit_xt_grid,
        "mean": ML.fit_mean,
    }
    res_value = {}
    for hz in D.HORIZONS:
        fitters = dict(value_models)
        if hz == D.HORIZON:
            fitters["boosting"] = ML.fit_boosting_regressor
        res_value[hz] = ML.lomo_reg(vt, fitters, y=vt["y_h"][hz])
        print(
            "value",
            hz,
            {
                k: (round(v["rmse"], 5), round(v["mse_skill"], 4), v["spearman"] and round(v["spearman"], 3))
                for k, v in res_value[hz].items()
            },
        )
    # ------------------------------------------------------------------ shot model (penalties excluded)
    res_shot = ML.lomo(st, {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate})
    ok = np.isfinite(st["xg"])
    provider = ML.metrics(st["y"][ok], st["xg"][ok])
    print(
        "pass",
        {
            n: (round(v["log_loss"], 4), round(v["brier"], 4), v["auc"] and round(v["auc"], 3))
            for n, v in res_pass.items()
        },
    )
    print(
        "shot",
        {
            n: (round(v["log_loss"], 4), round(v["brier"], 4), v["auc"] and round(v["auc"], 3))
            for n, v in res_shot.items()
        },
        "provider xG",
        provider,
    )

    # ------------------------------------------------------------------ advice on held-out matches, per horizon
    # the coach values pass targets with the arrival model (cut-back fix), fitted on the same horizon as V
    advice = {}
    for hz in D.HORIZONS:
        at = D.stack([D.arrival_table(m, p, horizon=hz) for m, p in zip(matches, pts)])
        recs = []
        for m in matches:
            tr = pt["match"] != m.match_id
            vtr = vt["match"] != m.match_id
            ttr = st["match"] != m.match_id
            atr = at["match"] != m.match_id
            models = A.from_fitted(
                {
                    "pass": ML.fit_logistic(pt["X"][tr], pt["y"][tr], pt["weight"][tr]),
                    "value": ML.fit_ridge(vt["X"][vtr], vt["y_h"][hz][vtr]),
                    "shot": ML.fit_logistic(st["X"][ttr], st["y"][ttr]),
                    "arrival": ML.fit_ridge(at["X"][atr], at["y"][atr]),
                }
            )
            xt = ML.fit_xt_grid(vt["X"][vtr], vt["y_h"][hz][vtr])
            w_fail = float(pt["weight"][pt["match"] == m.match_id][pt["y"][pt["match"] == m.match_id] == 0][0])
            recs += advice_records(m, models, xt, hz, w_fail)
        advice[hz] = advice_summary(recs)
        a = advice[hz]
        print(
            "advice",
            hz,
            a["passes"],
            {
                k: (round(v["spearman"], 4), [round(x, 4) for x in v["ci95"]])
                for k, v in a["outcome_correlation"].items()
            },
            [(b["rank"], b["n"], round(b["completion"], 3)) for b in a["completion_by_rank"]],
            a["agreement_context"]["coach"],
        )

    # ------------------------------------------------------------------ write
    res = {
        "labels": out["labels"],
        "pass": strip(res_pass),
        "pass_unweighted": strip(res_pass_unw),
        "pass_interceptor_removed": strip(res_noint),
        "pass_geometry_only": strip(res_geom),
        "value": {str(hz): (strip(r) if hz == D.HORIZON else summary(r)) for hz, r in res_value.items()},
        "shot": strip(res_shot),
        "shot_provider_xg": provider,
        "advice": {str(hz): a for hz, a in advice.items()},
        "main_horizon_s": D.HORIZON,
        "sync_audit": json.loads((OUT / "sync_audit.json").read_text(encoding="utf-8")),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    reliability_chart(
        {k: res_pass[k] for k in ("logistic", "boosting", "distance_only")},
        DOCS / "coach_pass_reliability.png",
        "Pass completion, held-out matches",
    )
    main_cal = {
        "ridge basis": res_value[D.HORIZON]["ridge_basis"]["calibration"],
        "xT grid": res_value[D.HORIZON]["xt_grid"]["calibration"],
    }
    decile_chart(
        main_cal,
        DOCS / "coach_value_calibration.png",
        f"Value: net xG in {D.HORIZON:.0f} s, held-out matches",
        "predicted (deciles)",
        "observed",
    )
    a = advice[D.HORIZON]
    decile_chart(
        {"expected outcome of the chosen pass": a["coach_deciles"]},
        DOCS / "coach_advice_calibration.png",
        f"Coach expectation vs net xG in {D.HORIZON:.0f} s",
        "score + V(now) (deciles)",
        "realised net xG",
    )
    bucket_chart(a["completion_by_rank"], DOCS / "coach_rank_completion.png")


if __name__ == "__main__":
    main()

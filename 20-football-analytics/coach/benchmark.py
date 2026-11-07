"""Benchmark of the coach against two published methods, expected threat (xT, Singh 2019) and VAEP (Decroos et al.,
KDD 2019), on the same held-out passes (coach insight, part C).

    python -m coach.benchmark            # from 20-football-analytics/ (~15 min on CPU)
    python -m coach.benchmark --reuse    # metrics and chart again from the cached pass rows (minutes)
                                         # -> docs/coach_benchmark_results.json + docs/coach_benchmark.png
    python -m coach.benchmark --set aleague   # the 20 A-League matches (raw store, proxy xG) -> *_aleague_*
    python -m coach.benchmark --set pooled    # all 27 matches, each scored with models fitted on the 26 others

For every pass of the 7 IDSSE matches that the match report scores (datasets.pass_samples), with every model fitted
WITHOUT the match (asserted): the coach (report.held_out_models, the site's recipe), xT (coach.xt on the other
matches' actions) and VAEP (coach.vaep.lomo). Values of the same pass:
- coach_score: the coach's score of the chosen pass, p * V(target) - (1 - p) * V_opp(loss) - V(now); ex ante (it does
  not know whether the pass was completed);
- coach_dv: V(target) - V(start) with the coach's value model; ex ante, ignores the risk;
- xt: xT(end) - xT(start) for a completed pass, -xT(start) for a failed one; ex post;
- vaep: the VAEP value of the pass action; ex post (it sees the result and where the ball went).
These are changes; the outcome (net xG in the next 15 s) is a level. Like-for-like levels are scored too (calibration
only): coach_after = coach_score + V(now) (the coach's expected net xG after the decision), xt_after = xT(start) +
xt (xT where the ball ended, 0 when lost), vaep_after = P_score - P_concede after the pass.
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
LEVELS = ("coach_after", "xt_after", "vaep_after")
DISAGREE = ("coach_score", "xt", "vaep")
N_DISAGREE = 10
PAIRS = (
    ("coach_after", "xt_after"),
    ("coach_after", "vaep_after"),
    ("xt_after", "vaep_after"),
    ("coach_dv", "xt"),
    ("coach_dv", "vaep"),
    ("xt", "vaep"),
)
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


def paired(rows, pairs=PAIRS, reps=EV.BOOTSTRAPS) -> dict:
    """Spearman with the net xG of method a minus that of method b, on the same passes and the same resamples."""
    out = {}
    sp = lambda rs, k: EV.spearman(col(rs, k), col(rs, "net_xg"))
    for a, b in pairs:
        fn = lambda rs, a=a, b=b: sp(rs, a) - sp(rs, b)
        out[f"{a}-{b}"] = {"diff": fn(rows), "ci95": EV.bootstrap_ci(rows, fn, reps)}
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


def pass_rows(m, models, xt_model, vaep_pred, acts, horizon=D.HORIZON) -> list:
    """One row per pass of evaluate.pass_decisions: the four values, the outcome and where / who (KEEP)."""
    row_of = {int(ev): k for k, ev in enumerate(acts["event"]) if ev >= 0}
    xt_val = xt_model.action_values(acts)
    xt_start = xt_model.xt_[XT.cell(acts["x0"], acts["y0"])]
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
                "v_now": v_now,
                "coach_after": ch["score"] + v_now,
                "xt_after": float(xt_start[k] + xt_val[k]),
                "vaep_after": float(vaep_pred["p_score"][k] - vaep_pred["p_concede"][k]),
                "net_xg": d["outcome"],
                "goal_15s": int(R.outcome(shots[e["half"]], d["t"], e["team"], bool(e["success"]), horizon) == "goal"),
                "scores_10": int(scores[k]),
            }
        )
    return out


def tree_nodes(model) -> int:
    """Nodes of a fitted HistGradientBoosting model (sklearn's internal predictors)."""
    return int(sum(len(p.nodes) for it in getattr(model, "_predictors", []) for p in it))


def cost(fitted, xt_model, vaep_models, timing, vaep_config) -> dict:
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
            "classifiers": {t: [n, V.CANDIDATES[n]] for t, n in vaep_config.items()},
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
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
    for k in LEVELS:
        dec = res["all"]["calibration"][k]["deciles"]
        axes[2].plot(range(1, len(dec) + 1), [c["observed"] for c in dec], "o-", label=k)
    axes[2].set(xlabel="decile of the method's state value after the pass", ylabel="mean net xG within 15 s")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarise(rows, reps=EV.BOOTSTRAPS) -> dict:
    comp = [r for r in rows if r["success"]]
    out = {"passes": len(rows), "n_completed": len(comp), "goals_15s": int(sum(r["goal_15s"] for r in rows))}
    for name, rs in (("all", rows), ("completed", comp)):
        out[name] = {
            "agreement": agreement(rs, reps=reps),
            "calibration": calibration(rs, METHODS + LEVELS, reps=reps),
            "paired_spearman_net_xg": paired(rs, reps=reps),
        }
    out["disagreements"] = disagreements(rows)
    return out


SETS = ("idsse", "aleague", "pooled")


def attach_proxy_xg(m, shots) -> int:
    """Give every shot of m (SkillCorner: no provider xG) the xG of cross_league.xg_proxy. `shots` is the match's
    cross_league.shot_rows table after xg_proxy filled its "xg"; rows are matched in order and checked on time and
    team. Returns the number of shots."""
    ev = [e for e in m.events if e["type"] == "shot" and e.get("half") is not None and e["team"] in (0, 1)]
    assert len(ev) == len(shots["xg"]), (len(ev), len(shots["xg"]))
    assert np.allclose([D.when(m, e) for e in ev], shots["t"]) and [e["team"] for e in ev] == list(shots["team"])
    for e, x in zip(ev, shots["xg"]):
        e["xg"], e["xg_source"] = float(x), "proxy"
    return len(ev)


def load_set(name: str) -> tuple:
    """(matches by id, extra shot rows for the coach's shot model, info) of a benchmark set: "idsse" (7 Bundesliga
    matches; shot model also on the A-League shots, the site's recipe), "aleague" (20 A-League matches from the raw
    store, xG = cross_league proxy; same-league training only) or "pooled" (all 27; each held-out match is scored
    with models fitted on the 26 others)."""
    from . import cross_league as CL
    from . import skillcorner as SC
    from .match import load
    from .prepare import MATCHES, OUT

    matches, info = {}, {}
    if name in ("idsse", "pooled"):
        matches.update({mid: load(OUT / mid) for mid in MATCHES})
    if name in ("aleague", "pooled"):
        idsse_d, sc_d = CL.load_all()
        px = CL.xg_proxy(idsse_d, sc_d)  # fitted on IDSSE shots only: no A-League labels leak into it
        info["xg_proxy"] = {k: v for k, v in px.items() if k != "weights"}
        missing = [mid for mid in SC.MATCHES if mid not in SC.stored_matches()]
        assert not missing, f"not in the raw store yet: {missing} (python -m coach.skillcorner --store-raw ...)"
        for mid, d in zip(SC.MATCHES, sc_d):
            m = SC.load_raw_match(mid)
            attach_proxy_xg(m, d["shot"])
            matches[str(m.match_id)] = m
    extra = CL.skillcorner_shots()[0] if name == "idsse" else None
    return matches, extra, info


def paths(name: str) -> tuple:
    """(row cache outside the repo, results JSON, chart) of a set; "idsse" keeps the original names."""
    from . import skillcorner as SC
    from .prepare import OUT

    if name == "idsse":
        return OUT / "benchmark_rows.json", DOCS / "coach_benchmark_results.json", DOCS / "coach_benchmark.png"
    return (
        SC.OUT / f"benchmark_rows_{name}.json",
        DOCS / f"coach_benchmark_{name}_results.json",
        DOCS / f"coach_benchmark_{name}.png",
    )


def main():
    from . import cross_league as CL

    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, default="idsse")
    ap.add_argument("--reuse", action="store_true", help="metrics from the cached pass rows")
    args = ap.parse_args()
    cache, results_path, chart_path = paths(args.set)
    matches, extra, info = load_set(args.set)
    acts = {mid: V.actions(m) for mid, m in matches.items()}
    vaep = V.lomo(acts, select_inner=True)
    print("vaep", {mid: v["config"] for mid, v in vaep.items()}, flush=True)
    tables = None
    if args.reuse and cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        rows, timing = saved["rows"], saved["timing"]
    else:
        tables = {mid: R.match_tables(m) for mid, m in matches.items()}
        rows, coach_s, iters = [], 0.0, {}
        for mid, m in matches.items():
            models, _, trained_on = R.held_out_models(mid, tables, extra)
            others = [o for o in matches if o != mid]
            assert mid not in trained_on and mid not in vaep[mid]["trained_on"] and mid not in others
            xt_model = XT.ExpectedThreat().fit(D.stack([acts[o] for o in others]))
            iters[mid] = xt_model.iterations_
            t0 = time.perf_counter()
            new = pass_rows(m, models, xt_model, vaep[mid], acts[mid])
            coach_s += time.perf_counter() - t0
            for r in new:
                r["league"] = CL.league_of(m)
            rows += new
            print(mid, len(new), "passes", flush=True)
        timing = {"coach_ms": round(1000 * coach_s / len(rows), 2), "xt_iterations": iters}
        cache.write_text(json.dumps({"rows": rows, "timing": timing}), encoding="utf-8")
    # cost: models fitted on every match of the set (size and timing only, never used for the held-out numbers)
    all_acts = D.stack(list(acts.values()))
    n_all = len(all_acts["type"])
    xt_all = XT.ExpectedThreat().fit(all_acts)
    t0 = time.perf_counter()
    xt_all.action_values(all_acts)
    timing["xt_ms"] = round(1000 * (time.perf_counter() - t0) / n_all, 5)
    configs = [vaep[mid]["config"] for mid in acts]
    common = {t: max(sorted(set(c[t] for c in configs)), key=[c[t] for c in configs].count) for t in V.TARGETS}
    vaep_all = V.fit(all_acts, common)
    t0 = time.perf_counter()
    V.predict(vaep_all, all_acts)
    timing["vaep_ms"] = round(1000 * (time.perf_counter() - t0) / n_all, 4)
    if tables is None:
        tables = {mid: R.match_tables(m) for mid, m in matches.items()}
    fitted = R.held_out_models(list(matches)[-1], tables, extra)[1]
    lab = {mid: V.labels(acts[mid]) for mid in acts}
    heldout = {}
    for i, (name, key) in enumerate((("scores", "p_score"), ("concedes", "p_concede"))):
        y = np.concatenate([lab[mid][i] for mid in acts])
        heldout[name] = ML.metrics(y, np.concatenate([vaep[mid][key] for mid in acts]))
        heldout[name]["constant_log_loss"] = ML.metrics(y, np.full(len(y), y.mean()))["log_loss"]
    types, counts = np.unique(all_acts["type"], return_counts=True)
    if args.set == "pooled":  # one summary per league of the held-out match (leagues are not mixed in a metric)
        leagues = sorted({r["league"] for r in rows})
        summary = {"by_league": {lg: summarise([r for r in rows if r["league"] == lg]) for lg in leagues}}
    else:
        summary = summarise(rows)
    res = {
        "set": args.set,
        "matches": list(matches),
        "horizon_s": D.HORIZON,
        "vaep_k": V.K,
        "xt_grid": [XT.NX, XT.NY],
        "actions": {str(t): int(c) for t, c in zip(types, counts)},
        "goals": int(all_acts["goal"].sum()),
        **info,
        **summary,
        "vaep_heldout": heldout,
        "vaep_config": {mid: vaep[mid]["config"] for mid in acts},
        "vaep_inner_log_loss": {mid: vaep[mid]["inner_log_loss"] for mid in acts},
        "xt_iterations": timing.get("xt_iterations"),
        "xt_grid_all_matches": [[round(float(v), 4) for v in r] for r in xt_all.grid()],
        "cost": cost(fitted, xt_all, vaep_all, timing, common),
    }
    DOCS.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    parts = res["by_league"] if args.set == "pooled" else {args.set: res}
    for lg, r in parts.items():
        if args.set != "pooled":
            chart(r, chart_path)
        for name in ("all", "completed"):
            print(lg, name, {k: round(v["spearman"], 3) for k, v in r[name]["agreement"].items()})
            print(lg, name, {k: round(v["spearman_net_xg"], 3) for k, v in r[name]["calibration"].items()})


if __name__ == "__main__":
    main()

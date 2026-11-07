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

from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from . import evaluate as EV
from . import models as ML

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

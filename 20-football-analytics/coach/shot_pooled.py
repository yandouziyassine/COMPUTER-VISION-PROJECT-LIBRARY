"""Held-out evidence for the website's pooled shot model (the export recipe of coach.export).

    python -m coach.shot_pooled           # from 20-football-analytics/; needs cv-data/skillcorner/derived/*.pkl

Leave-one-match-out over the 7 Bundesliga (IDSSE) matches. Each fold trains the shot logistic twice, on the other 6
Bundesliga matches only and on those plus every A-League shot (open play, features finite), and scores the held-out
Bundesliga shots (placed, no penalties: the rows coach.export fits on). Pooled out-of-fold log-loss, Brier and AUC,
per-match values, and a paired bootstrap over shots for the difference (pooled minus Bundesliga-only).
Written to docs/coach_shot_pooled_results.json.
"""

import json
import pickle

import numpy as np

from . import cross_league as CL
from . import datasets as D
from . import models as ML

OUT = CL.DOCS / "coach_shot_pooled_results.json"
KEYS = ("log_loss", "brier", "auc")


def idsse_shots(derived=None, ids=None) -> dict:
    """Placed non-penalty Bundesliga shots with finite features, from the cached derived tables (the rows of
    datasets.shot_table minus penalties)."""
    from . import skillcorner as SC
    from .prepare import MATCHES

    derived = SC.DERIVED if derived is None else derived
    rows = []
    for mid in MATCHES if ids is None else ids:
        s = pickle.loads((derived / f"idsse_{mid}.pkl").read_bytes())["shot"]
        keep = s["placed"] & ~s["penalty"] & np.isfinite(s["X"]).all(axis=1)
        rows.append({"X": s["X"][keep], "y": s["y"][keep], "match": s["match"][keep]})
    return D.stack(rows)


def _scores(y, p) -> dict:
    m = ML.metrics(y, p)
    return {k: m[k] for k in KEYS}


def _per_match(y, p, groups) -> dict:
    out = {}
    for g in np.unique(groups):
        sel = groups == g
        out[str(g)] = {"n": int(sel.sum()), "goals": int(y[sel].sum()), **_scores(y[sel], p[sel])}
    return out


def lomo_pooled(idsse: dict, extra: dict, reps: int = 2000, seed: int = 0) -> dict:
    """Leave-one-match-out over `idsse`; the pooled fit of each fold also gets every row of `extra`."""
    X, y, groups = idsse["X"], idsse["y"], idsse["match"]
    oof = {"idsse_only": np.zeros(len(y)), "pooled": np.zeros(len(y))}
    for g in np.unique(groups):
        test = groups == g
        oof["idsse_only"][test] = ML.fit_logistic(X[~test], y[~test]).predict_proba(X[test])[:, 1]
        pool_X, pool_y = np.vstack([X[~test], extra["X"]]), np.concatenate([y[~test], extra["y"]])
        oof["pooled"][test] = ML.fit_logistic(pool_X, pool_y).predict_proba(X[test])[:, 1]
    out = {
        "n_shots": int(len(y)),
        "goals": int(y.sum()),
        "extra_shots": int(len(extra["y"])),
        "extra_goals": int(extra["y"].sum()),
    }
    for name, p in oof.items():
        out[name] = {**_scores(y, p), "mean_pred": float(p.mean()), "per_match": _per_match(y, p, groups)}
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in KEYS}
    while len(boot["auc"]) < reps:
        idx = rng.integers(0, len(y), len(y))
        if not 0 < y[idx].sum() < len(idx):
            continue
        a, b = _scores(y[idx], oof["idsse_only"][idx]), _scores(y[idx], oof["pooled"][idx])
        for k in KEYS:
            boot[k].append(b[k] - a[k])
    base, pooled = out["idsse_only"], out["pooled"]
    out["difference"] = {
        k: {
            "mean": float(pooled[k] - base[k]),
            "ci95": [float(np.percentile(boot[k], 2.5)), float(np.percentile(boot[k], 97.5))],
        }
        for k in KEYS
    }
    out["note"] = "difference = pooled minus Bundesliga-only; 95 % interval from a paired bootstrap over held-out shots"
    return out


def main():
    idsse = idsse_shots()
    extra, ids = CL.skillcorner_shots()
    res = {
        "recipe": "coach.export shot model: logistic on placed non-penalty IDSSE shots (+ every open-play A-League shot)",
        "idsse_matches": sorted(set(idsse["match"].tolist())),
        "skillcorner_matches": ids,
        **lomo_pooled(idsse, extra),
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k in ("idsse_only", "pooled"):
        print(k, {m: round(res[k][m], 4) for m in KEYS})
    print("difference", json.dumps(res["difference"]))


if __name__ == "__main__":
    main()

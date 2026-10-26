"""Expected-goals (xG) model trained on StatsBomb open data (stage S4).

Data: StatsBomb open data (https://github.com/statsbomb/open-data), used under the StatsBomb Public Data User
Agreement; shots extracted by `stats/download_data.py`. Features: distance and angle to goal, body part, header,
set-piece / play pattern, first-time, under pressure, and from the shot's freeze frame (the positions StatsBomb
records at the moment of the shot): defenders inside the shooting cone, goalkeeper in the cone, goalkeeper distance.
Models: logistic regression and gradient boosting; split grouped by match (no match in both train and test);
log-loss, Brier score and calibration compared with StatsBomb's own `shot_statsbomb_xg`.

    python -m stats.xg            # from 20-football-analytics/, ~1 min on CPU
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
DOCS = HERE / "docs"
CV_DATA = Path(os.environ.get("CV_DATA", HERE.parents[1] / "cv-data"))
SHOTS = CV_DATA / "football" / "statsbomb" / "shots.csv"

GOAL = np.array([120.0, 40.0])  # StatsBomb pitch: 120 x 80 yards, goal centre
POSTS = np.array([[120.0, 36.0], [120.0, 44.0]])
YARD = 0.9144


def shot_geometry(x, y):
    """Distance to goal centre (m) and the angle (rad) subtended by the goal mouth, from StatsBomb coordinates."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    dist = np.hypot(GOAL[0] - x, GOAL[1] - y) * YARD
    a1 = np.arctan2(POSTS[0, 1] - y, POSTS[0, 0] - x)
    a2 = np.arctan2(POSTS[1, 1] - y, POSTS[1, 0] - x)
    ang = np.abs(a2 - a1)
    ang = np.where(ang > np.pi, 2 * np.pi - ang, ang)
    return dist, ang


def in_triangle(p, a, b, c):
    """Points p (n, 2) inside triangle abc (barycentric signs)."""

    def s(p1, p2, p3):
        return (p1[..., 0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[..., 1] - p3[1])

    d1, d2, d3 = s(p, a, b), s(p, b, c), s(p, c, a)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def freeze_frame_features(x, y, ff):
    """ff: list of [x, y, teammate, position]. Returns (defenders in cone, keeper in cone, keeper distance m,
    nearest defender distance m); NaN when the shot has no freeze frame."""
    if not ff:
        return np.nan, np.nan, np.nan, np.nan
    shooter = np.array([x, y])
    opp = np.array([[p[0], p[1]] for p in ff if not p[2] and p[3] != "Goalkeeper"]).reshape(-1, 2)
    gk = np.array([[p[0], p[1]] for p in ff if not p[2] and p[3] == "Goalkeeper"]).reshape(-1, 2)
    n_cone = int(in_triangle(opp, shooter, POSTS[0], POSTS[1]).sum()) if len(opp) else 0
    gk_cone = float(in_triangle(gk[:1], shooter, POSTS[0], POSTS[1]).any()) if len(gk) else 0.0
    gk_dist = float(np.hypot(*(gk[0] - GOAL)) * YARD) if len(gk) else 30.0
    near = float(np.min(np.hypot(*(opp - shooter).T)) * YARD) if len(opp) else 30.0
    return n_cone, gk_cone, gk_dist, near


BODY = ["Right Foot", "Left Foot", "Head", "Other"]
SHOT_TYPE = ["Open Play", "Free Kick", "Penalty", "Corner", "Kick Off"]
PATTERN = ["Regular Play", "From Corner", "From Free Kick", "From Throw In", "From Counter", "From Goal Kick",
           "From Keeper", "From Kick Off", "Other"]  # fmt: skip


def build_features(df):
    """DataFrame of shots (columns of download_data.shot_rows) -> (feature matrix, feature names)."""
    dist, ang = shot_geometry(df["x"], df["y"])
    cols = {"distance_m": dist, "log_distance": np.log1p(dist), "angle_rad": ang, "dist_x_angle": dist * ang}
    for b in BODY:
        cols[f"body_{b}"] = (df["body_part"] == b).to_numpy(float)
    cols["header"] = (df["body_part"] == "Head").to_numpy(float)
    for t in SHOT_TYPE:
        cols[f"type_{t}"] = (df["shot_type"] == t).to_numpy(float)
    for p in PATTERN:
        cols[f"pattern_{p}"] = (df["play_pattern"] == p).to_numpy(float)
    cols["first_time"] = df["first_time"].to_numpy(float)
    cols["under_pressure"] = df["under_pressure"].to_numpy(float)
    ff = [freeze_frame_features(x, y, json.loads(f)) for x, y, f in zip(df["x"], df["y"], df["freeze_frame"])]
    ff = np.array(ff, float)
    has = ~np.isnan(ff[:, 0])
    cols["has_freeze_frame"] = has.astype(float)
    for j, n in enumerate(["defenders_in_cone", "keeper_in_cone", "keeper_dist_m", "nearest_defender_m"]):
        cols[n] = np.where(has, ff[:, j], np.nanmedian(ff[:, j]))  # median fill + has_freeze_frame flag
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names


def reliability(y, p, bins=10):
    """Quantile-binned calibration curve: (mean predicted, observed rate, count) per bin."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    return [
        (float(p[k == i].mean()), float(y[k == i].mean()), int((k == i).sum()))
        for i in range(len(edges) - 1)
        if (k == i).any()
    ]


def scores(y, p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = np.clip(p, 1e-4, 1 - 1e-4)
    return {
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def models():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=0,
        ),
    }


def plot_calibration(curves, path, n_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot([0, 0.8], [0, 0.8], "k--", lw=0.8, label="perfect calibration")
    for (name, pts), mk in zip(curves.items(), "osD"):
        p, o, _ = zip(*pts)
        ax.plot(p, o, marker=mk, label=name)
    ax.set_xlabel("predicted xG (quantile bins)")
    ax.set_ylabel("observed goal rate")
    ax.set_title(f"xG calibration on {n_test} held-out shots (grouped by match)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(
        0.99,
        0.01,
        "Data: StatsBomb open data",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#c0392b",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    import pandas as pd
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shots", default=str(SHOTS))
    a = ap.parse_args()
    df = pd.read_csv(a.shots)
    df = df[df["statsbomb_xg"].notna()].reset_index(drop=True)
    X, names = build_features(df)
    y = df["goal"].to_numpy(int)
    groups = df["match_id"].to_numpy()
    sb = df["statsbomb_xg"].to_numpy(float)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
    res = {
        "data": {
            "shots": len(df),
            "matches": int(df["match_id"].nunique()),
            "competitions": sorted(df["competition"].unique().tolist()),
            "goal_rate": float(y.mean()),
            "shots_with_freeze_frame": float(np.mean([f != "[]" for f in df["freeze_frame"]])),
            "train_shots": len(tr),
            "test_shots": len(te),
            "test_matches": int(len(np.unique(groups[te]))),
        },
        "features": names,
        "test": {"base rate": scores(y[te], np.full(len(te), y[tr].mean())), "StatsBomb xG": scores(y[te], sb[te])},
        "cv5_grouped": {},
    }
    curves = {"StatsBomb xG": reliability(y[te], sb[te])}
    preds = {}
    for name, m in models().items():
        m.fit(X[tr], y[tr])
        preds[name] = m.predict_proba(X[te])[:, 1]
        res["test"][name] = scores(y[te], preds[name])
        curves[name] = reliability(y[te], preds[name])
        cv = []
        for ktr, kte in GroupKFold(n_splits=5).split(X, y, groups):
            mm = models()[name].fit(X[ktr], y[ktr])
            cv.append([scores(y[kte], mm.predict_proba(X[kte])[:, 1])["log_loss"], scores(y[kte], sb[kte])["log_loss"]])
        cv = np.array(cv)
        res["cv5_grouped"][name] = {
            "log_loss_mean": float(cv[:, 0].mean()),
            "log_loss_std": float(cv[:, 0].std()),
            "statsbomb_log_loss_mean": float(cv[:, 1].mean()),
        }
    # Without penalties (StatsBomb gives every penalty the same xG; this checks open-play quality).
    npen = te[df["shot_type"].to_numpy()[te] != "Penalty"]
    res["test_no_penalties"] = {"StatsBomb xG": scores(y[npen], sb[npen])}
    for name in models():
        p = preds[name][np.isin(te, npen)]
        res["test_no_penalties"][name] = scores(y[npen], p)
    res["test_total_xg"] = {"goals": int(y[te].sum()), "StatsBomb": float(sb[te].sum())} | {
        n: float(p.sum()) for n, p in preds.items()
    }
    res["calibration_curves"] = curves

    DOCS.mkdir(exist_ok=True)
    plot_calibration(curves, DOCS / "stats_xg_calibration.png", len(te))
    out = DOCS / "results_stats.json"
    full = json.loads(out.read_text()) if out.exists() else {}
    full["xg"] = res
    out.write_text(json.dumps(full, indent=2))
    for k, v in res["test"].items():
        print(f"{k:22s} log-loss {v['log_loss']:.4f}  Brier {v['brier']:.4f}  AUC {v['roc_auc']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()

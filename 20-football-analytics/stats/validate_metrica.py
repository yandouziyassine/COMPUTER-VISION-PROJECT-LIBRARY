"""Validate the statistics engine on Metrica Sports sample games 1 and 2 (stage S4).

1. Pass detection: control radius and minimum control time tuned on game 1, scored once on game 2 against the
   Metrica PASS events (precision / recall / F1).
2. Full-match statistics on the clean tracks (physical, possession, passes, formation, shape, PPDA-style pressing
   compared with the same ratio computed from the event data).
3. Noise-injection ablation: 10-minute windows of both games are corrupted with the errors measured in project 20
   (position error median 2.5 m / p90 6.2 m, ID swaps, missing frames) and the bias of every statistic relative to
   the clean tracks is reported with 95 % bootstrap confidence intervals over windows.

Writes docs/results_stats.json (key "metrica") and docs/stats_noise_ablation.png, docs/stats_pass_network.png.

    python -m stats.validate_metrica            # from 20-football-analytics/, ~10 min on a laptop CPU
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import metrica, noise, physical, tactics
from . import possession as P

HERE = Path(__file__).resolve().parents[1]
DOCS = HERE / "docs"
CV_DATA = Path(os.environ.get("CV_DATA", HERE.parents[1] / "cv-data"))
METRICA_DIR = CV_DATA / "football" / "metrica"

WINDOW_MIN = 10
CONDITIONS = {  # name: kwargs of noise.inject (None = clean)
    "clean": None,
    "independent noise, per frame": {"tau_s": 0.0},
    "independent noise, smooth (1 s)": {"tau_s": 1.0},
    "independent noise, smooth, half size": {"tau_s": 1.0, "pos_scale": 0.5},
    "camera-level noise (shared, 1 s) + 0.5 m local": {"tau_s": 1.0, "shared": True},
    "10 % frames missing": {"missing_frac": 0.10},
    "30 % frames missing": {"missing_frac": 0.30},
    "1 ID swap / min": {"swaps_per_min": 1.0},
    "5 ID swaps / min": {"swaps_per_min": 5.0},
    "combined: camera-level noise + 10 % missing + 1 swap/min": {
        "tau_s": 1.0, "shared": True, "missing_frac": 0.10, "swaps_per_min": 1.0,
    },
}  # fmt: skip
RELATIVE = ["distance_per90_m", "sprints", "passes", "compactness_m2", "ppda"]  # bias in % of the clean value
ABSOLUTE = ["top_speed", "possession_home", "defensive_line_m"]  # bias in the statistic's own unit


COMBINED = CONDITIONS["combined: camera-level noise + 10 % missing + 1 swap/min"]


def control_tracks(tr, prm):
    """Tracks used to decide who controls the ball: optionally smoothed first (helps against position noise)."""
    return physical.smooth_tracks(tr, prm["smooth_s"]) if prm.get("smooth_s") else tr


def tune_pass_detection(tr, gt, radii, min_controls, smooth=(0.0,)):
    grid = []
    for sm in smooth:
        ct = control_tracks(tr, {"smooth_s": sm})
        for r in radii:
            own = P.ball_owner(ct, r)
            for mc in min_controls:
                m = P.match_passes(P.detect_passes(ct, own, min_control=mc), gt)
                grid.append({"smooth_s": sm, "r_control": r, "min_control": mc, **m})
    best = max(grid, key=lambda g: g["f1"])
    return best, grid


def passes_and_possession(tr, prm):
    ct = control_tracks(tr, prm)
    own = P.ball_owner(ct, prm["r_control"])
    return P.detect_passes(ct, own, min_control=prm["min_control"]), P.possession(ct, own, prm["min_frames_possession"])


def window_stats(tr, prm, gt=None, prm_noisy=None):
    """Scalar statistics of one window (both teams pooled where it makes sense). With `prm_noisy`, passes and
    possession are also computed with the parameters tuned on noisy tracks (suffix _nt)."""
    rows = physical.physical_stats(tr)
    teams = [physical.team_physical(rows, k) for k in (0, 1)]
    passes, poss = passes_and_possession(tr, prm)
    shapes = [tactics.shape_stats(tr, k) for k in (0, 1)]
    forms = [tactics.formation(tr, k)[0] for k in (0, 1)]
    ppda = [tactics.ppda_counts(tr, passes, poss, k) for k in (0, 1)]
    out = {
        "distance_per90_m": float(np.mean([t["distance_per90_m"] for t in teams])),
        "top_speed": float(np.mean([t["mean_top_speed"] for t in teams])),
        "sprints": sum(t["sprints"] for t in teams),
        "possession_home": 100 * P.possession_share(poss),
        "passes": len(passes),
        "compactness_m2": float(np.mean([s["compactness_m2"] for s in shapes])),
        "defensive_line_m": float(np.mean([s["defensive_line_m"] for s in shapes])),
        "ppda_passes": sum(p for p, _ in ppda),
        "ppda_regains": sum(r for _, r in ppda),
        "formations": forms,
    }
    if gt is not None:
        out["pass_f1"] = P.match_passes(passes, gt)["f1"]
        out["pass_f1_team"] = P.match_passes(passes, gt, use_ids=False)["f1"]
    if prm_noisy is not None:
        pn, possn = passes_and_possession(tr, prm_noisy)
        out["passes_nt"] = len(pn)
        out["possession_home_nt"] = 100 * P.possession_share(possn)
        if gt is not None:
            out["pass_f1_nt"] = P.match_passes(pn, gt)["f1"]
    return out


def windows(tr):
    """(start, stop) of the 10-minute windows inside each half (a final piece shorter than 5 min is dropped)."""
    n = int(WINDOW_MIN * 60 * tr.fps)
    out = []
    for per in np.unique(tr.period):
        idx = np.flatnonzero(tr.period == per)
        a, b = idx[0], idx[-1] + 1
        for s in range(a, b, n):
            if min(s + n, b) - s >= n // 2:
                out.append((s, min(s + n, b)))
    return out


def gt_in(gt, s, e):
    return [dict(g, start=g["start"] - s, end=g["end"] - s) for g in gt if s <= g["start"] < e]


def bootstrap(values, n=2000, seed=0):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def bootstrap_ratio(num, den, n=2000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(num), (n, len(num)))
    r = num[i].sum(1) / np.maximum(den[i].sum(1), 1)
    return {
        "value": float(num.sum() / max(den.sum(), 1)),
        "ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))],
    }


def full_match(tr, ev, prm, gt):
    rows = physical.physical_stats(tr)
    passes, poss = passes_and_possession(tr, prm)
    out = {
        "physical": {k: physical.team_physical(rows, i) for i, k in enumerate(("home", "away"))},
        "top_speeds_m_s": sorted(round(r["max_speed"], 2) for r in rows)[-3:],
        "possession_home_pct": 100 * P.possession_share(poss),
        "passes_detected": len(passes),
        "passes_events": len(gt),
        "pass_detection": P.match_passes(passes, gt),
        "pass_detection_team_level": P.match_passes(passes, gt, use_ids=False),
        "halves": {},
    }
    for per in (1, 2):
        idx = np.flatnonzero(tr.period == per)
        s, e = idx[0], idx[-1] + 1
        w = tr.window(s, e)
        wp = [dict(p, start=p["start"] - s) for p in passes if s <= p["start"] < e]
        half = {}
        for k, name in enumerate(("home", "away")):
            sign = tactics.attack_sign(w, k)
            pp, rg = tactics.ppda_counts(w, wp, poss[s:e], k)
            ep, er = metrica.event_ppda_counts(tr, ev, k, sign, s, e)
            half[name] = {
                "formation": tactics.formation(w, k)[0],
                **tactics.shape_stats(w, k),
                "ppda_tracking": pp / max(rg, 1),
                "ppda_events": ep / max(er, 1),
                "ppda_counts_tracking": [pp, rg],
                "ppda_counts_events": [ep, er],
            }
        out["halves"][str(per)] = half
    return out, passes, poss


def plot_ablation(abl, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = [
        ("distance_per90_m", "Distance per 90 min (% bias)"),
        ("sprints", "Sprints (% bias)"),
        ("top_speed", "Mean top speed (m/s bias)"),
        ("possession_home", "Possession home (pp bias)"),
        ("passes", "Passes detected (% bias)"),
        ("pass_f1_nt", "Pass F1 vs events, noise-tuned detector (value)"),
        ("compactness_m2", "Compactness (% bias)"),
        ("defensive_line_m", "Defensive line (m bias)"),
        ("ppda", "PPDA-style (% bias)"),
    ]
    conds = [c for c in CONDITIONS if c != "clean"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    y = np.arange(len(conds))
    for ax, (key, title) in zip(axes.flat, stats):
        m = np.array([abl[c][key]["mean"] for c in conds])
        lo = np.array([abl[c][key]["ci95"][0] for c in conds])
        hi = np.array([abl[c][key]["ci95"][1] for c in conds])
        ref = abl["clean"][key]["mean"] if key.startswith("pass_f1") else 0
        ax.barh(y, m - ref, left=ref, color="#2e86c1", alpha=0.8)
        ax.errorbar(m, y, xerr=[m - lo, hi - m], fmt="none", ecolor="k", capsize=3, lw=1)
        ax.axvline(ref, color="k", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_yticks(y, conds, fontsize=8)
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        "Bias of each statistic when our measured tracking errors are injected into clean Metrica tracks\n"
        "(mean over 10-min windows of 2 matches, 95 % bootstrap CI; position error median 2.5 m / p90 6.2 m)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_pitch(ax, L=105, W=68):
    kw = dict(color="#666", lw=1, zorder=0)
    ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], **kw)
    ax.plot([L / 2, L / 2], [0, W], **kw)
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(L / 2 + 9.15 * np.cos(t), W / 2 + 9.15 * np.sin(t), **kw)
    for x0, d in ((0, 1), (L, -1)):
        ax.plot(
            [x0, x0 + d * 16.5, x0 + d * 16.5, x0], [W / 2 - 20.16, W / 2 - 20.16, W / 2 + 20.16, W / 2 + 20.16], **kw
        )
        ax.plot([x0, x0 + d * 5.5, x0 + d * 5.5, x0], [W / 2 - 9.16, W / 2 - 9.16, W / 2 + 9.16, W / 2 + 9.16], **kw)
    ax.set_xlim(-3, L + 3)
    ax.set_ylim(-3, W + 3)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_networks(panels, path, min_edge=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(panels), figsize=(7.5 * len(panels), 5.6))
    for ax, (title, g) in zip(np.atleast_1d(axes), panels):
        draw_pitch(ax)
        wmax = max([d["weight"] for *_, d in g.edges(data=True)] or [1])
        for a, b, d in g.edges(data=True):
            if d["weight"] < min_edge:
                continue
            (x0, y0), (x1, y1) = g.nodes[a]["pos"], g.nodes[b]["pos"]
            ax.annotate(
                "",
                (x1, y1),
                (x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.5 + 4 * d["weight"] / wmax, color="#1f618d", alpha=0.6,
                                shrinkA=9, shrinkB=9, connectionstyle="arc3,rad=0.08"),
            )  # fmt: skip
        for _, d in g.nodes(data=True):
            x, y = d["pos"]
            ax.scatter(x, y, s=80 + 8 * d["n_passes"], color="#e74c3c", edgecolor="k", zorder=3)
            ax.text(x, y, d["name"].replace("Player", ""), ha="center", va="center", fontsize=7, zorder=4)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(METRICA_DIR))
    ap.add_argument("--seeds", type=int, default=2, help="noise draws per window and condition")
    a = ap.parse_args()
    t0 = time.time()
    games = {g: metrica.load_game(a.data, g) for g in (1, 2)}
    gts = {g: metrica.true_passes(*games[g]) for g in games}

    best, grid = tune_pass_detection(games[1][0], gts[1], (0.5, 0.75, 1.0, 1.25, 1.5, 2.0), (1, 2, 3, 4, 5))
    prm = {
        "smooth_s": 0.0,
        "r_control": best["r_control"],
        "min_control": best["min_control"],
        "min_frames_possession": 5,
    }
    print(f"tuned on clean game 1: {prm} F1 {best['f1']:.3f}  ({time.time() - t0:.0f}s)")
    # Same search on game 1 corrupted with the combined errors, with optional smoothing before ball control.
    noisy = {g: noise.inject(games[g][0], np.random.default_rng(100 + g), **COMBINED) for g in games}
    best_n, grid_n = tune_pass_detection(noisy[1], gts[1], (1.0, 2.0, 3.0, 4.0, 5.0), (1, 3, 5, 10), (0.0, 1.0))
    prm_n = {k: best_n[k] for k in ("smooth_s", "r_control", "min_control")} | {"min_frames_possession": 5}
    held_out = {
        "clean_params_on_noisy_game2": P.match_passes(passes_and_possession(noisy[2], prm)[0], gts[2]),
        "noisy_params_on_noisy_game2": P.match_passes(passes_and_possession(noisy[2], prm_n)[0], gts[2]),
        "noisy_params_on_clean_game2": P.match_passes(passes_and_possession(games[2][0], prm_n)[0], gts[2]),
        "noisy_params_on_noisy_game2_team_level": P.match_passes(
            passes_and_possession(noisy[2], prm_n)[0], gts[2], use_ids=False
        ),
    }
    indep = noise.inject(games[2][0], np.random.default_rng(202), **(COMBINED | {"shared": False}))
    held_out["noisy_params_on_game2_independent_noise"] = P.match_passes(passes_and_possession(indep, prm_n)[0], gts[2])
    f1n = held_out["noisy_params_on_noisy_game2"]["f1"]
    print(f"tuned on noisy game 1: {prm_n} F1 {best_n['f1']:.3f}; noisy game 2 F1 {f1n:.3f} ({time.time() - t0:.0f}s)")

    res = {
        "params": prm,
        "params_noise_tuned": prm_n,
        "tuning_game1": {"best": best, "grid": grid},
        "tuning_noisy_game1": {"best": best_n, "grid": grid_n},
        "pass_detection_under_combined_noise": held_out,
        "matches": {},
    }
    passes_g, poss_g = {}, {}
    for g, (tr, ev) in games.items():
        res["matches"][f"game{g}"], passes_g[g], poss_g[g] = full_match(tr, ev, prm, gts[g])
        m = res["matches"][f"game{g}"]["pass_detection"]
        print(f"game {g}: pass P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f}")

    # Noise-injection ablation over 10-minute windows.
    per_cond = {c: [] for c in CONDITIONS}
    for g, (tr, _) in games.items():
        for wi, (s, e) in enumerate(windows(tr)):
            w, gt = tr.window(s, e), gt_in(gts[g], s, e)
            clean = window_stats(w, prm, gt, prm_n)
            per_cond["clean"].append((clean, clean))
            for ci, (c, kw) in enumerate(CONDITIONS.items()):
                if kw is None:
                    continue
                for seed in range(a.seeds):
                    rng = np.random.default_rng([g, wi, ci, seed])
                    per_cond[c].append((clean, window_stats(noise.inject(w, rng, **kw), prm, gt, prm_n)))
        print(f"game {g} ablation done ({time.time() - t0:.0f}s)")

    abl = {}
    for c, pairs in per_cond.items():
        d = {}
        for k in RELATIVE[:-1]:
            d[k] = bootstrap([100 * (n[k] - cl[k]) / cl[k] for cl, n in pairs if cl[k]])
        d["passes_nt"] = bootstrap([100 * (n["passes_nt"] - cl["passes_nt"]) / cl["passes_nt"] for cl, n in pairs])
        for k in ABSOLUTE + ["possession_home_nt"]:
            d[k] = bootstrap([n[k] - cl[k] for cl, n in pairs])
        d["pass_f1"] = bootstrap([n["pass_f1"] for _, n in pairs])
        d["pass_f1_nt"] = bootstrap([n["pass_f1_nt"] for _, n in pairs])
        d["pass_f1_team"] = bootstrap([n["pass_f1_team"] for _, n in pairs])
        d["formation_agreement"] = bootstrap(
            [np.mean([x == y for x, y in zip(cl["formations"], n["formations"])]) for cl, n in pairs]
        )
        clean_ppda = sum(cl["ppda_passes"] for cl, _ in pairs) / max(sum(cl["ppda_regains"] for cl, _ in pairs), 1)
        r = bootstrap_ratio([n["ppda_passes"] for _, n in pairs], [n["ppda_regains"] for _, n in pairs])
        d["ppda"] = {
            "mean": 100 * (r["value"] - clean_ppda) / clean_ppda,
            "ci95": [100 * (v - clean_ppda) / clean_ppda for v in r["ci95"]],
            "value": r["value"],
        }
        abl[c] = d
    res["ablation"] = {
        "windows": len(per_cond["clean"]),
        "seeds_per_window": a.seeds,
        "noise_model": "position error = log-normal-scaled isotropic Gaussian, median 2.5 m, p90 6.2 m (project 20 v1)",
        "conditions": {c: kw for c, kw in CONDITIONS.items()},
        "results": abl,
    }

    # What S3 must deliver: distance / sprint bias vs how slowly the camera-level error changes and its size.
    sweep = []
    sub = [(g, s, e) for g, (tr, _) in games.items() for s, e in windows(tr)][::3]
    for tau in (1.0, 3.0, 10.0, 30.0):
        for scale in (0.25, 0.5, 1.0):
            bias = {"distance_per90_m": [], "sprints": []}
            for g, s, e in sub:
                w = games[g][0].window(s, e)
                rng = np.random.default_rng([g, s, int(tau), int(100 * scale)])
                kw = {"tau_s": tau, "shared": True, "pos_scale": scale, "local_scale": 0.2 * scale}
                cl = [physical.team_physical(physical.physical_stats(w), j) for j in (0, 1)]
                rows = physical.physical_stats(noise.inject(w, rng, **kw))
                no = [physical.team_physical(rows, j) for j in (0, 1)]
                for key in bias:
                    c, n = sum(t[key] for t in cl), sum(t[key] for t in no)
                    bias[key].append(100 * (n - c) / c)
            sweep.append({"tau_s": tau, "median_error_m": 2.5 * scale} | {k: bootstrap(v) for k, v in bias.items()})
    res["camera_noise_sweep"] = {"windows": len(sub), "results": sweep}
    print(f"sweep done ({time.time() - t0:.0f}s)")

    DOCS.mkdir(exist_ok=True)
    plot_ablation(abl, DOCS / "stats_noise_ablation.png")
    tr = games[1][0]
    first = tr.period == 1
    half = tr.window(0, int(first.sum()))
    p1 = [p for p in passes_g[1] if p["start"] < first.sum()]
    noisy_half = noise.inject(half, np.random.default_rng(7), **COMBINED)
    pn, possn = passes_and_possession(noisy_half, prm_n)
    plot_networks(
        [
            ("Game 1, home team, 1st half: clean Metrica tracks", P.pass_network(half, p1, 0, poss_g[1][: first.sum()])),
            ("Same half, our errors injected (camera-level noise, 10 % missing, 1 swap/min), noise-tuned detector",
             P.pass_network(noisy_half, pn, 0, possn)),
        ],
        DOCS / "stats_pass_network.png",
    )  # fmt: skip

    out = DOCS / "results_stats.json"
    full = json.loads(out.read_text()) if out.exists() else {}
    full["metrica"] = res
    out.write_text(json.dumps(full, indent=2))
    print(f"wrote {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

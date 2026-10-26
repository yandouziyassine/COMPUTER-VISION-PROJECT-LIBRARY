"""W9 analysis: human vs model accuracy on the "which point is closer?" pairs.

Inputs: docs/w9_pairs.json (truth), docs/w9_model_answers.json (Kaggle job), and a folder of volunteer answer files
(answers_<session>.json, downloaded from the study page). Without answer files only the model side is reported
and the human side is marked "pending"; human data is never simulated.

Outputs docs/w9_results.json and docs/w9_results.png:
  - accuracy per category (bootstrap 95 % CI over pairs) for each model and, when available, humans
    (mean over volunteers, and the majority vote)
  - inter-annotator agreement: Fleiss' kappa and mean pairwise agreement
  - per pair: humans right / models right breakdown ("both right", "humans win", "models win", "both fail"),
    using the human majority vote and the majority of the models that passed the reproduction gate

Usage:
    python w9_analyze.py [--answers <folder with answers_*.json>] [--min-answers 250]
"""

import argparse
import json
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parent / "docs"
CATEGORIES = ["normal", "textureless", "mirror", "glass", "poster", "illusion"]
GATED = ["da2-large", "marigold-lcm", "da2-base", "da2-small", "dpt-hybrid"]  # relative models in the main table


# ---------- pure helpers (unit-tested) ----------


def fleiss_kappa(counts):
    """counts: (n_items, n_categories) number of raters choosing each category; equal raters per item."""
    c = np.asarray(counts, np.float64)
    n = c.sum(axis=1)
    if c.shape[0] == 0 or np.any(n != n[0]) or n[0] < 2:
        raise ValueError("Fleiss' kappa needs >= 2 raters, the same number on every item")
    n = n[0]
    p_j = c.sum(axis=0) / c.sum()
    P_i = ((c * c).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j**2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if P_e < 1 else 1.0


def pairwise_agreement(choices):
    """choices: list of {pair_id: 'a'|'b'} per rater -> mean share of identical answers over rater pairs."""
    vals = []
    for i in range(len(choices)):
        for j in range(i + 1, len(choices)):
            common = set(choices[i]) & set(choices[j])
            if common:
                vals.append(np.mean([choices[i][k] == choices[j][k] for k in common]))
    return float(np.mean(vals)) if vals else None


def bootstrap_mean(x, resamples=1000, seed=0):
    x = np.asarray(x, np.float64)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (resamples, len(x)))].mean(axis=1)
    return [round(float(x.mean()), 4), round(float(np.quantile(m, 0.025)), 4), round(float(np.quantile(m, 0.975)), 4)]


def majority(answers):
    """Majority answer ('a'/'b') of a list, None on a tie."""
    na = sum(a == "a" for a in answers)
    nb = len(answers) - na
    return None if na == nb else ("a" if na > nb else "b")


def error_kappa(correct_1, correct_2):
    """Error consistency (Geirhos et al., NeurIPS 2020): Cohen's kappa of trial-by-trial correctness.
    1 = the two observers fail on exactly the same trials; 0 = no more overlap than their accuracies imply."""
    c1, c2 = np.asarray(correct_1, bool), np.asarray(correct_2, bool)
    p1, p2 = c1.mean(), c2.mean()
    c_exp = p1 * p2 + (1 - p1) * (1 - p2)
    c_obs = np.mean(c1 == c2)
    return float((c_obs - c_exp) / (1 - c_exp)) if c_exp < 1 else float("nan")


def bootstrap_kappa(c1, c2, resamples=1000, seed=0):
    c1, c2 = np.asarray(c1, bool), np.asarray(c2, bool)
    rng = np.random.default_rng(seed)
    ks = [error_kappa(c1[i], c2[i]) for i in rng.integers(0, len(c1), (resamples, len(c1)))]
    ks = np.array([k for k in ks if np.isfinite(k)])
    k = error_kappa(c1, c2)
    if not len(ks) or not np.isfinite(k):
        return None
    return [round(k, 3), round(float(np.quantile(ks, 0.025)), 3), round(float(np.quantile(ks, 0.975)), 3)]


def outcome(human_right, model_right):
    return {
        (True, True): "both right",
        (True, False): "humans win",
        (False, True): "models win",
        (False, False): "both fail",
    }[(bool(human_right), bool(model_right))]


# ---------- analysis ----------


def per_category(pairs, correct):
    """correct: {pair_id: bool} -> {category: [mean, lo, hi]} plus 'all'."""
    out = {}
    for cat in [*CATEGORIES, "all"]:
        ids = [p["id"] for p in pairs if (cat == "all" or p["category"] == cat) and p["id"] in correct]
        out[cat] = bootstrap_mean([correct[i] for i in ids])
    return out


def load_humans(folder, min_answers):
    raters = []
    for f in sorted(Path(folder).glob("answers_*.json")):
        s = json.loads(f.read_text())
        ch = {a["id"]: a["choice"] for a in s.get("answers", [])}
        if len(ch) >= min_answers:
            ms = [a["ms"] for a in s["answers"] if isinstance(a.get("ms"), (int, float))]
            trials = [a for a in s["answers"] if a.get("id") and a.get("choice")]
            raters.append({"session": s.get("session", f.stem), "choices": ch, "ms": ms, "trials": trials})
    return raters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", help="folder with answers_*.json files from volunteers")
    ap.add_argument("--min-answers", type=int, default=250, help="drop incomplete sessions")
    ap.add_argument("--pairs", default=str(DOCS / "w9_pairs.json"), help="pair spec with truth (study or public set)")
    ap.add_argument("--model-answers", default=str(DOCS / "w9_model_answers.json"))
    ap.add_argument("--out", default="w9_results", help="output name in docs/ (.json and .png)")
    a = ap.parse_args()
    spec = json.loads(Path(a.pairs).read_text())
    pairs = spec["pairs"]
    if "counts" not in spec:
        spec["counts"] = {c: sum(p["category"] == c for p in pairs) for c in CATEGORIES}
    truth = {p["id"]: p["truth"] for p in pairs}
    cat_of = {p["id"]: p["category"] for p in pairs}
    models = json.loads(Path(a.model_answers).read_text())["models"]

    res = {"n_pairs": len(pairs), "counts": spec["counts"], "models": {}, "humans": "pending: no answers collected yet"}
    for name, m in models.items():
        correct = {k: v["answer"] == truth[k] for k, v in m["answers"].items()}
        res["models"][name] = per_category(pairs, correct)
    gated = [m for m in GATED if m in models]
    model_vote = {p["id"]: majority([models[m]["answers"][p["id"]]["answer"] for m in gated]) for p in pairs}
    res["model_majority"] = {"members": gated, **per_category(pairs, {k: v == truth[k] for k, v in model_vote.items()})}
    # B: error consistency between models, all pairs and per category
    names = list(models)
    corr = {m: {p["id"]: models[m]["answers"][p["id"]]["answer"] == p["truth"] for p in pairs} for m in names}
    res["error_consistency_models"] = {}
    for cat in ["all", *CATEGORIES]:
        ids = [p["id"] for p in pairs if cat == "all" or p["category"] == cat]
        res["error_consistency_models"][cat] = {
            f"{m1}|{m2}": bootstrap_kappa([corr[m1][i] for i in ids], [corr[m2][i] for i in ids])
            for j, m1 in enumerate(names)
            for m2 in names[j + 1 :]
        }
    # D: cue ablations, model side
    if all("by_transform" in m for m in models.values()):
        res["models_by_transform"] = {
            name: {
                tr: per_category(pairs, {k: v["answer"] == truth[k] for k, v in ans.items()})["all"]
                for tr, ans in m["by_transform"].items()
            }
            for name, m in models.items()
        }
    # mirrors/glass: what the depth *sensor* says about the surface point (it measures the reflection / sees through)
    sens = [p for p in pairs if p["category"] in ("mirror", "glass") and p.get("sensor_depth_on_surface")]
    sensor_correct = []
    for p in sens:
        surf, other = p["first_point"], "b" if p["first_point"] == "a" else "a"
        z_surf, z_other = p["sensor_depth_on_surface"], p[f"depth_{other}"]
        sensor_answer = surf if z_surf < z_other else other
        sensor_correct.append(sensor_answer == p["truth"])
    res["kinect_sensor_on_mirror_glass"] = {
        "pairs_with_sensor_reading": len(sens),
        "accuracy": bootstrap_mean(sensor_correct),
    }

    if a.answers:
        raters = load_humans(a.answers, a.min_answers)
        if len(raters) >= 2:
            choices = [r["choices"] for r in raters]
            per_rater = [per_category(pairs, {k: v == truth[k] for k, v in c.items()})["all"][0] for c in choices]
            vote = {p["id"]: majority([c[p["id"]] for c in choices if p["id"] in c]) for p in pairs}
            full = [p["id"] for p in pairs if all(p["id"] in c for c in choices)]
            counts = [[sum(c[i] == "a" for c in choices), sum(c[i] == "b" for c in choices)] for i in full]
            breakdown = {}
            for p in pairs:
                if vote[p["id"]] is None or model_vote[p["id"]] is None:
                    continue
                o = outcome(vote[p["id"]] == p["truth"], model_vote[p["id"]] == p["truth"])
                breakdown.setdefault(p["category"], {}).setdefault(o, 0)
                breakdown[p["category"]][o] += 1
            res["humans"] = {
                "raters": len(raters),
                "accuracy_mean_rater": per_rater and bootstrap_mean(per_rater),
                "majority_vote": per_category(pairs, {k: v == truth[k] for k, v in vote.items() if v}),
                "per_rater_by_category": {
                    cat: bootstrap_mean(
                        [
                            np.mean([c[p["id"]] == p["truth"] for p in pairs if p["category"] == cat and p["id"] in c])
                            for c in choices
                        ]
                    )
                    for cat in CATEGORIES
                },
                "fleiss_kappa": round(fleiss_kappa(counts), 4) if counts else None,
                "pairwise_agreement": pairwise_agreement(choices),
                "median_response_ms": float(np.median([m for r in raters for m in r["ms"]])),
                "outcomes_vs_model_majority": breakdown,
            }
            trials = [(r["session"], tr) for r in raters for tr in r["trials"]]
            right = lambda tr: tr["choice"] == truth[tr["id"]]  # noqa: E731
            # C: brief (200 ms) vs unlimited exposure; D: cue ablations (each trial has one condition)
            res["humans"]["by_exposure"] = {
                ex: {
                    cat: bootstrap_mean(
                        [
                            right(tr)
                            for _, tr in trials
                            if tr.get("exposure") == ex and (cat == "all" or cat_of[tr["id"]] == cat)
                        ]
                    )
                    for cat in ["all", *CATEGORIES]
                }
                for ex in ("free", "flash")
            }
            res["humans"]["by_transform"] = {
                trn: bootstrap_mean([right(tr) for _, tr in trials if tr.get("transform", "none") == trn])
                for trn in ("none", "rot180", "gray", "blur")
            }
            # B: human vs model error consistency on the same trials (model answer under the same transform)
            res["humans"]["error_consistency_vs_models"] = {}
            for m, mm in models.items():
                bt = mm.get("by_transform", {"none": mm["answers"]})
                hc, mc, cats = [], [], []
                for _, tr in trials:
                    ans = bt.get(tr.get("transform", "none"), {}).get(tr["id"])
                    if ans:
                        hc.append(right(tr))
                        mc.append(ans["answer"] == truth[tr["id"]])
                        cats.append(cat_of[tr["id"]])
                res["humans"]["error_consistency_vs_models"][m] = {
                    cat: bootstrap_kappa(
                        [h for h, c in zip(hc, cats) if cat == "all" or c == cat],
                        [x for x, c in zip(mc, cats) if cat == "all" or c == cat],
                    )
                    for cat in ["all", *CATEGORIES]
                }
        else:
            res["humans"] = f"pending: {len(raters)} complete session(s), need >= 2"
    (DOCS / f"{a.out}.json").write_text(json.dumps(res, indent=1))
    figure(res, a.out)
    for name, r in res["models"].items():
        print(f"{name:18s}", " ".join(f"{c}={r[c][0]:.2f}" for c in [*CATEGORIES, "all"]))
    print("humans:", res["humans"] if isinstance(res["humans"], str) else res["humans"]["majority_vote"]["all"])


def figure(res, out="w9_results"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(res["models"].items())
    if isinstance(res["humans"], dict):
        rows = [("humans (majority)", res["humans"]["majority_vote"])] + rows
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(rows) + 1.8))
    mat = np.array([[r[c][0] if r[c] else np.nan for c in CATEGORIES] for _, r in rows])
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(CATEGORIES)), [f"{c}\n(n={res['counts'][c]})" for c in CATEGORIES])
    ax.set_yticks(range(len(rows)), [n for n, _ in rows])
    title = "Which point is closer? accuracy by category (50 % = chance)"
    if isinstance(res["humans"], str):
        title += "\nhuman answers pending"
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(DOCS / f"{out}.png", dpi=110)


if __name__ == "__main__":
    main()

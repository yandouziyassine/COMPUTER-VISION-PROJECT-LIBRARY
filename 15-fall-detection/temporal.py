"""Temporal models (causal 1D-CNN / GRU) on raw keypoint sequences, evaluated exactly like the
gradient-boosted trees of evaluate.py: same videos, same grouped outer folds per seed, same nested
threshold choice (inner grouped 4-fold CV, best video-level F1), same alarm rule (10 frames).

Inputs per frame: keypoint coordinates and confidences, box, detection flag and their velocities
(temporal_model.sequence_inputs), standardised with training-fold statistics. Mirrored copies of the
training videos are added as augmentation. Run evaluate.py first (it caches the skeletons and writes
docs/results.json, which the comparison chart reads).

Usage:
    python temporal.py                          # TCN and GRU, outer seeds 0 1 2
    python temporal.py --arch tcn --seeds 0     # quicker
    python temporal.py --inputs features        # TCN on the 16 hand-crafted features instead
"""

import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from evaluate import aggregate, load_dataset, nested_cv, summarize  # noqa: E402
from fall import DOCS, OUT, features  # noqa: E402
from temporal_model import mirror, predict, sequence_inputs, train  # noqa: E402


def add_inputs(seqs, kind):
    for s in seqs:
        if kind == "keypoints":
            s["X"] = sequence_inputs(s["kps"], s["boxes"])
            s["Xm"] = sequence_inputs(*mirror(s["kps"], s["boxes"]))
        else:  # hand-crafted window features (nearly mirror-invariant, kept for symmetry with keypoints)
            s["X"] = s["F"].astype(np.float32)
            s["Xm"] = features(*mirror(s["kps"], s["boxes"]))[0].astype(np.float32)


def make_fit_predict(arch, steps, ensemble, log):
    calls = {"n": 0, "seconds": 0.0}

    def fit_predict(seqs, tr, te):
        t0 = time.time()
        train_x = [seqs[i]["X"] for i in tr] + [seqs[i]["Xm"] for i in tr]
        cat = np.concatenate(train_x)
        mu, sd = cat.mean(0), cat.std(0) + 1e-6
        data = [((x - mu) / sd, seqs[i]["y"].astype(np.float32)) for x, i in zip(train_x, list(tr) * 2)]
        models = [train(arch, data, steps=steps, seed=k) for k in range(ensemble)]
        out = {seqs[i]["name"]: np.mean([predict(m, (seqs[i]["X"] - mu) / sd) for m in models], axis=0) for i in te}
        calls["n"] += 1
        calls["seconds"] += time.time() - t0
        return out

    return fit_predict, calls


def comparison_chart(rows, path):
    """Sensitivity / specificity / F1 per model and protocol: mean over outer seeds, bars = min..max."""
    metrics = ("sensitivity", "specificity", "f1")
    fig, axes = plt.subplots(1, 3, figsize=(12, 0.55 * len(rows) + 1.4), sharey=True)
    y = np.arange(len(rows))
    for ax, m in zip(axes, metrics):
        mean = [r[m]["mean"] for r in rows]
        lo = [r[m]["mean"] - r[m]["min"] for r in rows]
        hi = [r[m]["max"] - r[m]["mean"] for r in rows]
        colors = ["tab:gray" if "0.5" in r["label"] else "tab:blue" for r in rows]
        ax.barh(y, mean, xerr=[lo, hi], color=colors, alpha=0.8, capsize=3)
        for yi, v in zip(y, mean):
            ax.text(0.02, yi, f"{v:.3f}", va="center", color="white", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_title(m)
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, [r["label"] for r in rows])
    axes[0].invert_yaxis()
    fig.suptitle("Video-level results, out-of-fold (mean over outer-CV seeds, whiskers = min..max)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", nargs="+", default=["tcn", "gru"], choices=["tcn", "gru"])
    ap.add_argument("--inputs", default="keypoints", choices=["keypoints", "features"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="outer-CV shuffling seeds")
    ap.add_argument("--steps", type=int, default=600, help="training steps per model")
    ap.add_argument("--ensemble", type=int, default=1, help="models (training seeds) averaged per fit")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    log_path = OUT / f"temporal_{args.inputs}.log"
    log_file = open(log_path, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    seqs, _ = load_dataset()
    add_inputs(seqs, args.inputs)
    res_path = DOCS / "temporal_results.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    for arch in args.arch:
        key = f"{arch}_{args.inputs}"
        fit_predict, calls = make_fit_predict(arch, args.steps, args.ensemble, log)
        per_seed = {}
        t0 = time.time()
        for seed in args.seeds:
            probs, nested_alarms, thresholds = nested_cv(seqs, fit_predict, seed, log=log)
            per_seed[seed] = summarize(seqs, probs, nested_alarms, thresholds)
            r = per_seed[seed]
            log(
                f"{key} seed {seed}: fixed 0.5 F1 {r['fixed_0.5']['f1']:.3f} "
                f"(sens {r['fixed_0.5']['sensitivity']:.3f}, spec {r['fixed_0.5']['specificity']:.3f}) | "
                f"nested F1 {r['nested']['f1']:.3f} (sens {r['nested']['sensitivity']:.3f}, "
                f"spec {r['nested']['specificity']:.3f})"
            )
            np.savez_compressed(OUT / f"oof_{key}_seed{seed}.npz", **probs)
        results[key] = {
            "steps": args.steps,
            "ensemble": args.ensemble,
            "seeds": args.seeds,
            "per_seed": per_seed,
            "aggregate": aggregate(per_seed),
            "fits": calls["n"],
            "seconds_per_fit": round(calls["seconds"] / max(calls["n"], 1), 1),
            "total_minutes": round((time.time() - t0) / 60, 1),
        }
        res_path.write_text(json.dumps(results, indent=1))
        log(f"{key}: {json.dumps(results[key]['aggregate'])}")

    # Comparison chart against the gradient-boosted trees (same seeds where available)
    hgb = json.loads((DOCS / "results.json").read_text())["nested_threshold"]["per_seed"]
    rows = []
    seeds = [str(s) for s in args.seeds]

    def agg(per_seed, protocol):
        a = aggregate({s: per_seed[s] for s in seeds if s in per_seed}, (protocol,))
        return a[protocol]

    names = {"tcn": "1D-CNN", "gru": "GRU", "keypoints": "keypoints", "features": "hand-crafted features"}
    for proto, tag in (("fixed_0.5", "thr 0.5"), ("nested", "nested thr")):
        rows.append({"label": f"Gradient boosting, features, {tag}", **agg(hgb, proto)})
        for key, r in results.items():
            arch, inputs = key.split("_")
            per_seed = {str(k): v for k, v in r["per_seed"].items()}
            rows.append({"label": f"{names[arch]}, {names[inputs]}, {tag}", **agg(per_seed, proto)})
    comparison_chart(rows, DOCS / "model_comparison.png")
    log(f"chart written ({len(rows)} rows, seeds {seeds})")


if __name__ == "__main__":
    main()

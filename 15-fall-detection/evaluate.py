"""Evaluate fall detection on UR Fall (30 falls + 40 daily activities, front camera).

Two detectors on the same pose features:
    rule      hand-written thresholds: horizontal posture AND a fast descent in the last 1.5 s
    learned   gradient-boosted trees on per-frame features, 5-fold cross-validation grouped by
              video (a video's frames are never split between train and test)

Sequence-level scoring: a fall video must raise an alarm (sensitivity); a daily-activity video,
including the 16 where the person lies down on purpose, must not (specificity).

Two threshold protocols for the learned model (same out-of-fold probabilities):
    fixed 0.5   the original protocol
    nested      inside each training fold, an inner grouped 4-fold CV gives validation probabilities
                for the training videos; the alarm threshold that maximises video-level F1 on them is
                applied to the held-out fold. No test video influences its own threshold.
The outer CV is repeated with several shuffling seeds (seed 0 = the original split).

Usage:
    python evaluate.py                 # poses (first run ~10 min), CV, figures, final model
    python evaluate.py --seeds 0       # only the original split
    python evaluate.py --gifs          # also write demo GIFs to cv-data (never to docs/: dataset licence)
"""

import argparse
import json
import time

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

from fall import DATA, DOCS, FPS, OUT, draw_skeleton, extract_poses, features, read_frames  # noqa: E402
from protocol import alarm_frames, alarms_at, choose_threshold, video_metrics  # noqa: E402

POSES = OUT / "poses"
NAMES = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]


def load_labels():
    cols = ["seq", "frame", "label"]
    df = pd.concat(
        [
            pd.read_csv(DATA / f, header=None, usecols=[0, 1, 2], names=cols)
            for f in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
        ]
    )
    return {s: dict(zip(g["frame"], g["label"])) for s, g in df.groupby("seq")}


def load_sequence(name):
    POSES.mkdir(exist_ok=True)
    cache = POSES / f"{name}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["kps"], z["boxes"]
    frames = read_frames(DATA / "videos" / f"{name}-cam0.mp4")
    kps, boxes = extract_poses(frames)
    np.savez_compressed(cache, kps=kps, boxes=boxes)
    return kps, boxes


def load_dataset():
    """All 70 videos: skeletons (cached), hand-crafted features, per-frame target and fall onset."""
    labels = load_labels()
    seqs = []
    for k, n in enumerate(NAMES):
        kps, boxes = load_sequence(n)
        F, fnames = features(kps, boxes)
        lab = np.array([labels.get(n, {}).get(t + 1, -1) for t in range(len(F))])
        onset = int(np.argmax(lab == 0)) if n.startswith("fall") and (lab == 0).any() else None
        # Learned target: the fall itself and the first 2 s after it began. Daily activities are
        # all negative, including lying down on purpose.
        y = np.zeros(len(F), int)
        if onset is not None:
            y[onset : onset + 2 * FPS] = 1
        seqs.append({"name": n, "kps": kps, "boxes": boxes, "F": F, "y": y, "onset": onset, "lab": lab})
        if k % 10 == 0:
            print(f"poses {k + 1}/{len(NAMES)}", flush=True)
    return seqs, fnames


def rule_prob(F, names):
    f = dict(zip(names, F.T))
    lying = (f["torso_angle"] > 55) | (f["aspect"] > 1.0)
    fast_drop = (f["max_v_head_1.5s"] > 0.8) | (f["max_v_top_1.5s"] > 0.8)  # frame-heights per second
    return (lying & fast_drop).astype(float)


def video_folds(seqs, idx, n_splits, seed):
    """Grouped, stratified folds over the videos `idx`. Split at frame level exactly as the original
    protocol did (StratifiedGroupKFold balances frame counts), then mapped back to video indices."""
    idx = np.asarray(idx)
    groups = np.concatenate([[i] * len(seqs[i]["F"]) for i in idx])
    cls = np.concatenate([[int(seqs[i]["onset"] is not None)] * len(seqs[i]["F"]) for i in idx])
    for tr, te in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(
        np.zeros((len(groups), 1)), cls, groups
    ):
        yield np.unique(groups[tr]), np.unique(groups[te])


def cross_val_probs(seqs, idx, fit_predict, n_splits, seed):
    probs = {}
    for tr, te in video_folds(seqs, idx, n_splits, seed):
        probs.update(fit_predict(seqs, tr, te))
    return probs


def nested_cv(seqs, fit_predict, seed, n_outer=5, n_inner=4, log=print):
    """Outer grouped CV for test probabilities; per outer fold, an inner grouped CV on the training
    videos chooses the alarm threshold. Returns out-of-fold probs, nested alarms and chosen thresholds."""
    onsets = {s["name"]: s["onset"] for s in seqs}
    probs, nested_alarms, thresholds = {}, {}, []
    for k, (tr, te) in enumerate(video_folds(seqs, range(len(seqs)), n_outer, seed)):
        inner = cross_val_probs(seqs, tr, fit_predict, n_inner, seed)
        thr, inner_f1 = choose_threshold(inner, {n: onsets[n] for n in inner})
        test = fit_predict(seqs, tr, te)
        probs.update(test)
        nested_alarms.update(alarms_at(test, thr))
        thresholds.append(thr)
        log(f"  seed {seed} fold {k}: threshold {thr:.2f} (inner F1 {inner_f1:.3f})")
    return probs, nested_alarms, thresholds


def summarize(seqs, probs, nested_alarms, thresholds):
    onsets = {s["name"]: s["onset"] for s in seqs}
    best_thr, _ = choose_threshold(probs, onsets)  # picked on the test probabilities: optimistic reference
    return {
        "fixed_0.5": video_metrics(alarms_at(probs, 0.5), onsets),
        "nested": video_metrics(nested_alarms, onsets),
        "nested_thresholds": thresholds,
        "posthoc_best": {"threshold": best_thr, **video_metrics(alarms_at(probs, best_thr), onsets)},
    }


def aggregate(per_seed, protocols=("fixed_0.5", "nested", "posthoc_best")):
    """Mean, min and max of sensitivity / specificity / F1 over the outer-CV seeds."""
    out = {}
    for p in protocols:
        out[p] = {}
        for m in ("sensitivity", "specificity", "f1"):
            v = [r[p][m] for r in per_seed.values()]
            out[p][m] = {"mean": round(float(np.mean(v)), 3), "min": min(v), "max": max(v)}
    return out


def make_hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)


def hgb_fit_predict(seqs, tr, te):
    X = np.concatenate([seqs[i]["F"] for i in tr])
    Y = np.concatenate([seqs[i]["y"] for i in tr])
    clf = make_hgb().fit(X, Y)
    return {seqs[i]["name"]: clf.predict_proba(seqs[i]["F"])[:, 1] for i in te}


def make_gif(name, prob, alarm, path, title):
    """Local inspection only: written to cv-data, never committed (UR Fall images can't be redistributed)."""
    frames = read_frames(DATA / "videos" / f"{name}-cam0.mp4")
    kps, _ = load_sequence(name)
    imgs = []
    step = max(3, len(frames) // 50)
    for t in range(0, len(frames), step):
        f = draw_skeleton(frames[t].copy(), np.nan_to_num(kps[t]))
        bar = int(prob[t] * f.shape[1])
        cv2.rectangle(f, (0, f.shape[0] - 10), (bar, f.shape[0]), (0, 0, 255) if prob[t] > 0.5 else (0, 200, 0), -1)
        if alarm is not None and t >= alarm:
            cv2.putText(f, "FALL ALARM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(f, title, (10, f.shape[0] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        imgs.append(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(colors=64))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 * step / FPS), loop=0, optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="outer-CV shuffling seeds")
    ap.add_argument("--gifs", action="store_true", help="write demo GIFs to cv-data/fall/outputs")
    args = ap.parse_args()
    DOCS.mkdir(exist_ok=True)
    t0 = time.time()
    seqs, fnames = load_dataset()
    onsets = {s["name"]: s["onset"] for s in seqs}

    # --- rule-based baseline
    rule_alarms = {s["name"]: alarm_frames(rule_prob(s["F"], fnames), 0.5, 5) for s in seqs}
    rule_res = video_metrics(rule_alarms, onsets)

    # --- learned, out-of-fold, fixed 0.5 and nested threshold, for each outer seed
    per_seed, probs0 = {}, None
    for seed in args.seeds:
        probs, nested_alarms, thresholds = nested_cv(seqs, hgb_fit_predict, seed)
        per_seed[seed] = summarize(seqs, probs, nested_alarms, thresholds)
        if probs0 is None:
            probs0, seed0 = probs, seed
        r = per_seed[seed]
        print(
            f"seed {seed}: fixed 0.5 F1 {r['fixed_0.5']['f1']:.3f} | nested F1 {r['nested']['f1']:.3f} "
            f"(sens {r['nested']['sensitivity']:.3f}, spec {r['nested']['specificity']:.3f})",
            flush=True,
        )
    learned_res = per_seed[seed0]["fixed_0.5"]
    learned_alarms = alarms_at(probs0, 0.5)

    # Operating curve on the out-of-fold probabilities of the first seed (descriptive only).
    sweep = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        r = video_metrics(alarms_at(probs0, thr), onsets)
        sweep.append({k: r[k] for k in ("sensitivity", "specificity", "f1", "median_latency_s")} | {"threshold": thr})

    adl_lying = sum(1 for s in seqs if s["onset"] is None and (s["lab"] == 1).any())
    res = {
        "videos": {"falls": 30, "daily_activities": 40, "adl_with_lying_down": adl_lying},
        "rule_based": rule_res,
        "learned_gradient_boosting_cv5": learned_res,
        "learned_threshold_sweep": sweep,
        "nested_threshold": {
            "outer_folds": 5,
            "inner_folds": 4,
            "criterion": "video-level F1, median of tied thresholds",
            "seeds": args.seeds,
            "per_seed": per_seed,
            "aggregate": aggregate(per_seed),
        },
    }
    print(json.dumps({k: res[k] for k in ("rule_based", "learned_gradient_boosting_cv5")}, indent=1))
    print(json.dumps(res["nested_threshold"]["aggregate"], indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))
    pd.DataFrame(
        [{"name": n, "fall": onsets[n] is not None, "alarm_frame": a} for n, a in learned_alarms.items()]
    ).to_csv(OUT / "per_video.csv", index=False)
    np.savez_compressed(OUT / "oof_hgb_seed0.npz", **probs0)

    # Final model on all data, for detect.py. Its threshold is chosen the same way as in the nested
    # protocol: grouped 5-fold CV probabilities over all 70 videos, best video-level F1.
    all_probs = cross_val_probs(seqs, range(len(seqs)), hgb_fit_predict, 5, 0)
    final_thr, _ = choose_threshold(all_probs, onsets)
    X = np.concatenate([s["F"] for s in seqs])
    Y = np.concatenate([s["y"] for s in seqs])
    joblib.dump({"model": make_hgb().fit(X, Y), "features": fnames, "threshold": final_thr}, OUT / "fall_model.joblib")
    print(f"final model saved, alarm threshold {final_thr:.2f}")

    # Out-of-fold probability traces of one fall and one deliberate lying down (chart only, no frames)
    lying_adl = next(
        s["name"]
        for s in seqs
        if s["onset"] is None and (s["lab"] == 1).sum() > 30 and learned_alarms[s["name"]] is None
    )
    if args.gifs:
        make_gif("fall-01", probs0["fall-01"], learned_alarms["fall-01"], OUT / "demo_fall.gif", "fall-01 (fall)")
        make_gif(lying_adl, probs0[lying_adl], None, OUT / "demo_lying_not_fall.gif", f"{lying_adl} (lying down)")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for n, c in (("fall-01", "tab:red"), (lying_adl, "tab:blue")):
        ax.plot(np.arange(len(probs0[n])) / FPS, probs0[n], color=c, label=n)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fall probability")
    ax.legend()
    ax.set_title("Out-of-fold fall probability")
    fig.tight_layout()
    fig.savefig(DOCS / "probability_traces.png", dpi=100)
    plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

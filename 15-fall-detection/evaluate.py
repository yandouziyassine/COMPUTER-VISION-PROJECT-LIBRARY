"""Evaluate fall detection on UR Fall (30 falls + 40 daily activities, front camera).

Two detectors on the same pose features:
    rule      hand-written thresholds: horizontal posture AND a fast descent in the last 1.5 s
    learned   gradient-boosted trees on per-frame features, 5-fold cross-validation grouped by
              video (a video's frames are never split between train and test)

Sequence-level scoring: a fall video must raise an alarm (sensitivity); a daily-activity video,
including the 16 where the person lies down on purpose, must not (specificity).

Usage:
    python evaluate.py
"""

import json

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

from fall import DATA, DOCS, FPS, OUT, alarm_frames, draw_skeleton, extract_poses, features, read_frames  # noqa: E402

POSES = OUT / "poses"


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


def rule_prob(F, names):
    f = dict(zip(names, F.T))
    lying = (f["torso_angle"] > 55) | (f["aspect"] > 1.0)
    fast_drop = (f["max_v_head_1.5s"] > 0.8) | (f["max_v_top_1.5s"] > 0.8)  # frame-heights per second
    return (lying & fast_drop).astype(float)


def score(alarms, seqs):
    rows = []
    for s in seqs:
        is_fall = s["name"].startswith("fall")
        a = alarms[s["name"]]
        detected = a is not None and (not is_fall or a >= s["onset"] - 15)  # alarms >0.5 s before the fall don't count
        latency = (a - s["onset"]) / FPS if (is_fall and detected) else None
        rows.append({"name": s["name"], "fall": is_fall, "alarm": detected, "latency_s": latency})
    df = pd.DataFrame(rows)
    falls, adls = df[df.fall], df[~df.fall]
    tp, fn = int(falls.alarm.sum()), int((~falls.alarm).sum())
    fp, tn = int(adls.alarm.sum()), int((~adls.alarm).sum())
    return {
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "accuracy": round((tp + tn) / len(df), 3),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "median_latency_s": round(float(falls.latency_s.dropna().median()), 2) if tp else None,
        "false_alarm_videos": adls[adls.alarm].name.tolist(),
        "missed_falls": falls[~falls.alarm].name.tolist(),
    }, df


def make_gif(name, prob, alarm, path, title):
    frames = read_frames(DATA / "videos" / f"{name}-cam0.mp4")
    kps, _ = load_sequence(name)
    imgs = []
    step = max(3, len(frames) // 50)  # at most ~50 GIF frames, so README GIFs stay < 2 MB
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
    DOCS.mkdir(exist_ok=True)
    labels = load_labels()
    names = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]

    seqs = []
    for k, n in enumerate(names):
        kps, boxes = load_sequence(n)
        F, fnames = features(kps, boxes)
        lab = np.array([labels.get(n, {}).get(t + 1, -1) for t in range(len(F))])
        onset = int(np.argmax(lab == 0)) if n.startswith("fall") and (lab == 0).any() else None
        # Learned target: the fall itself and the first 2 s after it began. Daily activities are
        # all negative, including lying down on purpose.
        y = np.zeros(len(F), int)
        if onset is not None:
            y[onset : onset + 2 * FPS] = 1
        seqs.append({"name": n, "F": F, "y": y, "onset": onset})
        if k % 10 == 0:
            print(f"poses {k + 1}/{len(names)}", flush=True)

    # --- rule-based baseline
    rule_alarms = {s["name"]: alarm_frames(rule_prob(s["F"], fnames), 0.5, 5) for s in seqs}
    rule_res, _ = score(rule_alarms, seqs)

    # --- learned, out-of-fold
    groups = np.concatenate([[i] * len(s["F"]) for i, s in enumerate(seqs)])
    X = np.concatenate([s["F"] for s in seqs])
    Y = np.concatenate([s["y"] for s in seqs])
    seq_class = np.concatenate([[int(s["name"].startswith("fall"))] * len(s["F"]) for s in seqs])
    oof = np.zeros(len(X))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, seq_class, groups):
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
        clf.fit(X[tr], Y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    learned_alarms, start = {}, 0
    probs = {}
    for s in seqs:
        p = oof[start : start + len(s["F"])]
        start += len(s["F"])
        probs[s["name"]] = p
        learned_alarms[s["name"]] = alarm_frames(p, 0.5, 10)
    learned_res, per_video = score(learned_alarms, seqs)

    # Operating curve: the alarm threshold trades missed falls against false alarms. Reported as a
    # curve on out-of-fold probabilities (picking the best point here would be optimistic).
    sweep = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        r, _ = score({n: alarm_frames(p, thr, 10) for n, p in probs.items()}, seqs)
        sweep.append(
            {
                "threshold": thr,
                "sensitivity": r["sensitivity"],
                "specificity": r["specificity"],
                "median_latency_s": r["median_latency_s"],
            }
        )

    adl_lying = sum(1 for n in names if n.startswith("adl") and 1 in labels.get(n, {}).values())
    res = {
        "videos": {"falls": 30, "daily_activities": 40, "adl_with_lying_down": adl_lying},
        "rule_based": rule_res,
        "learned_gradient_boosting_cv5": learned_res,
        "learned_threshold_sweep": sweep,
    }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))
    per_video.to_csv(OUT / "per_video.csv", index=False)

    # Final model on all data, for detect.py
    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=0)
    joblib.dump({"model": final.fit(X, Y), "features": fnames}, OUT / "fall_model.joblib")

    # Demos (out-of-fold probabilities, so the GIFs show test-time behavior)
    make_gif("fall-01", probs["fall-01"], learned_alarms["fall-01"], DOCS / "demo_fall.gif", "fall-01 (fall)")
    lying_adl = next(
        s["name"]
        for s in seqs
        if s["name"].startswith("adl")
        and (np.array([labels[s["name"]].get(t + 1, -1) for t in range(len(s["F"]))]) == 1).sum() > 30
        and learned_alarms[s["name"]] is None
    )
    make_gif(lying_adl, probs[lying_adl], None, DOCS / "demo_lying_not_fall.gif", f"{lying_adl} (lies down on purpose)")

    fig, ax = plt.subplots(figsize=(9, 3.2))
    for n, c in (("fall-01", "tab:red"), (lying_adl, "tab:blue")):
        ax.plot(np.arange(len(probs[n])) / FPS, probs[n], color=c, label=n)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fall probability")
    ax.legend()
    ax.set_title("Out-of-fold fall probability")
    fig.tight_layout()
    fig.savefig(DOCS / "probability_traces.png", dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()

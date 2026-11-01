"""Choose a per-category operating threshold for a target defect recall, without tuning on the images it is
evaluated on, and report the false-alarm rate it implies (run after run.py).

Protocol. PatchCore trains on good images only, so the only labelled defects are in the MVTec test set
(good + defective images). Each category's test set is split in two halves, stratified by defect type
("good" counts as one type), with a fixed seed:

1. calibration half: the threshold is the highest score that still flags >= `--recall` of its defective
   images (for fewer than 100 defects and a 99 % target, that is the lowest defect score);
2. held-out half: that threshold is applied unchanged; we count missed defects (recall) and good parts
   flagged as defective (false-alarm rate).

The halves then swap roles, and the whole thing is repeated over `--splits` random splits, so every image is
held out the same number of times. Reported numbers are averages over these held-out evaluations, never on
the half the threshold came from. As an optimistic reference only, the script also gives the false-alarm rate
at the same recall when the threshold is tuned on the full test set (what an oracle would get).

Usage:
    python threshold.py                      # 99 % recall target, 10 random splits
    python threshold.py --recall 0.95

Outputs:
    docs/thresholds.json, docs/threshold_chart.png
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import output_dir  # noqa: E402

SCORES = output_dir("mvtec") / "scores"
DOCS = Path(__file__).resolve().parent / "docs"
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip


def threshold_for_recall(defect_scores, recall):
    """Highest threshold t such that the share of defect scores >= t is at least `recall`."""
    s = np.sort(np.asarray(defect_scores, dtype=np.float64))
    k = int(np.floor((1 - recall) * len(s) + 1e-9))  # how many defects we may miss
    return float(s[min(k, len(s) - 1)])


def stratified_halves(groups, rng):
    """Boolean mask of half the items, taking half of every group (odd leftovers go to a random side)."""
    mask = np.zeros(len(groups), dtype=bool)
    for g in np.unique(groups):
        idx = rng.permutation(np.flatnonzero(groups == g))
        n = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
        mask[idx[:n]] = True
    return mask


def rates(scores, labels, t):
    flagged = scores >= t
    return float(flagged[labels == 1].mean()), float(flagged[labels == 0].mean())


def evaluate_category(scores, labels, defect, recall, n_splits, seed=0):
    rng = np.random.default_rng(seed)
    held_recall, held_far, thresholds, missed, false_alarms = [], [], [], 0, 0
    for _ in range(n_splits):
        a = stratified_halves(defect, rng)
        for cal, held in [(a, ~a), (~a, a)]:
            t = threshold_for_recall(scores[cal & (labels == 1)], recall)
            r, f = rates(scores[held], labels[held], t)
            held_recall.append(r)
            held_far.append(f)
            thresholds.append(t)
            missed += int(((scores < t) & held & (labels == 1)).sum())
            false_alarms += int(((scores >= t) & held & (labels == 0)).sum())
    n_eval = 2 * n_splits
    t_oracle = threshold_for_recall(scores[labels == 1], recall)
    oracle_recall, oracle_far = rates(scores, labels, t_oracle)
    return {
        "n_good": int((labels == 0).sum()),
        "n_defective": int((labels == 1).sum()),
        "heldout_recall_mean": round(float(np.mean(held_recall)), 4),
        "heldout_recall_min": round(float(np.min(held_recall)), 4),
        "heldout_false_alarm_mean": round(float(np.mean(held_far)), 4),
        "heldout_false_alarm_min": round(float(np.min(held_far)), 4),
        "heldout_false_alarm_max": round(float(np.max(held_far)), 4),
        # Per held-out evaluation, every image is seen once per split: divide totals by n_splits for a count
        # over one full test set.
        "missed_defects_per_test_set": round(missed / n_splits, 2),
        "false_alarms_per_test_set": round(false_alarms / n_splits, 2),
        "threshold_median": round(float(np.median(thresholds)), 4),
        "n_heldout_evaluations": n_eval,
        "oracle_recall": round(oracle_recall, 4),
        "oracle_false_alarm": round(oracle_far, 4),
    }


def plot(res, recall, path):
    cats = [c for c in CATEGORIES if c in res]
    far = np.array([res[c]["heldout_false_alarm_mean"] for c in cats]) * 100
    lo = far - np.array([res[c]["heldout_false_alarm_min"] for c in cats]) * 100
    hi = np.array([res[c]["heldout_false_alarm_max"] for c in cats]) * 100 - far
    oracle = np.array([res[c]["oracle_false_alarm"] for c in cats]) * 100
    rec = np.array([res[c]["heldout_recall_mean"] for c in cats]) * 100
    order = np.argsort(far)
    y = np.arange(len(cats))
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 0.42 * len(cats) + 1.6), sharey=True, width_ratios=[3, 1.3])
    ax.barh(
        y,
        far[order],
        xerr=[lo[order], hi[order]],
        color="#c0504d",
        alpha=0.85,
        capsize=2,
        ecolor="gray",
        label="held-out (mean, min-max)",
    )
    ax.scatter(oracle[order], y, marker="|", s=160, color="black", label="tuned on test (optimistic)", zorder=3)
    ax.set_yticks(y, [cats[i] for i in order])
    ax.set_xlabel("good parts flagged as defective (%)")
    ax.set_title(f"False-alarm rate at a {recall:.0%} defect-recall threshold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax2.barh(y, rec[order], color="#4f81bd", alpha=0.85)
    ax2.axvline(recall * 100, color="black", lw=1, ls="--")
    ax2.set_xlim(min(80, rec.min() - 2), 100.5)
    ax2.set_xlabel("held-out defect recall (%)")
    ax2.set_title("Recall actually reached")
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recall", type=float, default=0.99, help="target defect recall on the calibration half")
    ap.add_argument("--splits", type=int, default=10, help="random stratified half/half splits (x2 directions)")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    args = ap.parse_args()

    res = {}
    for cat in args.categories:
        f = SCORES / f"{cat}.npz"
        if not f.exists():
            print(f"{cat}: no scores (run run.py first), skipped")
            continue
        d = np.load(f)
        res[cat] = evaluate_category(d["scores"], d["labels"], d["defect"], args.recall, args.splits)
        r = res[cat]
        print(
            f"{cat:11s} held-out recall {r['heldout_recall_mean']:.3f} (min {r['heldout_recall_min']:.3f}) | "
            f"false alarms {r['heldout_false_alarm_mean']:.3f} | oracle {r['oracle_false_alarm']:.3f}"
        )
    if len(res) == len(CATEGORIES):
        keys = ["heldout_recall_mean", "heldout_false_alarm_mean", "oracle_false_alarm"]
        res["mean"] = {k: round(float(np.mean([res[c][k] for c in CATEGORIES])), 4) for k in keys}
        print("mean", res["mean"])
    out = {"protocol": __doc__.split("Usage:")[0].strip(), "target_recall": args.recall, "splits": args.splits}
    (DOCS / "thresholds.json").write_text(json.dumps({**out, "results": res}, indent=1))
    plot({c: v for c, v in res.items() if c != "mean"}, args.recall, DOCS / "threshold_chart.png")


if __name__ == "__main__":
    main()

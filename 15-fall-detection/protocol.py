"""Evaluation protocol shared by every detector: alarm rule, video-level scoring, threshold choice.

NumPy only (no Ultralytics, no torch), so the unit tests and both training scripts use the same code.

A video is scored as a whole: a fall video must raise an alarm no earlier than 0.5 s before the labelled
fall onset (sensitivity); a daily-activity video must raise none (specificity). F1 treats "fall" as the
positive class: F1 = 2 TP / (2 TP + FP + FN).
"""

import numpy as np

EARLY_FRAMES = 15  # alarms more than 0.5 s (at 30 fps) before the fall onset don't count as a detection
THRESHOLD_GRID = tuple(np.round(np.arange(0.30, 0.951, 0.05), 2))


def alarm_frames(prob, threshold=0.5, min_frames=10):
    """Raise an alarm when the per-frame fall probability stays above `threshold` for `min_frames` in a row.
    Returns the frame index of the first alarm, or None."""
    run = 0
    for t, p in enumerate(prob):
        run = run + 1 if p > threshold else 0
        if run >= min_frames:
            return t
    return None


def video_metrics(alarms, onsets, fps=30):
    """alarms: {video: first alarm frame or None}; onsets: {video: fall onset frame, or None for a non-fall}.
    Returns counts, sensitivity, specificity, F1, accuracy, median latency (s) and the error lists."""
    tp = fn = fp = tn = 0
    latencies, missed, false_alarms = [], [], []
    for name, a in alarms.items():
        onset = onsets[name]
        if onset is not None:
            if a is not None and a >= onset - EARLY_FRAMES:
                tp += 1
                latencies.append((a - onset) / fps)
            else:
                fn += 1
                missed.append(name)
        elif a is not None:
            fp += 1
            false_alarms.append(name)
        else:
            tn += 1
    n = tp + fn + fp + tn
    return {
        "sensitivity": round(tp / max(tp + fn, 1), 3),
        "specificity": round(tn / max(tn + fp, 1), 3),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 3),
        "accuracy": round((tp + tn) / max(n, 1), 3),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "median_latency_s": round(float(np.median(latencies)), 2) if latencies else None,
        "missed_falls": missed,
        "false_alarm_videos": false_alarms,
    }


def alarms_at(probs, threshold, min_frames=10):
    return {name: alarm_frames(p, threshold, min_frames) for name, p in probs.items()}


def choose_threshold(probs, onsets, grid=THRESHOLD_GRID, min_frames=10):
    """Pick the alarm threshold that maximises video-level F1 on (validation) probabilities.
    Ties are common with few videos: take the median of the tied thresholds, i.e. the middle of the
    best plateau, which is less fragile than its edge. Returns (threshold, best F1)."""
    f1s = np.array([video_metrics(alarms_at(probs, t, min_frames), onsets)["f1"] for t in grid])
    best = np.flatnonzero(f1s == f1s.max())
    return float(np.asarray(grid)[best[len(best) // 2]]), float(f1s.max())

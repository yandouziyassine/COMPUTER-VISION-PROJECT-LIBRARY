"""Check the speed calibration against the road's own markings.

Speeds are only as good as the pixel → meter mapping (the SOURCE polygon assumed to be a 25 m × 250 m patch of
road). Road markings give an independent ruler:
- across the road: the distance between lane lines (motorway lanes are 3.5–3.75 m wide in Europe);
- along the road: the period of the dashed lane lines (dash + gap).

This script removes the cars (median of frames spread over the video), warps the road to a bird's-eye view using
the same calibration as traffic.py, finds the white markings and measures them in "calibrated meters".
If the calibration were exact, lanes would measure ~3.5–3.75 m and dash periods would match a standard pattern.

Usage:
    python calibration_check.py
"""

import json

import cv2
import matplotlib
import numpy as np

from traffic import DOCS, SOURCE, TARGET_H, TARGET_W, get_sample_video

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PX_PER_M = 20
LANE_M, PERIOD_M = 3.75, 18.0  # common European motorway standards (assumed, not known for this road)


def background(video, n=25):
    """Median of n frames spread over the video: moving cars disappear, markings stay."""
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, total - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def birds_eye(img):
    target = np.array([[0, 0], [TARGET_W, 0], [TARGET_W, TARGET_H], [0, TARGET_H]], np.float32) * PX_PER_M
    m = cv2.getPerspectiveTransform(SOURCE, target)
    return cv2.warpPerspective(img, m, (TARGET_W * PX_PER_M, TARGET_H * PX_PER_M))


def marking_mask(top):
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    v, s = hsv[..., 2].astype(np.int16), hsv[..., 1]
    local = cv2.blur(v, (41, 41))
    return ((v > local + 35) & (s < 60)).astype(np.uint8)  # brighter than the asphalt around it, and colorless


def line_positions(mask, min_frac=0.08):
    """x positions (m) of longitudinal markings: columns where enough of the road length is white."""
    prof = mask.mean(axis=0)
    prof = np.convolve(prof, np.ones(5) / 5, mode="same")
    peaks = [x for x in range(2, len(prof) - 2) if prof[x] >= min_frac and prof[x] == prof[x - 2 : x + 3].max()]
    merged = []
    for x in peaks:  # one peak per marking (they are ~0.15–0.25 m wide)
        if merged and x - merged[-1][-1] <= 0.5 * PX_PER_M:
            merged[-1].append(x)
        else:
            merged.append([x])
    return [float(np.mean(g)) / PX_PER_M for g in merged], prof


def dash_runs(mask, x_m, half_width=0.25):
    """Lengths (m) of white runs and gaps along one dashed line."""
    x = int(x_m * PX_PER_M)
    col = mask[:, max(0, x - int(half_width * PX_PER_M)) : x + int(half_width * PX_PER_M) + 1].max(axis=1)
    col = cv2.morphologyEx(col.reshape(-1, 1), cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)).ravel()
    edges = np.flatnonzero(np.diff(col.astype(np.int8))) + 1
    runs = np.split(col, edges)
    lens = [(int(r[0]), len(r) / PX_PER_M) for r in runs[1:-1]]  # drop the truncated first/last runs
    dashes = [length for white, length in lens if white and length >= 1.0]
    gaps = [length for white, length in lens if not white and length >= 1.0]
    return dashes, gaps


def main():
    bg = background(get_sample_video())
    top = birds_eye(bg)
    mask = marking_mask(top)
    lines, prof = line_positions(mask)
    widths = np.diff(lines).round(2).tolist()

    dashed = []
    for x in lines:
        d, g = dash_runs(mask, x)
        if len(d) >= 3 and len(g) >= 3:
            dashed.append(
                {
                    "x_m": round(x, 2),
                    "dash_m": round(float(np.median(d)), 2),
                    "gap_m": round(float(np.median(g)), 2),
                    "period_m": round(float(np.median(d) + np.median(g)), 2),
                    "n_dashes": len(d),
                }
            )
    lanes = [w for w in widths if 2.8 <= w <= 4.2]  # lane-to-lane spacings only
    periods = [d["period_m"] for d in dashed if 10 <= d["period_m"] <= 25]  # motorway lane dashes, not barrier posts
    res = {
        "assumed_patch_m": [TARGET_W, TARGET_H],
        "marking_x_m": [round(x, 2) for x in lines],
        "spacing_between_markings_m": widths,
        "dashed_lines": dashed,
        "lane_width_m": round(float(np.mean(lanes)), 2) if lanes else None,
        "dash_period_m": round(float(np.median(periods)), 2) if periods else None,
    }
    # If the road follows common European motorway standards (3.75 m lanes, 6 m dash + 12 m gap), how far off is
    # the calibration? Speed scales with the along-road factor.
    if lanes:
        res["across_scale_if_3.75m_lanes"] = round(LANE_M / res["lane_width_m"], 3)
    if periods:
        res["along_scale_if_18m_period"] = round(PERIOD_M / res["dash_period_m"], 3)
    print(json.dumps(res, indent=1))
    (DOCS / "calibration_check.json").write_text(json.dumps(res, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 7), gridspec_kw={"width_ratios": [1, 1.4]})
    view = cv2.cvtColor(top, cv2.COLOR_BGR2RGB)[: 120 * PX_PER_M]
    axes[0].imshow(view, extent=[0, TARGET_W, 120, 0], aspect="auto")
    for x in lines:
        axes[0].axvline(x, color="orange", lw=0.8, alpha=0.7)
    axes[0].set_xlabel("across the road (calibrated m)")
    axes[0].set_ylabel("along the road (calibrated m)")
    axes[0].set_title("Bird's-eye view (first 120 m)")
    axes[1].plot(np.arange(len(prof)) / PX_PER_M, prof, color="#333")
    for x, w in zip(lines[:-1], widths):
        axes[1].annotate(f"{w:.2f} m", ((x + x + w) / 2, prof.max() * 1.02), ha="center", fontsize=8)
    axes[1].set_xlabel("across the road (calibrated m)")
    axes[1].set_ylabel("share of road length that is white")
    axes[1].set_title("Lane markings and their spacing")
    fig.tight_layout()
    fig.savefig(DOCS / "calibration_check.jpg", dpi=110)


if __name__ == "__main__":
    main()

"""Evaluate the football pipeline.

1. Detection mAP per class (player, ball) on the test split.
2. Team assignment (no team labels exist in the dataset): reported as how cleanly the two jersey
   clusters separate (silhouette score) and how many players end up "other"; plus visual examples.
3. Ball possession demo on test frames.

Usage:
    python evaluate.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import data_dir, output_dir  # noqa: E402  (must come before ultralytics)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from analytics import assign_teams, possession, torso_color  # noqa: E402

DATA = data_dir("football")
OUT = output_dir("football")
DOCS = Path(__file__).resolve().parent / "docs"
WEIGHTS = OUT / "runs" / "yolo11n_football" / "weights" / "best.pt"
TEST = DATA / "yolo" / "test"
TEAM_COLORS = {0: (255, 80, 0), 1: (0, 80, 255), -1: (200, 200, 200)}


def detect(model, img):
    r = model(img, imgsz=960, conf=0.25, verbose=False)[0]
    b, c, s = r.boxes.xyxy.numpy(), r.boxes.cls.numpy().astype(int), r.boxes.conf.numpy()
    players = b[c == 1]
    balls = b[c == 0]
    ball = balls[np.argmax(s[c == 0])] if len(balls) else None
    return players, ball


def main():
    DOCS.mkdir(exist_ok=True)
    model = YOLO(str(WEIGHTS))
    m = model.val(
        data=str(DATA / "football_test.yaml"),
        imgsz=960,
        batch=4,
        device="cpu",
        plots=False,
        verbose=False,
        project=str(OUT / "runs"),
        name="val",
        exist_ok=True,
    )
    per_class = {
        model.names[c]: {"AP50": round(float(m.box.ap50[i]), 4), "AP50-95": round(float(m.box.ap[i]), 4)}
        for i, c in enumerate(m.box.ap_class_index)
    }
    res = {
        "detection": {
            "mAP50": round(float(m.box.map50), 4),
            "mAP50-95": round(float(m.box.map), 4),
            "per_class": per_class,
            "ms_per_image": round(float(m.speed["inference"]), 1),
        }
    }

    images = sorted((TEST / "images").glob("*.jpg"))
    sils, other_share, poss = [], [], []
    for p in images:
        img = cv2.imread(str(p))
        players, ball = detect(model, img)
        labels, _ = assign_teams(img, players)
        feats = [torso_color(img, b) for b in players]
        valid = [i for i, f in enumerate(feats) if f is not None and labels[i] >= 0]
        if len(set(labels[valid])) == 2 and len(valid) >= 6:
            sils.append(silhouette_score(np.array([feats[i] for i in valid]), labels[valid]))
        if len(players):
            other_share.append(float((labels == -1).mean()))
        team, _ = possession(ball, players, labels)
        poss.append(team)
    res["teams"] = {
        "frames": len(images),
        "frames_scored": len(sils),
        "mean_silhouette": round(float(np.mean(sils)), 3) if sils else None,
        "mean_share_other": round(float(np.mean(other_share)), 3) if other_share else None,
        "note": "no team ground truth; silhouette in [-1, 1], higher = cleaner color split",
    }
    res["possession"] = {
        "frames_with_ball_and_owner": int(sum(t is not None and t >= 0 for t in poss)),
        "frames": len(images),
    }
    print(json.dumps(res, indent=1))
    (DOCS / "results.json").write_text(json.dumps(res, indent=1))

    # Examples: frames where the ball was found, annotated with teams and possession
    with_ball = [p for p, t in zip(images, poss) if t is not None] or images
    picks = with_ball[:: max(1, len(with_ball) // 4)][:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    for ax, p in zip(axes.ravel(), picks):
        img = cv2.imread(str(p))
        players, ball = detect(model, img)
        labels, _ = assign_teams(img, players)
        team, k = possession(ball, players, labels)
        for i, (x1, y1, x2, y2) in enumerate(players.astype(int)):
            cv2.rectangle(img, (x1, y1), (x2, y2), TEAM_COLORS[int(labels[i])], 2)
            if i == k:
                cv2.putText(img, "ball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if ball is not None:
            bx1, by1, bx2, by2 = ball.astype(int)
            cv2.circle(img, ((bx1 + bx2) // 2, (by1 + by2) // 2), 12, (0, 255, 255), 3)
        title = "possession: " + (
            {0: "team A", 1: "team B", -1: "other"}.get(team, "none") if team is not None else "none"
        )
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")
        ax.set_title(title)
    fig.suptitle("Blue/orange: teams by jersey color (unsupervised) · grey: referee/keeper/unsure · yellow: ball")
    fig.tight_layout()
    fig.savefig(DOCS / "examples.jpg", dpi=60)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Python <-> TypeScript parity fixtures for the website's coach (phase 2).

    python -m coach.parity_fixtures                               # from 20-football-analytics/
    python -m coach.parity_fixtures --export <dir> --out <file>   # defaults: cv-data site-export -> site tests

Reads exactly what the site ships (coach.export: match.json, h0.bin, h1.bin, models.json) and rebuilds each frame
state the way the browser does: int16 cm -> float64 metres, 12.5 fps, velocity = backward difference over 2 frames
(0.16 s, training's 4 frames at 25 fps), attack direction from match.json. It finds the ball carrier with the site's
rule and ranks the options with the library's own features.py / advice.py on the exported weights. It writes the
state, features and ranked options of up to `per_half` frames per half to the site's tests/fixtures/coach-parity.json;
the site's vitest asserts |diff| < 1e-6 and the same option order.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import advice as A
from . import features as F
from .models import linear_predictor
from .prepare import CV_DATA, HERE

NAN = -32768
BACK_SITE = 2  # frames at 12.5 fps = 0.16 s, the training lag (features.BACK = 4 frames at 25 fps)
CARRIER_M = 2.0  # the nearest player of the team in possession has the ball when within this distance (m)
TIE = 1e-9  # options closer than this could swap order between languages: such frames are skipped
EXPORT = CV_DATA / "football-coach" / "site-export"
SITE_FIXTURE = HERE.parents[1] / "yandouziyassine.github.io" / "tests" / "fixtures" / "coach-parity.json"


def load_half(export: Path, meta: dict, half: int) -> SimpleNamespace:
    """xy [F, P, 2] and ball [F, 2] in float64 metres (NaN = absent), P = every player of the match."""
    info = meta["halves"][half]
    n = len(info["slots"])
    raw = np.fromfile(export / f"h{half}.bin", dtype="<i2").reshape(info["frames"], n + 1, 2)
    val = np.where(raw == NAN, np.nan, raw / 100.0)
    xy = np.full((info["frames"], len(meta["players"]), 2), np.nan)
    xy[:, info["slots"]] = val[:, :n]
    return SimpleNamespace(xy=xy, ball=val[:, n])


def carrier_at(meta: dict, half: int, h, team_of, frame: int):
    """(player, team) of the ball carrier, or None: ball in play, team in possession known, and that team's nearest
    player within CARRIER_M of the ball (same rule as the site's carrierAt)."""
    owner = int(meta["owner"][half][frame]) - 1
    if meta["alive"][half][frame] != "1" or owner < 0 or np.isnan(h.ball[frame, 0]):
        return None
    cand = np.flatnonzero((team_of == owner) & ~np.isnan(h.xy[frame, :, 0]))
    if not len(cand):
        return None
    d = np.hypot(h.xy[frame, cand, 0] - h.ball[frame, 0], h.xy[frame, cand, 1] - h.ball[frame, 1])
    k = int(np.argmin(d))
    return (int(cand[k]), owner) if d[k] <= CARRIER_M else None


def predictors(models: dict) -> dict:
    return {k: linear_predictor(models[k]) for k in ("pass", "value", "shot")}


def frame_record(meta: dict, half: int, h, team_of, frame: int, models: dict):
    found = carrier_at(meta, half, h, team_of, frame)
    if found is None:
        return None
    player, team = found
    m = SimpleNamespace(halves={half: h}, team=team_of)
    direction = meta["halves"][half]["attack"][team]
    s = F.state(m, half, frame, team, fps=meta["fps"], back_frames=BACK_SITE, direction=direction)
    row = int(np.flatnonzero(s["att_idx"] == player)[0])
    opts = A.options(s, row, predictors(models))
    scores = [o["score"] for o in opts]
    if any(abs(a - b) < TIE for a, b in zip(scores, scores[1:])):
        return None
    here = s["att"][row]
    return {
        "half": half,
        "frame": frame,
        "carrier": player,
        "team": team,
        "carrier_row": row,
        "state": {
            "att": s["att"].tolist(),
            "att_v": s["att_v"].tolist(),
            "def": s["def"].tolist(),
            "def_v": s["def_v"].tolist(),
            "att_idx": s["att_idx"].tolist(),
            "def_idx": s["def_idx"].tolist(),
            "ball": s["ball"].tolist(),
            "direction": int(direction),
        },
        "pass_features": [
            {"target": int(s["att_idx"][j]), **F.pass_features(s, row, j)} for j in range(len(s["att"])) if j != row
        ],
        "value_features": F.value_features(here),
        "shot_features": F.shot_features(here, s["def"]),
        "options": opts,
    }


def candidate_frames(meta: dict, half: int, per_half: int) -> list:
    """Open-play pass frames of the half (the passer has the ball there), evenly thinned to about 3 x per_half."""
    frames = sorted(
        {
            e["frame"]
            for e in meta["events"]
            if e["type"] == "pass" and e.get("half") == half and e.get("frame") is not None and not e.get("set_piece")
        }
    )
    return frames[:: max(1, len(frames) // (3 * per_half))]


def build(export: Path, per_half: int = 15) -> dict:
    meta = json.loads((export / "match.json").read_text(encoding="utf-8"))
    models = json.loads((export / "models.json").read_text(encoding="utf-8"))
    team_of = np.array([p["team"] for p in meta["players"]])
    frames = []
    for half in range(len(meta["halves"])):
        h = load_half(export, meta, half)
        got = 0
        for f in candidate_frames(meta, half, per_half):
            rec = frame_record(meta, half, h, team_of, f, models)
            if rec is not None:
                frames.append(rec)
                got += 1
            if got == per_half:
                break
    return {
        "generated_by": "COMPUTER-VISION-PROJECT-LIBRARY/20-football-analytics/coach/parity_fixtures.py",
        "match_id": meta["match_id"],
        "fps": meta["fps"],
        "back_frames": BACK_SITE,
        "carrier_m": CARRIER_M,
        "tolerance": 1e-6,
        "models": models,
        "frames": frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=EXPORT)
    ap.add_argument("--out", type=Path, default=SITE_FIXTURE)
    ap.add_argument("--per-half", type=int, default=15)
    args = ap.parse_args()
    fx = build(args.export, args.per_half)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fx, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {len(fx['frames'])} frames to {args.out}")


if __name__ == "__main__":
    main()

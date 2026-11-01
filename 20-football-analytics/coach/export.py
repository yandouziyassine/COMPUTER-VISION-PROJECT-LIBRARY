"""Export for the website (phase 2): logistic weights as JSON and one match as int16 centimetres at 10 fps.

    python -m coach.export --match J03WOH --site ../../yandouziyassine.github.io/public/football

Frame rate: the 25 fps source is subsampled by a whole step (fps=10 -> step 2 -> 12.5 fps, written in match.json).
Binary layout (little-endian int16): for each frame, P players then the ball, each (x, y) in cm; NaN -> -32768.
match.json holds players (id index, team, shirt, name), team names/colours, frames per half, fps, owner/alive per
frame (as strings of digits), passes and shots with frame indices, and the data credit.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import datasets as D
from . import models as ML
from .match import load
from .prepare import MATCHES, OUT

NAN = -32768
CREDIT = "Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0"


def pack_match(m, out: Path, fps: int = 10):
    out.mkdir(parents=True, exist_ok=True)
    step = max(1, int(round(25 / fps)))  # 25 fps source: fps=10 -> every 2nd frame = 12.5 fps (stored as such)
    chunks, frames, owner, alive = [], [], [], []
    for h in m.halves:
        idx = np.arange(0, len(h.t), step)
        pos = np.concatenate([h.xy[idx], h.ball[idx, None, :2]], axis=1) * 100
        chunks.append(np.where(np.isnan(pos), NAN, np.round(pos)).astype("<i2"))
        frames.append(len(idx))
        owner.append("".join(str(int(o) + 1) for o in h.owner[idx]))
        alive.append("".join("1" if a else "0" for a in h.alive[idx]))
    np.concatenate(chunks).tofile(out / "match.bin")
    events = [
        {
            **{k: e[k] for k in e if k not in ("t",)},
            "frame": (e["frame"] // step) if e.get("frame") is not None else None,
        }
        for e in m.events
    ]
    meta = {
        "match_id": m.match_id,
        "fps": 25 / step,
        "frames": frames,
        "players": [{"team": int(t), "shirt": int(s), "name": n} for t, s, n in zip(m.team, m.shirt, m.names)],
        "meta": m.meta,
        "owner": owner,
        "alive": alive,
        "events": events,
        "credit": CREDIT,
    }
    (out / "match.json").write_text(json.dumps(meta, ensure_ascii=False))


def export_models(out: Path):
    matches = [load(OUT / mid) for mid in MATCHES]
    weights = {}
    for name, build, feats in (
        ("pass", D.pass_table, D.F.PASS_FEATURES),
        ("value", D.value_table, D.F.VALUE_FEATURES),
        ("shot", D.shot_table, D.F.SHOT_FEATURES),
    ):
        t = D.stack([build(m) for m in matches])
        weights[name] = ML.export_logistic(ML.fit_logistic(t["X"], t["y"]), feats)
    weights["pitch_control"] = {"reaction": 0.7, "vmax": 5.0, "sigma": 0.45}
    (out / "models.json").write_text(json.dumps(weights, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    pack_match(load(OUT / args.match), args.site)
    export_models(args.site)
    print("wrote", sorted(p.name for p in args.site.iterdir()))


if __name__ == "__main__":
    main()

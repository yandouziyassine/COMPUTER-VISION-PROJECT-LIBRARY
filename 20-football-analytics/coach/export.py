"""Export for the website (phase 2): model weights as JSON and one match as int16 centimetres, one file per half.

    python -m coach.export --match J03WOH --site ../../cv-data/football-coach/site-export

Frame rate: the 25 fps source is subsampled by a whole step (fps=10 -> step 2 -> 12.5 fps, written in match.json).
h<i>.bin (little-endian int16): for each frame of half i, the players who appear in that half (match.json
halves[i].slots, ascending player index), then the ball, each (x, y) in cm; NaN -> -32768. z<i>.bin (little-endian
int16): the ball's height in cm for each frame of half i (NaN -> -32768), used only by the site's 3D view; h<i>.bin
keeps its layout so the parity fixtures are unchanged. The browser loads the first
half first and the second when needed. match.json (UTF-8) holds players (team, shirt, name; the list index is the
player index used by slots and events), team names/colours, per half the frame count, the slots and the attack
directions (+1/-1 for home, guest), fps, owner/alive per frame (strings of digits), passes and shots with frame and
nominal indices at the site's rate (frame None when sync could not place the event, `drop` says why), and the data
credit. models.json holds the weights of the pass (logistic), value (ridge, with its basis definition) and shot
(logistic) models, trained WITHOUT the exported match. The pass and value models learn from the Bundesliga matches
only; the shot model also learns from every A-League shot (SkillCorner, cross_league.skillcorner_shots) because
pooling raised the held-out Bundesliga shot AUC (see shot_pooled); "shot_trained_on" lists what it was fitted on.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import advice as A
from . import cross_league as CL
from . import datasets as D
from . import geometry as G
from . import models as ML
from . import pitch_control as PC
from .match import load
from .prepare import MATCHES, OUT

NAN = -32768
CREDIT = "Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0"
SYNC_ONLY = ("t", "t_sync", "calc_frame", "dist", "sync_shift_s")  # sync diagnostics, not needed by the site


def _site_event(e: dict, step: int) -> dict:
    """An event with frame indices at the site's rate (frame stays None when sync could not place it)."""
    out = {k: v for k, v in e.items() if k not in SYNC_ONLY}
    out["frame"] = int(e["frame"]) // step if e.get("frame") is not None else None
    if e.get("nominal") is not None:
        out["nominal"] = int(e["nominal"]) // step
    return out


def pack_match(m, out: Path, fps: int = 10):
    out.mkdir(parents=True, exist_ok=True)
    (out / "match.bin").unlink(missing_ok=True)  # the single-file layout used before the per-half split
    step = max(1, int(round(25 / fps)))  # 25 fps source: fps=10 -> every 2nd frame = 12.5 fps (stored as such)
    halves, owner, alive = [], [], []
    for i, h in enumerate(m.halves):
        idx = np.arange(0, len(h.t), step)
        slots = np.flatnonzero(~np.isnan(h.xy[idx, :, 0]).all(axis=0))  # players seen at least once in this half
        pos = np.concatenate([h.xy[idx][:, slots], h.ball[idx, None, :2]], axis=1) * 100
        np.where(np.isnan(pos), NAN, np.round(pos)).astype("<i2").tofile(out / f"h{i}.bin")
        z = h.ball[idx, 2] * 100  # ball height (DFL tracking Z, metres) -> cm
        np.where(np.isnan(z), NAN, np.round(z)).astype("<i2").tofile(out / f"z{i}.bin")
        halves.append(
            {"frames": len(idx), "slots": slots.tolist(), "attack": [int(d) for d in G.attack_dirs(h, m.team)]}
        )
        owner.append("".join(str(int(o) + 1) for o in h.owner[idx]))
        alive.append("".join("1" if a else "0" for a in h.alive[idx]))
    meta = {
        "match_id": m.match_id,
        "fps": 25 / step,
        "halves": halves,
        "players": [{"team": int(t), "shirt": int(s), "name": n} for t, s, n in zip(m.team, m.shirt, m.names)],
        "meta": m.meta,
        "owner": owner,
        "alive": alive,
        "events": [_site_event(e, step) for e in m.events],
        "credit": CREDIT,
    }
    (out / "match.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def training_tables(train) -> dict:
    """Stacked pass / value / shot (penalties out) tables of the training matches."""
    passes = [D.pass_table(m) for m in train]
    return {
        "pass": D.stack(passes),
        "value": D.stack([D.value_table(m) for m in train]),
        "shot": D.drop_penalties(D.stack([D.shot_table(m) for m in train])),
        "arrival": D.stack([D.arrival_table(m, pt) for m, pt in zip(train, passes)]),
    }


def fit_models(tables: dict, extra_shots=None) -> dict:
    """The website's three models (fitted sklearn objects) from training tables; `extra_shots` (X, y, match: open-play
    shots of another league) is added to the shot model's rows only. Also used by coach.report."""
    p, v, s = tables["pass"], tables["value"], tables["shot"]
    if extra_shots is not None and len(extra_shots["y"]):
        s = {"X": np.vstack([s["X"], extra_shots["X"]]), "y": np.concatenate([s["y"], extra_shots["y"]])}
    fitted = {
        "pass": ML.fit_logistic(p["X"], p["y"], p["weight"]),
        "value": ML.fit_ridge(v["X"], v["y"]),
        "shot": ML.fit_logistic(s["X"], s["y"]),
    }
    if "arrival" in tables:  # value of a pass that arrives (cut-back fix), same basis as the value model
        fitted["arrival"] = ML.fit_ridge(tables["arrival"]["X"], tables["arrival"]["y"])
    return fitted


def export_models(out: Path, matches, exclude=None, extra_shots=None):
    """Fit on every match but `exclude` (the replay match must stay unseen) and write models.json. `extra_shots`
    (X, y, match: open-play shots of another league) is added to the shot model's training rows only."""
    train = [m for m in matches if m.match_id != exclude]
    tables = training_tables(train)
    fitted = fit_models(tables, extra_shots)
    n_extra = 0 if extra_shots is None else len(extra_shots["y"])
    weights = {
        "trained_without": exclude,
        "trained_on": [m.match_id for m in train],
        "shot_trained_on": {
            "idsse": [m.match_id for m in train],
            "skillcorner_matches": len(set(extra_shots["match"].tolist())) if n_extra else 0,
            "shots": len(tables["shot"]["y"]) + n_extra,
        },
        "pass": ML.export_logistic(fitted["pass"], D.F.PASS_FEATURES),
        "value": {
            **ML.export_ridge(fitted["value"], D.F.VALUE_FEATURES),
            "horizon_s": D.HORIZON,
            "target": "net expected goals (provider xG for minus against) within horizon_s, team in possession",
            "basis": D.F.VALUE_BASIS,
        },
        "arrival": {
            **ML.export_ridge(fitted["arrival"], D.F.VALUE_FEATURES),
            "horizon_s": D.HORIZON,
            "target": "net expected goals within horizon_s after a COMPLETED pass, at the target's position at the "
            "release; used for the V(target) of pass options only",
            "basis": D.F.VALUE_BASIS,
        },
        "shot": ML.export_logistic(fitted["shot"], D.F.SHOT_FEATURES),
        "pitch_control": {"reaction": PC.REACTION, "vmax": PC.VMAX, "sigma": PC.SIGMA},
        "advice": {
            "carry_dist": A.CARRY_DIST,
            "carry_angles_deg": list(A.CARRY_ANGLES),
            "shot_range": A.SHOT_RANGE,
            "score": "p * V(target) + (1 - p) * (-V_opp(loss point)) - V(now); shot: xG - V(now); "
            "V(target) of a pass = the arrival model, of a carry = the value model",
        },
    }
    (out / "models.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    ap.add_argument("--idsse-only-shots", action="store_true", help="fit the shot model on the Bundesliga shots only")
    args = ap.parse_args()
    pack_match(load(OUT / args.match), args.site)
    extra = None if args.idsse_only_shots else CL.skillcorner_shots()[0]
    export_models(args.site, [load(OUT / mid) for mid in MATCHES], exclude=args.match, extra_shots=extra)
    print("wrote", sorted(p.name for p in args.site.iterdir()))


if __name__ == "__main__":
    main()

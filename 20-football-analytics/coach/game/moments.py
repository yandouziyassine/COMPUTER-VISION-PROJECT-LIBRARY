"""One match -> Playmaker moment candidates: public file (no answers), answer file, interest features."""

import numpy as np

from .. import evaluate as EV
from .. import explain as E
from .. import features as F
from .. import geometry as G
from .. import models as ML
from .. import report as R
from . import CONTENT_VERSION, SCHEMA_VERSION
from . import inputs as GI
from . import quality as GQ
from . import scoring as GS

LEAD_UP_S = 3.0
AFTER_S = 6.0  # "what really happened": tracking after the release, shown only in the reveal
IDSSE_CREDIT = "Bundesliga tracking and event data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0"
CREDITS = {
    "Bundesliga": IDSSE_CREDIT,
    "2. Bundesliga": IDSSE_CREDIT,
    "A-League": "Tracking and event data: SkillCorner open data (MIT)",
}
FIRST_DIVISION = {"J03WMX", "J03WN1"}  # the other IDSSE matches are 2. Bundesliga 2022/23


def league_name(m, source: str) -> str:
    """Display league of a match; `source` is cross_league.league_of(m) ("idsse" / "skillcorner")."""
    if source == "skillcorner":
        return "A-League"
    return "Bundesliga" if str(m.match_id) in FIRST_DIVISION else "2. Bundesliga"


CUTBACK_X, CUTBACK_WIDE, CUTBACK_CENTRAL = 36.0, (12.0, 25.0), 8.0


def cm(a) -> list:
    return [int(round(float(v) * 100)) for v in a]


def fill_gaps(track: np.ndarray, max_gap: int = 6) -> np.ndarray:
    """[T, 2] positions: linear interpolation over NaN gaps of at most `max_gap` frames between two known points (the
    broadcast tracking loses the ball for a few frames); longer gaps and the ends stay NaN."""
    out = np.array(track, float)
    known = np.flatnonzero(~np.isnan(out[:, 0]))
    for a, b in zip(known[:-1], known[1:]):
        if 1 < b - a <= max_gap + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            out[a + 1 : b] = out[a] + t[:, None] * (out[b] - out[a])
    return out


def cm_or_none(a):
    return None if np.isnan(a[0]) else cm(a)


def is_cutback(here, to) -> bool:
    return bool(
        here[0] >= CUTBACK_X
        and CUTBACK_WIDE[0] <= abs(here[1]) <= CUTBACK_WIDE[1]
        and to[0] >= CUTBACK_X
        and abs(to[1]) < CUTBACK_CENTRAL
    )


def weights_of(fitted: dict) -> dict:
    return {
        "pass": ML.export_logistic(fitted["pass"], F.PASS_FEATURES),
        "shot": ML.export_logistic(fitted["shot"], F.SHOT_FEATURES),
    }


def match_moments(m, held_out, refits, league: str, require_full: bool = True) -> list:
    """held_out = report.held_out_models(...) for m; refits = refits.refit_models(...) for m.
    require_full=False keeps moments that fail the 22-players check (synthetic test matches only)."""
    models, fitted, trained_on = held_out
    assert str(m.match_id) not in [str(t) for t in trained_on]
    weights = weights_of(fitted)
    fps = F.match_fps(m)
    step = max(1, int(round(fps / 12.5)))
    back = int(round(LEAD_UP_S * fps))
    shots = R.shots_with_goals(m)
    out = []
    for d in EV.pass_decisions(m, models):
        p, e = d["sample"], d["sample"]["event"]
        s, carrier, half, frame = p["state"], p["carrier"], e["half"], e["frame"]
        if frame - back < 0:
            continue
        inp = GI.inputs(s, carrier)
        opts = GI.options(inp)
        main = GI.parts(inp, models)
        refit = np.array([GI.scores(inp, rm) for rm, _ in refits])
        share = GS.share_top(refit)
        pts = GS.points(main["score"], share)
        h = m.halves[half]
        idx = list(s["att_idx"]) + list(s["def_idx"])
        frames_src = list(range(frame - back, frame + 1, step))
        if frames_src[-1] != frame:
            frames_src.append(frame)
        lead = G.normalise(h.xy[frames_src][:, idx], s["direction"])
        ball = fill_gaps(G.normalise(h.ball[frames_src, :2], s["direction"]))
        after_src = list(range(frame, min(len(h.t), frame + int(round(AFTER_S * fps)) + 1), step))
        after = G.normalise(h.xy[after_src][:, idx], s["direction"])
        after_ball = fill_gaps(G.normalise(h.ball[after_src, :2], s["direction"]))
        n_att, n_def = len(s["att_idx"]), len(s["def_idx"])
        bad = GQ.problems(n_att, n_def, float(np.linalg.norm(s["ball"] - s["att"][carrier])), lead, len(opts))
        if require_full:
            if bad:
                continue
        elif [b for b in bad if b != "players"]:
            continue
        pro_key = ("pass", int(d["chosen"]["target"]))
        keys = [GI.key(o) for o in opts]
        expl = E.explain_options(
            s, carrier, [_as_advice(o, main, i) for i, o in enumerate(opts)], weights, main["v_now"]
        )
        mid = f"{m.match_id}-{half}-{frame}"
        player_of = {int(pid): i for i, pid in enumerate(idx)}
        public_opts = []
        for o in opts:
            po = {"k": o["k"], "kind": o["kind"], "to": cm(o["to"])}
            if o["kind"] == "pass":
                po["player"] = player_of[o["target"]]
            if o["kind"] == "carry":
                po["angle"] = int(o["angle"])
            public_opts.append(po)
        home = m.meta.get("home", "Home")
        away = m.meta.get("guest", "Away")
        t_start = float(h.t[0])
        public = {
            "schema": SCHEMA_VERSION,
            "id": mid,
            "league": league,
            "home": home,
            "away": away,
            "attacking": "home" if int(e["team"]) == 0 else "away",
            "minute": int((float(h.t[frame]) - t_start) // 60) + 45 * half,
            "fps": fps / step,
            "carrier": player_of[int(s["att_idx"][carrier])],
            "players": [
                {"side": 0 if i < n_att else 1, "shirt": int(m.shirt[pid]), "name": str(m.names[pid])}
                for i, pid in enumerate(idx)
            ],
            "lead_up": [[None if np.isnan(xy[0]) else cm(xy) for xy in f] for f in lead],
            "ball": [cm_or_none(b) for b in ball],
            "options": public_opts,
            "credit": CREDITS[league],
        }
        answer = {
            "schema": SCHEMA_VERSION,
            "id": mid,
            "options": [
                {
                    "k": o["k"],
                    "score": round(float(main["score"][i]), 6),
                    "p": round(float(main["p"][i]), 4),
                    "share_top": round(float(share[i]), 3),
                    "points": int(pts[i]),
                    "reasons": expl[i]["phrases"][:3],
                }
                for i, o in enumerate(opts)
            ],
            "ai_pick": int(np.argmax(main["score"])),
            "pro_pick": keys.index(pro_key),
            "outcome": {
                "kind": R.outcome(shots[half], d["t"], int(e["team"]), bool(e["success"])),
                "net_xg_15s": round(float(d["outcome"]), 4),
            },
            "after": {
                "fps": fps / step,
                "frames": [[None if np.isnan(xy[0]) else cm(xy) for xy in f] for f in after],
                "ball": [cm_or_none(b) for b in after_ball],
            },
            "models": {"content_version": CONTENT_VERSION, "trained_without": str(m.match_id), "refits": len(refits)},
        }
        here = s["att"][carrier]
        out.append(
            {
                "public": public,
                "answer": answer,
                "features": {
                    "match": str(m.match_id),
                    "x": float(here[0]),
                    "v_now": main["v_now"],
                    "iqr": float(np.subtract(*np.percentile(main["score"], [75, 25]))),
                    "all_best": bool((pts == 100).all()),
                    "obvious": bool(share.max() == 1.0 and (pts == 100).sum() == 1),
                    "cutback": is_cutback(here, d["chosen"]["to"]) and bool(e["success"]),
                    "ai_is_pro": answer["ai_pick"] == answer["pro_pick"],
                    "shot_after": answer["outcome"]["kind"] in ("shot", "goal"),
                    "difficulty": float(1 - pts.mean() / 100),
                },
            }
        )
    return out


def _as_advice(o: dict, parts: dict, i: int) -> dict:
    """advice.options-shaped option for explain.explain_options."""
    a = {"kind": o["kind"], "to": o["to"], "p": float(parts["p"][i]), "value": float(parts["value"][i])}
    a["score"] = float(parts["score"][i])
    if o["kind"] == "pass":
        a["target"] = o["target"]
        a["v_lost"] = float(parts["v_lost"][i])
    elif o["kind"] == "carry":
        a["angle"] = o["angle"]
        a["v_lost"] = float(parts["v_lost"][i])
    return a

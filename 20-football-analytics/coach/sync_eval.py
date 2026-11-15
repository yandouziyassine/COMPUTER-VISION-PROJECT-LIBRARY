"""Coach v2 data check: event sync "legacy" vs "release" and failed-pass target inference "ray" vs "direction".

    python -m coach.sync_eval            # from 20-football-analytics/ (~2 min, the 7 converted Bundesliga matches)

Reads the converted matches (cv-data/football-coach) and re-syncs them in memory with both methods (nothing is
written but docs/coach_sync_results.json). Per match and pooled:

1. Passes placed, drop reasons, clock offsets, and checks that neither method optimises: ball speed jump across the
   chosen frame (a kick shows a jump), completed passes whose recipient touches the ball next / comes within 2 m within
   6 s, failed passes followed by an opponent's touch or a dead ball; also the logged spot's distance (which "legacy"
   minimises, so it is not a fair check) and the training rows each gives (open-play passes in the pass table).
2. Timing against an independent reference: the provider's CalculatedFrame of the shots. Each method places the shots
   with use_calc=False and the error to the CalculatedFrame is measured.
3. Failed-pass target inference: the DFL events do not record the intended receiver of a failed pass. Proxy: the
   completed passes with a clean reception, their ball path cut at 40 / 60 / 80 % (as if intercepted there), the
   receiver inferred by each rule. Where the provider records the intended receiver of failed passes (SkillCorner
   A-League, raw store on the USB key), the real accuracy is measured too (skipped when the store is offline).
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from . import datasets as D
from . import features as F
from . import sync
from .match import load
from .prepare import MATCHES, OUT

DOCS = Path(__file__).resolve().parents[1] / "docs"
CUTS = (0.4, 0.6, 0.8)


def shot_timing(m, method: str) -> list:
    """Error (s) between each method's frame (provider frame ignored) and the provider's CalculatedFrame, per shot."""
    mm = copy.deepcopy(m)
    sync.attach_frames(mm, method, use_calc=False)
    fps = sync._fps(mm)
    out = []
    for e in mm.events:
        if e["type"] != "shot" or e.get("calc_frame") is None or e.get("half") is None:
            continue
        h = mm.halves[e["half"]]
        k = int(np.searchsorted(h.n, e["calc_frame"]))
        if k >= len(h.n) or h.n[k] != e["calc_frame"]:
            continue
        out.append(None if e.get("frame") is None else (e["frame"] - k) / fps)
    return out


def target_proxy(m) -> dict:
    """Completed passes received cleanly (release sync on m), path cut at CUTS: rule -> list of hits (bool)."""
    out = {f"{r}@{q}": [] for r in D.TARGET_RULES for q in CUTS}
    for e in m.events:
        if e["type"] != "pass" or not e.get("success") or e.get("follow") != "recipient":
            continue
        if e.get("set_piece") or e.get("cross") or e.get("reception") is None or e["reception"] - e["frame"] < 5:
            continue
        f, rec = e["frame"], e["reception"]
        h = m.halves[e["half"]]
        s = F.state(m, e["half"], f, e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where or e["recipient"] not in where:
            continue
        rows = [j for p, j in where.items() if p != e["player"]]
        u0, speed = D.ball_direction(m, e, s)
        for q in CUTS:
            point = h.ball[f + int(round(q * (rec - f))), :2] * s["direction"]
            for r in D.TARGET_RULES:
                got = D.infer_target(s, where[e["player"]], rows, point, True, u0, speed, r)
                out[f"{r}@{q}"].append(got == where[e["recipient"]])
    return out


def target_known(m) -> dict:
    """Failed passes whose intended receiver is recorded (SkillCorner): rule -> hits, the record ignored."""
    out = {r: [] for r in D.TARGET_RULES}
    for e in m.events:
        if e["type"] != "pass" or e.get("success") or not e.get("target_known") or e.get("frame") is None:
            continue
        if e.get("set_piece") or e.get("cross") or e["team"] < 0 or e.get("player") is None:
            continue
        s = F.state(m, e["half"], e["frame"], e["team"])
        where = {p: i for i, p in enumerate(s["att_idx"])}
        if e["player"] not in where or e.get("recipient") not in where:
            continue
        for r in D.TARGET_RULES:
            found = D.intended_target(m, e, s, where, r)
            out[r].append(found is not None and found[0] == where[e["recipient"]])
    return out


def pass_rows(m, rule) -> int:
    """Open-play passes that reach the pass table (placed, passer tracked, target known or inferred)."""
    return sum(1 for _ in D.pass_samples(m, rule))


def _share(v, f):
    v = [x for x in v if x is not None]
    return float(np.mean([f(x) for x in v])) if v else None


def evaluate(mids=MATCHES) -> dict:
    res = {"matches": {}, "pooled": {}}
    pooled = {"shots": {k: [] for k in sync.METHODS}, "proxy": {}, "frames": [], "rows": {}}
    for mid in mids:
        m0 = load(OUT / mid)
        r = {}
        frames = {}
        for method in sync.METHODS:
            m = copy.deepcopy(m0)
            r[method] = sync.attach_frames(m, method)
            frames[method] = [e.get("frame") for e in m.events]
            for rule in D.TARGET_RULES if method == "release" else ("ray",):
                n = pass_rows(m, rule)
                r[method][f"pass_table_rows_{rule}"] = n
                pooled["rows"][f"{method}_{rule}"] = pooled["rows"].get(f"{method}_{rule}", 0) + n
            st = shot_timing(m0, method)
            pooled["shots"][method] += st
            if method == "release":
                for k, v in target_proxy(m).items():
                    pooled["proxy"].setdefault(k, []).extend(v)
        common = [
            (b - a) / sync._fps(m0)
            for e, a, b in zip(m0.events, frames["legacy"], frames["release"])
            if e["type"] == "pass" and a is not None and b is not None
        ]
        pooled["frames"] += common
        r["common_passes"] = len(common)
        res["matches"][mid] = r
        print(mid, {k: (r[k]["passes_kept"], r[k]["passes"]) for k in sync.METHODS}, flush=True)
    P = res["pooled"]
    for method in sync.METHODS:
        audits = [res["matches"][mid][method] for mid in mids]
        n_pass = sum(a["passes"] for a in audits)
        kept = sum(a["passes_kept"] for a in audits)
        q = {}
        for key in audits[0]["quality"]:
            if key == "passes_placed":
                continue
            w = np.array([a["quality"]["passes_placed"] for a in audits], float)
            v = np.array([a["quality"][key] for a in audits], float)
            q[key + "_match_weighted"] = float(np.nansum(w * v) / w[np.isfinite(v)].sum())
        st = pooled["shots"][method]
        drops = {}
        for a in audits:
            for k, v in a["dropped"].items():
                drops[k] = drops.get(k, 0) + v
        P[method] = {
            "passes": n_pass,
            "passes_kept": kept,
            "passes_kept_share": kept / n_pass,
            "shots_kept": sum(a["shots_kept"] for a in audits),
            "shots": sum(a["shots"] for a in audits),
            "dropped": drops,
            "quality": q,
            "shots_vs_calc_frame": {
                "n": len(st),
                "placed": sum(x is not None for x in st),
                "within_0.2s": _share(st, lambda x: abs(x) <= 0.2),
                "within_0.4s": _share(st, lambda x: abs(x) <= 0.4),
                "median_abs_error_s": float(np.median([abs(x) for x in st if x is not None])),
            },
        }
    P["pass_table_rows"] = pooled["rows"]
    fr = np.array(pooled["frames"])
    P["common_passes"] = {
        "n": int(len(fr)),
        "release_minus_legacy_median_s": float(np.median(fr)),
        "within_0.2s": float(np.mean(np.abs(fr) <= 0.2)),
        "release_later_share": float(np.mean(fr > 0.2)),
    }
    P["target_proxy"] = {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in pooled["proxy"].items()}
    res["target_known_aleague"] = aleague_targets()
    return res


def aleague_targets() -> dict:
    """Real accuracy of the target rules on SkillCorner failed passes with a recorded receiver (raw store needed)."""
    from . import skillcorner as SC

    try:
        stored = SC.stored_matches()
    except OSError:
        stored = []
    if not stored:
        return {"skipped": "SkillCorner raw store offline (python -m coach.skillcorner --store-raw ...)"}
    hits = {r: [] for r in D.TARGET_RULES}
    for mid in stored:
        for r, v in target_known(SC.load_raw_match(mid)).items():
            hits[r] += v
    return {"matches": len(stored), **{r: {"n": len(v), "accuracy": float(np.mean(v))} for r, v in hits.items()}}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=DOCS / "coach_sync_results.json")
    args = ap.parse_args()
    from .stamp import generated_by

    res = {"generated_by": generated_by(), **evaluate()}
    args.out.write_text(json.dumps(_clean(res), indent=1), encoding="utf-8")
    print(json.dumps(_clean(res["pooled"]), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

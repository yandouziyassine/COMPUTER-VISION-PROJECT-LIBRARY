"""Ghosts and explanations, checked (coach insight, part B): association on held-out passes and the cost per pause.

    python -m coach.ghosts_eval      # from 20-football-analytics/ -> docs/coach_ghosts_results.json (~15 min on CPU)

Association (not causation). For every usable pass of the 7 IDSSE matches (datasets.pass_samples), with pass and
value models trained WITHOUT that match: go back ghosts.HORIZON_S (1.5 s) before the release; if the passer had the
ball then (ball in play, his team in possession, ball within CARRIER_M of him), compute the ghost runs of his
teammates at that anchor frame. For each ghost teammate still on the pitch at the release: progress = how much of the
ghost run he actually made by the release (projection of his real displacement on the ghost vector, divided by its
length); followed = progress >= FOLLOW. Outcomes at the release: did he receive the pass, the score of the pass to him
(advice.py's formula on the real state) and its change since the anchor. Followed vs not followed, with 95 % intervals
resampling whole matches. Moving towards a spot the model likes raises the model's score of a pass there partly by
construction (the defenders did react in reality, they don't in the ghost), so "realised share" (mean change / mean
predicted gain) is the more telling number; "received" is the behavioural outcome.

Timing: ghosts per paused frame in Python on the parity frames (exported weights), with model evaluation counts.
"""

import json
import time
from pathlib import Path

import numpy as np

from . import advice as A
from . import evaluate as EV
from . import features as F
from . import ghosts as GH

FOLLOW = 0.5
CARRIER_M = 2.0  # same rule as the site's carrier (parity_fixtures.CARRIER_M)
DOCS = Path(__file__).resolve().parents[1] / "docs"


def progress(start, end, ghost) -> float:
    """Share of the ghost run start -> ghost made by moving start -> end (projection, can be < 0 or > 1)."""
    g = np.asarray(ghost, float) - np.asarray(start, float)
    d = np.asarray(end, float) - np.asarray(start, float)
    return float(d @ g / max(float(g @ g), 1e-9))


def anchor_state(m, e: dict, back: int):
    """(state, passer row) at `back` frames before the release when the passer had the ball there, else None."""
    f = int(e["frame"]) - back
    h = m.halves[e["half"]]
    if f < 0 or not h.alive[f] or int(h.owner[f]) != e["team"] or np.isnan(h.ball[f, 0]):
        return None
    pos = h.xy[f, e["player"]]
    if np.isnan(pos[0]) or float(np.hypot(*(pos - h.ball[f, :2]))) > CARRIER_M:
        return None
    s = F.state(m, e["half"], f, e["team"])
    rows = np.flatnonzero(s["att_idx"] == e["player"])
    return (s, int(rows[0])) if len(rows) else None


def association_records(m, models: dict, pass_samples=None) -> list:
    """One record per ghost teammate per usable pass (see the module docstring)."""
    from . import datasets as D

    back = int(round(GH.HORIZON_S * F.match_fps(m)))
    out = []
    for p in pass_samples if pass_samples is not None else D.pass_samples(m):
        e, s_r, c_r = p["event"], p["state"], p["carrier"]
        found = anchor_state(m, e, back)
        if found is None:
            continue
        s_a, c_a = found
        chosen = int(s_r["att_idx"][p["target"]])
        v_now = float(A.value_at(models, [s_r["att"][c_r]])[0])
        for g in GH.ghost_runs(s_a, c_a, models):
            rows = np.flatnonzero(s_r["att_idx"] == g["player"])
            if not len(rows) or rows[0] == c_r:
                continue
            j = int(rows[0])
            score = float(GH.pass_scores([s_r], c_r, j, models, v_now)[0])
            prog = progress(g["from"], s_r["att"][j], g["to"])
            out.append(
                {
                    "match": m.match_id,
                    "event": int(p["index"]),
                    "player": g["player"],
                    "progress": prog,
                    "followed": prog >= FOLLOW,
                    "received": g["player"] == chosen,
                    "gain": g["gain"],
                    "score_anchor": g["score_now"],
                    "score_release": score,
                    "change": score - g["score_now"],
                }
            )
    return out


def _mean(rs, key):
    return float(np.mean([float(r[key]) for r in rs])) if rs else float("nan")


def _diff(rs, key):
    """mean(key | followed) - mean(key | not followed)."""
    return _mean([r for r in rs if r["followed"]], key) - _mean([r for r in rs if not r["followed"]], key)


def association_summary(recs: list) -> dict:
    """Followed vs not followed: n, P(received), mean score at the release and mean change since the anchor, the
    score of the passes they received, and the share of the predicted gain realised; differences with match-bootstrap
    95 % intervals (evaluate.bootstrap_ci)."""
    groups = {"followed": [r for r in recs if r["followed"]], "not_followed": [r for r in recs if not r["followed"]]}
    out = {
        "label": "association only, not causation",
        "records": len(recs),
        "passes": len({(r["match"], r["event"]) for r in recs}),
    }
    for name, rs in groups.items():
        rec = [r for r in rs if r["received"]]
        out[name] = {
            "n": len(rs),
            "received": _mean(rs, "received"),
            "score_release": _mean(rs, "score_release"),
            "change": _mean(rs, "change"),
            "predicted_gain": _mean(rs, "gain"),
            "realised_share": _mean(rs, "change") / _mean(rs, "gain") if rs else float("nan"),
            "received_n": len(rec),
            "received_score": _mean(rec, "score_release"),
        }
    received = lambda rs: _diff([r for r in rs if r["received"]], "score_release")
    out["difference"] = {
        k: {"value": _diff(recs, k), "ci95": EV.bootstrap_ci(recs, lambda rs, k=k: _diff(rs, k))}
        for k in ("received", "score_release", "change")
    }
    out["difference"]["received_score"] = {"value": received(recs), "ci95": EV.bootstrap_ci(recs, received)}
    return out


class Counter:
    """Wraps advice model functions and counts the rows each one evaluates."""

    def __init__(self, models: dict):
        self.rows = {k: 0 for k in models}
        self.models = {k: self._wrap(k, fn) for k, fn in models.items()}

    def _wrap(self, k, fn):
        def run(X):
            self.rows[k] += len(np.atleast_2d(X))
            return fn(X)

        return run


def timing(frames: list, models: dict) -> dict:
    """Python wall time per paused frame for ghost_runs and ghost_defence, plus the model rows and advice.options
    calls they need. `frames`: (state, carrier row) pairs."""
    res = {"frames": len(frames), "runs_ms": [], "defence_ms": [], "runs_rows": [], "defence_rows": [], "ghosts": []}
    for s, c in frames:
        cnt = Counter(models)
        t0 = time.perf_counter()
        runs = GH.ghost_runs(s, c, cnt.models)
        res["runs_ms"].append(1000 * (time.perf_counter() - t0))
        res["runs_rows"].append(dict(cnt.rows))
        cnt = Counter(models)
        t0 = time.perf_counter()
        dfn = GH.ghost_defence(s, c, cnt.models)
        res["defence_ms"].append(1000 * (time.perf_counter() - t0))
        res["defence_rows"].append(dict(cnt.rows))
        res["ghosts"].append([len(runs), len(dfn)])
    steps = lambda n_def: sum(n_def - k for k in range(GH.MAX_GHOSTS))
    n_def = [len(s["def"]) for s, _ in frames]
    return {
        "frames": len(frames),
        "runs_ms_mean": float(np.mean(res["runs_ms"])),
        "defence_ms_mean": float(np.mean(res["defence_ms"])),
        "defence_ms_max": float(np.max(res["defence_ms"])),
        "runs_pass_rows_mean": float(np.mean([r["pass"] for r in res["runs_rows"]])),
        "defence_pass_rows_mean": float(np.mean([r["pass"] for r in res["defence_rows"]])),
        "defence_options_calls_max": int(max(GH.FAN_N * steps(n) + 1 for n in n_def)),
        "ghost_runs_per_frame": float(np.mean([g[0] for g in res["ghosts"]])),
        "ghost_defenders_per_frame": float(np.mean([g[1] for g in res["ghosts"]])),
    }


def parity_frames(fx: dict) -> list:
    """(state, carrier row) of every frame of a parity fixture (coach.parity_fixtures)."""
    out = []
    for rec in fx["frames"]:
        st = rec["state"]
        s = {k: np.array(st[k], float) for k in ("att", "att_v", "def", "def_v", "ball")}
        s.update({"att_idx": np.array(st["att_idx"], int), "def_idx": np.array(st["def_idx"], int)})
        s["direction"] = st["direction"]
        out.append((s, rec["carrier_row"]))
    return out


def main():
    from . import parity_fixtures as PF
    from . import report as R
    from .match import load
    from .prepare import MATCHES, OUT

    fx = json.loads(PF.SITE_FIXTURE.read_text(encoding="utf-8"))
    tim = timing(parity_frames(fx), PF.predictors(fx["models"]))
    print("timing", tim, flush=True)
    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: R.match_tables(m) for mid, m in matches.items()}
    recs = []
    for mid, m in matches.items():
        models, _, _ = R.held_out_models(mid, tables)
        got = association_records(m, models)
        recs += got
        print(mid, len(got), sum(r["followed"] for r in got), flush=True)
    res = {
        "params": {**GH.params(), "follow": FOLLOW},
        "timing_python": tim,
        "association": association_summary(recs),
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_ghosts_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res["association"], indent=1))


if __name__ == "__main__":
    main()

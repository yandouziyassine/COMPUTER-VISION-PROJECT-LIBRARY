"""Match report (coach insight, part A): decision quality and missed opportunities in one match.

    python -m coach.report      # from 20-football-analytics/: held-out summary over the 7 IDSSE matches
                                # -> docs/coach_report_results.json + docs/coach_report_gaps.png

For every pass the sync kept (datasets.pass_samples), with models trained WITHOUT the match (asserted in build): the
coach's options at the release frame (advice.options: passes, carries, shot), the pass the player chose, the best
option, gap = score(best) - score(chosen) >= 0 in expected goals, the chosen option's rank among all options, and what
happened within HORIZON s: goal / shot by the passing team, else kept / lost (the pass completed or not), with the net
xG. The per-pass work is evaluate.pass_decisions, shared with the held-out advice evaluation.

Frames: "frame_src" indexes the source half (25 fps IDSSE, 10 fps SkillCorner); "frame" is the index at the site's
rate, frame_src // step with step = round(fps_src / 12.5): 2 for IDSSE (the 12.5 fps of export.pack_match), 1 for
SkillCorner. "event" is the index in m.events, which is also the index in the site's match.json events.

"Missed opportunity" means the model valued another option more, not a mistake (see CAVEATS). Player and team numbers
carry n and too_few (n < MIN_DECISIONS): below it the site shows "too few decisions" instead of the numbers.
"""

import json
from pathlib import Path

import numpy as np

from . import advice as A
from . import datasets as D
from . import evaluate as EV
from . import export as X
from . import features as F

SITE_FPS = 12.5
HORIZON = D.HORIZON
MIN_DECISIONS = 20
MIN_BEST_P = 0.5
N_HIGHLIGHTS = 10
THIRD_M = 52.5 / 3  # pitch thirds at x = -17.5 / +17.5 m, attack-normalised (the passing team attacks +x)
THIRDS = ("defensive", "middle", "attacking")
DOCS = Path(__file__).resolve().parents[1] / "docs"


def site_step(fps_src: float, site_fps: float = SITE_FPS) -> int:
    """Whole subsampling step from the source frame rate to the site's (25 -> 2, as export.pack_match; 10 -> 1)."""
    return max(1, int(round(fps_src / site_fps)))


def third(x: float) -> str:
    return "defensive" if x < -THIRD_M else "attacking" if x > THIRD_M else "middle"


def option_key(o: dict) -> dict:
    """What identifies an option: its kind, plus the target player (pass) or the direction in degrees (carry)."""
    key = {"kind": o["kind"]}
    if o["kind"] == "pass":
        key["target"] = int(o["target"])
    elif o["kind"] == "carry":
        key["angle"] = int(o["angle"])
    return key


def shots_with_goals(m) -> list:
    """Per half: array (time on the tracking clock, team, goal 0/1) of every shot with a team, sorted by time."""
    out = []
    for hi in range(len(m.halves)):
        rows = sorted(
            (D.when(m, e), e["team"], float(bool(e.get("goal"))))
            for e in m.events
            if e["type"] == "shot" and e.get("half") == hi and e["team"] in (0, 1)
        )
        out.append(np.array(rows, float).reshape(-1, 3))
    return out


def outcome(shots, t: float, team: int, success: bool, horizon: float = HORIZON) -> str:
    """goal / shot when the passing team scores / shoots in (t, t + horizon], else kept / lost (pass completed or not).
    `shots`: one half of shots_with_goals."""
    if len(shots):
        mine = (shots[:, 0] > t) & (shots[:, 0] <= t + horizon) & (shots[:, 1] == team)
        if (shots[mine, 2] > 0).any():
            return "goal"
        if mine.any():
            return "shot"
    return "kept" if success else "lost"


def _r(x, nd=4):
    return round(float(x), nd)


def decision_row(d: dict, step: int, shots: list, horizon: float = HORIZON) -> dict:
    """One decision from an evaluate.pass_decisions record (options sorted by coach score, best first)."""
    p, opts, chosen = d["sample"], d["options"], d["chosen"]
    e, s = p["event"], p["state"]
    best = opts[0]
    scores = np.array([o["score"] for o in opts], float)
    return {
        "event": int(p["index"]),
        "half": int(e["half"]),
        "frame": int(e["frame"]) // step,
        "frame_src": int(e["frame"]),
        "team": int(e["team"]),
        "carrier": int(e["player"]),
        "target": int(chosen["target"]),
        "best": option_key(best),
        "gap": _r(max(0.0, best["score"] - chosen["score"]), 5),
        "rank": 1 + next(i for i, o in enumerate(opts) if o is chosen),
        "options": len(opts),
        "p_chosen": _r(chosen["p"], 3),
        "p_best": _r(best["p"], 3),
        "score_chosen": _r(chosen["score"], 5),
        "above_median": _r(chosen["score"] - float(np.median(scores)), 5),
        "third": third(float(s["att"][p["carrier"]][0])),
        "outcome": outcome(shots[e["half"]], d["t"], e["team"], bool(e["success"]), horizon),
        "net_xg": _r(d["outcome"], 4),
    }


CAVEATS = [
    "Everything here is the model's opinion. A missed opportunity means the model valued another option more, "
    "not that the player made a mistake.",
    "The value model is weak: on held-out matches it explains about 2 % of the variance of the net xG in the next "
    "15 s, little more than 'closer to the goal is better'.",
    "The coach is bolder than professional players: its top-ranked options are riskier, and pros complete more of the "
    "passes it ranks lower.",
    "Only passes the event sync placed on the tracking are included (about two thirds of open-play passes in the "
    "Bundesliga data); the intended receiver of a failed pass is inferred from the ball's path.",
    f"Player and team numbers need at least {MIN_DECISIONS} decisions; below that they are flagged too_few.",
]
FIELDS = {
    "event": "index in the match's events (= match.json events)",
    "half": "0 or 1",
    "frame": "frame index in the half at the site's rate (fps)",
    "frame_src": "frame index in the half at the source rate (fps_src)",
    "team": "0 home, 1 guest",
    "carrier": "player index (match.json players) of the passer",
    "target": "player index of the receiver the passer chose (given or inferred)",
    "best": "the coach's best option: kind (pass / carry / shot), target player (pass) or angle in degrees (carry, "
    "relative to the attacking direction)",
    "gap": "score(best) - score(chosen), expected goals, >= 0",
    "rank": "rank of the chosen pass among all options by coach score (1 = the coach's best)",
    "options": "number of options",
    "p_chosen": "success probability of the chosen pass",
    "p_best": "success probability of the best option",
    "score_chosen": "coach score of the chosen pass (expected change of the net xG)",
    "above_median": "score(chosen) - median score of the options, expected goals",
    "third": "pitch third of the passer (defensive / middle / attacking, for the passing team)",
    "outcome": "within horizon_s: goal / shot by the passing team, else kept / lost (pass completed or not)",
    "net_xg": "xG of the passing team minus xG against within horizon_s",
}


def summarise(rows) -> dict:
    """n, share of choices that were the coach's top-1 / top-3 option, mean and total gap (expected goals);
    too_few when n < MIN_DECISIONS."""
    n = len(rows)
    gaps = np.array([r["gap"] for r in rows], float)
    ranks = np.array([r["rank"] for r in rows], int)
    return {
        "n": n,
        "too_few": n < MIN_DECISIONS,
        "top1": _r(np.mean(ranks == 1), 3) if n else None,
        "top3": _r(np.mean(ranks <= 3), 3) if n else None,
        "mean_gap": _r(gaps.mean(), 5) if n else None,
        "total_gap": _r(gaps.sum(), 4),
    }


def by_player(m, rows) -> dict:
    """Per passer (key: player index as a string): name, team, shirt and summarise()."""
    out = {}
    for pid in sorted({r["carrier"] for r in rows}):
        info = {"name": m.names[pid], "team": int(m.team[pid]), "shirt": int(m.shirt[pid])}
        out[str(pid)] = {**info, **summarise([r for r in rows if r["carrier"] == pid])}
    return out


def by_team(m, rows) -> dict:
    """Per team ("0" home, "1" guest): name, summarise() and the same per pitch third of the passer."""
    out = {}
    for t in (0, 1):
        mine = [r for r in rows if r["team"] == t]
        out[str(t)] = {
            "name": m.meta.get(("home", "guest")[t], str(t)),
            **summarise(mine),
            "by_third": {k: summarise([r for r in mine if r["third"] == k]) for k in THIRDS},
        }
    return out


def highlights(rows, k: int = N_HIGHLIGHTS, min_best_p: float = MIN_BEST_P) -> dict:
    """missed: the k largest gaps among decisions whose best option succeeds with p >= min_best_p (the coach is not
    asking for a miracle); best: the k decisions where the chosen pass was the coach's best option, by how far its score
    is above the median option (the player found the rare good option). Entries point into "decisions"."""
    missed = [i for i, r in enumerate(rows) if r["gap"] > 0 and r["p_best"] >= min_best_p]
    missed = sorted(missed, key=lambda i: (-rows[i]["gap"], i))[:k]
    best = sorted((i for i, r in enumerate(rows) if r["rank"] == 1), key=lambda i: (-rows[i]["above_median"], i))[:k]
    return {
        "missed": [{"decision": i, "gap": rows[i]["gap"]} for i in missed],
        "best": [{"decision": i, "above_median": rows[i]["above_median"]} for i in best],
    }


def gap_summary(rows) -> dict:
    """summarise() plus the gap distribution, the kind of the best option and the highlight candidate counts."""
    out = summarise(rows)
    if not rows:
        return out
    gaps = np.array([r["gap"] for r in rows], float)
    kinds = [r["best"]["kind"] for r in rows]
    q = np.percentile(gaps, [50, 75, 90, 99])
    out.update(
        {
            "gap_median": _r(q[0], 5),
            "gap_p75": _r(q[1], 5),
            "gap_p90": _r(q[2], 5),
            "gap_p99": _r(q[3], 5),
            "gap_max": _r(gaps.max(), 5),
            "best_kind": {kd: _r(kinds.count(kd) / len(kinds), 3) for kd in ("pass", "carry", "shot")},
            "missed_candidates": sum(r["gap"] > 0 and r["p_best"] >= MIN_BEST_P for r in rows),
            "best_candidates": sum(r["rank"] == 1 for r in rows),
        }
    )
    return out


def build(m, models: dict, trained_on, horizon: float = HORIZON) -> dict:
    """The report of match m. `models`: advice.from_fitted-style functions; `trained_on`: every match id any of the
    models learned from (asserted not to contain m)."""
    trained_on = [str(t) for t in trained_on]
    assert str(m.match_id) not in trained_on, f"match {m.match_id} is in the training set"
    fps_src = F.match_fps(m)
    step = site_step(fps_src)
    shots = shots_with_goals(m)
    rows = [decision_row(d, step, shots, horizon) for d in EV.pass_decisions(m, models, horizon)]
    open_play = sum(
        e["type"] == "pass" and not e.get("set_piece") and not e.get("cross") and e["team"] >= 0 for e in m.events
    )
    return {
        "match_id": str(m.match_id),
        "fps": fps_src / step,
        "fps_src": fps_src,
        "horizon_s": horizon,
        "trained_without": str(m.match_id),
        "trained_on": trained_on,
        "thresholds": {
            "min_decisions": MIN_DECISIONS,
            "missed_min_best_p": MIN_BEST_P,
            "highlights": N_HIGHLIGHTS,
            "third_x_m": round(THIRD_M, 2),
        },
        "fields": FIELDS,
        "caveats": CAVEATS,
        "summary": {
            **gap_summary(rows),
            "open_play_passes": open_play,
            "usable_share": _r(len(rows) / open_play, 3) if open_play else None,
        },
        "teams": by_team(m, rows),
        "players": by_player(m, rows),
        "highlights": highlights(rows),
        "decisions": rows,
    }


def match_tables(m) -> dict:
    """The training tables of one match (pass, value at the main horizon, shot with penalties flagged)."""
    return {"pass": D.pass_table(m), "value": D.value_table(m), "shot": D.shot_table(m)}


def held_out_models(target: str, tables: dict, extra_shots=None) -> tuple:
    """(advice functions, fitted models, trained_on) for match `target` with the site's recipe (export.fit_models):
    pass and value models on the other matches of `tables` (match id -> match_tables), shot model on their
    non-penalty shots plus the `extra_shots` rows (X, y, match) of matches other than `target`."""
    others = [mid for mid in tables if mid != target]
    stacked = {k: D.stack([tables[mid][k] for mid in others]) for k in ("pass", "value", "shot")}
    stacked["shot"] = D.drop_penalties(stacked["shot"])
    trained_on = list(others)
    if extra_shots is not None:
        keep = extra_shots["match"] != target
        extra_shots = {k: v[keep] for k, v in extra_shots.items()}
        trained_on += sorted(set(extra_shots["match"].tolist()) - set(trained_on))
    fitted = X.fit_models(stacked, extra_shots)
    return A.from_fitted(fitted), fitted, trained_on


GAP_BINS = (0.0, 1e-9, 0.0025, 0.005, 0.01, 0.02, 0.04, 1.0)


def held_out_summary(reports) -> dict:
    """Per match and pooled numbers of several held-out reports: gap distribution, top-1 / top-3, candidates."""
    rows = [r for rep in reports for r in rep["decisions"]]
    ranks = [min(r["rank"], 10) for r in rows]
    counts, _ = np.histogram([r["gap"] for r in rows], bins=GAP_BINS)
    outcomes = [r["outcome"] for r in rows]
    share = {k: _r(outcomes.count(k) / len(rows), 3) for k in ("goal", "shot", "kept", "lost")} if rows else {}
    return {
        "matches": {rep["match_id"]: rep["summary"] for rep in reports},
        "trained_on": {rep["match_id"]: rep["trained_on"] for rep in reports},
        "pooled": {
            **gap_summary(rows),
            "open_play_passes": sum(rep["summary"]["open_play_passes"] for rep in reports),
            "outcome_share": share,
        },
        "rank_counts": {("10+" if k == 10 else str(k)): ranks.count(k) for k in range(1, 11)},
        "gap_histogram": {"edges": list(GAP_BINS), "counts": counts.tolist(), "note": "first bin: gap = 0"},
    }


def gap_chart(summary: dict, path: Path) -> None:
    """Two plain panels: distribution of the gap, and the coach's rank of the pass the player chose."""
    plt = EV._pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    h = summary["gap_histogram"]
    e = h["edges"]
    labels = ["0"] + [f"{e[i] * 1000:g}–{e[i + 1] * 1000:g}" for i in range(1, len(e) - 2)] + [f">{e[-2] * 1000:g}"]
    axes[0].bar(labels, h["counts"], color="#4c78a8")
    axes[0].set(xlabel="gap to the coach's best option (milli-xG)", ylabel="passes")
    rc = summary["rank_counts"]
    axes[1].bar(list(rc), list(rc.values()), color="#4c78a8")
    axes[1].set(xlabel="coach rank of the pass the player chose (all options)", ylabel="passes")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    from . import cross_league as CL
    from .match import load
    from .prepare import MATCHES, OUT

    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: match_tables(m) for mid, m in matches.items()}
    extra = CL.skillcorner_shots()[0]
    reports = []
    for mid, m in matches.items():
        models, _, trained_on = held_out_models(mid, tables, extra)
        reports.append(build(m, models, trained_on))
        s = reports[-1]["summary"]
        print(mid, s["n"], s["top1"], s["top3"], s["gap_median"], s["gap_p90"], s["missed_candidates"], flush=True)
    res = held_out_summary(reports)
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_report_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    gap_chart(res, DOCS / "coach_report_gaps.png")
    print(json.dumps(res["pooled"], indent=1))


if __name__ == "__main__":
    main()

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

from pathlib import Path

import numpy as np

from . import datasets as D

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

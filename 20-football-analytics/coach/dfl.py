"""Streaming parser for the DFL / IDSSE XML files (match information, raw positions, raw events).

The positions file is ~400 MB: it is read with iterparse and each element is cleared after use, so memory stays at
the size of the output arrays (~25 MB per match). Players are stored in the order they first appear.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from .match import Half

SECTIONS = {"firstHalf": 0, "secondHalf": 1}
SHOT_RESULTS = ("SuccessfulShot", "SavedShot", "ShotWide", "BlockedShot", "OtherShot")
SET_PIECES = ("KickOff", "FreeKick", "CornerKick", "ThrowIn", "GoalKick", "Penalty")


def epoch(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def parse_info(path) -> dict:
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    teams, players, colors = {}, {}, []
    for team in root.iter("Team"):
        idx = 0 if team.get("Role") == "home" else 1
        teams[team.get("TeamId")] = idx
        colors.append((idx, team.get("PlayerShirtMainColor")))
        for p in team.iter("Player"):
            players[p.get("PersonId")] = {
                "team": idx,
                "shirt": int(p.get("ShirtNumber")),
                "name": p.get("Shortname"),
                "position": p.get("PlayingPosition"),
            }
    meta = {
        "match_id": general.get("MatchId"),
        "home": general.get("HomeTeamName"),
        "guest": general.get("GuestTeamName"),
        "colors": [c for _, c in sorted(colors)],
    }
    return {"teams": teams, "players": players, "meta": meta}


def parse_positions(path, info):
    """Return (player_ids, [Half, Half]). Frames of a half = union of all frame numbers seen in that half."""
    raw = {0: {}, 1: {}}  # half -> person -> (N list, x list, y list)
    ball = {0: {}, 1: {}}  # half -> N -> (t, x, y, z, owner, alive)
    order = []
    current = None
    for event, el in ET.iterparse(path, events=("start", "end")):
        if event == "start" and el.tag == "FrameSet":
            half = SECTIONS[el.get("GameSection")]
            pid = el.get("PersonId")
            is_ball = el.get("TeamId") == "BALL"
            if not is_ball and pid not in order:
                order.append(pid)
            current = (half, pid, is_ball, [], [], [])
        elif event == "end" and el.tag == "Frame":
            half, pid, is_ball, ns, xs, ys = current
            n = int(el.get("N"))
            if is_ball:
                ball[half][n] = (
                    epoch(el.get("T")),
                    float(el.get("X")),
                    float(el.get("Y")),
                    float(el.get("Z", 0)),
                    int(el.get("BallPossession", 0)) - 1,
                    el.get("BallStatus") == "1",
                )
            else:
                ns.append(n)
                xs.append(float(el.get("X")))
                ys.append(float(el.get("Y")))
            el.clear()
        elif event == "end" and el.tag == "FrameSet":
            half, pid, is_ball, ns, xs, ys = current
            if not is_ball:
                raw[half][pid] = (np.array(ns), np.array(xs, np.float32), np.array(ys, np.float32))
            el.clear()
    halves = []
    for half in (0, 1):
        frames = sorted(set(ball[half]) | {n for ns, _, _ in raw[half].values() for n in ns})
        idx = {n: i for i, n in enumerate(frames)}
        F, P = len(frames), len(order)
        xy = np.full((F, P, 2), np.nan, np.float32)
        for p, pid in enumerate(order):
            if pid in raw[half]:
                ns, xs, ys = raw[half][pid]
                rows = np.array([idx[n] for n in ns])
                xy[rows, p, 0], xy[rows, p, 1] = xs, ys
        b = np.full((F, 3), np.nan, np.float32)
        t = np.full(F, np.nan)
        owner = np.full(F, -1, np.int8)
        alive = np.zeros(F, bool)
        for n, (tt, x, y, z, o, a) in ball[half].items():
            i = idx[n]
            t[i], b[i], owner[i], alive[i] = tt, (x, y, z), o, a
        n_arr = np.array(frames, np.int32)
        known = ~np.isnan(t)  # frames without a ball sample: time from the 25 fps clock
        t = np.interp(n_arr, n_arr[known], t[known]) if known.any() else t
        halves.append(Half(n=n_arr, t=t, xy=xy, ball=b, owner=owner, alive=alive))
    return order, halves


def parse_events(path, info) -> list:
    """Passes (Play with Pass/Cross child) and shots, in time order; ids kept as PersonId strings."""
    teams = info["teams"]
    out = []
    for ev in ET.parse(path).getroot().iter("Event"):
        t = epoch(ev.get("EventTime"))
        x, y = float(ev.get("X-Position", "nan")), float(ev.get("Y-Position", "nan"))
        for child in ev:
            set_piece = child.tag if child.tag in SET_PIECES else None
            play = child if child.tag == "Play" else child.find("Play")
            if play is not None and (play.find("Pass") is not None or play.find("Cross") is not None):
                out.append(
                    {
                        "type": "pass",
                        "t": t,
                        "x": x,
                        "y": y,
                        "team": teams.get(play.get("Team"), -1),
                        "player": play.get("Player"),
                        "recipient": play.get("Recipient"),
                        "success": play.get("Evaluation") == "successfullyCompleted",
                        "cross": play.find("Cross") is not None,
                        "set_piece": set_piece,
                    }
                )
            for shot in child.iter("ShotAtGoal"):  # also nested in <Penalty> / <FreeKick>
                result = next((c.tag for c in shot if c.tag in SHOT_RESULTS), None)
                frame, xg = ev.get("CalculatedFrame"), shot.get("xG")
                out.append(
                    {
                        "type": "shot",
                        "t": t,
                        "x": x,
                        "y": y,
                        "team": teams.get(shot.get("Team"), -1),
                        "player": shot.get("Player"),
                        "result": result,
                        "goal": result == "SuccessfulShot",
                        "set_piece": child.tag if child.tag in ("Penalty", "FreeKick") else None,
                        "xg": float(xg) if xg else None,  # provider's expected goals
                        "calc_frame": int(frame) if frame else None,  # provider frame number (not an index)
                    }
                )
    out.sort(key=lambda e: e["t"])
    return out

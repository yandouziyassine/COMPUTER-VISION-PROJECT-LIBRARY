"""SkillCorner open data (A-League 2024/25, broadcast tracking at 10 fps, MIT) -> the coach's Match format.

    python -m coach.skillcorner                  # from 20-football-analytics/: all 20 matches, restartable
    python -m coach.skillcorner --only 1886347
    python -m coach.skillcorner --store-raw D:/cv-data/skillcorner-raw   # raw files kept (benchmark), restartable

Source: github.com/SkillCorner/opendata (MIT; credit SkillCorner). The 89 MB tracking .jsonl of each match is a
Git LFS file: it is streamed from media.githubusercontent.com and parsed in memory, never written to disk (--store-raw
keeps its compact form, below); the match
sheet, dynamic events and phases of play (raw.githubusercontent.com) are read in memory too. Only the small derived
tables (cross_league.derive: pass / value / shot rows, ~1 MB a match) and audit.json are written to
cv-data/skillcorner/. With --keep-tracking the compact tracking is cached as <id>_tracking.npz (int16 centimetres,
delta-coded in time, with the is_detected mask: ~2 MB instead of 89 MB).

Conventions (checked on the data, see audit()):
- Tracking is in metres, centred on the centre spot, x along the long side; each pitch has its own size
  (pitch_length x pitch_width, e.g. 104 x 68). Positions are rescaled to the standard 105 x 68 used by the coach.
- Broadcast tracking: players off screen get extrapolated positions with is_detected = False. Positions are kept;
  the mask is stored next to the match (detected) and its share is reported.
- Frames are 10 fps video frames; events carry frame numbers on the same clock (no clock offset to estimate).
- Event coordinates (dynamic_events.csv) are in pitch metres with the attacking team towards +x (verified against
  the tracking position of the player in possession); they are rotated back to the tracking frame and stored in the
  0..105 x 0..68 convention of the other provider (meta["event_axis"] = "same_y").
- Passes = player_possession rows that end with a pass; the intended receiver (player_targeted_id) is given by
  SkillCorner for completed and failed passes alike. Set-piece deliveries are not player possessions in this data.
  Crosses are not flagged: a pass from a wide channel to a target in the penalty area counts as a cross.
- Shots = player_possession rows that end with a shot; goal = the next game interruption is goal_for. SkillCorner
  has no expected goals: e["xg"] is None and e["xg_missing"] is True.
- Ball in play = inside a phase of play (SkillCorner defines phases only while the ball is in play). Owner = the
  tracking file's possession group (home 0 / away 1, else -1).
"""

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np

from .match import Half, Match

FPS = 10.0
STD_LENGTH, STD_WIDTH = 105.0, 68.0
NAN16 = -32768
GROUPS = {"home team": 0, "away team": 1}
HERE = Path(__file__).resolve().parents[1]
CV_DATA = Path(os.environ.get("CV_DATA", HERE.parents[1] / "cv-data"))
OUT = CV_DATA / "skillcorner"
REPO = "SkillCorner/opendata"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/master/data"
LFS_URL = f"https://media.githubusercontent.com/media/{REPO}/master/data"
CREDIT = "Tracking and event data: SkillCorner open data (MIT)"


# ------------------------------------------------------------------------------------------- tracking


def read_tracking(lines) -> dict:
    """Parse the tracking .jsonl (an iterable of lines) into arrays. Frames without a period (before kick-off, half
    time) are skipped. Players are columns in the order they first appear; NaN where a player is absent."""
    frames, periods, ball, ball_det, poss_pid, poss_grp, rows = [], [], [], [], [], [], []
    cols = {}
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.strip():
            continue
        fr = json.loads(line)
        if fr.get("period") is None:
            continue
        frames.append(int(fr["frame"]))
        periods.append(int(fr["period"]))
        b = fr.get("ball_data") or {}
        ball.append([np.nan if b.get(k) is None else float(b[k]) for k in ("x", "y", "z")])
        ball_det.append(bool(b.get("is_detected")))
        p = fr.get("possession") or {}
        poss_pid.append(-1 if p.get("player_id") is None else int(p["player_id"]))
        poss_grp.append(GROUPS.get(p.get("group"), -1))
        row = []
        for pl in fr.get("player_data") or []:
            if pl.get("x") is None or pl.get("y") is None:
                continue
            c = cols.setdefault(int(pl["player_id"]), len(cols))
            row.append((c, float(pl["x"]), float(pl["y"]), bool(pl.get("is_detected"))))
        rows.append(row)
    F, P = len(frames), len(cols)
    xy = np.full((F, P, 2), np.nan, np.float32)
    det = np.zeros((F, P), bool)
    for i, row in enumerate(rows):
        if row:
            c, x, y, d = zip(*row)
            c = np.array(c)
            xy[i, c, 0], xy[i, c, 1], det[i, c] = x, y, d
    return {
        "frame": np.array(frames, np.int32),
        "period": np.array(periods, np.int8),
        "player_ids": np.array(sorted(cols, key=cols.get), np.int64),
        "xy": xy,
        "detected": det,
        "ball": np.array(ball, np.float32).reshape(F, 3),
        "ball_detected": np.array(ball_det, bool),
        "poss_player": np.array(poss_pid, np.int64),
        "poss_group": np.array(poss_grp, np.int8),
    }


def _encode(a):
    """float metres -> int16 cm, NaN -> NAN16, then differences along time (small numbers compress well)."""
    q = np.where(np.isnan(a), NAN16, np.round(np.nan_to_num(a) * 100)).astype(np.int32)
    present = q != NAN16
    filled = np.where(present, q, 0)
    d = np.diff(filled, axis=0, prepend=np.zeros_like(filled[:1]))
    assert np.abs(d).max(initial=0) < 32767
    return d.astype(np.int16), present


def _decode(d, present):
    q = np.cumsum(d.astype(np.int64), axis=0)
    return np.where(present, q / 100.0, np.nan).astype(np.float32)


def save_tracking(raw: dict, path: Path) -> None:
    xy_d, xy_ok = _encode(raw["xy"])
    b_d, b_ok = _encode(raw["ball"])
    np.savez_compressed(
        path,
        frame=raw["frame"],
        period=raw["period"],
        player_ids=raw["player_ids"],
        xy_d=xy_d,
        xy_ok=np.packbits(xy_ok[..., 0], axis=None),
        det=np.packbits(raw["detected"], axis=None),
        ball_d=b_d,
        ball_ok=np.packbits(b_ok, axis=None),
        ball_detected=raw["ball_detected"],
        poss_player=raw["poss_player"],
        poss_group=raw["poss_group"],
        shape=np.array(raw["xy"].shape[:2], np.int64),
    )


def load_tracking(path: Path) -> dict:
    z = np.load(path)
    F, P = (int(v) for v in z["shape"])
    ok = np.unpackbits(z["xy_ok"], count=F * P).reshape(F, P).astype(bool)
    b_ok = np.unpackbits(z["ball_ok"], count=F * 3).reshape(F, 3).astype(bool)
    return {
        "frame": z["frame"],
        "period": z["period"],
        "player_ids": z["player_ids"],
        "xy": _decode(z["xy_d"], np.repeat(ok[..., None], 2, axis=2)),
        "detected": np.unpackbits(z["det"], count=F * P).reshape(F, P).astype(bool),
        "ball": _decode(z["ball_d"], b_ok),
        "ball_detected": z["ball_detected"],
        "poss_player": z["poss_player"],
        "poss_group": z["poss_group"],
    }


# ------------------------------------------------------------------------------------- match sheet, events


def parse_sheet(sheet: dict) -> dict:
    home = sheet["home_team"]["id"]
    players = {
        int(p["id"]): {
            "team": 0 if p["team_id"] == home else 1,
            "shirt": int(p.get("number") or 0),
            "name": p.get("short_name") or "?",
            "position": (p.get("player_role") or {}).get("acronym"),
        }
        for p in sheet["players"]
    }
    kit = lambda k: (sheet.get(k) or {}).get("jersey_color")
    meta = {
        "match_id": str(sheet["id"]),
        "home": sheet["home_team"].get("short_name"),
        "guest": sheet["away_team"].get("short_name"),
        "colors": [kit("home_team_kit"), kit("away_team_kit")],
        "score": [sheet.get("home_team_score"), sheet.get("away_team_score")],
        "date": sheet.get("date_time"),
        "pitch": [sheet["pitch_length"], sheet["pitch_width"]],
        "home_team_side": sheet.get("home_team_side"),
        "fps": FPS,
        "source": "SkillCorner open data, A-League 2024/25",
        "credit": CREDIT,
        "event_axis": "same_y",
    }
    return {"players": players, "teams": {sheet["home_team"]["id"]: 0, sheet["away_team"]["id"]: 1}, "meta": meta}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_events(rows, info: dict, pidx: dict, scale) -> tuple:
    """Passes and shots from the dynamic events (player_possession rows), in frame order. Returns (events, offside):
    offside passes are counted but left out (neither completed nor failed)."""
    sx, sy = scale
    out, offside = [], 0
    for r in rows:
        if r.get("event_type") != "player_possession" or r.get("end_type") not in ("pass", "shot"):
            continue
        n = int(r["frame_end"])
        sign = -1.0 if r.get("attacking_side") == "right_to_left" else 1.0  # events: attacking team towards +x
        x, y = _num(r.get("x_end")), _num(r.get("y_end"))
        ex = sign * x * sx + STD_LENGTH / 2 if x is not None else float("nan")
        ey = sign * y * sy + STD_WIDTH / 2 if y is not None else float("nan")
        team = info["teams"].get(int(r["team_id"]), -1) if r.get("team_id") else -1
        base = {
            "n": n,
            "period": int(r["period"]),
            "t": n / FPS,
            "x": ex,
            "y": ey,
            "team": team,
            "player": pidx.get(int(r["player_id"])) if r.get("player_id") else None,
            "set_piece": None,
        }
        after = r.get("game_interruption_after") or ""
        if r["end_type"] == "pass":
            if r.get("pass_outcome") == "offside":
                offside += 1
                continue
            target = r.get("player_targeted_id")
            wide = r.get("channel_end") in ("wide_left", "wide_right")
            out.append(
                {
                    "type": "pass",
                    **base,
                    "recipient": pidx.get(int(float(target))) if target else None,
                    "success": r.get("pass_outcome") == "successful",
                    "target_known": True,
                    "cross": wide and r.get("player_targeted_penalty_area_pass") == "True",
                    "xpass": _num(r.get("player_targeted_xpass_completion")),  # SkillCorner's own completion model
                }
            )
        else:
            before = r.get("game_interruption_before") or ""
            out.append(
                {
                    "type": "shot",
                    **base,
                    "goal": after == "goal_for",
                    "result": "goal" if after == "goal_for" else None,
                    "set_piece": "Penalty" if before.startswith("penalty") else None,
                    "header": r.get("is_header") == "True",
                    "xg": None,
                    "xg_missing": True,
                }
            )
    out.sort(key=lambda e: e["n"])
    return out, offside


def phases_alive(frames: np.ndarray, phases) -> np.ndarray:
    alive = np.zeros(len(frames), bool)
    for p in phases:
        alive |= (frames >= int(p["frame_start"])) & (frames <= int(p["frame_end"]))
    return alive


# --------------------------------------------------------------------------------------------- the match

MAX_ACTOR_BALL_M = 3.0  # same tolerance as the event-to-ball distance used for the other provider


def _pct(a, q):
    return float(np.nanpercentile(a, q)) if len(a) else float("nan")


def build_match(sheet: dict, raw: dict, event_rows, phase_rows) -> Match:
    """Match in the coach's format (+ m.detected / m.ball_detected: per-half is_detected masks, m.meta["sync"])."""
    info = parse_sheet(sheet)
    length, width = info["meta"]["pitch"]
    sx, sy = STD_LENGTH / length, STD_WIDTH / width
    ids = [int(p) for p in raw["player_ids"]]
    unknown = {"team": -1, "shirt": 0, "name": "?", "position": None}
    players = [info["players"].get(p, unknown) for p in ids]
    alive_all = phases_alive(raw["frame"], phase_rows) if phase_rows else raw["poss_group"] >= 0
    halves, detected, ball_detected = [], [], []
    for period in (1, 2):
        sel = raw["period"] == period
        n = raw["frame"][sel].astype(np.int32)
        halves.append(
            Half(
                n=n,
                t=n / FPS,
                xy=(raw["xy"][sel] * np.array([sx, sy], np.float32)).astype(np.float32),
                ball=(raw["ball"][sel] * np.array([sx, sy, 1.0], np.float32)).astype(np.float32),
                owner=raw["poss_group"][sel].astype(np.int8),
                alive=alive_all[sel],
            )
        )
        detected.append(raw["detected"][sel])
        ball_detected.append(raw["ball_detected"][sel])
    events, offside = parse_events(event_rows, info, {p: i for i, p in enumerate(ids)}, (sx, sy))
    m = Match(
        info["meta"]["match_id"],
        [str(p) for p in ids],
        np.array([p["team"] for p in players], np.int8),
        np.array([p["shirt"] for p in players], np.int16),
        [p["name"] for p in players],
        halves,
        events,
        dict(info["meta"]),
    )
    m.detected, m.ball_detected = detected, ball_detected
    m.meta["sync"] = attach_frames(m, offside)
    return m


def attach_frames(m, offside: int = 0) -> dict:
    """Events carry the provider's frame number: place each on its half and frame index, then measure the distance
    from the ball to the acting player (passer / shooter) and to the event location at that frame. Events whose actor
    is more than MAX_ACTOR_BALL_M from the ball are dropped ("far"), like the other provider's events far from the
    ball; their time stays in t_sync for the value labels."""
    from .sync import event_xy

    for e in m.events:
        e["half"], e["drop"], e["sync"], e["frame"], e["t_sync"] = e["period"] - 1, None, None, None, e["t"]
        if not 0 <= e["half"] < len(m.halves):
            e["half"], e["drop"] = None, "no_half"
            continue
        h = m.halves[e["half"]]
        i = int(np.searchsorted(h.n, e["n"]))
        if i >= len(h.n) or h.n[i] != e["n"]:
            e["drop"] = "no_frame"
            continue
        e["nominal"] = i
        ball = h.ball[i, :2]
        e["dist"] = float(np.hypot(*(event_xy(m, e) - ball))) if np.isfinite(e["x"]) else float("nan")
        a = e.get("player")
        e["actor_ball_m"] = float(np.hypot(*(h.xy[i, a] - ball))) if a is not None else float("nan")
        e["actor_detected"] = bool(m.detected[e["half"]][i, a]) if a is not None else None
        if not np.isfinite(e["actor_ball_m"]):
            e["drop"] = "no_position"
        elif e["actor_ball_m"] > MAX_ACTOR_BALL_M:
            e["drop"] = "far"
        else:
            e["frame"], e["sync"] = i, "provider"
    placed = [e for e in m.events if np.isfinite(e.get("actor_ball_m", np.nan))]
    kept = [e for e in m.events if e["frame"] is not None]
    passes = [e for e in m.events if e["type"] == "pass"]
    shots = [e for e in m.events if e["type"] == "shot"]
    reasons = {}
    for e in m.events:
        if e["frame"] is None:
            reasons[e["drop"]] = reasons.get(e["drop"], 0) + 1
    share = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    return {
        "events": len(m.events),
        "kept": len(kept),
        "dropped": reasons,
        "passes": len(passes),
        "passes_kept": sum(e["frame"] is not None for e in passes),
        "offside_dropped": offside,
        "shots": len(shots),
        "shots_kept": sum(e["frame"] is not None for e in shots),
        "goals": sum(e["goal"] for e in shots),
        "score": m.meta.get("score"),
        # at the provider's frame, before dropping: passer / shooter to the ball, and event location to the ball
        "actor_ball_median_m": _pct([e["actor_ball_m"] for e in placed], 50),
        "actor_ball_p90_m": _pct([e["actor_ball_m"] for e in placed], 90),
        "actor_ball_within_3m": share([e["actor_ball_m"] <= MAX_ACTOR_BALL_M for e in placed]),
        "event_ball_median_m": _pct([e["dist"] for e in placed], 50),
        "actor_detected_share": share([bool(e["actor_detected"]) for e in placed]),
    }


def detection_report(m) -> dict:
    """Share of on-pitch player positions that are detected (not extrapolated), overall and while the ball is in
    play, and the share of in-play frames with a detected ball."""
    present = np.concatenate([~np.isnan(h.xy[..., 0]) for h in m.halves])
    det = np.concatenate(m.detected) & present
    alive = np.concatenate([h.alive for h in m.halves])
    bdet = np.concatenate(m.ball_detected)
    ratio = lambda a, b: float(a.sum() / b.sum()) if b.sum() else float("nan")
    return {
        "players_detected": ratio(det, present),
        "players_detected_in_play": ratio(det[alive], present[alive]),
        "players_per_frame": float(present.sum(axis=1).mean()),
        "players_per_frame_in_play": float(present[alive].sum(axis=1).mean()) if alive.any() else 0.0,
        "detected_per_frame_in_play": float(det[alive].sum(axis=1).mean()) if alive.any() else 0.0,
        "ball_detected_in_play": float(bdet[alive].mean()) if alive.any() else float("nan"),
        "in_play_share": float(alive.mean()),
    }


# ------------------------------------------------------------------------------------ download + prepare

MATCHES = [
    "1874553", "1886347", "1899585", "1925299", "1927964", "1953632", "1959846", "1986691", "1996435", "1996436",
    "2006229", "2006363", "2007448", "2007721", "2010085", "2011166", "2013725", "2015213", "2016236", "2017461",
]  # fmt: skip
DERIVED = OUT / "derived"
MIN_FREE_GB = 9.0  # owner's limit for drive C:


class _Counted:
    """Iterate the lines of an HTTP response while counting the bytes."""

    def __init__(self, response):
        self.response, self.bytes = response, 0

    def __iter__(self):
        for line in self.response:
            self.bytes += len(line)
            yield line


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def free_gb(path: Path = OUT) -> float:
    path = Path(path)
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 2**30


def fetch_match(mid: str, keep_tracking: bool = False) -> tuple:
    """(Match, bytes downloaded). The tracking is streamed and parsed in memory; with keep_tracking it is also cached
    as <id>_tracking.npz (~2 MB), and a cached file is used instead of the download."""
    base = f"{RAW_URL}/matches/{mid}/{mid}"
    sheet_b, events_b, phases_b = (
        _get(f"{base}_{k}") for k in ("match.json", "dynamic_events.csv", "phases_of_play.csv")
    )
    rows = lambda b: list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
    nbytes = len(sheet_b) + len(events_b) + len(phases_b)
    cached = OUT / f"{mid}_tracking.npz"
    if cached.exists():
        raw = load_tracking(cached)
    else:
        with urllib.request.urlopen(f"{LFS_URL}/matches/{mid}/{mid}_tracking_extrapolated.jsonl", timeout=300) as r:
            lines = _Counted(r)
            raw = read_tracking(lines)
        nbytes += lines.bytes
        if keep_tracking:
            save_tracking(raw, cached)
    return build_match(json.loads(sheet_b), raw, rows(events_b), rows(phases_b)), nbytes


# Raw store (benchmark on the A-League, coach insight C): the three small files as downloaded and the compact tracking
# (save_tracking) of each match under RAW_DIR/<id>/, on the USB key by default. Restartable: complete files are
# skipped; every file is written to <name>.part and renamed, so an interrupted run (the key can disconnect) never
# leaves a truncated file. The raw 89 MB .jsonl is not kept: the compact file is exact to 1 cm.
RAW_DIR = Path(os.environ.get("SKILLCORNER_RAW", "D:/cv-data/skillcorner-raw"))
RAW_FILES = ("match.json", "dynamic_events.csv", "phases_of_play.csv")


def _atomic(path: Path, write) -> None:
    """write(file object) into <path>.part, then rename it to path."""
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        write(f)
    os.replace(tmp, path)


@contextlib.contextmanager
def _open_lfs(url: str):
    with urllib.request.urlopen(url, timeout=300) as r:
        yield _Counted(r)


def store_raw(mid: str, raw_dir: Path = RAW_DIR, get=_get, open_lines=_open_lfs) -> int:
    """Download what is missing of match `mid` into raw_dir/<mid>/; returns the bytes downloaded (0 if complete)."""
    d = Path(raw_dir) / mid
    d.mkdir(parents=True, exist_ok=True)
    nbytes = 0
    for k in RAW_FILES:
        path = d / f"{mid}_{k}"
        if not path.exists():
            data = get(f"{RAW_URL}/matches/{mid}/{mid}_{k}")
            _atomic(path, lambda f, data=data: f.write(data))
            nbytes += len(data)
    trk = d / f"{mid}_tracking.npz"
    if not trk.exists():
        with open_lines(f"{LFS_URL}/matches/{mid}/{mid}_tracking_extrapolated.jsonl") as lines:
            raw = read_tracking(lines)
            nbytes += getattr(lines, "bytes", 0)
        _atomic(trk, lambda f: save_tracking(raw, f))
    return nbytes


def stored_matches(raw_dir: Path = RAW_DIR) -> list:
    """Ids of the matches whose four raw files are all in raw_dir."""
    names = lambda mid: [f"{mid}_{k}" for k in RAW_FILES] + [f"{mid}_tracking.npz"]
    root = Path(raw_dir)
    if not root.exists():
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and all((d / n).exists() for n in names(d.name)))


def load_raw_match(mid: str, raw_dir: Path = RAW_DIR) -> Match:
    """The match rebuilt from the raw store (same parsing as fetch_match)."""
    d = Path(raw_dir) / mid
    sheet_b, events_b, phases_b = ((d / f"{mid}_{k}").read_bytes() for k in RAW_FILES)
    rows = lambda b: list(csv.DictReader(io.StringIO(b.decode("utf-8-sig"))))
    raw = load_tracking(d / f"{mid}_tracking.npz")
    return build_match(json.loads(sheet_b), raw, rows(events_b), rows(phases_b))


def main():
    import pickle

    from .cross_league import derive

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=MATCHES)
    ap.add_argument("--keep-tracking", action="store_true", help="also cache the compact tracking (~2 MB a match)")
    ap.add_argument("--max-download-gb", type=float, default=3.0)
    ap.add_argument("--store-raw", type=Path, help="only download the raw files (compact tracking) into this folder")
    args = ap.parse_args()
    if args.store_raw:
        total = 0
        for mid in args.only:
            n = store_raw(mid, args.store_raw)
            total += n
            print(mid, n // 2**20, "MB downloaded" if n else "already stored", flush=True)
        print(f"downloaded {total / 2**30:.2f} GB; stored: {len(stored_matches(args.store_raw))} matches")
        return
    DERIVED.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    total = sum(a.get("downloaded_bytes", 0) for a in audit.values())
    for mid in args.only:
        if (DERIVED / f"{mid}.pkl").exists() and mid in audit:
            print("have", mid)
            continue
        if total > args.max_download_gb * 2**30 - 100e6:
            raise SystemExit(f"download budget reached ({total / 2**30:.2f} GB)")
        if free_gb() < MIN_FREE_GB:
            print(f"warning: {free_gb():.2f} GB free on the data drive (limit {MIN_FREE_GB} GB); derived tables only")
        m, nbytes = fetch_match(mid, keep_tracking=args.keep_tracking and free_gb() > MIN_FREE_GB + 0.1)
        d = derive(m)
        (DERIVED / f"{mid}.pkl").write_bytes(pickle.dumps(d))
        total += nbytes
        audit[mid] = {
            "downloaded_bytes": nbytes,
            "home": m.meta["home"],
            "guest": m.meta["guest"],
            "pitch": m.meta["pitch"],
            "sync": m.meta["sync"],
            "detection": d["detection"],
            "pass_rows": int(len(d["pass"]["y"])),
            "value_rows": int(len(d["value"]["t"])),
        }
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(mid, nbytes // 2**20, "MB", m.meta["sync"], d["detection"], flush=True)
    print(f"downloaded {total / 2**30:.2f} GB in total; {free_gb():.2f} GB free")


if __name__ == "__main__":
    main()

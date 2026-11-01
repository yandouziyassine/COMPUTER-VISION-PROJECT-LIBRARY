"""In-memory match: per-half NumPy arrays in metres (centre spot = origin), plus the events list.

Storage: <stem>.npz (arrays) + <stem>.json (ids, names, events). Positions are NaN when a player is not on the pitch.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FPS = 25.0


@dataclass
class Half:
    n: np.ndarray  # [F] int32 frame numbers from the provider
    t: np.ndarray  # [F] float64 epoch seconds (UTC)
    xy: np.ndarray  # [F, P, 2] float32 player positions, NaN = not on the pitch
    ball: np.ndarray  # [F, 3] float32 x, y, z
    owner: np.ndarray  # [F] int8 team in possession (0 home, 1 guest, -1 unknown)
    alive: np.ndarray  # [F] bool ball in play


@dataclass
class Match:
    match_id: str
    player_ids: list
    team: np.ndarray  # [P] int8, 0 home / 1 guest
    shirt: np.ndarray  # [P] int16
    names: list
    halves: list
    events: list = field(default_factory=list)  # dicts, see dfl.parse_events
    meta: dict = field(default_factory=dict)  # team names, colours, axis convention


def save(m: Match, stem: Path) -> None:
    stem = Path(stem)
    arrays = {"team": m.team, "shirt": m.shirt}
    for i, h in enumerate(m.halves):
        for k in ("n", "t", "xy", "ball", "owner", "alive"):
            arrays[f"h{i}_{k}"] = getattr(h, k)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    info = {
        "match_id": m.match_id,
        "player_ids": m.player_ids,
        "names": m.names,
        "halves": len(m.halves),
        "events": m.events,
        "meta": m.meta,
    }
    stem.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")


def load(stem: Path) -> Match:
    stem = Path(stem)
    info = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    z = np.load(stem.with_suffix(".npz"))
    halves = [
        Half(**{k: z[f"h{i}_{k}"] for k in ("n", "t", "xy", "ball", "owner", "alive")}) for i in range(info["halves"])
    ]
    return Match(
        info["match_id"],
        info["player_ids"],
        z["team"],
        z["shirt"],
        info["names"],
        halves,
        info["events"],
        info.get("meta", {}),
    )


def frame_at(h: Half, t: float) -> int:
    """Index of the frame nearest to epoch time t (clamped to the half)."""
    i = int(np.searchsorted(h.t, t))
    if i <= 0:
        return 0
    if i >= len(h.t):
        return len(h.t) - 1
    return i if h.t[i] - t < t - h.t[i - 1] else i - 1


def velocity(xy: np.ndarray, fps: float = FPS, k: int = 2) -> np.ndarray:
    """Central difference over ±k frames (one-sided at the edges); works on [F, 2] or [F, P, 2]."""
    v = np.zeros_like(xy, dtype=np.float32)
    v[k:-k] = (xy[2 * k :] - xy[: -2 * k]) * fps / (2 * k)
    v[:k] = (xy[k : 2 * k] - xy[:k]) * fps / k
    v[-k:] = (xy[-k:] - xy[-2 * k : -k]) * fps / k
    return v

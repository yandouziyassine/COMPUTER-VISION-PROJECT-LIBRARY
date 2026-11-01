"""Convert the IDSSE raw XML (USB key) into one compact match per id in cv-data/football-coach/.

    python -m coach.prepare                       # from 20-football-analytics/
    python -m coach.prepare --raw D:/football-coach/idsse-raw --only J03WOH

Restartable: a match whose .npz exists is skipped (the USB key can disconnect). Writes sync_audit.json next to the
matches. Raw data: Bassek et al. (2025), DFL / Sportec Solutions, CC BY 4.0.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import dfl, sync
from .match import Match, save

HERE = Path(__file__).resolve().parents[1]
CV_DATA = Path(os.environ.get("CV_DATA", HERE.parents[1] / "cv-data"))
OUT = CV_DATA / "football-coach"
RAW = Path(os.environ.get("IDSSE_RAW", "D:/football-coach/idsse-raw"))
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]


def find(raw: Path, kind: str, mid: str) -> Path:
    hits = sorted(raw.glob(f"DFL_{kind}_*_DFL-MAT-{mid}.xml"))
    if not hits:
        raise FileNotFoundError(f"{kind} file for {mid} not in {raw}")
    return hits[0]


def convert(raw: Path, mid: str) -> dict:
    info = dfl.parse_info(find(raw, "02_01_matchinformation", mid))
    ids, halves = dfl.parse_positions(find(raw, "04_03_positions_raw_observed", mid), info)
    events = dfl.parse_events(find(raw, "03_02_events_raw", mid), info)
    pidx = {p: i for i, p in enumerate(ids)}
    for e in events:  # PersonId -> player index (None when the person has no tracking, e.g. unknown recipient)
        for k in ("player", "recipient"):
            if k in e:
                e[k] = pidx.get(e[k])
    unknown = {"team": -1, "shirt": 0, "name": "?", "position": None}  # tracked but not in the match sheet
    players = {p: info["players"].get(p, unknown) for p in ids}
    m = Match(
        mid,
        ids,
        np.array([players[p]["team"] for p in ids], np.int8),
        np.array([players[p]["shirt"] for p in ids], np.int16),
        [players[p]["name"] for p in ids],
        halves,
        events,
        dict(info["meta"]),
    )
    report = sync.attach_frames(m)
    save(m, OUT / mid)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--only", nargs="*", default=MATCHES)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit_path = OUT / "sync_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    for mid in args.only:
        if (OUT / f"{mid}.npz").exists() and mid in audit:
            print("have", mid)
            continue
        audit[mid] = convert(args.raw, mid)
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(mid, audit[mid], flush=True)


if __name__ == "__main__":
    main()

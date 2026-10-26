"""Download the open data used by the statistics engine (stage S4) into ../cv-data/football/.

- Metrica Sports sample games 1 and 2 (tracking + events CSV, ~160 MB). No formal licence: the repo asks users to be
  responsible and to acknowledge Metrica Sports when results are made public.
- StatsBomb open data (StatsBomb Public Data User Agreement: research/analysis use, no redistribution, StatsBomb
  attribution on publications). Only shots are kept: each match's event file is downloaded, its shots are written to
  one compact table (shots.csv), and the raw JSON is not stored.

Nothing downloaded here is committed or redistributed.
"""

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CV_DATA = Path(os.environ.get("CV_DATA", Path(__file__).resolve().parents[3] / "cv-data"))
OUT = CV_DATA / "football"
METRICA = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
SB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Men's competitions from 2018 on (competition_id, season_id): recent, varied leagues and tournaments.
SB_COMPETITIONS = [
    (9, 281), (1267, 107), (16, 4), (223, 282), (43, 106), (43, 3), (1238, 108),
    (11, 90), (11, 4), (7, 235), (7, 108), (44, 107), (55, 282), (55, 43),
]  # fmt: skip


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def download_metrica():
    dst = OUT / "metrica"
    dst.mkdir(parents=True, exist_ok=True)
    for g in (1, 2):
        for kind in ("RawEventsData", "RawTrackingData_Home_Team", "RawTrackingData_Away_Team"):
            name = f"Sample_Game_{g}_{kind}.csv"
            if not (dst / name).exists():
                urllib.request.urlretrieve(f"{METRICA}/Sample_Game_{g}/{name}", dst / name)
    print("metrica ok:", dst)


def shot_rows(match, events):
    """Flatten the shots of one match (one row per shot, freeze frame kept as JSON text)."""
    rows = []
    for e in events:
        if e["type"]["name"] != "Shot":
            continue
        s = e["shot"]
        rows.append(
            {
                "match_id": match["match_id"],
                "competition": match["competition"]["competition_name"],
                "season": match["season"]["season_name"],
                "id": e["id"],
                "period": e["period"],
                "minute": e["minute"],
                "x": e["location"][0],
                "y": e["location"][1],
                "body_part": s["body_part"]["name"],
                "shot_type": s["type"]["name"],
                "technique": s["technique"]["name"],
                "play_pattern": e["play_pattern"]["name"],
                "first_time": bool(s.get("first_time", False)),
                "under_pressure": bool(e.get("under_pressure", False)),
                "outcome": s["outcome"]["name"],
                "goal": int(s["outcome"]["name"] == "Goal"),
                "statsbomb_xg": s.get("statsbomb_xg"),
                "freeze_frame": json.dumps(
                    [
                        [p["location"][0], p["location"][1], int(p["teammate"]), p["position"]["name"]]
                        for p in s.get("freeze_frame", [])
                    ]
                ),
            }
        )
    return rows


def download_statsbomb():
    import pandas as pd

    dst = OUT / "statsbomb"
    dst.mkdir(parents=True, exist_ok=True)
    matches = []
    for cid, sid in SB_COMPETITIONS:
        matches += fetch_json(f"{SB}/matches/{cid}/{sid}.json")
    print("statsbomb matches:", len(matches))

    def one(m):
        try:
            return shot_rows(m, fetch_json(f"{SB}/events/{m['match_id']}.json"))
        except Exception as ex:  # a missing file should not stop the whole download
            print("skip", m["match_id"], ex)
            return []

    with ThreadPoolExecutor(8) as pool:
        rows = [r for rs in pool.map(one, matches) for r in rs]
    pd.DataFrame(rows).to_csv(dst / "shots.csv", index=False)
    print("statsbomb shots:", len(rows), "->", dst / "shots.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["metrica", "statsbomb"])
    a = ap.parse_args()
    if a.only != "statsbomb":
        download_metrica()
    if a.only != "metrica":
        download_statsbomb()

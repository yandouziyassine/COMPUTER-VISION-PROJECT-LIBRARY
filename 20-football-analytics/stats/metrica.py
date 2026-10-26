"""Loader for the Metrica Sports sample games (CSV format, games 1 and 2) -> Tracks + ground-truth events."""

import numpy as np
import pandas as pd

from .tracks import Tracks

PITCH = (105.0, 68.0)  # metres, as stated in the Metrica sample-data README


def _read_tracking(path):
    head = pd.read_csv(path, skiprows=2, nrows=0).columns
    names = [c for c in head[3:] if not c.startswith("Unnamed")]
    df = pd.read_csv(path, skiprows=2)
    xy = df.iloc[:, 3:].to_numpy(float).reshape(len(df), -1, 2)
    return df["Period"].to_numpy(int), df["Frame"].to_numpy(int), names, xy


def load_game(folder, game):
    """Returns (Tracks, events DataFrame). Home = team 0, away = team 1; coordinates in metres."""
    stem = f"{folder}/Sample_Game_{game}_RawTrackingData"
    per_h, fr_h, names_h, xy_h = _read_tracking(f"{stem}_Home_Team.csv")
    per_a, fr_a, names_a, xy_a = _read_tracking(f"{stem}_Away_Team.csv")
    assert np.array_equal(fr_h, fr_a) and names_h[-1] == "Ball" and names_a[-1] == "Ball"
    ball = xy_h[:, -1]
    xy = np.concatenate([xy_h[:, :-1], xy_a[:, :-1]], axis=1) * PITCH
    names = names_h[:-1] + names_a[:-1]
    team = np.r_[np.zeros(len(names_h) - 1, int), np.ones(len(names_a) - 1, int)]
    tr = Tracks(xy=xy, team=team, ball=ball * PITCH, fps=25.0, period=per_h, names=names, pitch=PITCH, frame=fr_h)
    ev = pd.read_csv(f"{folder}/Sample_Game_{game}_RawEventsData.csv")
    return tr, ev


def true_passes(tracks, events):
    """Metrica PASS events (completed passes) as dicts aligned with `tracks` frame indices."""
    index = {f: i for i, f in enumerate(tracks.frame)}
    pid = {n: i for i, n in enumerate(tracks.names)}
    out = []
    for _, e in events[events["Type"] == "PASS"].iterrows():
        if e["From"] not in pid or e["To"] not in pid or e["Start Frame"] not in index:
            continue
        out.append(
            {
                "start": index[e["Start Frame"]],
                "end": index.get(e["End Frame"], index[e["Start Frame"]]),
                "from": pid[e["From"]],
                "to": pid[e["To"]],
                "team": 0 if e["Team"] == "Home" else 1,
                "x0": e["Start X"] * PITCH[0],
            }
        )
    return out


def event_ppda_counts(tracks, events, team, sign, start=0, stop=None):
    """Event-data version of `tactics.ppda_counts` for frames [start, stop): opponent PASS events started in the
    opponent's own 60 % and RECOVERY events of `team` in that zone (sign = attack direction of `team`)."""
    from .tactics import in_build_up_zone

    stop = tracks.n_frames if stop is None else stop
    lo, hi = tracks.frame[start], tracks.frame[stop - 1]
    ev = events[(events["Start Frame"] >= lo) & (events["Start Frame"] <= hi)]
    name = "Home" if team == 0 else "Away"
    x = ev["Start X"].to_numpy(float) * PITCH[0]
    zone = in_build_up_zone(x, sign, PITCH)
    passes = int(((ev["Type"] == "PASS") & (ev["Team"] != name) & zone).sum())
    regains = int(((ev["Type"] == "RECOVERY") & (ev["Team"] == name) & zone).sum())
    return passes, regains

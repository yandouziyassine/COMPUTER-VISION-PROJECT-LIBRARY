"""report.json for the website's replay match, with models trained WITHOUT that match (coach insight, part A).

    python -m coach.export_report --match J03WOH --site ../../cv-data/football-coach/site-export

Same model recipe as coach.export (pass and value models on the other Bundesliga matches, shot model also on the
A-League shots), see report.held_out_models. Checked against the site's match.json when it is there: same frame rate,
and every decision's frame is the frame of the same event in match.json. The file must stay under MAX_BYTES (compact
JSON). Kept apart from coach.export on purpose (like export_profiles).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from . import models as ML
from . import report as R

MAX_BYTES = 300_000


def check_against_site(report: dict, site_match: dict) -> None:
    """Same frame rate as the site's match.json, and each decision sits on the frame of its event there."""
    assert abs(float(site_match["fps"]) - report["fps"]) < 1e-9, (site_match["fps"], report["fps"])
    for d in report["decisions"]:
        e = site_match["events"][d["event"]]
        assert e["type"] == "pass" and e["half"] == d["half"] and e["frame"] == d["frame"], (d, e)
        assert 0 <= d["frame"] < site_match["halves"][d["half"]]["frames"], d


def write(report: dict, site: Path) -> Path:
    text = json.dumps(report, ensure_ascii=False, separators=(",", ":"))
    size = len(text.encode("utf-8"))
    assert size <= MAX_BYTES, f"report.json is {size} bytes (limit {MAX_BYTES})"
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    (site / "report.json").write_text(text, encoding="utf-8")
    return site / "report.json"


def main():
    from . import cross_league as CL
    from .match import load
    from .prepare import MATCHES, OUT

    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    matches = {mid: load(OUT / mid) for mid in MATCHES}
    tables = {mid: R.match_tables(m) for mid, m in matches.items() if mid != args.match}  # the target's are never used
    models, fitted, trained_on = R.held_out_models(args.match, tables, CL.skillcorner_shots()[0])
    rep = R.build(matches[args.match], models, trained_on)
    site_match = args.site / "match.json"
    if site_match.exists():
        check_against_site(rep, json.loads(site_match.read_text(encoding="utf-8")))
        print("frames checked against", site_match)
    site_models = args.site / "models.json"
    if site_models.exists():
        w = json.loads(site_models.read_text(encoding="utf-8"))
        same = w.get("trained_without") == args.match and np.allclose(
            ML.export_logistic(fitted["pass"], R.F.PASS_FEATURES)["coef"], w["pass"]["coef"]
        )
        print("same pass model as the site's models.json:", bool(same))
    path = write(rep, args.site)
    s = rep["summary"]
    print("wrote", path, path.stat().st_size, "bytes,", s["n"], "decisions,", len(rep["players"]), "passers")
    print({k: s[k] for k in ("top1", "top3", "gap_median", "gap_p90", "missed_candidates", "best_candidates")})


if __name__ == "__main__":
    main()

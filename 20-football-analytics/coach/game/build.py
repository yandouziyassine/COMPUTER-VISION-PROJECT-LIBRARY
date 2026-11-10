"""Build a Playmaker content release from the 27 open matches (models held out per match, 20 refits).

    python -m coach.game.build --out D:/cv-data/playmaker --start 2026-12-01 --days 60
    python -m coach.game.build --validate D:/cv-data/playmaker/content-v1
    python -m coach.game.build --set idsse --out ../../cv-data/playmaker-dev --days 20   # dev release, no USB key

Checkpoints: one candidates file per match in <out>/cache (restartable when the USB key disconnects)."""

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path

import jsonschema

from ..stamp import generated_by
from . import CONTENT_VERSION
from . import days as GD
from . import evaluate_content as EC

SCHEMAS = Path(__file__).resolve().parent / "schemas"


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)


def write_release(root: Path, cands: dict, days: list, start: str) -> Path:
    out = Path(root) / f"content-v{CONTENT_VERSION}"
    first = dt.date.fromisoformat(start)
    used = {f["id"] for d in days for f in d}
    for mid, c in cands.items():
        if mid in used or not days:
            _write(out / "public" / "moments" / f"{mid}.json", c["public"])
            _write(out / "answers" / f"{mid}.json", c["answer"])
    for i, d in enumerate(days):
        date = (first + dt.timedelta(days=i)).isoformat()
        _write(
            out / "public" / "days" / f"{date}.json",
            {
                "schema": 1,
                "content_version": CONTENT_VERSION,
                "day_number": i + 1,
                "date": date,
                "moments": [f["id"] for f in d],
            },
        )
    # a sample moment for the intro preview on launch day (never used in a daily set; public file only, no answer)
    spare = [c for mid, c in cands.items() if mid not in used and c["features"]["x"] > 52.5 / 3]
    if spare:
        _write(out / "public" / "sample.json", max(spare, key=lambda c: c["features"]["iqr"])["public"])
    index = [{"date": (first + dt.timedelta(days=i)).isoformat(), "day_number": i + 1} for i in range(len(days))]
    _write(out / "public" / "days" / "index.json", {"days": index})
    shutil.copytree(SCHEMAS, out / "schemas", dirs_exist_ok=True)
    _write(
        out / "release.json",
        {
            "content_version": CONTENT_VERSION,
            "start": start,
            "days": len(days),
            "moments": len(used) if days else len(cands),
            "generated_by": generated_by(),
        },
    )
    return out


def validate_release(out: Path) -> list:
    out = Path(out)
    sch = {
        n: json.loads((SCHEMAS / f"{n}.schema.json").read_text(encoding="utf-8")) for n in ("day", "moment", "answer")
    }
    errors = []
    if (out / "public" / "sample.json").exists():
        try:
            jsonschema.validate(json.loads((out / "public" / "sample.json").read_text(encoding="utf-8")), sch["moment"])
        except jsonschema.ValidationError as e:
            errors.append(f"sample.json: {e.message}")
    for kind, folder in (("day", "public/days"), ("moment", "public/moments"), ("answer", "answers")):
        for f in sorted(g for g in (out / folder).glob("*.json") if g.name != "index.json"):
            try:
                jsonschema.validate(json.loads(f.read_text(encoding="utf-8")), sch[kind])
            except jsonschema.ValidationError as e:
                errors.append(f"{f.name}: {e.message}")
    for f in (out / "public" / "days").glob("????-??-??.json"):
        for mid in json.loads(f.read_text(encoding="utf-8"))["moments"]:
            if (
                not (out / "public" / "moments" / f"{mid}.json").exists()
                or not (out / "answers" / f"{mid}.json").exists()
            ):
                errors.append(f"{f.name}: missing files for {mid}")
    for f in (out / "public").rglob("*.json"):
        text = f.read_text(encoding="utf-8")
        if '"points"' in text or '"pro_pick"' in text or '"share_top"' in text:
            errors.append(f"{f.name}: answer field in a public file")
    return errors


def candidates(cache: Path, n_refits: int, match_set: str = "pooled") -> tuple:
    """All matches -> ({moment id: candidate}, matches, tables); per-match results cached as JSON in `cache`."""
    from .. import benchmark as B
    from .. import cross_league as CL
    from .. import report as R
    from . import moments as GM
    from . import refits as GR

    matches, _, _ = B.load_set(match_set)
    tables = {mid: R.match_tables(m) for mid, m in matches.items()}
    out = {}
    for mid, m in matches.items():
        f = cache / f"{mid}.json"
        if not f.exists():
            league = GM.league_name(m, CL.league_of(m))
            cands = GM.match_moments(m, R.held_out_models(mid, tables), GR.refit_models(mid, tables, n_refits), league)
            _write(f, cands)
            print(mid, len(cands), "moments", flush=True)
        for c in json.loads(f.read_text(encoding="utf-8")):
            pub = c["public"]
            if "scoreline" not in pub:  # caches written before the scoreline field
                _, half, frame = pub["id"].rsplit("-", 2)
                pub["scoreline"] = GM.score_at(m, int(half), int(frame))
            out[pub["id"]] = c
    return out, matches, tables


def rescore(c: dict) -> None:
    """Recompute the option points (and the features that depend on them) of one candidate from its stored scores."""
    import numpy as np

    from . import scoring as GS

    opts = c["answer"]["options"]
    pts = GS.points(np.array([o["score"] for o in opts]), np.array([o["share_top"] for o in opts]))
    for o, p in zip(opts, pts):
        o["points"] = int(p)
    c["features"]["all_best"] = bool((pts == 100).all())
    c["features"]["difficulty"] = float(1 - pts.mean() / 100)


def refresh_released_reasons(cands: dict, days: list, matches: dict, tables: dict) -> int:
    """Recompute the reasons (with their effect sign) of the released moments with the held-out main models."""
    from .. import report as R
    from . import moments as GM

    by_match = {}
    for d in days:
        for f in d:
            by_match.setdefault(str(f["match"]), []).append(f["id"])
    n = 0
    for mid, ids in by_match.items():
        fresh = GM.refresh_reasons(matches[mid], R.held_out_models(mid, tables), ids)
        for i in ids:
            for o, r in zip(cands[i]["answer"]["options"], fresh[i]):
                o["reasons"] = r
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("D:/cv-data/playmaker"))
    ap.add_argument("--start", default="2026-12-01")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--refits", type=int, default=20)
    ap.add_argument("--set", default="pooled", choices=("pooled", "idsse"), help="idsse: Bundesliga only (no USB key)")
    ap.add_argument("--validate", type=Path)
    args = ap.parse_args()
    if args.validate:
        errs = validate_release(args.validate)
        print("\n".join(errs) or "release OK")
        raise SystemExit(1 if errs else 0)
    cands, matches, tables = candidates(args.out / "cache", args.refits, args.set)
    for c in cands.values():  # points under the current rule (caches may hold an older one)
        rescore(c)
    # moments where a trivial strategy (shoot / most advanced pass / nearest pass) already scores 100 are not shown:
    # with them a player who never looks at the defence beats the pros (coach.game.evaluate_content)
    feats = [{**c["features"], "id": mid} for mid, c in cands.items() if not EC.trivial_hit(c["public"], c["answer"])]
    print(len(feats), "candidates after the trivial-strategy filter", flush=True)
    days = GD.daily_sets(GD.rank_pool(feats, args.pool), args.days)
    print(len(cands), "candidates,", len(days), "days")
    print("reasons refreshed for", refresh_released_reasons(cands, days, matches, tables), "moments", flush=True)
    out = write_release(args.out, cands, days, args.start)
    errs = validate_release(out)
    print("\n".join(errs) or f"release OK: {out}")


if __name__ == "__main__":
    main()

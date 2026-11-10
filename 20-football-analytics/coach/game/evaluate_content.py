"""Playmaker content audit: do the points mean anything, and how hard is the game?

python -m coach.game.evaluate_content [--cache D:/cv-data/playmaker/cache] [--release D:/cv-data/playmaker/content-v1]

1. Validity, on every candidate moment (16,678, models held out per match): the points and the coach score of the
   option the professional actually chose vs what happened in the next 15 s (net xG, shot or goal, possession kept).
   95 % CIs from bootstrapping whole matches.
2. Difficulty, on the released moments: expected points of the pro, a random pick, the AI and trivial strategies that
   only read the public geometry. A trivial strategy at >= 70 points or >= 50 % of 100s makes the game exploitable.
3. Close calls: how many options score 100 per released moment.
4. Everything per league (A-League vs Bundesliga + 2. Bundesliga).
No model fitting: reads the candidate cache and the release only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

BUCKETS = ("100", "50-99", "0-49")
EXPLOIT_MEAN, EXPLOIT_FULL = 70.0, 0.5
LEAGUE_GROUPS = {"A-League": "A-League", "Bundesliga": "Bundesliga (1+2)", "2. Bundesliga": "Bundesliga (1+2)"}
HERE = Path(__file__).resolve().parents[2]


def bucket_of(points: int) -> str:
    return "100" if points >= 100 else ("50-99" if points >= 50 else "0-49")


def candidate_row(c: dict) -> dict:
    """One cache entry -> the pro's pick: points, coach score, outcome (kind, net xG), league group, match."""
    a = c["answer"]
    pro = next(o for o in a["options"] if o["k"] == a["pro_pick"])
    return {
        "match": str(c["features"]["match"]) if "features" in c else c["public"]["id"].rsplit("-", 2)[0],
        "league": LEAGUE_GROUPS.get(c["public"]["league"], c["public"]["league"]),
        "points": int(pro["points"]),
        "score": float(pro["score"]),
        "net_xg": float(a["outcome"]["net_xg_15s"]),
        "kind": a["outcome"]["kind"],
    }


def as_arrays(rows: list) -> dict:
    """Pro rows -> column arrays (match, points, score, net_xg, shot = shot or goal, kept = kind != lost)."""
    return {
        "match": np.array([r["match"] for r in rows], dtype=object),
        "points": np.array([r["points"] for r in rows], int),
        "score": np.array([r["score"] for r in rows], float),
        "net_xg": np.array([r["net_xg"] for r in rows], float),
        "shot": np.array([r["kind"] in ("shot", "goal") for r in rows], bool),
        "kept": np.array([r["kind"] != "lost" for r in rows], bool),
    }


def _bucket_masks(points: np.ndarray) -> dict:
    return {"100": points >= 100, "50-99": (points >= 50) & (points < 100), "0-49": points < 50}


def bucket_stats(rows) -> dict:
    """Per points bucket of the pro's pick: n, mean net xG in 15 s, P(shot or goal), completion (kind != lost).
    rows: list of candidate rows or as_arrays(...)."""
    a = rows if isinstance(rows, dict) else as_arrays(rows)
    out = {}
    for b, m in _bucket_masks(a["points"]).items():
        n = int(m.sum())
        out[b] = {
            "n": n,
            "net_xg": float(a["net_xg"][m].mean()) if n else None,
            "p_shot": float(a["shot"][m].mean()) if n else None,
            "completion": float(a["kept"][m].mean()) if n else None,
        }
    return out


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(stats.spearmanr(x, y)[0])


def match_bootstrap(rows, stat, n_boot: int = 1000, seed: int = 0) -> dict:
    """stat(arrays) -> float or dict of floats, on as_arrays columns; resamples whole matches with replacement.
    Returns {"value", "lo", "hi"} (2.5 / 97.5 percentiles; dicts are bootstrapped key by key, None values skipped)."""
    a = rows if isinstance(rows, dict) else as_arrays(rows)
    groups = [np.flatnonzero(a["match"] == g) for g in sorted(set(a["match"].tolist()))]
    rng = np.random.default_rng(seed)
    point = stat(a)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        draws.append(stat({k: v[idx] for k, v in a.items()}))

    def ci(vals, v):
        vals = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
        if v is None or not len(vals):
            return {"value": v, "lo": None, "hi": None}
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {"value": float(v), "lo": float(lo), "hi": float(hi)}

    if isinstance(point, dict):
        return {k: ci([d.get(k) for d in draws], point[k]) for k in point}
    return ci(draws, point)


def validity(rows, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bucket table (with match-bootstrap CIs on each number) and the two Spearman correlations with net xG."""
    a = rows if isinstance(rows, dict) else as_arrays(rows)

    def flat(x):
        s = bucket_stats(x)
        out = {f"{b}.{m}": s[b][m] for b in BUCKETS for m in ("net_xg", "p_shot", "completion")}
        out["sp_points"] = spearman(x["points"], x["net_xg"])
        out["sp_score"] = spearman(x["score"], x["net_xg"])
        return out

    table = bucket_stats(a)
    cis = match_bootstrap(a, flat, n_boot, seed)
    for b in BUCKETS:
        for m in ("net_xg", "p_shot", "completion"):
            c = cis[f"{b}.{m}"]
            table[b][m + "_ci"] = [c["lo"], c["hi"]]
    return {
        "n": int(len(a["points"])),
        "matches": len(set(a["match"].tolist())),
        "buckets": table,
        "spearman_points_netxg": cis["sp_points"],
        "spearman_score_netxg": cis["sp_score"],
    }


# ---- strategies (public geometry only) -------------------------------------------------------------------------


def _xy_at_release(public: dict, i: int):
    lead = public.get("lead_up")
    if lead:
        p = lead[-1][i]
        if p is not None:
            return np.array(p, float)
    return None


def _carrier_xy(public: dict) -> np.ndarray:
    p = _xy_at_release(public, public["carrier"])
    if p is None and public.get("ball") and public["ball"][-1] is not None:
        p = np.array(public["ball"][-1], float)
    return np.zeros(2) if p is None else p


def _target_xy(public: dict, o: dict) -> np.ndarray:
    p = _xy_at_release(public, o["player"])
    return np.array(o["to"], float) if p is None else p


def _passes(public: dict) -> list:
    return [o for o in public["options"] if o["kind"] == "pass"]


def pick_most_advanced(public: dict) -> int:
    """Pass to the teammate with the largest x at release (attack-normalised: toward the goal)."""
    ps = _passes(public)
    return ps[int(np.argmax([_target_xy(public, o)[0] for o in ps]))]["k"]


def pick_nearest(public: dict) -> int:
    c, ps = _carrier_xy(public), _passes(public)
    return ps[int(np.argmin([np.linalg.norm(_target_xy(public, o) - c) for o in ps]))]["k"]


def pick_longest(public: dict) -> int:
    c, ps = _carrier_xy(public), _passes(public)
    return ps[int(np.argmax([np.linalg.norm(_target_xy(public, o) - c) for o in ps]))]["k"]


def pick_shoot(public: dict) -> int:
    shot = [o for o in public["options"] if o["kind"] == "shot"]
    return shot[0]["k"] if shot else pick_most_advanced(public)


def pick_carry_forward(public: dict) -> int:
    """Carry straight ahead (angle 0); the carry closest to 0 degrees if there is none, else the most advanced pass."""
    cs = [o for o in public["options"] if o["kind"] == "carry"]
    if not cs:
        return pick_most_advanced(public)
    return cs[int(np.argmin([abs(o["angle"]) for o in cs]))]["k"]


TRIVIAL = {
    "most_advanced_pass": pick_most_advanced,
    "nearest_teammate": pick_nearest,
    "shoot_if_possible": pick_shoot,
    "carry_forward": pick_carry_forward,
    "longest_pass": pick_longest,
}


def points_by_k(answer: dict) -> dict:
    return {o["k"]: int(o["points"]) for o in answer["options"]}


def strategy_points(moments: list) -> dict:
    """moments: [(public, answer)] -> {strategy: [expected points per moment]} for pro, random, AI and TRIVIAL."""
    out = {name: [] for name in ("pro", "random", "ai", *TRIVIAL)}
    for public, answer in moments:
        pts = points_by_k(answer)
        out["pro"].append(pts[answer["pro_pick"]])
        out["random"].append(float(np.mean(list(pts.values()))))
        out["ai"].append(pts[answer["ai_pick"]])
        for name, pick in TRIVIAL.items():
            out[name].append(pts[pick(public)])
    return out


def strategy_table(moments: list) -> dict:
    """{strategy: {"mean", "full" (share of 100s; for random: mean share of 100-point options), "exploit"}}."""
    res = {}
    for name, v in strategy_points(moments).items():
        v = np.asarray(v, float)
        if name == "random":
            full = float(np.mean([np.mean([o["points"] == 100 for o in a["options"]]) for _, a in moments]))
        else:
            full = float(np.mean(v == 100)) if len(v) else float("nan")
        mean = float(v.mean()) if len(v) else float("nan")
        res[name] = {
            "mean": round(mean, 2),
            "full": round(full, 4),
            "exploit": bool(name in TRIVIAL and (mean >= EXPLOIT_MEAN or full >= EXPLOIT_FULL)),
        }
    return res


def close_calls(moments: list) -> dict:
    """Distribution of the number of 100-point options per moment, and the share of moments where the pro scores 100."""
    n100 = [sum(o["points"] == 100 for o in a["options"]) for _, a in moments]
    pro = [points_by_k(a)[a["pro_pick"]] == 100 for _, a in moments]
    dist = {str(k): int(v) for k, v in zip(*np.unique(n100, return_counts=True))} if n100 else {}
    return {
        "n_options_at_100": dist,
        "mean_options_at_100": float(np.mean(n100)) if n100 else None,
        "mean_options": float(np.mean([len(a["options"]) for _, a in moments])) if moments else None,
        "pro_scores_100": float(np.mean(pro)) if pro else None,
    }


# ---- I/O ---------------------------------------------------------------------------------------------------------


def load_cache(cache_dir: Path):
    """-> (pro rows, slim (public, answer) pairs: last lead-up frame only, no 'after', interest features + id)."""
    rows, pairs, feats = [], [], []
    for f in sorted(Path(cache_dir).glob("*.json")):
        for c in json.loads(f.read_text(encoding="utf-8")):
            rows.append(candidate_row(c))
            pub = dict(c["public"])
            pub["lead_up"] = pub["lead_up"][-1:] if pub.get("lead_up") else None
            pub["ball"] = pub["ball"][-1:] if pub.get("ball") else None
            ans = {k: c["answer"][k] for k in ("options", "ai_pick", "pro_pick", "outcome")}
            pairs.append((pub, ans))
            feats.append({**c["features"], "id": pub["id"]})
    return rows, pairs, feats


# ---- what if: candidate fixes, replayed through the real daily selection ----------------------------------------

FILTER_TRIVIAL = ("most_advanced_pass", "shoot_if_possible", "nearest_teammate")


def trivial_hit(public: dict, answer: dict, names=FILTER_TRIVIAL) -> bool:
    """True when one of the named trivial strategies scores 100 on the moment."""
    pts = points_by_k(answer)
    return any(pts[TRIVIAL[n](public)] == 100 for n in names)


def rescore(answer: dict, power: float = 1.0) -> dict:
    """The answer with the 0-90 partial credit replaced by 90 * ((r - rank) / (r - 1)) ** power (power 1 = current
    rule; 100-point options and the 45 of a lone non-best option unchanged)."""
    rest = sorted((o for o in answer["options"] if o["points"] < 100), key=lambda o: -o["score"])
    r = len(rest)
    new = {o["k"]: 45 if r == 1 else int(round(90 * ((r - i) / (r - 1)) ** power)) for i, o in enumerate(rest, 1)}
    return {**answer, "options": [{**o, "points": new.get(o["k"], o["points"])} for o in answer["options"]]}


def what_if(pairs: list, feats: list, n_days: int = 60, pool: int = 1500) -> dict:
    """Rebuild the daily sets (days.rank_pool + days.daily_sets, as coach.game.build) under each fix and score the
    strategies on the selected moments."""
    from . import days as GD

    by_id = {p["id"]: (p, a) for p, a in pairs}
    variants = {
        "current": (lambda p, a: True, 1.0),
        "steeper_credit": (lambda p, a: True, 2.0),
        "drop_trivial_100": (lambda p, a: not trivial_hit(p, a), 1.0),
        "drop_trivial_100+steeper_credit": (lambda p, a: not trivial_hit(p, a), 2.0),
    }
    out = {}
    for name, (keep, power) in variants.items():
        fs = [f for f in feats if keep(*by_id[f["id"]])]
        days = GD.daily_sets(GD.rank_pool(fs, pool), n_days)
        sel = [(by_id[f["id"]][0], rescore(by_id[f["id"]][1], power)) for d in days for f in d]
        out[name] = {"pool": len(fs), "days": len(days), "power": power, "strategies": strategy_table(sel)}
    return out


def load_release(release_dir: Path) -> list:
    root = Path(release_dir)
    index = json.loads((root / "public" / "days" / "index.json").read_text(encoding="utf-8"))
    out = []
    for d in index["days"]:
        day = json.loads((root / "public" / "days" / f"{d['date']}.json").read_text(encoding="utf-8"))
        for mid in day["moments"]:
            pub = json.loads((root / "public" / "moments" / f"{mid}.json").read_text(encoding="utf-8"))
            ans = json.loads((root / "answers" / f"{mid}.json").read_text(encoding="utf-8"))
            out.append((pub, ans))
    return out


def by_league(items: list, league_of) -> dict:
    out = {}
    for x in items:
        out.setdefault(LEAGUE_GROUPS.get(league_of(x), league_of(x)), []).append(x)
    return dict(sorted(out.items()))


def evaluate(rows: list, cand_pairs: list, released: list, n_boot: int = 1000, seed: int = 0, feats=None) -> dict:
    res = {
        "validity": {"all": validity(rows, n_boot, seed)},
        "difficulty_released": {"all": strategy_table(released)},
        "difficulty_candidates": {"all": strategy_table(cand_pairs)},
        "close_calls_released": {"all": close_calls(released)},
        "close_calls_candidates": {"all": close_calls(cand_pairs)},
        "n_released": len(released),
        "n_candidates": len(cand_pairs),
    }
    for lg, rs in by_league(rows, lambda r: r["league"]).items():
        res["validity"][lg] = validity(rs, n_boot, seed)
    for lg, ms in by_league(released, lambda m: m[0]["league"]).items():
        res["difficulty_released"][lg] = strategy_table(ms)
        res["close_calls_released"][lg] = close_calls(ms)
    if feats is not None:
        res["what_if"] = what_if(cand_pairs, feats)
    res["exploits"] = sorted(
        f"{scope}/{lg}/{name}"
        for scope in ("difficulty_released", "difficulty_candidates")
        for lg, t in res[scope].items()
        for name, s in t.items()
        if s["exploit"]
    )
    return res


def chart(res: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    groups = [g for g in res["validity"] if g != "all"]
    names = ["all", *groups]
    w = 0.8 / len(names)
    colors = ["#334155", "#2563eb", "#d97706", "#059669"]
    for j, g in enumerate(names):
        b = res["validity"][g]["buckets"]
        y = [b[k]["net_xg"] or 0 for k in BUCKETS]
        err = np.array(
            [
                [
                    max(0.0, y[i] - (b[k]["net_xg_ci"][0] if b[k]["net_xg_ci"][0] is not None else y[i]))
                    for i, k in enumerate(BUCKETS)
                ],
                [
                    max(0.0, (b[k]["net_xg_ci"][1] if b[k]["net_xg_ci"][1] is not None else y[i]) - y[i])
                    for i, k in enumerate(BUCKETS)
                ],
            ]
        )
        x = np.arange(len(BUCKETS)) + (j - (len(names) - 1) / 2) * w
        a1.bar(x, y, w, yerr=err, capsize=3, color=colors[j % len(colors)], label=f"{g} (n={res['validity'][g]['n']})")
    a1.axhline(0, color="#94a3b8", lw=0.8)
    a1.set_xticks(range(len(BUCKETS)), [f"{k} pts" for k in BUCKETS])
    a1.set_ylabel("net xG in the next 15 s (pro's pick)")
    a1.set_title("Do higher points predict better outcomes?\n(95 % CI, bootstrap over matches)", fontsize=10)
    a1.legend(fontsize=8, frameon=False)
    t = res["difficulty_released"]["all"]
    order = sorted(t, key=lambda k: t[k]["mean"])
    cols = ["#dc2626" if t[k]["exploit"] else ("#64748b" if k in TRIVIAL else "#2563eb") for k in order]
    a2.barh(order, [t[k]["mean"] for k in order], color=cols)
    for i, k in enumerate(order):
        a2.text(
            t[k]["mean"] + 1, i, f"{t[k]['mean']:.0f}  ({100 * t[k]['full']:.0f} % at 100)", va="center", fontsize=8
        )
    a2.axvline(EXPLOIT_MEAN, color="#dc2626", ls="--", lw=0.8)
    a2.set_xlim(0, 125)
    a2.set_xlabel("mean points per moment")
    a2.set_title(f"Released moments (n={res['n_released']}): strategies\n(red = exploit, dashed = 70 pts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", default="D:/cv-data/playmaker/cache")
    ap.add_argument("--release", default="D:/cv-data/playmaker/content-v1")
    ap.add_argument("--out", default=str(HERE / "docs" / "playmaker_content_eval.json"))
    ap.add_argument("--png", default=str(HERE / "docs" / "playmaker_content_eval.png"))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rows, cand, feats = load_cache(Path(a.cache))
    released = load_release(Path(a.release))
    res = evaluate(rows, cand, released, a.boot, a.seed, feats)
    from .. import stamp

    res = {
        "generated_by": stamp.generated_by(),
        "bootstrap": {"resamples": a.boot, "seed": a.seed, "unit": "match"},
        **res,
    }
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    chart(res, Path(a.png))
    print_summary(res)


def print_summary(res: dict) -> None:
    for g, v in res["validity"].items():
        print(f"validity [{g}] n={v['n']} matches={v['matches']}")
        for b in BUCKETS:
            s = v["buckets"][b]
            if s["n"]:
                print(
                    f"  {b:>6}: n={s['n']:5d} net_xg={s['net_xg']:+.4f} [{s['net_xg_ci'][0]:+.4f},{s['net_xg_ci'][1]:+.4f}]"
                    f" p_shot={s['p_shot']:.3f} completion={s['completion']:.3f}"
                )
        for key in ("spearman_points_netxg", "spearman_score_netxg"):
            c = v[key]
            print(f"  {key}: {c['value']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]")
    for scope in ("difficulty_released", "difficulty_candidates"):
        for g, t in res[scope].items():
            print(f"{scope} [{g}]")
            for name, s in t.items():
                flag = "  EXPLOIT" if s["exploit"] else ""
                print(f"  {name:20s} mean={s['mean']:6.1f} at100={100 * s['full']:5.1f}%{flag}")
    for scope in ("close_calls_released", "close_calls_candidates"):
        for g, c in res[scope].items():
            print(f"{scope} [{g}] {c}")
    for name, w in res.get("what_if", {}).items():
        t = w["strategies"]
        best = max(TRIVIAL, key=lambda k: t[k]["mean"])
        print(
            f"what_if {name:32s} pool={w['pool']:5d} days={w['days']} pro={t['pro']['mean']:.1f} "
            f"random={t['random']['mean']:.1f} best trivial={best} {t[best]['mean']:.1f} "
            f"({100 * max(t[k]['full'] for k in TRIVIAL):.0f} % at 100 max)"
        )
    print("EXPLOIT: " + ", ".join(res["exploits"]) if res["exploits"] else "no exploit")


if __name__ == "__main__":
    main()

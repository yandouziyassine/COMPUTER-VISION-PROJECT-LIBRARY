"""Held-out test of the player profiles: do the offsets improve the pass and shot models on matches the profiles
were not built from?

    python -m coach.profiles_eval      # from 20-football-analytics/, after coach.cross_league (~5 min)

Leave-one-match-out. For each held-out match the base model AND the player offsets come only from the other matches;
the offsets are residuals against out-of-fold expectations of the training matches (nested leave-one-match-out),
and k / a are picked by nested held-out log-loss inside the training matches. Data: the derived tables cached by
coach.cross_league (no tracking needed); passer / shooter ids from the IDSSE match JSON and, for SkillCorner, from
the dynamic-events CSV (streamed, only the ids are cached in cv-data/skillcorner/profiles/players.json).
Writes docs/coach_profiles_results.json and docs/coach_profiles.png (matplotlib imported only for the chart).
"""

import csv
import io
import json
import pickle
from pathlib import Path

import numpy as np

from . import datasets as D
from . import models as ML
from . import profiles as PR

DOCS = Path(__file__).resolve().parents[1] / "docs"
BOOTSTRAPS = 2000
RULE = (
    "offsets change the advice only if the match-bootstrap 95 % CI of the held-out log-loss difference "
    "(with - without) lies entirely below 0; otherwise the profiles are display-only"
)


def subset(table, sel):
    return {k: v[sel] for k, v in table.items() if isinstance(v, np.ndarray) and len(v) == len(sel)}


def oof(table, fit):
    """Leave-one-match-out predictions of `fit` on every row of `table` (weights used when present)."""
    X, y, g, w = table["X"], table["y"], table["match"], table.get("weight")
    p = np.full(len(y), np.nan)
    for m in np.unique(g):
        test = g == m
        model = fit(X[~test], y[~test]) if w is None else fit(X[~test], y[~test], w[~test])
        p[test] = model.predict_proba(X[test])[:, 1]
    return p


def lomo_profiles(table, keys, fit=None, base=None, scales=PR.A_GRID) -> dict:
    """Held-out model without (p0) and with (p1) player offsets. Either `fit` (refitted per fold; the training
    expectations behind the offsets are its inner out-of-fold predictions) or `base` (fixed expectations per row, e.g.
    the provider's xG). Per row: off (offset used), has (player had a profile); per fold: k, a, inner gain."""
    y, g = table["y"], table["match"]
    w = table["weight"] if "weight" in table else np.ones(len(y))
    keys = np.asarray(keys, dtype=str)
    p0, p1, off = (np.full(len(y), np.nan) for _ in range(3))
    has = np.zeros(len(y), bool)
    folds = {}
    for m in np.unique(g):
        test, train = g == m, g != m
        if fit is not None:
            tr = subset(table, train)
            p0[test] = fit(tr["X"], tr["y"], w[train]).predict_proba(table["X"][test])[:, 1]
            p_in = oof(tr, fit)
        else:
            p0[test], p_in = base[test], base[train]
        sel = PR.choose_k(g[train], keys[train], y[train], p_in, w[train], scales=scales)
        prof = PR.offsets(keys[train], y[train], p_in, w[train], k=sel["k"])
        off[test] = PR.offset_vector(keys[test], prof)
        has[test] = [k in prof for k in keys[test]]
        p1[test] = PR.apply(p0[test], off[test], sel["a"])
        folds[str(m)] = {"k": sel["k"], "a": sel["a"], "inner_gain": sel["base_log_loss"] - sel["log_loss"]}
    return {"p0": p0, "p1": p1, "off": off, "has": has, "folds": folds, "w": w}


def brier_rows(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def bootstrap_diff(match, y, p0, p1, w, fn=PR.row_loss, reps=BOOTSTRAPS, seed=0) -> dict:
    """Weighted mean of fn(with) - fn(without) and its 95 % interval, resampling whole matches with replacement."""
    d = w * (fn(y, p1) - fn(y, p0))
    ids, inv = np.unique(match, return_inverse=True)
    sd, sw = np.bincount(inv, weights=d), np.bincount(inv, weights=w)
    pick = np.random.default_rng(seed).integers(0, len(ids), size=(reps, len(ids)))
    boots = sd[pick].sum(axis=1) / sw[pick].sum(axis=1)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    return {"mean": float(d.sum() / w.sum()), "ci95": ci}


def quantiles(x) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {"n": 0}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {
        "n": int(len(x)),
        **{k: float(v) for k, v in zip(("p5", "p25", "median", "p75", "p95"), q)},
        "mean_abs": float(np.abs(x).mean()),
        "max_abs": float(np.abs(x).max()),
    }


def summarize(table, r) -> dict:
    y, g, w = table["y"], table["match"], r["w"]
    ll = bootstrap_diff(g, y, r["p0"], r["p1"], w)
    return {
        "rows": int(len(y)),
        "without": ML.metrics(y, r["p0"], w),
        "with": ML.metrics(y, r["p1"], w),
        "log_loss_diff": ll,
        "brier_diff": bootstrap_diff(g, y, r["p0"], r["p1"], w, fn=brier_rows),
        "coverage": float(r["has"].mean()),
        "coverage_weighted": float(w[r["has"]].sum() / w.sum()),
        "held_out_offsets": quantiles(r["off"][r["has"]]),
        "folds": r["folds"],
        "decision": "advice" if ll["ci95"][1] < 0 else "display-only",
    }


# ------------------------------------------------------------------------------------------- data


def idsse_keys(d, info):
    """Passer keys of the derived pass rows (event_idx -> event -> PersonId) and shooter keys of the derived shot rows
    (cross_league.shot_rows order: shots with a half and a team)."""
    ids, ev = info["player_ids"], info["events"]

    def key(e):
        return f"dfl:{ids[e['player']]}" if e.get("player") is not None else ""

    passes = [key(ev[i]) for i in d["pass"]["event_idx"]]
    shots = [key(e) for e in ev if e["type"] == "shot" and e.get("half") is not None and e["team"] in (0, 1)]
    if len(shots) != len(d["shot"]["y"]):
        raise ValueError(f"{len(shots)} shots in the events, {len(d['shot']['y'])} derived rows")
    return np.array(passes, dtype=str), np.array(shots, dtype=str)


def idsse_data() -> list:
    """The 7 IDSSE matches: derived tables (coach.cross_league cache) + keys, match info and the team of each player."""
    from . import skillcorner as SC
    from .prepare import MATCHES, OUT

    out = []
    for mid in MATCHES:
        path = SC.DERIVED / f"idsse_{mid}.pkl"
        if not path.exists():
            raise SystemExit(f"{path} missing: run python -m coach.cross_league first")
        d = pickle.loads(path.read_bytes())
        d["info"] = json.loads((OUT / f"{mid}.json").read_text(encoding="utf-8"))
        d["team"] = np.load(OUT / f"{mid}.npz")["team"]
        d["pass_keys"], d["shot_keys"] = idsse_keys(d, d["info"])
        out.append(d)
    return out


class _Same(dict):
    """pidx stand-in for skillcorner.parse_events: keeps SkillCorner player ids instead of tracking columns."""

    def get(self, key, default=None):
        return key


def sc_event_players(sheet: dict, event_rows) -> list:
    """[actor id, recipient id] of every event, in skillcorner.parse_events order (the order behind event_idx)."""
    from . import skillcorner as SC

    events, _ = SC.parse_events(event_rows, SC.parse_sheet(sheet), _Same(), (1.0, 1.0))
    return [[e["player"], e.get("recipient")] for e in events]


def check_target_mapping(recipients, target_idx) -> int:
    """Rebuilt events name receivers by SkillCorner id, the derived rows by tracking column: the event order is right
    only if the two correspond one to one. Returns the number of distinct receivers."""
    fwd, back = {}, {}
    for r, t in zip(recipients, target_idx):
        t = int(t)
        if r is None or fwd.setdefault(r, t) != t or back.setdefault(t, r) != r:
            raise ValueError(f"event order mismatch: recipient {r} vs target column {t}")
    return len(fwd)


def sc_data() -> list:
    """The 20 SkillCorner matches: derived tables + passer keys. Match sheet and dynamic events are streamed once
    (~5 MB a match); only actor / recipient ids, names, teams and roles are cached."""
    from . import skillcorner as SC

    cache = SC.OUT / "profiles" / "players.json"
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    out = []
    for mid in SC.MATCHES:
        d = pickle.loads((SC.DERIVED / f"{mid}.pkl").read_bytes())
        if mid not in store:
            base = f"{SC.RAW_URL}/matches/{mid}/{mid}"
            sheet = json.loads(SC._get(f"{base}_match.json"))
            rows = list(csv.DictReader(io.StringIO(SC._get(f"{base}_dynamic_events.csv").decode("utf-8-sig"))))
            info = SC.parse_sheet(sheet)
            store[mid] = {
                "actors": sc_event_players(sheet, rows),
                "players": {
                    str(k): {f: v[f] for f in ("team", "name", "position")} for k, v in info["players"].items()
                },
                "teams": [info["meta"]["home"], info["meta"]["guest"]],
            }
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(store), encoding="utf-8")
        actors = store[mid]["actors"]
        idx = d["pass"]["event_idx"]
        check_target_mapping([actors[i][1] for i in idx], d["pass"]["target_idx"])
        keys = [f"skillcorner:{actors[i][0]}" if actors[i][0] is not None else "" for i in idx]
        d["pass_keys"] = np.array(keys, dtype=str)
        d["players"], d["teams"] = store[mid]["players"], store[mid]["teams"]
        out.append(d)
    return out


def pass_rows(parts):
    return D.stack([d["pass"] for d in parts]), np.concatenate([d["pass_keys"] for d in parts])


def shot_rows(parts):
    """Placed non-penalty shots with a provider xG and features (the shots of the coach's evaluation)."""
    st, keys = D.stack([d["shot"] for d in parts]), np.concatenate([d["shot_keys"] for d in parts])
    keep = st["placed"] & ~st["penalty"] & np.isfinite(st["xg"]) & np.isfinite(st["X"]).all(axis=1)
    return subset(st, keep), keys[keep]


# ------------------------------------------------------------------------------------------ results


def final_profile(table, keys, p_oof) -> tuple:
    """k / a chosen on every match (nested held-out) and the offsets of every player: the 'best k' and the
    distribution shown in the README."""
    w = table.get("weight")
    sel = PR.choose_k(table["match"], keys, table["y"], p_oof, w)
    return sel, PR.offsets(keys, table["y"], p_oof, w, k=sel["k"])


def profile_summary(sel, prof) -> dict:
    off = np.array([v["logit"] for v in prof.values()])
    n = np.array([v["n"] for v in prof.values()])
    return {
        "k": sel["k"],
        "a": sel["a"],
        "held_out_log_loss": sel["log_loss"],
        "base_log_loss": sel["base_log_loss"],
        "coverage": sel["coverage"],
        "curve_a1": sel.get("curve_a1"),
        "players": len(prof),
        "offsets": quantiles(off),
        "rows_per_player": quantiles(n),
        "share_abs_offset_over_0.1": float(np.mean(np.abs(off) > 0.1)) if len(off) else 0.0,
    }


def _pyplot():
    """matplotlib only for the chart (the tests and CI import this module without it)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


LABELS = {
    "pass_idsse": "pass, Bundesliga",
    "pass_skillcorner": "pass, A-League",
    "finishing_idsse": "finishing vs provider xG",
    "shot_provider_xg": "shot, provider xG",
    "shot_logistic": "shot, coach logistic",
}


def chart(finals, res, path):
    """Offsets vs sample size, held-out log-loss change vs k, and the with-minus-without log-loss with its 95 % CI."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for name, (sel, prof) in finals.items():
        n = [v["n"] for v in prof.values()]
        label = f"{LABELS[name]} (k = {sel['k']:g})"
        axes[0].scatter(n, [v["logit"] for v in prof.values()], s=10, alpha=0.6, label=label)
        curve = sel["curve_a1"]
        axes[1].plot([float(k) for k in curve], [v - sel["base_log_loss"] for v in curve.values()], "o-", label=label)
    axes[0].set(xscale="log", xlabel="rows of the player (passes / shots)", ylabel="shrunk offset (logit)")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("Player offsets vs sample size")
    axes[0].legend(fontsize=7)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_ylim(-0.003, 0.015)  # unshrunk finishing (k = 0) is far above: see the JSON curve_a1
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set(xlabel="k (prior strength)", ylabel="held-out log-loss change (a = 1)")
    axes[1].set_title("Choosing k (all matches, nested)")
    axes[1].legend(fontsize=7)
    rows = ("pass_idsse", "pass_skillcorner", "shot_provider_xg", "shot_logistic")
    for i, key in enumerate(rows):
        d = res[key]["log_loss_diff"]
        err = [[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]]
        axes[2].errorbar(d["mean"], i, xerr=err, fmt="o", capsize=4, color="#4c78a8")
    axes[2].set_yticks(range(len(rows)), [LABELS[r] for r in rows])
    axes[2].axvline(0, color="k", lw=0.8)
    axes[2].set(xlabel="log-loss with - without offsets (95 % CI)")
    axes[2].set_title("Held-out effect (leave-one-match-out)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _line(name, s):
    return (
        f"{name}: log-loss {s['without']['log_loss']:.4f} -> {s['with']['log_loss']:.4f}, diff {s['log_loss_diff']}, "
        f"coverage {s['coverage']:.3f}, folds {[(f['k'], f['a']) for f in s['folds'].values()]}"
    )


def main():
    from . import skillcorner as SC

    idsse, sc = idsse_data(), sc_data()
    res, finals = {}, {}
    for name, parts in (("idsse", idsse), ("skillcorner", sc)):
        pt, keys = pass_rows(parts)
        res[f"pass_{name}"] = summarize(pt, lomo_profiles(pt, keys, fit=ML.fit_logistic))
        s1 = summarize(pt, lomo_profiles(pt, keys, fit=ML.fit_logistic, scales=(1.0,)))
        res[f"pass_{name}"]["a_fixed_1"] = {k: s1[k] for k in ("with", "log_loss_diff", "folds")}
        finals[f"pass_{name}"] = final_profile(pt, keys, oof(pt, ML.fit_logistic))
        print(_line(f"pass {name}", res[f"pass_{name}"]), flush=True)
    st, sk = shot_rows(idsse)
    res["shot_provider_xg"] = summarize(st, lomo_profiles(st, sk, base=st["xg"]))
    res["shot_logistic"] = summarize(st, lomo_profiles(st, sk, fit=ML.fit_logistic))
    finals["finishing_idsse"] = final_profile(st, sk, st["xg"])
    for k in ("shot_provider_xg", "shot_logistic"):
        print(_line(k, res[k]), flush=True)
    desc = PR.skillcorner_descriptors(SC.OUT / "aggregates")
    sc_keys = set(np.concatenate([d["pass_keys"] for d in sc]).tolist())
    out = {
        **res,
        "profiles_all_matches": {k: profile_summary(*v) for k, v in finals.items()},
        "skillcorner_descriptors": {
            "players": len(desc),
            "tracked_passers": len(sc_keys),
            "tracked_passers_with_descriptors": len(sc_keys & set(desc)),
        },
        "decision": {"pass": res["pass_idsse"]["decision"], "shot": res["shot_provider_xg"]["decision"], "rule": RULE},
        "grids": {"k": list(PR.K_GRID), "a": list(PR.A_GRID)},
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_profiles_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    chart(finals, res, DOCS / "coach_profiles.png")
    print(json.dumps(out["profiles_all_matches"], indent=1)[:3000])
    print("descriptors", out["skillcorner_descriptors"], "decision", out["decision"])


if __name__ == "__main__":
    main()

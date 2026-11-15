"""Coach v2: does the better data (release sync + direction target rule) make a better coach, and should the
zone x context value / arrival terms ship? Held-out, with 95 % intervals from resampling whole matches.

    python -m coach.coach_v2              # from 20-football-analytics/ (first run builds the Bundesliga tables, ~10 min)
    python -m coach.coach_v2 --rebuild    # rebuild them (after a change to sync.py / datasets.py)
    python -m coach.coach_v2 --advice     # + full advice on every held-out pass, v1- vs v2-trained (~30 min)

Bundesliga data configs (the 20 A-League matches come from the model lab cache: their events carry SkillCorner frame
numbers and recorded receivers, so neither change applies to them):

    v1            sync "legacy",  failed-pass targets "ray"        what the coach shipped with
    v2            sync "release", targets "ray"                    coach v2 (shipped)
    v2_direction  sync "release", targets "direction"              tried: better receiver guesses, worse advice

Every comparison scores the SAME held-out rows, the v2 rows of the held-out match (the best estimate of what happened),
so a model trained on v1 data and one trained on v2 data answer the same questions; the end-to-end check is also
reported on the passes that v1 placed too ("common"), so a gain cannot come from the recovered passes alone.

Recipes: "bundesliga" = 7-match leave-one-match-out on Bundesliga tables only (the website's recipe: pass logistic,
value and arrival ridge); "pooled" = 27-match leave-one-match-out on all matches (the model lab's recipe; the game
content's models are pooled too). Models: "current" (logistic / ridge / ridge on the 10-feature basis) and "zxc" (ridge
with zone x context terms for value and arrival, model_lab.interactions). End to end = Spearman between the coach score
of the pro's chosen pass (advice.options' formula) and the net xG of the next 15 s.

Ship rule, fixed before looking at the numbers: a data or model change ships only if the end-to-end gain is >= 0 at the
point estimate in BOTH recipes and the lower end of its pooled 95 % interval is > -0.005 (not measurably worse); zxc
additionally needs a pooled end-to-end interval above zero (it costs a port of the context features to the browser).
Writes docs/coach_v2_results.json.
"""

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import datasets as D
from . import model_lab as L
from . import sync
from .match import load
from .prepare import MATCHES, OUT

DOCS = Path(__file__).resolve().parents[1] / "docs"
DATA = {"v1": ("legacy", "ray"), "v2": ("release", "ray"), "v2_direction": ("release", "direction")}
MODELS = {
    "current": ("current_logistic", "current_ridge", "current_ridge"),
    "zxc": ("current_logistic", "ridge_zone_x_context", "ridge_zone_x_context"),
}
NOT_WORSE = -0.005


def cache_path() -> Path:
    return OUT / "coach_v2_tables.pkl"


def build_tables(config: str) -> dict:
    method, rule = DATA[config]
    old = D.TARGET_RULE
    D.TARGET_RULE = rule
    try:
        out = {}
        for mid in MATCHES:
            m = load(OUT / mid)
            sync.attach_frames(m, method)
            out[mid] = L.match_lab_tables(m)
            print(f"  {config} {mid}: {len(out[mid]['pass']['y'])} passes", flush=True)
        return out
    finally:
        D.TARGET_RULE = old


def load_tables(rebuild=False) -> dict:
    path = cache_path()
    if path.exists() and not rebuild:
        return pickle.loads(path.read_bytes())
    tables = {c: build_tables(c) for c in DATA}
    path.write_bytes(pickle.dumps(tables))
    return tables


def aleague_tables() -> dict:
    """The 20 A-League matches' lab tables (model lab cache), with a dummy event index on their decision rows."""
    cache = pickle.loads(L.cache_path().read_bytes())
    out = {}
    for mid, t in cache.items():
        if L.league(mid) != "aleague":
            continue
        t = copy.copy(t)
        t["decisions"] = {**t["decisions"], "event": np.full(len(t["decisions"]["outcome"]), -1)}
        out[mid] = t
    return out


def predictions(train: dict, test: dict, mids, combos: dict) -> dict:
    """Leave-one-match-out predictions on the test tables. train / test: match id -> lab tables; combos: name ->
    (pass, value, arrival) model names. Returns name -> {pass, pass_noint, value, arrival, score} held-out arrays,
    aligned with the stacked test tables."""
    stack = lambda tabs, key: D.stack([tabs[m][key] for m in mids])
    T = {k: stack(test, k) for k in ("pass", "value", "arrival", "decisions")}
    out = {
        c: {
            k: np.zeros(len(T[t]["y"] if t != "decisions" else T[t]["outcome"]))
            for k, t in (
                ("pass", "pass"),
                ("pass_noint", "pass"),
                ("value", "value"),
                ("value_mean", "value"),
                ("arrival", "arrival"),
                ("arrival_mean", "arrival"),
                ("score", "decisions"),
            )
        }
        for c in combos
    }
    for mid in mids:
        others = [x for x in mids if x != mid]
        P, V, A = (D.stack([train[x][k] for x in others]) for k in ("pass", "value", "arrival"))
        te = {k: T[k]["match"] == mid for k in T}
        fitted = {}
        for c, (pn, vn, an) in combos.items():
            if ("p", pn) not in fitted:
                fitted[("p", pn)] = L.fit_pass(pn, P["X_full"], P["y"], P["weight"], P["match"])
                fitted[("pn", pn)] = L.fit_pass(pn, P["X_full_noint"], P["y"], P["weight"], P["match"])
            if ("v", vn) not in fitted:
                fitted[("v", vn)] = L.fit_value(vn, V["X_full"], V["y"])
            if ("a", an) not in fitted:
                fitted[("a", an)] = L.fit_value(an, A["X_full"], A["y"])
            pm, pnm, vm, am = fitted[("p", pn)], fitted[("pn", pn)], fitted[("v", vn)], fitted[("a", an)]
            o, dec = out[c], T["decisions"]
            o["pass"][te["pass"]] = pm.predict_proba(T["pass"]["X_full"][te["pass"]])[:, 1]
            o["pass_noint"][te["pass"]] = pnm.predict_proba(T["pass"]["X_full_noint"][te["pass"]])[:, 1]
            o["value"][te["value"]] = vm.predict(T["value"]["X_full"][te["value"]])
            o["value_mean"][te["value"]] = V["y"].mean()
            o["arrival"][te["arrival"]] = am.predict(T["arrival"]["X_full"][te["arrival"]])
            o["arrival_mean"][te["arrival"]] = A["y"].mean()
            p = pm.predict_proba(dec["pass"][te["decisions"]])[:, 1]
            o["score"][te["decisions"]] = L.chosen_score(
                p,
                am.predict(dec["target"][te["decisions"]]),
                vm.predict(dec["loss"][te["decisions"]]),
                vm.predict(dec["now"][te["decisions"]]),
            )
    return {"pred": out, "test": T}


def metrics(pred: dict, T: dict, sel=None) -> dict:
    """Component and end-to-end metrics of one prediction set on the test tables (sel: decision-row mask)."""
    P, V, A, dec = T["pass"], T["value"], T["arrival"], T["decisions"]
    mse_skill = lambda y, p, m: 1 - L._mse(y, p) / L._mse(y, m)
    s = np.ones(len(dec["outcome"]), bool) if sel is None else sel
    lg = np.array([L.league(x) for x in dec["match"]])
    return {
        "pass_log_loss": L._ll(P["y"], pred["pass"], P["weight"]),
        "pass_auc": L._auc(P["y"], pred["pass"], P["weight"]),
        "pass_log_loss_noint": L._ll(P["y"], pred["pass_noint"], P["weight"]),
        "pass_auc_noint": L._auc(P["y"], pred["pass_noint"], P["weight"]),
        "value_mse_skill": mse_skill(V["y"], pred["value"], pred["value_mean"]),
        "arrival_mse_skill": mse_skill(A["y"], pred["arrival"], pred["arrival_mean"]),
        "end_to_end": L._sp(pred["score"][s], dec["outcome"][s]),
        "end_to_end_per_league": {
            k: L._sp(pred["score"][s & (lg == k)], dec["outcome"][s & (lg == k)])
            for k in ("idsse", "aleague")
            if (s & (lg == k)).any()
        },
        "n_decisions": int(s.sum()),
    }


def gains(new: dict, old: dict, T: dict, sel=None, reps=L.BOOT) -> dict:
    """Match-bootstrap 95 % intervals of new - old: end-to-end Spearman, pass log-loss (old - new: > 0 = better),
    value / arrival MSE (old - new)."""
    P, V, A, dec = T["pass"], T["value"], T["arrival"], T["decisions"]
    s = np.ones(len(dec["outcome"]), bool) if sel is None else sel
    g = dec["match"][s]
    y, sn, so = dec["outcome"][s], new["score"][s], old["score"][s]
    out = {"end_to_end": L.boot_diff(g, lambda i: L._sp(sn[i], y[i]), lambda i: L._sp(so[i], y[i]), reps)}
    if sel is None:
        w = P["weight"]
        out["pass_log_loss"] = L.boot_diff(
            P["match"],
            lambda i: L._ll(P["y"][i], old["pass"][i], w[i]),
            lambda i: L._ll(P["y"][i], new["pass"][i], w[i]),
            reps,
        )
        out["value_mse"] = L.boot_diff(
            V["match"], lambda i: L._mse(V["y"][i], old["value"][i]), lambda i: L._mse(V["y"][i], new["value"][i]), reps
        )
        out["arrival_mse"] = L.boot_diff(
            A["match"],
            lambda i: L._mse(A["y"][i], old["arrival"][i]),
            lambda i: L._mse(A["y"][i], new["arrival"][i]),
            reps,
        )
    return out


def common_mask(tables: dict, T: dict) -> np.ndarray:
    """Decision rows (v2, stacked) whose pass v1 placed too (A-League rows: always True)."""
    placed = {(mid, int(e)) for mid, t in tables["v1"].items() for e in t["decisions"]["event"]}
    dec = T["decisions"]
    return np.array(
        [L.league(mid) == "aleague" or (mid, int(e)) in placed for mid, e in zip(dec["match"], dec["event"])], bool
    )


def run(tables: dict, reps=L.BOOT) -> dict:
    t0 = time.time()
    al = aleague_tables()
    res = {"data_configs": {k: list(v) for k, v in DATA.items()}, "models": {k: list(v) for k, v in MODELS.items()}}
    res["rows"] = {
        c: {k: int(sum(len(t[k]["y"]) for t in tables[c].values())) for k in ("pass", "arrival", "value")} for c in DATA
    }
    for recipe in ("bundesliga", "pooled"):
        mids = list(MATCHES) + (list(al) if recipe == "pooled" else [])
        test = {**tables["v2"], **al}
        preds, T = {}, None
        for c in DATA:
            train = {**tables[c], **al}
            combos = MODELS if c == "v2" else {"current": MODELS["current"]}
            r = predictions(train, test, mids, combos)
            T = r["test"]
            for mname, p in r["pred"].items():
                preds[f"{c}/{mname}"] = p
            print(f"{recipe} {c} ({time.time() - t0:.0f} s)", flush=True)
        common = common_mask(tables, T)
        out = {"n_matches": len(mids), "metrics": {}, "metrics_common": {}, "gains": {}}
        for k, p in preds.items():
            out["metrics"][k] = metrics(p, T)
            out["metrics_common"][k] = metrics(p, T, common)
        for new, old in (
            ("v2/current", "v1/current"),
            ("v2_direction/current", "v2/current"),
            ("v2_direction/current", "v1/current"),
            ("v2/zxc", "v2/current"),
            ("v2/zxc", "v1/current"),
        ):
            out["gains"][f"{new} vs {old}"] = {
                "all": gains(preds[new], preds[old], T, None, reps),
                "common": gains(preds[new], preds[old], T, common, reps)["end_to_end"],
            }
        res[recipe] = out
    res["verdicts"] = verdicts(res)
    res["seconds"] = round(time.time() - t0)
    return res


RANK_BUCKETS = ((1, 1), (2, 3), (4, 6), (7, 99))


def advice_stats(tables: dict, configs=("v1", "v2"), reps=L.BOOT) -> dict:
    """Full advice (advice.options) on the v2 pass decisions of each Bundesliga match, with the website's models
    (export.fit_models; shot model + the A-League shots) trained on each data config's tables of the 6 other matches:
    the coach's rank of the pro's pass (top-1 / top-3 of all options), completion of the pro's pass by that rank, and
    whether the pro's receiver is among the coach's top-3 pass options (vs random and the 3 nearest teammates)."""
    from . import advice as A
    from . import cross_league as CL
    from . import evaluate as EV
    from . import export as X

    extra = CL.skillcorner_shots()[0]
    matches = {}
    for mid in MATCHES:
        matches[mid] = load(OUT / mid)
        sync.attach_frames(matches[mid], DATA["v2"][0])
    out, per = {}, {}
    for c in configs:
        rows = []
        for mid, m in matches.items():
            others = [x for x in MATCHES if x != mid]
            stacked = {k: D.stack([tables[c][x][k] for x in others]) for k in ("pass", "value", "arrival", "shot")}
            stacked["shot"] = D.drop_penalties(stacked["shot"])
            models = A.from_fitted(X.fit_models(stacked, extra))
            for d in EV.pass_decisions(m, models):
                opts, chosen, p = d["options"], d["chosen"], d["sample"]
                passes = [o for o in opts if o["kind"] == "pass"]
                here = p["state"]["att"][p["carrier"]]
                near = sorted(passes, key=lambda o: float(np.hypot(o["to"][0] - here[0], o["to"][1] - here[1])))
                rows.append(
                    {
                        "match": mid,
                        "rank": 1 + next(i for i, o in enumerate(opts) if o is chosen),
                        "pass_rank": 1 + next(i for i, o in enumerate(passes) if o is chosen),
                        "n_pass": len(passes),
                        "nearest3": any(o is chosen for o in near[:3]),
                        "success": bool(p["event"]["success"]),
                        "best_kind": opts[0]["kind"],
                        "gap": opts[0]["score"] - chosen["score"],
                    }
                )
            print(f"  advice {c} {mid}: {len(rows)} decisions", flush=True)
        per[c] = rows
        r = np.array([x["rank"] for x in rows])
        ok = np.array([x["success"] for x in rows])
        out[c] = {
            "n": len(rows),
            "top1": float(np.mean(r == 1)),
            "top3": float(np.mean(r <= 3)),
            "receiver_top3": float(np.mean([x["pass_rank"] <= 3 for x in rows])),
            "receiver_top3_random": float(np.mean([min(3, x["n_pass"]) / x["n_pass"] for x in rows])),
            "receiver_nearest3": float(np.mean([x["nearest3"] for x in rows])),
            "completion_by_rank": {
                f"{lo}-{hi}" if hi < 99 else f"{lo}+": float(ok[(r >= lo) & (r <= hi)].mean())
                for lo, hi in RANK_BUCKETS
                if ((r >= lo) & (r <= hi)).any()
            },
            "best_is_pass": float(np.mean([x["best_kind"] == "pass" for x in rows])),
            "gap_mean": float(np.mean([x["gap"] for x in rows])),
        }
    if len(configs) == 2:
        a, b = (per[c] for c in configs)
        g = np.array([x["match"] for x in a])
        t3 = lambda rows: lambda i: float(np.mean([rows[k]["pass_rank"] <= 3 for k in i]))
        out["receiver_top3_gain"] = L.boot_diff(g, t3(b), t3(a), reps)
    return out


def verdicts(res) -> dict:
    out = {}
    for name, key, strict in (
        ("data v2", "v2/current vs v1/current", False),
        ("direction target rule", "v2_direction/current vs v2/current", False),
        ("zone x context", "v2/zxc vs v2/current", True),
    ):
        b, p = res["bundesliga"]["gains"][key]["all"]["end_to_end"], res["pooled"]["gains"][key]["all"]["end_to_end"]
        ok = b["delta"] >= 0 and p["delta"] >= 0 and p["ci"][0] > NOT_WORSE
        if strict:
            ok = ok and p["ci"][0] > 0
        out[name] = {"ship": bool(ok), "bundesliga": b, "pooled": p}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--reps", type=int, default=L.BOOT)
    ap.add_argument("--out", type=Path, default=DOCS / "coach_v2_results.json")
    ap.add_argument("--advice", action="store_true", help="only the full-advice comparison, merged into --out")
    args = ap.parse_args()
    tables = load_tables(args.rebuild)
    from .stamp import generated_by

    if args.advice:
        res = json.loads(args.out.read_text(encoding="utf-8"))
        res["advice"] = L._clean(advice_stats(tables, reps=args.reps))
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res["advice"], indent=1))
        return
    res = run(tables, args.reps)
    res = {"generated_by": generated_by(), **res}
    args.out.write_text(json.dumps(L._clean(res), indent=1), encoding="utf-8")
    for recipe in ("bundesliga", "pooled"):
        for k, v in res[recipe]["metrics"].items():
            print(recipe, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    print(json.dumps(L._clean(res["verdicts"]), indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()

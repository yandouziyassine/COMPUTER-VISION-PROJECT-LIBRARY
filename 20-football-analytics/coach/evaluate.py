"""Evaluate the AI coach on the 7 IDSSE matches (leave-one-match-out) and write docs/coach_results.json + charts.

    python -m coach.evaluate          # from 20-football-analytics/, after coach.prepare (~10-20 min on CPU)

Charts only (no broadcast imagery exists in this dataset; positions are drawn by us).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import advice as A  # noqa: E402
from . import datasets as D  # noqa: E402
from . import features as F  # noqa: E402
from . import models as ML  # noqa: E402
from .match import load  # noqa: E402
from .prepare import MATCHES, OUT  # noqa: E402

DOCS = Path(__file__).resolve().parents[1] / "docs"


def strip(r):
    return {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in r.items()}


def advice_hit_rate(matches, fitted) -> dict:
    """For held-out matches: is the pro's actual receiver the coach's best pass option / in its top 3?

    Only completed passes are used (the actual receiver is only recorded for those). Reference rankings on the same
    passes: the success probability alone, the nearest teammate, and a random order (expected value).
    """
    rules = {
        "coach": lambda o, a: o["score"],
        "pass_probability_only": lambda o, a: o["p"],
        "nearest_teammate": lambda o, a: -np.hypot(o["to"][0] - a[0], o["to"][1] - a[1]),
    }
    hits = {k: [0, 0] for k in rules}
    n = 0
    random1 = random3 = 0.0
    for m in matches:
        models = A.from_fitted({k: v[m.match_id] for k, v in fitted.items()})
        for e in m.events:
            if e["type"] != "pass" or e.get("frame") is None or e.get("set_piece") or e.get("cross"):
                continue
            s = F.state(m, e["half"], e["frame"], e["team"])
            where = {p: i for i, p in enumerate(s["att_idx"])}
            if e["player"] not in where or e["recipient"] not in where:
                continue
            opts = [o for o in A.options(s, where[e["player"]], models) if o["kind"] == "pass"]
            here = s["att"][where[e["player"]]]
            n += 1
            random1 += 1 / len(opts)
            random3 += min(3, len(opts)) / len(opts)
            for k, f in rules.items():
                ranked = [o["target"] for o in sorted(opts, key=lambda o: -f(o, here))]
                hits[k][0] += ranked[:1] == [e["recipient"]]
                hits[k][1] += e["recipient"] in ranked[:3]
    n = max(n, 1)
    out = {"passes": n, "top1": hits["coach"][0] / n, "top3": hits["coach"][1] / n}
    out["reference"] = {k: {"top1": v[0] / n, "top3": v[1] / n} for k, v in hits.items() if k != "coach"}
    out["reference"]["random"] = {"top1": random1 / n, "top3": random3 / n}
    return out


def reliability_chart(results, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for name, r in results.items():
        pts = [(b["mean_pred"], b["observed"]) for b in r["reliability"] if b["n"] >= 20]
        if pts:
            ax.plot(*zip(*pts), "o-", label=f"{name} (Brier {r['brier']:.3f})")
    ax.set(xlabel="predicted", ylabel="observed", title=title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    matches = [load(OUT / mid) for mid in MATCHES]
    tables = {
        "pass": D.stack([D.pass_table(m) for m in matches]),
        "value": D.stack([D.value_table(m) for m in matches]),
        "shot": D.stack([D.shot_table(m) for m in matches]),
    }
    i = F.PASS_FEATURES.index
    results = {
        "pass": ML.lomo(
            tables["pass"],
            {
                "logistic": ML.fit_logistic,
                "boosting": ML.fit_boosting,
                "distance_only": ML.column(i("distance")),
                "target_control_only": ML.column(i("target_control")),
                "base_rate": ML.fit_base_rate,
            },
        ),
        "value": ML.lomo(
            tables["value"],
            {
                "logistic": ML.fit_logistic,
                "boosting": ML.fit_boosting,
                "goal_distance_only": ML.column(F.VALUE_FEATURES.index("goal_dist")),
                "base_rate": ML.fit_base_rate,
            },
        ),
        "shot": ML.lomo(tables["shot"], {"logistic": ML.fit_logistic, "base_rate": ML.fit_base_rate}),
    }
    # held-out models per match for the advice sanity check (logistic everywhere: it is what the site runs)
    fitted = {k: {} for k in tables}
    for k, t in tables.items():
        for g in np.unique(t["match"]):
            fitted[k][str(g)] = ML.fit_logistic(t["X"][t["match"] != g], t["y"][t["match"] != g])
    out = {k: strip(v) for k, v in results.items()}
    out["advice"] = advice_hit_rate(matches, fitted)
    out["sync_audit"] = json.loads((OUT / "sync_audit.json").read_text())
    DOCS.mkdir(exist_ok=True)
    (DOCS / "coach_results.json").write_text(json.dumps(out, indent=2))
    reliability_chart(
        {k: results["pass"][k] for k in ("logistic", "boosting", "distance_only")},
        DOCS / "coach_pass_reliability.png",
        "Pass success, held-out matches",
    )
    reliability_chart(
        {k: results["value"][k] for k in ("logistic", "boosting")},
        DOCS / "coach_value_reliability.png",
        "Shot within 10 s, held-out matches",
    )
    for k, r in out.items():
        if k in ("pass", "value", "shot"):
            print(
                k,
                {
                    n: (round(v["log_loss"], 4), round(v["brier"], 4), v["auc"] and round(v["auc"], 3))
                    for n, v in r.items()
                },
            )
    print("advice", out["advice"])


if __name__ == "__main__":
    main()

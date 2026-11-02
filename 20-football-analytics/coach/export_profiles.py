"""profiles.json for the website's replay match, built WITHOUT that match (player profiles, phase 1).

    python -m coach.export_profiles --match J03WOH --site ../../cv-data/football-coach/site-export

pass_skill: the passer's offset against the coach's pass model (logistic, the model of models.json, trained without
the match); the training passes' expectations are out-of-fold among the training matches. finishing: goals against
the provider's xG (placed non-penalty shots). k and a: nested held-out log-loss within the training matches. Whether
the offsets may change the advice comes from the held-out evaluation (docs/coach_profiles_results.json "decision",
written by coach.profiles_eval); without it: display-only, a = 0. Players are keyed by their index in the match (the
match.json players list); the DFL PersonId is kept in "id" and "mapping". Kept apart from coach.export on purpose.
"""

import argparse
import json
from pathlib import Path

from . import models as ML
from . import profiles as PR
from . import profiles_eval as PE

RESULTS = PE.DOCS / "coach_profiles_results.json"
SOURCE = "open-data: IDSSE events of the other matches (pass model residuals; goals vs provider xG)"
USAGE = "pass: logit(p) += a.pass * players[passer].pass_skill.logit; shot: logit(xG) += a.shot * finishing.logit"


def build(match_id: str, parts=None, decision=None) -> dict:
    parts = PE.idsse_data() if parts is None else parts
    target = next(d for d in parts if d["match"] == match_id)
    train = [d for d in parts if d["match"] != match_id]
    pt, pk = PE.pass_rows(train)
    p_in = PE.oof(pt, ML.fit_logistic)
    sel_pass = PR.choose_k(pt["match"], pk, pt["y"], p_in, pt["weight"])
    st, sk = PE.shot_rows(train)
    sel_shot = PR.choose_k(st["match"], sk, st["y"], st["xg"])
    provider = PR.OpenDataProvider(
        {"pass_skill": (pk, pt["y"], p_in, pt["weight"]), "finishing": (sk, st["y"], st["xg"], None)},
        {"pass_skill": sel_pass["k"], "finishing": sel_shot["k"]},
    )
    if decision is None:
        decision = json.loads(RESULTS.read_text(encoding="utf-8"))["decision"] if RESULTS.exists() else {}
    effect = {m: decision.get(m, "display-only") for m in ("pass", "shot")}
    info = target["info"]
    players = [
        {"id": f"dfl:{pid}", "name": name, "team": int(t)}
        for pid, name, t in zip(info["player_ids"], info["names"], target["team"])
    ]
    names = [info["meta"]["home"], info["meta"]["guest"]]
    played = [{d["info"]["meta"]["home"], d["info"]["meta"]["guest"]} for d in train]
    teams = {str(t): {"name": names[t], "matches_in_profiles": sum(names[t] in s for s in played)} for t in (0, 1)}
    return PR.build(
        players,
        [provider],
        source=SOURCE,
        built_without=[match_id],
        trained_on=[d["match"] for d in train],
        k={"pass": sel_pass["k"], "shot": sel_shot["k"]},
        a={m: sel["a"] if effect[m] == "advice" else 0.0 for m, sel in (("pass", sel_pass), ("shot", sel_shot))},
        a_fitted={"pass": sel_pass["a"], "shot": sel_shot["a"]},
        advice_effect="advice" if "advice" in effect.values() else "display-only",
        advice_effect_by_model=effect,
        usage=USAGE,
        teams=teams,
    )


def write(profiles: dict, site: Path) -> Path:
    site = Path(site)
    site.mkdir(parents=True, exist_ok=True)
    PR.save(profiles, site / "profiles.json")
    return site / "profiles.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", default="J03WOH")
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()
    path = write(build(args.match), args.site)
    prof = PR.load(path)
    print("wrote", path, path.stat().st_size, "bytes,", len(prof["players"]), "players, k", prof["k"], "a", prof["a"])


if __name__ == "__main__":
    main()

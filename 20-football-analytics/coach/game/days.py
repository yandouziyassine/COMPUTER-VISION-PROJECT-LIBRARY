"""Interest score and daily sets of 5 moments (spec section 4)."""

import numpy as np

THIRD_X = 52.5 / 3
ZONES = ("build", "middle", "middle", "final", "final")


def zone(x: float) -> str:
    return "build" if x < -THIRD_X else ("final" if x > THIRD_X else "middle")


def interest(f: dict, v_rank: float) -> float:
    bonus = 1.0 + 0.5 * f["cutback"] + 0.3 * (not f["ai_is_pro"]) + 0.3 * f["shot_after"]
    return f["iqr"] * (0.5 + v_rank) * bonus


def rank_pool(feats: list, size: int = 1500) -> list:
    """Drop all-best and obvious moments, sort by interest, keep the top `size`."""
    keep = [f for f in feats if not f["all_best"] and not f["obvious"]]
    if not keep:
        return []
    v = np.array([f["v_now"] for f in keep])
    ranks = v.argsort().argsort() / max(1, len(v) - 1)
    scored = sorted(zip(keep, ranks), key=lambda t: -interest(t[0], float(t[1])))
    return [f for f, _ in scored[:size]]


def daily_sets(ranked: list, n_days: int) -> list:
    """Greedy: for each day take the most interesting unused moment per zone slot, distinct matches, no match of the
    previous day, at least one followed by a shot or goal; order the 5 by difficulty."""
    used, prev, days = set(), set(), []
    for _ in range(n_days):
        day, matches = [], set()
        for slot in ZONES:
            pick = next(
                (
                    f
                    for f in ranked
                    if f["id"] not in used and zone(f["x"]) == slot and f["match"] not in matches | prev
                ),
                None,
            )
            if pick is None:
                return days
            day.append(pick)
            matches.add(pick["match"])
            used.add(pick["id"])
        if not any(f["shot_after"] for f in day):
            swap = next(
                (
                    (i, f)
                    for i, slot in enumerate(ZONES)
                    if slot == "final"
                    for f in ranked
                    if f["id"] not in used
                    and f["shot_after"]
                    and zone(f["x"]) == "final"
                    and f["match"] not in (matches - {day[i]["match"]}) | prev
                ),
                None,
            )
            if swap is None:
                return days
            i, f = swap
            used.discard(day[i]["id"])
            matches.discard(day[i]["match"])
            day[i] = f
            used.add(f["id"])
            matches.add(f["match"])
        days.append(sorted(day, key=lambda f: f["difficulty"]))
        prev = matches
    return days

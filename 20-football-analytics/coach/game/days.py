"""Interest score and daily sets of 5 moments (spec section 4)."""

import numpy as np

THIRD_X = 52.5 / 3
ZONES = ("build", "middle", "middle", "final", "final")


def zone(x: float) -> str:
    return "build" if x < -THIRD_X else ("final" if x > THIRD_X else "middle")


def _pct(values) -> np.ndarray:
    v = np.asarray(values, float)
    return v.argsort().argsort() / max(1, len(v) - 1)


def interest(f: dict, iqr_rank: float, v_rank: float) -> float:
    """Percentile ranks within the moment's zone (a spread of 0.01 xG means more in the own third than near goal)."""
    bonus = 1.0 + 0.5 * f["cutback"] + 0.3 * (not f["ai_is_pro"]) + 0.3 * f["shot_after"]
    return (0.25 + iqr_rank) * (0.5 + v_rank) * bonus


def rank_pool(feats: list, size: int = 1500) -> list:
    """Drop all-best and obvious moments; rank each zone by interest (percentiles within the zone); keep the top of
    each zone in the daily-slot proportions (1 build-up : 2 middle : 2 final); most interesting first overall."""
    keep = [f for f in feats if not f["all_best"] and not f["obvious"]]
    share = {z: ZONES.count(z) / len(ZONES) for z in set(ZONES)}
    out = []
    for z, frac in share.items():
        zf = [f for f in keep if zone(f["x"]) == z]
        if not zf:
            continue
        iq, vr = _pct([f["iqr"] for f in zf]), _pct([f["v_now"] for f in zf])
        scored = sorted(((f, interest(f, float(a), float(b))) for f, a, b in zip(zf, iq, vr)), key=lambda t: -t[1])
        out += scored[: int(round(size * frac))]
    return [f for f, _ in sorted(out, key=lambda t: -t[1])]


def daily_sets(ranked: list, n_days: int) -> list:
    """Greedy: for each day take the most interesting unused moment per zone slot, distinct matches, no match of the
    previous day (relaxed only when the pool has too few matches), at least one followed by a shot or goal; order the
    5 by difficulty."""
    used, prev, days = set(), set(), []
    for _ in range(n_days):
        day, matches = [], set()
        for slot in ZONES:
            free = [f for f in ranked if f["id"] not in used and zone(f["x"]) == slot and f["match"] not in matches]
            # yesterday's matches only when the pool has too few matches to avoid them (small dev releases)
            pick = next((f for f in free if f["match"] not in prev), free[0] if free else None)
            if pick is None:
                return days
            day.append(pick)
            matches.add(pick["match"])
            used.add(pick["id"])
        if not any(f["shot_after"] for f in day):
            swaps = [
                (i, f)
                for i, slot in enumerate(ZONES)
                if slot == "final"
                for f in ranked
                if f["id"] not in used
                and f["shot_after"]
                and zone(f["x"]) == "final"
                and f["match"] not in matches - {day[i]["match"]}
            ]
            swap = next((t for t in swaps if t[1]["match"] not in prev), swaps[0] if swaps else None)
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

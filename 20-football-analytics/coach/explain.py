"""Why the coach likes or dislikes an option (coach insight, part B): exact model terms turned into short phrases.

Pass and shot models are logistic regressions exported as JSON weights (models.export_linear): logit =
intercept + sum_k coef_k * (x_k - mean_k) / scale_k, so each feature's contribution coef_k * (x_k - mean_k) / scale_k
is exact (they sum with the intercept to the logit; positive = raises the success probability). The value terms are
the exact split of the score (advice.py): score = p * gain + (1 - p) * loss, with gain = V(target) - V(now) (the
change if the pass arrives) and loss = -V_opp(loss point) - V(now) (the change if it is lost there). A shot scores
xG - V(now). A carry's p is pitch control, not a fitted model: its reasons are the control and the value terms.

Phrases (2-3 per option): for a pass or a shot the two features with the largest |contribution| (ties: feature order)
rendered with PHRASES, then the value phrase; for a carry its pitch control, then the value phrase. PHRASES and fmt are the whole recipe: the website's TypeScript port copies them (the parity fixture ships the
table) and must produce the same strings.
"""

import math

import numpy as np

# How a feature value becomes text. v = fmt(|x| * scale, nd). Variant by "split":
#   "mean": "hi" if x >= mean (training mean, from the weights) else "lo";  "zero": "hi" if x >= 0 else "lo";
#   "count": n = fmt(|x|, 0): "0" if n == "0", "1" if n == "1", else "n";  "none": "text".
# Placeholder {v} in every template.
DEG = 180.0 / math.pi
PHRASES = {
    "pass": {
        "distance": {"split": "mean", "scale": 1.0, "nd": 0, "hi": "long pass ({v} m)", "lo": "short pass ({v} m)"},
        "forward": {"split": "zero", "scale": 1.0, "nd": 0, "hi": "{v} m forward", "lo": "{v} m backward"},
        "abs_angle": {"split": "none", "scale": DEG, "nd": 0, "text": "{v}° off straight ahead"},
        "lane_defenders": {
            "split": "count",
            "scale": 1.0,
            "nd": 0,
            "0": "no defender in the lane",
            "1": "1 defender in the lane",
            "n": "{v} defenders in the lane",
        },
        "receiver_space": {"split": "none", "scale": 1.0, "nd": 1, "text": "receiver has {v} m of space"},
        "passer_pressure": {"split": "none", "scale": 1.0, "nd": 1, "text": "nearest defender {v} m from the passer"},
        "target_control": {"split": "none", "scale": 100.0, "nd": 0, "text": "{v} % pitch control at the receiver"},
        "target_x": {
            "split": "zero",
            "scale": 1.0,
            "nd": 0,
            "hi": "receiver {v} m into the opponent half",
            "lo": "receiver {v} m inside his own half",
        },
        "target_goal_dist": {"split": "none", "scale": 1.0, "nd": 0, "text": "receiver {v} m from goal"},
    },
    "shot": {
        "goal_dist": {"split": "none", "scale": 1.0, "nd": 0, "text": "shot from {v} m"},
        "goal_angle": {"split": "none", "scale": DEG, "nd": 0, "text": "{v}° of goal mouth visible"},
        "cone_defenders": {
            "split": "count",
            "scale": 1.0,
            "nd": 0,
            "0": "no defender in the shot cone",
            "1": "1 defender in the shot cone",
            "n": "{v} defenders in the shot cone",
        },
    },
    # value phrases: {gain} / {loss} / {xg} / {keep} are signed(x) ("+0.021", "-0.004"), {pct} = fmt(100 p, 0)
    "value": {
        "pass": "{gain} xG if it arrives, {loss} if lost there",
        "carry_control": "{pct} % pitch control where the carry ends",
        "carry": "{gain} xG if kept, {loss} if lost there",
        "shot": "{xg} xG for the shot, {keep} if the ball is kept here",
    },
}
TOP_FEATURES = 2
FRAGILE = 1e-6  # a value this close to a rounding or variant boundary could print differently in TypeScript


def fmt(y: float, nd: int) -> str:
    """Round half away from zero to nd decimals: floor(|y| * 10^nd + 0.5), then a plain decimal string. The site
    computes Math.floor(Math.abs(y) * 10 ** nd + 0.5) (same IEEE operations), so both print the same digits; a
    minus sign only when the rounded value is not 0."""
    n = math.floor(abs(y) * 10**nd + 0.5)
    s = str(n).rjust(nd + 1, "0")
    s = s if nd == 0 else s[:-nd] + "." + s[-nd:]
    return ("-" if y < 0 and n > 0 else "") + s


def signed(y: float, nd: int = 3) -> str:
    """fmt with an explicit sign: "+0.021", "-0.004" ("+0.000" when it rounds to 0)."""
    s = fmt(y, nd)
    return s if s.startswith("-") else "+" + s


def _round_margin(y: float, nd: int) -> float:
    """How far |y| * 10^nd is from the nearest .5 rounding boundary."""
    t = abs(y) * 10**nd
    return abs(t - math.floor(t) - 0.5)


def contributions(weights: dict, x: dict) -> list:
    """[{feature, x, contribution}] in the model's feature order; contribution = coef * (x - mean) / scale."""
    return [
        {"feature": f, "x": float(x[f]), "contribution": float(c * (float(x[f]) - mu) / sd)}
        for f, mu, sd, c in zip(weights["features"], weights["mean"], weights["scale"], weights["coef"])
    ]


def feature_phrase(table: dict, weights: dict, feature: str, x: float) -> tuple:
    """(text, margin): the phrase of one feature value, and its distance to the nearest boundary that would change
    the text (rounding, or the hi / lo split)."""
    spec = table[feature]
    y = abs(x) * spec["scale"]
    v, margin = fmt(y, spec["nd"]), _round_margin(y, spec["nd"])
    split = spec["split"]
    if split == "mean":
        mu = weights["mean"][weights["features"].index(feature)]
        key, margin = ("hi" if x >= mu else "lo"), min(margin, abs(x - mu))
    elif split == "zero":
        key, margin = ("hi" if x >= 0 else "lo"), min(margin, abs(x))
    elif split == "count":
        key = v if v in ("0", "1") else "n"
    else:
        key = "text"
    return spec[key].replace("{v}", v), margin


def value_phrase(option: dict, v_now: float) -> tuple:
    """(texts, terms, margin): the value phrases of an option (a carry first states its pitch control), the exact
    value terms and the distance of the printed numbers to a rounding boundary."""
    t = PHRASES["value"]
    if option["kind"] == "shot":
        terms = {"xg": float(option["p"]), "keep": float(v_now)}
        text = t["shot"].replace("{xg}", signed(terms["xg"])).replace("{keep}", signed(terms["keep"]))
        return [text], terms, min(_round_margin(v, 3) for v in terms.values())
    terms = {"gain": float(option["value"]) - v_now, "loss": -float(option["v_lost"]) - v_now}
    kind = "carry" if option["kind"] == "carry" else "pass"
    texts = [t[kind].replace("{gain}", signed(terms["gain"])).replace("{loss}", signed(terms["loss"]))]
    margin = min(_round_margin(v, 3) for v in terms.values())
    if kind == "carry":
        pct = float(option["p"]) * 100.0
        texts.insert(0, t["carry_control"].replace("{pct}", fmt(pct, 0)))
        margin = min(margin, _round_margin(pct, 0))
    return texts, terms, margin


def explain(option: dict, v_now: float, weights: dict, features: dict | None = None) -> dict:
    """Explanation of one advice.options option. `weights`: the exported JSON weights ("pass", "shot"); `features`:
    pass_features (pass) or shot_features (shot) of the option, unused for a carry. Returns contributions (pass /
    shot, model order), intercept, value terms, phrases (pass / shot: the top TOP_FEATURES features by |contribution|
    then the value phrase; carry: its pitch control then the value phrase), signs (+1 helps / -1 hurts / 0 value phrase) and fragile (a boundary within FRAGILE: the TypeScript text
    could differ there, the parity test skips the strings)."""
    out = {"kind": option["kind"], "contributions": [], "intercept": None}
    phrases, signs, margin = [], [], math.inf
    model = {"pass": "pass", "shot": "shot"}.get(option["kind"])
    if model is not None:
        w = weights[model]
        cs = contributions(w, features)
        out["contributions"], out["intercept"] = cs, float(w["intercept"])
        mags = np.array([abs(c["contribution"]) for c in cs])
        order = sorted(range(len(cs)), key=lambda i: (-mags[i], i))
        top = order[:TOP_FEATURES]
        if len(order) > TOP_FEATURES:  # the cut between the kept and the next feature must not be a near tie
            margin = min(margin, mags[order[TOP_FEATURES - 1]] - mags[order[TOP_FEATURES]])
        for a, b in zip(top, top[1:]):
            margin = min(margin, mags[a] - mags[b])
        for i in top:
            text, mg = feature_phrase(PHRASES[model], w, cs[i]["feature"], cs[i]["x"])
            phrases.append(text)
            signs.append(1 if cs[i]["contribution"] >= 0 else -1)
            margin = min(margin, mg, mags[i])
    texts, terms, mg = value_phrase(option, v_now)
    phrases += texts
    signs += [0] * len(texts)
    margin = min(margin, mg)
    out.update({"value_terms": terms, "phrases": phrases, "signs": signs, "fragile": bool(margin < FRAGILE)})
    return out


def explain_options(s: dict, carrier: int, opts: list, weights: dict, v_now: float) -> list:
    """explain() for every option of advice.options(s, carrier, ...), in the same order."""
    from . import features as F

    rows = {int(s["att_idx"][j]): j for j in range(len(s["att"]))}
    here = s["att"][carrier]
    out = []
    for o in opts:
        if o["kind"] == "pass":
            f = F.pass_features(s, carrier, rows[o["target"]])
        elif o["kind"] == "shot":
            f = F.shot_features(here, s["def"])
        else:
            f = None
        out.append(explain(o, v_now, weights, f))
    return out

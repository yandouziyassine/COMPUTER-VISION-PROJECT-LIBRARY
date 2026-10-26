"""W4 · What drives the error: a mixed-effects regression of per-pixel error on measured attributes.

For every model, on the 654 NYU test images (cached v2 predictions, raw sensor depth as truth, Eigen crop):
  y = log(|aligned depth - gt| / gt + 0.01)      (log relative error; 0.01 keeps exact hits finite)
  attributes: surface (mirror / window+glass / screen / other, NYU labels), texture (local gradient energy, tercile),
              thin structure (inside a labeled segment narrower than 5 px), distance bin (< 2 m, 2-4 m, > 4 m),
              dark (luminance < 0.15)
  y ~ attributes + (1 | image)   (random intercept per image: some photos are just harder)
exp(coefficient) = the factor by which an attribute multiplies the typical relative error with the others held
fixed. Alignment: least squares in each model's own space (disparity or depth), so a metric model's global scale
error does not swamp the per-pixel effects.

Caveat: inside mirrors the sensor measures the reflected room, so the "mirror" factor compares the model with a
wrong reference; it measures disagreement with the sensor, not error against the physical surface (see W9).

Usage:
    python w4_attributes.py --preds <folder with preds/<model>__nyu__*.npz> [--per-image 150]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import models_v2 as M
import protocol as P

DOCS = Path(__file__).resolve().parent / "docs"
SURFACES = {"mirror": ["mirror"], "glass": ["window", "glass"], "screen": ["television", "monitor", "computer"]}


def texture_energy(gray, k=9):
    """Mean squared gradient magnitude in a k x k window (gray in [0, 1])."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.blur(gx * gx + gy * gy, (k, k))


def thin_mask(labels, width=5):
    """Pixels of labeled segments whose local width is below `width` px (distance to the segment edge < width/2)."""
    thin = np.zeros(labels.shape, bool)
    for c in np.unique(labels):
        if c == 0:
            continue
        seg = (labels == c).astype(np.uint8)
        dist = cv2.distanceTransform(np.pad(seg, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        # a pixel is in a thin part if the largest inscribed distance in its 5x5 neighbourhood is small
        local_max = cv2.dilate(dist, np.ones((width, width), np.uint8))
        thin |= (seg > 0) & (local_max < width / 2)
    return thin


def image_rows(rgb, gt, labels, pred, kind, ds, proto, names, rng, n):
    mask = P.valid_mask(gt, ds)
    method = "lsq_disparity" if kind == "disparity" else "lsq_depth"
    d = P.align(pred, gt, mask, method, ds["max_depth"], ds["min_depth"])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    tex = texture_energy(gray)
    thin = thin_mask(labels)
    surface = np.full(gt.shape, "other", object)
    for s, cls in SURFACES.items():
        ids = [names.index(c) + 1 for c in cls if c in names]
        surface[np.isin(labels, ids)] = s
    v, u = np.nonzero(mask)
    # oversample rare surfaces so the regression sees them (weights are not needed for the fixed effects' meaning)
    rare = np.isin(surface[v, u], list(SURFACES))
    pick = np.concatenate(
        [rng.choice(np.nonzero(~rare)[0], min(n, (~rare).sum()), replace=False), np.nonzero(rare)[0][:n]]
    )
    v, u = v[pick], u[pick]
    err = np.abs(d[v, u] - gt[v, u]) / gt[v, u]
    return {
        "y": np.log(err + 0.01),
        "surface": surface[v, u],
        "tex": tex[v, u],
        "thin": thin[v, u],
        "dist": gt[v, u],
        "dark": gray[v, u] < 0.15,
    }


def fit(rows, image_ids):
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({k: np.concatenate([r[k] for r in rows]) for k in rows[0]})
    df["image"] = np.concatenate([[i] * len(r["y"]) for i, r in zip(image_ids, rows)])
    df["texture"] = pd.qcut(df["tex"], 3, labels=["low", "mid", "high"]).astype(str)
    df["distance"] = pd.cut(df["dist"], [0, 2, 4, 100], labels=["lt2", "2to4", "gt4"]).astype(str)
    df["thin"] = df["thin"].astype(int)
    df["dark"] = df["dark"].astype(int)
    formula = (
        "C(surface, Treatment('other')) + C(texture, Treatment('high')) + thin + C(distance, Treatment('2to4')) + dark"
    )
    method, var_img = "mixed model, random intercept per image (REML)", None
    try:
        r = smf.mixedlm("y ~ " + formula, df, groups=df["image"]).fit(reml=True)
        var_img = float(r.cov_re.iloc[0, 0])
        ok = r.converged and var_img > 1e-6
    except (np.linalg.LinAlgError, ValueError):
        ok = False
    if ok:
        params, ci = r.fe_params, r.conf_int()
    else:
        # The random-intercept fit was singular: use the within-image estimator instead (image fixed effects,
        # i.e. every variable demeaned per image), with confidence intervals clustered by image.
        method = "image fixed effects (within estimator), CIs clustered by image; mixed-model fit was singular"
        X = smf.ols("y ~ " + formula, df).exog  # design matrix with intercept
        names = smf.ols("y ~ " + formula, df).exog_names
        Xd = X - pd.DataFrame(X).groupby(df["image"].values).transform("mean").values
        yd = df["y"].values - df.groupby("image")["y"].transform("mean").values
        import statsmodels.api as sm

        keep = [j for j, n in enumerate(names) if n != "Intercept"]
        r = sm.OLS(yd, Xd[:, keep]).fit(cov_type="cluster", cov_kwds={"groups": df["image"].values})
        params = pd.Series(r.params, index=[names[j] for j in keep])
        ci = pd.DataFrame(r.conf_int(), index=params.index)
    out = {}
    for name in params.index:
        if name == "Intercept":
            continue
        key = (
            name.replace("C(surface, Treatment('other'))[T.", "surface=")
            .replace("C(texture, Treatment('high'))[T.", "texture=")
            .replace("C(distance, Treatment('2to4'))[T.", "distance=")
            .rstrip("]")
        )
        out[key] = [round(float(np.exp(x)), 3) for x in (params[name], ci.loc[name, 0], ci.loc[name, 1])]
    return out, len(df), var_img, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--per-image", type=int, default=150)
    a = ap.parse_args()
    from nyu import NYU

    data = NYU()
    proto = P.load_protocol()
    ds = proto["datasets"]["nyu"]
    idx = data.test_indices()
    cache = [(data.rgb(i), data.raw(i).astype(np.float32), data.labels(i)) for i in idx]
    res = {
        "reference": "surface=other, texture=high, thin=0, distance 2-4 m, dark=0",
        "note": "values: [factor, 95% CI low, high] on the typical relative error",
    }
    for model in M.MODELS:
        files = sorted(Path(a.preds).rglob(f"preds/{model}__nyu__*.npz"))
        if not files:
            continue
        preds = {}
        for f in files:
            z = np.load(f)
            preds.update({k: z[k] for k in z.files})
        rng = np.random.default_rng(0)
        rows, ids = [], []
        for i, (rgb, gt, lab) in zip(idx, cache):
            rows.append(
                image_rows(
                    rgb,
                    gt,
                    lab,
                    preds[f"nyu_{i:04d}"].astype(np.float32),
                    M.kind(model),
                    ds,
                    proto,
                    data.names,
                    rng,
                    a.per_image,
                )
            )
            ids.append(i)
        factors, n, var_img, method = fit(rows, ids)
        res[model] = {"pixels": n, "method": method, "image_random_effect_var": var_img, "factors": factors}
        print(model, json.dumps(factors), flush=True)
        (DOCS / "w4_attributes.json").write_text(json.dumps(res, indent=1))
    figure(res)


def figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [m for m in M.MODELS if m in res]
    keys = list(res[models[0]]["factors"])
    fig, ax = plt.subplots(figsize=(10, 0.35 * len(keys) * 1.0 + 2))
    for j, m in enumerate(models):
        f = res[m]["factors"]
        y = np.arange(len(keys)) + (j - len(models) / 2) * 0.09
        vals = np.array([f[k] for k in keys])
        ax.errorbar(vals[:, 0], y, xerr=[vals[:, 0] - vals[:, 1], vals[:, 2] - vals[:, 0]], fmt="o", ms=3, label=m)
    ax.axvline(1, color="gray", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(keys)), keys)
    ax.set_xlabel("multiplies the typical relative error by (mixed model, 95 % CI; 1 = no effect)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("What drives depth error on NYU (other attributes held fixed, image as random effect)")
    fig.tight_layout()
    fig.savefig(DOCS / "w4_attributes.png", dpi=110)


if __name__ == "__main__":
    main()

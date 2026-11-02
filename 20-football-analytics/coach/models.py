"""Model fitting and leave-one-match-out (LOMO) evaluation shared by the pass, value and shot models."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y, w=None):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    return pipe.fit(X, y) if w is None else pipe.fit(X, y, logisticregression__sample_weight=w)


def fit_boosting(X, y, w=None):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=0, early_stopping=False
    ).fit(X, y, sample_weight=w)


def fit_base_rate(X, y, w=None):
    return DummyClassifier(strategy="prior").fit(X, y, sample_weight=w)


def column(i, cols=None):
    """Baseline that uses one feature (or the feature indices `cols`), still a logistic regression."""
    cols = [i] if cols is None else list(cols)
    return lambda X, y, w=None: _Column(cols, fit_logistic(X[:, cols], y, w))


def column_reg(i):
    """Regression baseline that uses one feature only (e.g. goal distance), still a Ridge."""
    return lambda X, y, w=None: _Column([i], fit_ridge(X[:, [i]], y))


class _Column:
    def __init__(self, cols, model):
        self.cols, self.model = cols, model

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, self.cols])

    def predict(self, X):
        return self.model.predict(X[:, self.cols])


def fit_ridge(X, y, w=None):
    """Value model: standardised fixed basis features + Ridge, exported as plain weights."""
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)


def fit_mean(X, y, w=None):
    return DummyRegressor(strategy="mean").fit(X, y)


def fit_boosting_regressor(X, y, w=None):
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=200, random_state=0, early_stopping=False
    ).fit(X, y)


class XTGrid:
    """xT-style baseline: mean target per cell of a location grid (x, |y|), shrunk towards the global mean."""

    def __init__(self, ix=0, iy=1, nx=14, ny=5, shrink=30.0):
        self.ix, self.iy, self.nx, self.ny, self.shrink = ix, iy, nx, ny, shrink

    def _cell(self, X):
        cx = np.clip(((X[:, self.ix] + 52.5) / 105 * self.nx).astype(int), 0, self.nx - 1)
        cy = np.clip((X[:, self.iy] / 34 * self.ny).astype(int), 0, self.ny - 1)
        return cx * self.ny + cy

    def fit(self, X, y):
        cell = self._cell(X)
        self.mean_ = float(np.mean(y))
        n = np.bincount(cell, minlength=self.nx * self.ny)
        total = np.bincount(cell, weights=y, minlength=self.nx * self.ny)
        self.table_ = (total + self.shrink * self.mean_) / (n + self.shrink)
        return self

    def predict(self, X):
        return self.table_[self._cell(X)]


def fit_xt_grid(X, y, w=None):
    return XTGrid().fit(X, y)


def export_linear(model, names, link="logit") -> dict:
    """Weights of a StandardScaler + (LogisticRegression | Ridge) pipeline: score = ((x - mean) / scale) . coef + b,
    probability = sigmoid(score) for link "logit", the score itself for "identity"."""
    scaler, last = model.steps[0][1], model.steps[-1][1]
    coef = np.ravel(last.coef_)
    return {
        "features": list(names),
        "link": link,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": coef.tolist(),
        "intercept": float(np.ravel(last.intercept_)[0]),
    }


def export_logistic(model, names) -> dict:
    return export_linear(model, names, "logit")


def export_ridge(model, names) -> dict:
    return export_linear(model, names, "identity")


def linear_predictor(w: dict):
    """Inverse of export_linear: X[n, k] -> P(y = 1) (link "logit") or the prediction (link "identity"), computed from
    the exported JSON weights alone, exactly what the website computes."""
    mean, scale, coef = (np.asarray(w[k], float) for k in ("mean", "scale", "coef"))

    def predict(X):
        z = ((np.atleast_2d(np.asarray(X, float)) - mean) / scale) @ coef + w["intercept"]
        return 1.0 / (1.0 + np.exp(-z)) if w["link"] == "logit" else z

    return predict


def reliability(y, p, bins=10, w=None):
    w = np.ones(len(y)) if w is None else w
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(sel.sum()),
                "mean_pred": float(np.average(p[sel], weights=w[sel])) if sel.any() else None,
                "observed": float(np.average(y[sel], weights=w[sel])) if sel.any() else None,
            }
        )
    return out


def metrics(y, p, w=None) -> dict:
    """Log-loss, Brier, AUC (sample-weighted when w is given; n is always the row count)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.average(y, weights=w)),
        "log_loss": float(log_loss(y, p, sample_weight=w, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p, sample_weight=w)),
        "auc": float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else None,
    }


def lomo(table, fitters: dict, weighted: bool = True) -> dict:
    """Each model is trained on all matches but one and scored on the held-out match; metrics on the pooled
    out-of-fold predictions, plus per-match metrics for the spread. When the table has a "weight" column (pass table:
    reweights the failed passes to the true completion rate) it is used for fitting and scoring unless weighted=False."""
    X, y, groups = table["X"], table["y"], table["match"]
    w = table.get("weight") if weighted else None
    result = {}
    for name, fit in fitters.items():
        oof = np.zeros(len(y))
        per = {}
        for g in np.unique(groups):
            test = groups == g
            model = fit(X[~test], y[~test]) if w is None else fit(X[~test], y[~test], w[~test])
            oof[test] = model.predict_proba(X[test])[:, 1]
            per[str(g)] = metrics(y[test], oof[test], None if w is None else w[test])
        result[name] = {
            **metrics(y, oof, w),
            "per_match": per,
            "reliability": reliability(y, oof, w=w),
            "oof": oof,
        }
    return result


def reg_metrics(y, p) -> dict:
    return {
        "n": int(len(y)),
        "mean": float(np.mean(y)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "r2": float(1 - np.sum((y - p) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)),
        "pearson": float(np.corrcoef(y, p)[0, 1]) if np.ptp(p) > 1e-12 and np.ptp(y) > 0 else None,
        "spearman": float(spearmanr(y, p)[0]) if np.ptp(p) > 1e-12 and np.ptp(y) > 0 else None,
    }


def calibration(y, p, bins=10) -> list:
    """Deciles of the prediction: mean prediction vs mean outcome."""
    order = np.argsort(p, kind="stable")
    out = []
    for chunk in np.array_split(order, bins):
        out.append({"n": int(len(chunk)), "mean_pred": float(p[chunk].mean()), "observed": float(y[chunk].mean())})
    return out


def lomo_reg(table, fitters: dict, y=None) -> dict:
    """Leave-one-match-out for regressors (fit(X, y) -> object with predict). `mse_skill` is 1 - MSE / MSE of the
    per-fold training mean (the "mean" fitter, when given)."""
    X, groups = table["X"], table["match"]
    y = table["y"] if y is None else y
    result = {}
    for name, fit in fitters.items():
        oof = np.zeros(len(y))
        per = {}
        for g in np.unique(groups):
            test = groups == g
            oof[test] = fit(X[~test], y[~test]).predict(X[test])
            per[str(g)] = reg_metrics(y[test], oof[test])
        result[name] = {**reg_metrics(y, oof), "per_match": per, "calibration": calibration(y, oof), "oof": oof}
    if "mean" in result:
        base = result["mean"]["rmse"] ** 2
        for r in result.values():
            r["mse_skill"] = float(1 - r["rmse"] ** 2 / base)
    return result

"""Model fitting and leave-one-match-out (LOMO) evaluation shared by the pass, value and shot models."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, y)


def fit_boosting(X, y):
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=300, random_state=0, early_stopping=False
    ).fit(X, y)


def fit_base_rate(X, y):
    return DummyClassifier(strategy="prior").fit(X, y)


def column(i):
    """Baseline that uses one feature only (e.g. pass distance), still a logistic regression."""
    return lambda X, y: _Column(i, fit_logistic(X[:, [i]], y))


def column_reg(i):
    """Regression baseline that uses one feature only (e.g. goal distance), still a Ridge."""
    return lambda X, y: _Column(i, fit_ridge(X[:, [i]], y))


class _Column:
    def __init__(self, i, model):
        self.i, self.model = i, model

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, [self.i]])

    def predict(self, X):
        return self.model.predict(X[:, [self.i]])


def fit_ridge(X, y):
    """Value model: standardised fixed basis features + Ridge, exported as plain weights."""
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(X, y)


def fit_mean(X, y):
    return DummyRegressor(strategy="mean").fit(X, y)


def fit_boosting_regressor(X, y):
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


def fit_xt_grid(X, y):
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


def reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": int(sel.sum()),
                "mean_pred": float(p[sel].mean()) if sel.any() else None,
                "observed": float(y[sel].mean()) if sel.any() else None,
            }
        )
    return out


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
    }


def lomo(table, fitters: dict) -> dict:
    """Each model is trained on all matches but one and scored on the held-out match; metrics on the pooled
    out-of-fold predictions, plus per-match metrics for the spread."""
    X, y, groups = table["X"], table["y"], table["match"]
    result = {}
    for name, fit in fitters.items():
        oof = np.zeros(len(y))
        per = {}
        for g in np.unique(groups):
            test = groups == g
            model = fit(X[~test], y[~test])
            oof[test] = model.predict_proba(X[test])[:, 1]
            per[str(g)] = metrics(y[test], oof[test])
        result[name] = {**metrics(y, oof), "per_match": per, "reliability": reliability(y, oof), "oof": oof}
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

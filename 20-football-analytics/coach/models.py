"""Model fitting and leave-one-match-out (LOMO) evaluation shared by the pass, value and shot models."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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


class _Column:
    def __init__(self, i, model):
        self.i, self.model = i, model

    def predict_proba(self, X):
        return self.model.predict_proba(X[:, [self.i]])


def export_logistic(model, names) -> dict:
    scaler, lr = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
    return {
        "features": list(names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
    }


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

"""Clasificador de tipo de fallo + baseline de referencia.

Dos vías:
  * Baseline fuerte: gradient boosting (XGBoost/LightGBM) en GPU sobre las features
    de vibración/corriente. Fija el listón que la red debe superar.
  * El ensemble final combina RUL + health-index + este clasificador.

Umbral de decisión con COSTE ASIMÉTRICO (un falso negativo cuesta fn_weight× más
que un falso positivo) → matriz de confusión con FN minimizados (requisito clave).

Librerías perezosas.
"""
from __future__ import annotations

from typing import Any


def train_gbm(X, y, cfg: dict[str, Any]):
    """Entrena XGBoost en GPU si está; si no, LightGBM; si no, sklearn. Devuelve el modelo."""
    fn_w = float(cfg.get("target", {}).get("fn_weight", 1.0))
    try:
        import xgboost as xgb
        # scale_pos_weight refleja el coste de FN (clase 'fallo' = positiva)
        clf = xgb.XGBClassifier(
            tree_method="hist", device="cuda", n_estimators=600, max_depth=8,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=fn_w, eval_metric="aucpr")
        clf.fit(X, y)
        return ("xgboost", clf)
    except Exception:
        pass
    try:
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(device_type="gpu", n_estimators=600, max_depth=8,
                                 learning_rate=0.05, class_weight={0: 1.0, 1: fn_w})
        clf.fit(X, y)
        return ("lightgbm", clf)
    except Exception:
        pass
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
    clf.fit(X, y)
    return ("sklearn", clf)


def best_threshold(y_true, y_score, fn_weight: float = 5.0):
    """Elige el umbral que minimiza el COSTE total (FN pesa fn_weight×). Devuelve
    (umbral, matriz de confusión, accuracy) — reporta FP/FN explícitos."""
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    thr_grid = np.unique(np.quantile(y_score, np.linspace(0.01, 0.99, 99)))
    best = None
    for thr in thr_grid:
        pred = (y_score >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp + fn_weight * fn
        acc = (tp + tn) / max(1, len(y_true))
        cand = {"threshold": float(thr), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "accuracy": round(acc, 4), "cost": cost}
        if best is None or cost < best["cost"]:
            best = cand
    return best


def hard_negatives(y_true, y_pred, X):
    """Devuelve los ejemplos FALSO NEGATIVO (fallo real no detectado) para reinyectar
    con más peso en la recalibración → reduce FN con el tiempo."""
    import numpy as np
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mask = (y_pred == 0) & (y_true == 1)
    return X[mask], mask.sum()

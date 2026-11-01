"""Train, select, evaluate and persist the T20 chase win-probability model.

Protocol (to avoid leakage and over-claiming):
  1. Matches are split chronologically: test = latest 20% of matches, validation =
     latest 15% of the remaining matches, fit = the rest. No match spans partitions.
  2. Candidate models are fitted on `fit` and compared on `validation` by log loss.
  3. The selected model is refitted on all training matches (fit + validation).
  4. Every candidate, plus baselines, is scored ONCE on the untouched test set.
  5. The reported test metrics are the only numbers presented as model performance.

Baselines:
  * majority  - always predicts the training-set chase win rate.
  * rrr_only  - logistic regression using only the required run rate.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import FEATURES, chronological_split


def _candidate_models():
    return {
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=50, max_depth=12,
            n_jobs=-1, random_state=config.RANDOM_STATE),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=config.RANDOM_STATE),
    }


def _metrics(y_true, proba) -> dict:
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    pred = (proba >= 0.5).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "brier": brier_score_loss(y_true, proba),
    }
    out["roc_auc"] = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    return {k: float(v) for k, v in out.items()}


def _calibration(y_true, proba, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({"bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
                     "mean_predicted": float(np.mean(proba[m])),
                     "observed_rate": float(np.mean(y_true[m])),
                     "count": int(m.sum())})
    return rows


def train(states: pd.DataFrame, log=print) -> dict:
    started = time.time()
    fit_m, val_m, test_m = chronological_split(states)
    X = states[FEATURES].to_numpy()
    y = states["label"].to_numpy()
    X_fit, y_fit = X[fit_m], y[fit_m]
    X_val, y_val = X[val_m], y[val_m]
    X_tr, y_tr = X[fit_m | val_m], y[fit_m | val_m]
    X_te, y_te = X[test_m], y[test_m]

    rng = np.random.default_rng(config.RANDOM_STATE)

    # ---- stage 1: select by validation log loss --------------------------
    val_scores = {}
    for name, model in _candidate_models().items():
        fit_X, fit_y = X_fit, y_fit
        if name == "random_forest" and len(fit_y) > config.RF_TRAIN_SAMPLE:
            pick = rng.choice(len(fit_y), config.RF_TRAIN_SAMPLE, replace=False)
            fit_X, fit_y = fit_X[pick], fit_y[pick]
        model.fit(fit_X, fit_y)
        val_scores[name] = float(log_loss(y_val, np.clip(model.predict_proba(X_val)[:, 1], 1e-6, 1 - 1e-6), labels=[0, 1]))
        log(f"  validation log loss {name}: {val_scores[name]:.4f}")
    selected = min(val_scores, key=val_scores.get)

    # ---- stage 2: refit on all training matches and score the test set once ----
    results = {}
    trained = {}
    for name, model in _candidate_models().items():
        fit_X, fit_y = X_tr, y_tr
        if name == "random_forest" and len(fit_y) > config.RF_TRAIN_SAMPLE:
            pick = rng.choice(len(fit_y), config.RF_TRAIN_SAMPLE, replace=False)
            fit_X, fit_y = fit_X[pick], fit_y[pick]
        model.fit(fit_X, fit_y)
        trained[name] = model
        proba = model.predict_proba(X_te)[:, 1]
        results[name] = {"test": _metrics(y_te, proba), "validation_log_loss": val_scores[name]}
        log(f"  test {name}: acc={results[name]['test']['accuracy']:.3f} auc={results[name]['test']['roc_auc']:.3f}")

    # ---- baselines ------------------------------------------------------------
    majority = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    results["baseline_majority"] = {"test": _metrics(y_te, majority.predict_proba(X_te)[:, 1])}
    rrr_idx = FEATURES.index("rrr")
    rrr_only = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    rrr_only.fit(X_tr[:, [rrr_idx]], y_tr)
    results["baseline_rrr_only"] = {"test": _metrics(y_te, rrr_only.predict_proba(X_te[:, [rrr_idx]])[:, 1])}

    best_model = trained[selected]
    best_proba_test = best_model.predict_proba(X_te)[:, 1]

    # ---- explainability: permutation importance on a test sample ------------
    sample = rng.choice(len(y_te), min(5000, len(y_te)), replace=False)
    perm = permutation_importance(best_model, X_te[sample], y_te[sample], n_repeats=3,
                                  random_state=config.RANDOM_STATE, scoring="roc_auc")
    importance = sorted(
        [{"feature": f, "importance_mean": float(m), "importance_std": float(s)}
         for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)],
        key=lambda d: d["importance_mean"], reverse=True)

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": selected,
        "selection_rule": "lowest log loss on the validation partition (matches before the test set)",
        "features": FEATURES,
        "label": "chasing side won the match (1) or lost (0)",
        "data": {
            "chase_states_total": int(len(states)),
            "matches_total": int(states["match_id"].nunique()),
            "fit_states": int(fit_m.sum()),
            "validation_states": int(val_m.sum()),
            "test_states": int(test_m.sum()),
            "fit_matches": int(states.loc[fit_m, "match_id"].nunique()),
            "validation_matches": int(states.loc[val_m, "match_id"].nunique()),
            "test_matches": int(states.loc[test_m, "match_id"].nunique()),
            "test_period": [str(states.loc[test_m, "date"].min()), str(states.loc[test_m, "date"].max())],
            "chase_win_rate_test": float(y_te.mean()),
        },
        "models": results,
        "calibration_test": _calibration(y_te, best_proba_test),
        "permutation_importance_test_auc": importance,
        "limitations": [
            "Trained on T20 chases of 20-over matches only, from the uploaded Cricsheet data.",
            "Features describe the score state only; team strength, venue, pitch and player quality are not modelled.",
            "Train, validation and test partitions are grouped by match to prevent match overlap. Metrics are calculated over pooled ball-states, not averaged per match; states within a match are correlated and should not be interpreted as independent match-level observations.",
            "Probabilities are model estimates from historical frequencies, not guarantees for any live match.",
            "Test metrics come from a single chronological hold-out; performance on future seasons may differ.",
        ],
        "training_seconds": round(time.time() - started, 1),
    }

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "name": selected, "features": FEATURES}, config.MODEL_FILE)
    config.EVAL_FILE.write_text(json.dumps(report, indent=2))
    return report


_cache: dict = {}


def load_model():
    """Return the saved model bundle, or None if it has not been trained yet."""
    if "bundle" not in _cache:
        _cache["bundle"] = joblib.load(config.MODEL_FILE) if config.MODEL_FILE.exists() else None
    return _cache["bundle"]


def clear_model_cache():
    _cache.clear()


def predict_win_probability(features: pd.DataFrame) -> np.ndarray | None:
    bundle = load_model()
    if bundle is None:
        return None
    return bundle["model"].predict_proba(features[bundle["features"]].to_numpy())[:, 1]

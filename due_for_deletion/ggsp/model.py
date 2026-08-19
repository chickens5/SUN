# ggsp/model.py
#
# STAGE 4 — Train, evaluate, and cache the GBR model.
#
# This module is responsible for supervised learning.  It takes the OMNI
# 3h DataFrame (built by omni_client.py) and produces:
#   1. A fitted sklearn Pipeline (StandardScaler → GradientBoostingRegressor)
#      that is saved to cache/model.joblib for future runs
#   2. A metrics dict with MAE, R², persistence baseline, CV results, and
#      new 95% confidence intervals on CV metrics
#   3. A test_frame DataFrame (held-out predictions vs observed) for plotting
#
# Why sklearn Pipeline instead of bare GBR?
#   StandardScaler re-centres and scales each feature to zero mean / unit variance
#   before the GBR sees it.  This doesn't change GBR training at all (trees are
#   scale-invariant), but it ensures that future changes — like adding a linear
#   layer or regularisation — won't be thrown off by wildly different feature scales.
#   It also makes the saved model self-contained: the scaler is baked in so you
#   never need to separately track the scaler state.
#
# Why chronological split (not random)?
#   Solar wind is strongly autocorrelated in time.  A random split would let
#   future observations "leak" into the training set — the model would see
#   test-set Kp values one or two lags away in the training set, making it look
#   much better than it really is.  The chronological split is the only honest
#   choice for time series.
#
# New in this refactor:
#   - CV 95% confidence intervals (mean ± 1.96 × std / sqrt(n_folds)) added
#     to the metrics dict and printed alongside the fold mean±std.
#   - _make_gbr_pipeline() extracted from a nested function to a module-level
#     function so it can be called from tests directly.
#
# Used by:  pipeline.py (Stage 4 of run_pipeline)
# Imports:  config.py

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance as _skl_perm_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FEATURE_COLUMNS, PipelineConfig, _EXPECTED_N_FEATURES


def make_gbr_pipeline(config: PipelineConfig) -> Pipeline:
    """Build a fresh unfitted sklearn Pipeline: StandardScaler → GBR.

    Extracted from the nested closure in the old monolith so it can be
    called by tests (e.g., 'fit on tiny dummy data, assert it predicts').
    All hyperparameters come from config so no magic numbers live here.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )),
    ])


def fit_and_evaluate_model(data_3h: pd.DataFrame, config: PipelineConfig):
    """STAGE 4 — Train the GBR and evaluate on the chronological held-out set.

    Input:  omni_3h (Stage 2 output) — DataFrame with FEATURE_COLUMNS + 'Kp'.
    Output: (fitted Pipeline, metrics dict, test_frame DataFrame)

    The fitted Pipeline is the only artifact passed forward to inference.
    test_frame is used for eval/evaluate.py and the held-out Kp plot.
    metrics is written to cache/model_metrics.json alongside model.joblib.

    CV note: CV folds use min(n_estimators, 100) trees for speed.  This means
    CV metrics are slightly pessimistic vs. the final model (which uses 200).
    The 95% CI printed alongside mean±std accounts for fold-to-fold variance
    but NOT for the 100-vs-200 tree difference — keep that in mind when
    comparing CV MAE to final MAE.
    """
    # ── Data validation ────────────────────────────────────────────────────────
    if len(data_3h) < 32:
        raise ValueError(
            f"Only {len(data_3h)} 3-hour OMNI rows — need at least 32. "
            "Use a wider --years range."
        )

    missing = [c for c in FEATURE_COLUMNS if c not in data_3h.columns]
    if missing:
        raise ValueError(
            f"Training frame missing feature columns: {missing}. "
            "Check OMNI parser and FEATURE_COLUMNS are in sync."
        )

    all_nan = [c for c in FEATURE_COLUMNS if data_3h[c].isna().all()]
    if all_nan:
        raise ValueError(
            f"Feature columns are entirely NaN: {all_nan}. "
            "Check _parse_omni2_yearly_line column indices."
        )

    # ── Train / test chronological split ──────────────────────────────────────
    # split_idx is the index of the first TEST row.  All rows before it train,
    # all rows from split_idx onward are the held-out evaluation window.
    split_idx = int(config.train_fraction * len(data_3h))
    split_idx = max(1, min(split_idx, len(data_3h) - 1))

    X = data_3h[FEATURE_COLUMNS].values
    y = data_3h["Kp"].values

    # Hard-fail here so any FEATURE_COLUMNS / X-builder mismatch is obvious.
    assert X.shape[1] == _EXPECTED_N_FEATURES, (
        f"Training X width {X.shape[1]} != expected {_EXPECTED_N_FEATURES}. "
        "FEATURE_COLUMNS and all X-building paths must stay in sync."
    )

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # ── TimeSeriesSplit cross-validation ───────────────────────────────────────
    # Expanding window (each fold adds more training data), gap=3 rows (9h).
    # The gap prevents the autoregressive kp_lag_3h/6h/9h from leaking the
    # last training-set Kp directly into the first test-set row.
    #
    # We cap at CV_MAX_ROWS for speed (30K rows ≈ 10 years of 3h data).
    # For 35yr runs this halves CV wall time from ~60s to ~20s.
    #
    # NEW: we compute 95% CI = mean ± 1.96 × (std / sqrt(n_folds)) on MAE, R², CSI.
    # This gives you a sense of how stable the fold scores are — a wide CI with
    # only 5 folds means you should interpret the scores cautiously.
    CV_MAX_ROWS = 30_000
    cv_results = {}

    if config.use_cv and len(X) >= 32 * config.n_cv_folds:
        X_cv = X[-CV_MAX_ROWS:] if len(X) > CV_MAX_ROWS else X
        y_cv = y[-CV_MAX_ROWS:] if len(y) > CV_MAX_ROWS else y
        tscv = TimeSeriesSplit(n_splits=config.n_cv_folds, gap=3)
        cv_maes, cv_r2s, cv_csis = [], [], []

        # CV folds use fewer estimators for speed — still representative of the
        # model's behaviour, just with a slightly shallower ensemble.
        cv_n_est = min(config.n_estimators, 100)

        for _fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X_cv), 1):
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue
            fold_model = Pipeline([
                ("scaler", StandardScaler()),
                ("gbr", GradientBoostingRegressor(
                    n_estimators=cv_n_est,
                    max_depth=config.max_depth,
                    learning_rate=config.learning_rate,
                    subsample=config.subsample,
                    min_samples_split=config.min_samples_split,
                    min_samples_leaf=config.min_samples_leaf,
                    random_state=config.random_state,
                )),
            ])
            fold_model.fit(X_cv[tr_idx], y_cv[tr_idx])
            yp = fold_model.predict(X_cv[te_idx])
            yt = y_cv[te_idx]

            cv_maes.append(float(mean_absolute_error(yt, yp)))
            cv_r2s.append(float(r2_score(yt, yp)))

            # G1 CSI on this fold.
            obs_pos  = yt >= 5.0
            pred_pos = yp >= 5.0
            tp = np.sum( obs_pos &  pred_pos)
            fp = np.sum(~obs_pos &  pred_pos)
            fn = np.sum( obs_pos & ~pred_pos)
            denom = tp + fp + fn
            cv_csis.append(float(tp / denom) if denom > 0 else 0.0)

        if cv_maes:
            n_f = len(cv_maes)
            # 95% CI = mean ± 1.96 × (std / sqrt(n_folds)).
            # With only 5 folds the t-distribution would give a slightly wider CI
            # (t_{4,0.025} ≈ 2.78 vs 1.96 for z), but 1.96 is conventional and
            # the difference is small for reporting purposes.
            mae_ci95 = 1.96 * float(np.std(cv_maes))  / (n_f ** 0.5)
            r2_ci95  = 1.96 * float(np.std(cv_r2s))   / (n_f ** 0.5)
            csi_ci95 = 1.96 * float(np.std(cv_csis))  / (n_f ** 0.5)

            cv_results = {
                "cv_mae_mean":       float(np.mean(cv_maes)),
                "cv_mae_std":        float(np.std(cv_maes)),
                "cv_mae_ci95":       mae_ci95,    # NEW — 95% CI half-width
                "cv_r2_mean":        float(np.mean(cv_r2s)),
                "cv_r2_std":         float(np.std(cv_r2s)),
                "cv_r2_ci95":        r2_ci95,     # NEW
                "cv_storm_csi_mean": float(np.mean(cv_csis)),
                "cv_storm_csi_ci95": csi_ci95,    # NEW
                "cv_n_folds":        n_f,
                "cv_n_estimators_used": cv_n_est, # documents the 100-vs-200 difference
            }
            print(
                f"[CV]  {n_f}-fold  "
                f"MAE={cv_results['cv_mae_mean']:.3f}±{cv_results['cv_mae_std']:.3f}"
                f"  (95% CI ±{mae_ci95:.3f})  "
                f"R²={cv_results['cv_r2_mean']:.3f}±{cv_results['cv_r2_std']:.3f}"
                f"  G1-CSI={cv_results['cv_storm_csi_mean']:.3f}±{csi_ci95:.3f}"
            )

    # ── Final model — trained on full chronological train split ───────────────
    # This is the model we save and use for live inference.  CV above was purely
    # to evaluate generalisation; it never touches the final fitted model.
    model = make_gbr_pipeline(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Persistence baseline ───────────────────────────────────────────────────
    # Persistence forecast: Kp(t) = Kp(t − 3h).
    # kp_lag_3h is one of the FEATURE_COLUMNS so we extract it by index.
    # This is the hardest baseline to beat for short-horizon Kp forecasting.
    kp_lag_col_idx = FEATURE_COLUMNS.index("kp_lag_3h")
    persistence_pred = X_test[:, kp_lag_col_idx]
    persistence_mae  = float(mean_absolute_error(y_test, persistence_pred))
    persistence_r2   = float(r2_score(y_test, persistence_pred))

    mean_baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "mae":             float(mean_absolute_error(y_test, y_pred)),
        "r2":              float(r2_score(y_test, y_pred)),
        "baseline_mae":    float(mean_absolute_error(y_test, mean_baseline)),
        "persistence_mae": persistence_mae,
        "persistence_r2":  persistence_r2,
        "test_count":      int(len(y_test)),
        **cv_results,
    }

    # ── Permutation feature importance ────────────────────────────────────────
    # Shuffles each feature column independently and measures how much the
    # model's MAE degrades.  A large degradation means that feature was important.
    # This is more reliable than the GBR's built-in split-based importances
    # because it reflects actual held-out performance, not training-set usage.
    if len(X_test) >= 20:
        try:
            perm = _skl_perm_importance(
                model, X_test, y_test,
                n_repeats=5, random_state=config.random_state,
            )
            ranked = sorted(
                zip(FEATURE_COLUMNS, perm.importances_mean),
                key=lambda kv: kv[1], reverse=True,
            )
            print("[Permutation importance — ranked by mean MAE increase on hold-out]")
            for feat, imp in ranked:
                print(f"  {feat:<20}  {imp:+.4f}")
        except Exception as perm_exc:
            print(f"[WARNING] Permutation importance failed: {perm_exc}", file=sys.stderr)

    # ── Post-fit sanity checks ─────────────────────────────────────────────────
    skill_ratio = metrics["mae"] / persistence_mae if persistence_mae > 0 else float("nan")
    if metrics["r2"] < 0.0:
        print(
            f"[WARNING] Model R²={metrics['r2']:.3f} is negative. "
            f"Persistence MAE={persistence_mae:.3f}, Model MAE={metrics['mae']:.3f}. "
            "Consider a wider --years window.",
            file=sys.stderr,
        )
    elif metrics["mae"] > 2.0:
        print(
            f"[WARNING] Model MAE={metrics['mae']:.3f} is unexpectedly high (>2.0). "
            "Typical well-fitted models score MAE < 0.7 on multi-year data.",
            file=sys.stderr,
        )
    else:
        improvement = (
            100.0 * (1.0 - metrics["mae"] / metrics["baseline_mae"])
            if metrics["baseline_mae"] > 0 else 0.0
        )
        beats = "beats" if skill_ratio < 1.0 else "WORSE than"
        print(
            f"[OK] Model: MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  "
            f"({improvement:.1f}% over mean-baseline)  "
            f"skill_ratio={skill_ratio:.3f} ({beats} persistence)  "
            f"test_n={metrics['test_count']}"
        )

    test_frame = pd.DataFrame(
        {"Kp_true": y_test, "Kp_pred": y_pred},
        index=data_3h.index[split_idx:],
    )
    return model, metrics, test_frame

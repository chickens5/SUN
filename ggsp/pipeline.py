# ggsp/pipeline.py
#
# The ORCHESTRATOR — run_pipeline() calls every other module in order.
#
# This file intentionally does NO computation itself.  Its only job is to
# wire the eight pipeline stages together, handle the model cache, and
# return a fully populated outputs dict.  Think of it as the conductor of
# an orchestra: it doesn't play any instrument, but it tells each section
# when to play and passes data between them.
#
# Stage order (must not be reordered without thought — data dependencies):
#   1  noaa_client.load_noaa_data()           live solar wind
#   2  omni_client.load_omni_data()            training archive
#   3  features.build_noaa_3h_features()       live inference frame
#   4  model.fit_and_evaluate_model()          GBR training + eval
#      (or load from cache/model.joblib)
#   5  eval.evaluate.print_evaluation_report() honest skill report
#   6  sunspot_pipeline.run_sunspot_pipeline() solar cycle weights
#   7a forecast.build_forecast_scenarios()     scenario solar wind arrays
#   7b forecast.predict_scenario_kp()         deterministic Kp trajectory
#   7c forecast.predict_scenario_kp_ensemble() stochastic storm probability
#   8  _serialize_outputs() → JSON            React frontend output
#
# Cache logic (Stage 4):
#   The model is cached as cache/model.joblib with metadata in
#   cache/model_meta.json (start_year, num_years, train_fraction).
#   If the requested training window matches the cached metadata, we skip
#   retraining and load the cached model — saving 30–120 seconds per run.
#   Pass refit=True (or --refit) to force retraining regardless of cache.
#
# Used by:  __main__.py  (the CLI entry point)
# Imports:  all other ggsp modules + optional eval, sunspot_pipeline

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

from .config import (
    FEATURE_COLUMNS, MODEL_CACHE_PATH, MODEL_META_PATH,
    PipelineConfig, _REPO_DIR,
)
from .features import build_noaa_3h_features
from .forecast import (
    build_forecast_scenarios, predict_scenario_kp,
    predict_scenario_kp_ensemble, weighted_ensemble,
)
from .model import fit_and_evaluate_model
from .noaa_client import load_noaa_data
from .omni_client import load_omni_data
from .physics import kp_label
from .viz import plot_forecast, plot_test_results


def run_pipeline(
    config: PipelineConfig | None = None,
    make_plots: bool = True,
    json_output_path: str | None = None,
    refit: bool = False,
    static_weights: bool = False,
) -> dict:
    """Orchestrate end-to-end data → model → forecast → export workflow.

    Parameters
    ----------
    config           : PipelineConfig instance.  None → use all defaults.
    make_plots       : If True, show matplotlib figures at the end of the run.
    json_output_path : If given, write the serialised outputs to this .json file.
    refit            : Force model retraining even if a valid cache exists.
    static_weights   : Skip the sunspot cycle modifier; use config.scenario_weights.

    Returns
    -------
    outputs : dict with keys sources, counts, metrics, latest, forecast, sunspot_info
    """
    cfg = config or PipelineConfig()
    rng = np.random.default_rng(cfg.random_state)

    # ── Stage 1: NOAA live data ────────────────────────────────────────────────
    plasma_df, mag_df, kp_df, noaa_source = load_noaa_data(cfg)

    # ── Stage 2: OMNI training archive ────────────────────────────────────────
    # Load OMNI BEFORE building NOAA features — we need omni_kp_hist to source
    # kp_lag_27d for the live inference row (NOAA 7-day feed can't reach back 27d).
    omni_3h, omni_source = load_omni_data(cfg)

    # ── Stage 3: Live inference features ──────────────────────────────────────
    noaa_3h = build_noaa_3h_features(
        plasma_df, mag_df, kp_df,
        omni_kp_hist=omni_3h["Kp"],
    )

    # ── Stage 4: Model (train or load from cache) ──────────────────────────────
    model_meta = {
        "start_year":     cfg.omni_start_year,
        "num_years":      cfg.omni_num_years,
        "train_fraction": cfg.train_fraction,
    }
    model      = None
    metrics    = {}
    test_frame = pd.DataFrame()

    if not refit and _JOBLIB_AVAILABLE and MODEL_CACHE_PATH.exists() and MODEL_META_PATH.exists():
        try:
            with open(MODEL_META_PATH, encoding="utf-8") as _f:
                cached_meta = json.load(_f)
            if cached_meta == model_meta:
                model = _joblib.load(MODEL_CACHE_PATH)
                metrics_path = _REPO_DIR / "cache" / "model_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, encoding="utf-8") as _f:
                        metrics = json.load(_f)
                print("[cache] Model loaded from cache (--refit to retrain).")
        except Exception as cache_exc:
            print(f"[WARNING] Model cache load failed ({cache_exc}); retraining.", file=sys.stderr)
            model = None

    if model is None:
        model, metrics, test_frame = fit_and_evaluate_model(omni_3h, cfg)
        if _JOBLIB_AVAILABLE:
            try:
                MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _joblib.dump(model, MODEL_CACHE_PATH)
                with open(MODEL_META_PATH, "w", encoding="utf-8") as _f:
                    json.dump(model_meta, _f)
                metrics_path = _REPO_DIR / "cache" / "model_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as _f:
                    json.dump(metrics, _f)
                print(f"[cache] Model saved → {MODEL_CACHE_PATH.relative_to(_REPO_DIR)}")
            except Exception as save_exc:
                print(f"[WARNING] Model cache save failed: {save_exc}", file=sys.stderr)

    # ── Stage 5: Honest evaluation report ─────────────────────────────────────
    # Only runs when we just trained (test_frame is not empty).
    # When loading from cache, we skip this — the report was printed when the
    # model was originally trained and the metrics are in model_metrics.json.
    if not test_frame.empty and "kp_lag_3h" in omni_3h.columns:
        try:
            # Add the SUN/ root to sys.path so `eval.evaluate` is importable
            # regardless of which directory the user launched Python from.
            import importlib, pathlib as _pl
            _sun_root = str(_pl.Path(__file__).parent.parent)
            if _sun_root not in sys.path:
                sys.path.insert(0, _sun_root)
            from eval.evaluate import print_evaluation_report
            split_idx = int(cfg.train_fraction * len(omni_3h))
            split_idx = max(1, min(split_idx, len(omni_3h) - 1))
            kp_lag_test = omni_3h["kp_lag_3h"].values[split_idx:]
            print_evaluation_report(
                y_true=test_frame["Kp_true"].values,
                y_pred=test_frame["Kp_pred"].values,
                kp_lag_3h=kp_lag_test,
                model_metrics=metrics,
            )
        except ImportError:
            pass
        except Exception as eval_exc:
            print(f"[WARNING] Evaluation report failed: {eval_exc}", file=sys.stderr)

    # ── Latest real-time prediction ────────────────────────────────────────────
    # Prefer the live NOAA 3h frame; fall back to OMNI if NOAA fetch failed.
    if len(noaa_3h) >= 1 and all(c in noaa_3h.columns for c in FEATURE_COLUMNS):
        latest_features = noaa_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time     = noaa_3h.index[-1]
        latest_actual   = float(kp_df["Kp"].iloc[-1]) if not kp_df.empty else float(omni_3h["Kp"].iloc[-1])
    else:
        latest_features = omni_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time     = omni_3h.index[-1]
        latest_actual   = float(omni_3h["Kp"].iloc[-1])
    latest_pred = float(np.clip(model.predict(latest_features)[0], 0, 9))

    # ── Stage 6: Solar cycle weight modifier ──────────────────────────────────
    # Calls sunspot_pipeline as a sub-pipeline to get storm_rate_modifier and
    # adjusted scenario weights.  Graceful fallback to static weights if SIDC
    # is unreachable.
    effective_weights = dict(cfg.scenario_weights)
    sunspot_info: dict = {}
    if not static_weights:
        try:
            _sun_root = str(_REPO_DIR)
            if _sun_root not in sys.path:
                sys.path.insert(0, _sun_root)
            from sunspot_pipeline import run_sunspot_pipeline as _run_ssn
            ssn_results = _run_ssn(make_plots=False)
            gi = ssn_results.get("ggsp_integration", {})
            if gi and "recommended_scenario_weights" in gi:
                effective_weights = gi["recommended_scenario_weights"]
                cur = ssn_results.get("current", {})
                sunspot_info = {
                    "storm_rate_modifier": gi.get("storm_rate_modifier"),
                    "f107_proxy":          gi.get("f107_proxy"),
                    "ssn_normalized":      gi.get("ssn_normalized"),
                    "solar_activity_tier": cur.get("solar_activity_tier"),
                    "adjusted_weights":    effective_weights,
                    "source":              "sunspot_pipeline",
                }
                print(
                    f"[OK] Sunspot modifier={gi['storm_rate_modifier']:.3f}  "
                    f"→ weights: Q={effective_weights['Quiet']:.3f}  "
                    f"M={effective_weights['Moderate']:.3f}  "
                    f"A={effective_weights['Active']:.3f}"
                )
        except Exception as ssn_exc:
            print(
                f"[WARNING] Sunspot modifier unavailable ({ssn_exc}); using static weights.",
                file=sys.stderr,
            )
            sunspot_info = {"source": "static_fallback", "error": str(ssn_exc)}
    else:
        sunspot_info = {"source": "static_weights_flag"}

    # ── Stage 7: Forecast ──────────────────────────────────────────────────────
    forecast_seed = noaa_3h if len(noaa_3h) >= 4 else omni_3h
    forecast_times, scenarios, fc_seed_source = build_forecast_scenarios(
        forecast_seed, cfg, rng, omni_kp_hist=omni_3h["Kp"],
    )
    if len(noaa_3h) < 4:
        fc_seed_source = "omni_fallback"

    # Seed Kp lags from the most-recent real observations.
    if not kp_df.empty:
        kp_seed = kp_df["Kp"].dropna().tail(3).values
    else:
        kp_seed = omni_3h["Kp"].tail(3).values
    if len(kp_seed) < 3:
        kp_seed = np.pad(kp_seed, (3 - len(kp_seed), 0), constant_values=2.0)

    kp_forecasts = predict_scenario_kp(
        model, scenarios, kp_seed,
        forecast_times=forecast_times,
        omni_kp_hist=omni_3h["Kp"],
    )
    kp_weighted = weighted_ensemble(kp_forecasts, effective_weights)

    ensemble_result = predict_scenario_kp_ensemble(
        model, scenarios, kp_seed,
        weights=effective_weights,
        forecast_times=forecast_times,
        omni_kp_hist=omni_3h["Kp"],
        n_draws=500,
        rng=rng,
    )
    storm_prob_per_window = ensemble_result["storm_prob_per_window"]
    storm_chance_pct      = float(100.0 * ensemble_result["storm_prob_72h"])

    # ── Assemble outputs ───────────────────────────────────────────────────────
    outputs = {
        "sources": {"noaa": noaa_source, "omni": omni_source},
        "counts": {
            "plasma_rows":  len(plasma_df),
            "mag_rows":     len(mag_df),
            "kp_rows":      len(kp_df),
            "omni_3h_rows": len(omni_3h),
        },
        "metrics": metrics,
        "latest": {
            "time":         latest_time,
            "predicted_kp": latest_pred,
            "observed_kp":  latest_actual,
            "category":     kp_label(latest_pred),
        },
        "forecast": {
            "times":                     forecast_times,
            "kp_by_scenario":            kp_forecasts,
            "kp_weighted":               kp_weighted,
            "mean_weighted_kp":          float(np.mean(kp_weighted)),
            "peak_weighted_kp":          float(np.max(kp_weighted)),
            "storm_prob_per_window":     storm_prob_per_window,
            "storm_prob_72h_pct":        storm_chance_pct,
            "storm_chance_percent":      storm_chance_pct,   # legacy key for React
            "storm_probability_windows": float(np.mean(storm_prob_per_window)),
            "forecast_seed_source":      fc_seed_source,
        },
        "sunspot_info": sunspot_info,
    }

    # ── Stage 8: Optional plots ────────────────────────────────────────────────
    if make_plots:
        if not test_frame.empty:
            plot_test_results(test_frame, metrics)
        else:
            print("[INFO] Skipping held-out plot — model loaded from cache (--refit to regenerate).")
        plot_forecast(forecast_times, scenarios, kp_forecasts, kp_weighted)
        plt.show()

    # ── Stage 9: JSON export ───────────────────────────────────────────────────
    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_outputs(outputs), f, indent=2)

    return outputs


def _serialize_outputs(outputs: dict) -> dict:
    """Convert all numpy types / pandas timestamps to JSON-serialisable Python types.

    The React frontend reads this JSON, so every value must be a plain Python
    type (list, dict, float, str, None).  Numpy scalars and arrays are not
    JSON-serialisable by default — this function handles the conversion.
    """
    forecast = outputs["forecast"]

    def _to_python(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, float) and v != v:   # NaN check
            return None
        return v

    metrics_clean = {k: _to_python(v) for k, v in outputs["metrics"].items()}
    storm_prob_arr = forecast.get("storm_prob_per_window")

    return {
        "sources": outputs["sources"],
        "counts":  outputs["counts"],
        "metrics": metrics_clean,
        "latest": {
            **outputs["latest"],
            "time": outputs["latest"]["time"].isoformat(),
        },
        "forecast": {
            **{k: _to_python(v) for k, v in forecast.items()
               if k not in ("times", "kp_by_scenario", "kp_weighted", "storm_prob_per_window")},
            "times":                 [t.isoformat() for t in forecast["times"]],
            "kp_by_scenario":        {k: v.tolist() for k, v in forecast["kp_by_scenario"].items()},
            "kp_weighted":           forecast["kp_weighted"].tolist(),
            "storm_prob_per_window": storm_prob_arr.tolist() if storm_prob_arr is not None else [],
        },
        "sunspot_info": outputs.get("sunspot_info", {}),
    }

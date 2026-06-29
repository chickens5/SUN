"""
GGSP v8 upgrades — makes the 72-hour forecast meaningful and honest.

Layered on top of ggsp_pipeline_v7.py: the heavy, already-correct code
(data loading, feature engineering, model fitting) is reused untouched.
This module only adds/replaces the forecasting + evaluation + serialization
parts that needed to change.

What v8 adds over v7:
  1. evaluate_forecast_skill() — a real backtest of the 72h ensemble against
     persistence and climatology, per horizon bin. This is the credibility piece:
     until now there was no measurement of whether the 72h curve has any skill.
  2. fetch_noaa_kp_forecast() + compare_to_noaa_forecast() — pulls NOAA SWPC's
     official 3-day Kp forecast and reports agreement, so the app can overlay
     "our ensemble vs NOAA official" instead of presenting an unanchored curve.
  3. predict_scenario_kp_clipped() — caps the autoregressive Kp lags at the
     training 95th percentile so the model never extrapolates outside the regime
     it was trained on during a runaway high-Kp scenario.
  4. add_event_overlays() — injects discrete physical events (sudden commencement,
     CIR, switchback) into the Moderate/Active scenarios for realistic diversity
     instead of smooth Gaussian priors.
  5. serialize_outputs_v8() — emits forecast_metrics, noaa_forecast, and a plain
     disclaimer alongside the v7 contract, bumping schema_version to "v8".

Recommended config change (full solar cycle, not just the active tail):
    PipelineConfig(omni_start_year=2013, omni_num_years=13)

Entry point: run_pipeline_v8(config, json_output_path=...) returns
(raw_outputs, serialized_dict) and writes the JSON if a path is given.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

from ggsp_pipeline_v7 import (
    PipelineConfig,
    load_noaa_data,
    build_noaa_3h_features,
    load_omni_data,
    fit_and_evaluate_model,
    build_forecast_scenarios,
    weighted_ensemble,
    newell_coupling,
    kp_label,
    _serialize_outputs,
    _fetch_json,
    FEATURE_COLUMNS,
)

NOAA_KP_FORECAST_URL = (
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json"
)


# ---------------------------------------------------------------------------
# 1. Extrapolation-safe autoregressive prediction
# ---------------------------------------------------------------------------
def predict_scenario_kp_clipped(model, scenarios, kp_seed, kp_cap: float):
    """Autoregressive 72h Kp forecast per scenario, with the lag features clipped
    to [0, kp_cap]. kp_cap should be the training 95th-percentile Kp.

    During a strong scenario the loop feeds its own predictions back as lags; if a
    predicted Kp climbs to, say, 8.5, the next step would be asked to interpret a
    lag value that barely appears in training data. Clipping keeps every inference
    inside the regime the model actually learned, which removes runaway artifacts
    without flattening genuine storm growth (the wind-derived features still push
    Kp up; only the self-referential lag term is bounded).
    """
    seed = (
        list(kp_seed[-3:])
        if len(kp_seed) >= 3
        else [2.0] * (3 - len(kp_seed)) + list(kp_seed)
    )

    kp_forecasts = {}
    for name, scenario in scenarios.items():
        bz_arr = np.asarray(scenario["bz_gsm"], dtype=float)
        by_arr = np.asarray(scenario["by_gsm"], dtype=float)
        spd_arr = np.asarray(scenario["speed"], dtype=float)
        den_arr = np.asarray(scenario["density"], dtype=float)
        bt_arr = np.asarray(scenario["bt"], dtype=float)

        bz_min_arr = np.array(
            [bz_arr[max(0, i - 2):i + 1].min() for i in range(len(bz_arr))]
        )
        coupling = newell_coupling(spd_arr, bt_arr, by_arr, bz_arr)
        p_dyn_arr = den_arr * (spd_arr / 100.0) ** 2

        steps = len(bz_arr)
        kp_pred_arr = np.zeros(steps)
        kp_history = list(seed)
        for i in range(steps):
            lag_3h = min(kp_history[-1], kp_cap)
            lag_6h = min(kp_history[-2], kp_cap)
            lag_9h = min(kp_history[-3], kp_cap)

            x = np.array([[
                spd_arr[i], den_arr[i],
                by_arr[i], bz_arr[i], bz_min_arr[i], bt_arr[i],
                coupling[i], p_dyn_arr[i],
                lag_3h, lag_6h, lag_9h,
            ]])
            kp_step = float(np.clip(model.predict(x)[0], 0, 9))
            kp_pred_arr[i] = kp_step
            kp_history.append(kp_step)

        kp_forecasts[name] = kp_pred_arr
    return kp_forecasts


# ---------------------------------------------------------------------------
# 2. Discrete-event overlays for scenario realism
# ---------------------------------------------------------------------------
def add_event_overlays(scenarios, rng, steps: int) -> None:
    """Mutate Moderate/Active scenario arrays in place to include discrete solar-wind
    events. Real wind is not a smooth Gaussian walk; these add the sharp structures
    the priors miss. predict_scenario_kp_* recomputes coupling/p_dyn/bz_min from the
    base arrays, so editing speed/density/Bz/Bt/By here is enough to propagate.
    """
    for name in ("Moderate", "Active"):
        sc = scenarios[name]
        speed = np.asarray(sc["speed"], dtype=float)
        density = np.asarray(sc["density"], dtype=float)
        bz = np.asarray(sc["bz_gsm"], dtype=float)
        bt = np.asarray(sc["bt"], dtype=float)

        active = name == "Active"

        # Sudden commencement: brief density pulse + speed jump (shock arrival).
        if rng.random() < (0.25 if active else 0.15):
            t = int(rng.integers(2, max(3, steps - 2)))
            density[t:t + 2] = np.clip(density[t:t + 2] * 3.0, 0.5, 60)
            speed[t:t + 2] = np.clip(speed[t:t + 2] + 120, 250, 900)

        # Co-rotating interaction region: ~30h of elevated speed + density.
        if rng.random() < (0.40 if active else 0.25):
            t = int(rng.integers(2, max(3, steps - 10)))
            speed[t:t + 10] = np.clip(speed[t:t + 10] * 1.4, 250, 900)
            density[t:t + 10] = np.clip(density[t:t + 10] * 1.8, 0.5, 60)

        # Magnetic switchback: sharp southward Bz spike over one step.
        if rng.random() < 0.20:
            t = int(rng.integers(1, max(2, steps - 1)))
            bz[t] = -abs(bz[t]) - rng.uniform(2.0, 6.0)

        # Keep |B| physically consistent with the deepened Bz.
        bt = np.maximum(bt, np.abs(bz) + 1.0)

        sc["speed"], sc["density"], sc["bz_gsm"], sc["bt"] = speed, density, bz, bt


def build_forecast_scenarios_v8(seed_3h, cfg, rng, enable_events: bool = True):
    """v7 scenario builder + optional discrete-event overlays."""
    forecast_times, scenarios = build_forecast_scenarios(seed_3h, cfg, rng)
    if enable_events:
        add_event_overlays(scenarios, rng, cfg.forecast_steps_3h)
    return forecast_times, scenarios


# ---------------------------------------------------------------------------
# 3. Real forecast-skill backtest
# ---------------------------------------------------------------------------
def evaluate_forecast_skill(omni_3h, model, cfg, rng, n_origins: int = 150):
    """Backtest the autoregressive ensemble forecast against persistence and
    climatology, reported per horizon bin (0-24h, 24-48h, 48-72h).

    Method: sample n_origins times from history; from each origin run the exact
    production forecast path (base scenarios, weighted ensemble, lag-clipped) and
    compare the predicted Kp at each 3h step to what was actually observed. Two
    standard references: persistence (Kp stays at its origin value) and climatology
    (Kp = training mean). Beating climatology is easy; beating persistence at
    48-72h is the real test — and an honest result if it doesn't.

    Events are intentionally OFF here: they are a presentation-layer realism overlay
    on the band, not a point-skill improvement, and their randomness would only add
    noise to the estimate. Returns a JSON-serializable dict (or None if too little
    data). Cost scales with n_origins (~150 keeps it well under a minute).
    """
    kp = omni_3h["Kp"].to_numpy(dtype=float)
    n = len(kp)
    horizon = int(cfg.forecast_steps_3h)
    lo, hi = 9, n - horizon - 1
    if hi <= lo:
        return None

    clim = float(np.mean(kp))
    kp_cap = float(np.quantile(kp, 0.95))

    count = min(n_origins, hi - lo)
    origins = rng.choice(np.arange(lo, hi), size=count, replace=False)

    bins = ("0-24h", "24-48h", "48-72h")
    ens = {b: [] for b in bins}
    per = {b: [] for b in bins}
    clm = {b: [] for b in bins}

    def bin_for(step: int) -> str:
        return bins[0] if step <= 8 else (bins[1] if step <= 16 else bins[2])

    for o in origins:
        seed_3h = omni_3h.iloc[o - 9:o + 1]
        try:
            _, scenarios = build_forecast_scenarios(seed_3h, cfg, rng)
        except Exception:
            continue
        kp_seed = kp[o - 2:o + 1]
        kpf = predict_scenario_kp_clipped(model, scenarios, kp_seed, kp_cap)
        kw = weighted_ensemble(kpf, cfg.scenario_weights)
        persist = kp[o]
        for step in range(1, horizon + 1):
            obs = kp[o + step]
            b = bin_for(step)
            ens[b].append(abs(kw[step - 1] - obs))
            per[b].append(abs(persist - obs))
            clm[b].append(abs(clim - obs))

    def mae(d):
        return {b: (float(np.mean(v)) if v else None) for b, v in d.items()}

    all_ens = float(np.mean([e for v in ens.values() for e in v]))
    all_per = float(np.mean([e for v in per.values() for e in v]))
    all_clm = float(np.mean([e for v in clm.values() for e in v]))

    ens_mae, per_mae, clm_mae = mae(ens), mae(per), mae(clm)
    horizons = {
        b: {
            "ensemble_mae": ens_mae[b],
            "persistence_mae": per_mae[b],
            "climatology_mae": clm_mae[b],
        }
        for b in bins
    }

    return {
        "method": "sampled autoregressive backtest",
        "n_origins": int(count),
        "kp_cap": kp_cap,
        "horizons": horizons,
        "overall": {
            "ensemble_mae": all_ens,
            "persistence_mae": all_per,
            "climatology_mae": all_clm,
            "skill_vs_climatology_pct": float(100.0 * (1.0 - all_ens / all_clm)) if all_clm > 0 else 0.0,
            "skill_vs_persistence_pct": float(100.0 * (1.0 - all_ens / all_per)) if all_per > 0 else 0.0,
        },
    }


# ---------------------------------------------------------------------------
# 4. NOAA official 3-day Kp forecast ingestion + comparison
# ---------------------------------------------------------------------------
def _parse_noaa_kp_forecast(raw) -> pd.Series | None:
    """Parse the SWPC forecast payload (list of dicts) into a tz-aware Kp Series,
    keeping only forward-looking 'predicted' rows. time_tag is UTC-naive in the
    feed, so it is localized to UTC explicitly.
    """
    if not isinstance(raw, list) or not raw:
        return None
    rows = [
        r for r in raw
        if isinstance(r, dict) and r.get("observed") == "predicted"
    ]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True, errors="coerce")
    df["kp"] = pd.to_numeric(df["kp"], errors="coerce")
    df = df.dropna(subset=["time_tag", "kp"]).set_index("time_tag").sort_index()
    if df.empty:
        return None
    return df["kp"]


def fetch_noaa_kp_forecast(timeout: int = 20) -> pd.Series | None:
    """Return NOAA SWPC's official 3-day Kp forecast as a tz-aware Series, or None
    if the feed is unreachable/empty. Never raises — the comparison is optional.
    """
    try:
        raw = _fetch_json(NOAA_KP_FORECAST_URL, timeout)
    except Exception:
        return None
    return _parse_noaa_kp_forecast(raw)


def compare_to_noaa_forecast(forecast_times, kp_weighted, noaa_kp: pd.Series | None):
    """Align our weighted ensemble to NOAA's official forecast on our 3h grid and
    report agreement. noaa_kp_aligned is index-aligned with forecast_times (null
    where NOAA has no value within tolerance) so the app can overlay both curves.
    """
    if noaa_kp is None or len(noaa_kp) == 0:
        return None
    idx = pd.DatetimeIndex(forecast_times)
    ours = pd.Series(np.asarray(kp_weighted, dtype=float), index=idx)
    aligned = noaa_kp.reindex(idx, method="nearest", tolerance=pd.Timedelta("90min"))
    mask = aligned.notna()
    if int(mask.sum()) == 0:
        return None
    agreement = float(np.mean(np.abs(ours[mask].to_numpy() - aligned[mask].to_numpy())))
    return {
        "overlap_points": int(mask.sum()),
        "agreement_mae": agreement,
        "noaa_kp_aligned": [
            (float(v) if pd.notna(v) else None) for v in aligned.to_numpy()
        ],
    }


# ---------------------------------------------------------------------------
# 5. Extended serialization
# ---------------------------------------------------------------------------
def serialize_outputs_v8(outputs, *, forecast_skill, noaa_comparison, disclaimer):
    base = _serialize_outputs(outputs)
    base["schema_version"] = "v8"
    base["disclaimer"] = disclaimer
    base["forecast_metrics"] = forecast_skill        # may be None
    base["noaa_forecast"] = noaa_comparison          # may be None
    return base


DEFAULT_DISCLAIMER = (
    "Current Kp is a ~30-minute-ahead nowcast from live DSCOVR solar wind. "
    "The 72-hour curve is a model ensemble over plausible solar-wind regimes, not a "
    "deterministic forecast. See forecast_metrics for measured skill versus persistence "
    "and climatology, and noaa_forecast for agreement with NOAA's official 3-day Kp forecast."
)


# ---------------------------------------------------------------------------
# 6. Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline_v8(
    config: PipelineConfig | None = None,
    make_plots: bool = False,
    json_output_path: str | None = None,
    skill_origins: int = 150,
    enable_events: bool = True,
):
    """End-to-end v8 run. Reuses v7 data loading + model fitting, then applies the
    v8 forecast/skill/comparison/serialization. Returns (raw_outputs, serialized).
    """
    cfg = config or PipelineConfig()
    rng = np.random.default_rng(cfg.random_state)

    plasma_df, mag_df, kp_df, noaa_source = load_noaa_data(cfg)
    noaa_3h = build_noaa_3h_features(plasma_df, mag_df, kp_df)

    omni_3h, omni_source = load_omni_data(cfg)
    model, metrics, _ = fit_and_evaluate_model(omni_3h, cfg)

    kp_cap = float(np.quantile(omni_3h["Kp"], 0.95))

    # Latest nowcast (unchanged from v7).
    if len(noaa_3h) >= 1 and all(c in noaa_3h.columns for c in FEATURE_COLUMNS):
        latest_features = noaa_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time = noaa_3h.index[-1]
        latest_actual = float(kp_df["Kp"].iloc[-1]) if not kp_df.empty else float(omni_3h["Kp"].iloc[-1])
    else:
        latest_features = omni_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time = omni_3h.index[-1]
        latest_actual = float(omni_3h["Kp"].iloc[-1])
    latest_pred = float(np.clip(model.predict(latest_features)[0], 0, 9))

    # 72h ensemble forecast (events + lag clipping).
    forecast_seed = noaa_3h if len(noaa_3h) >= 4 else omni_3h
    forecast_times, scenarios = build_forecast_scenarios_v8(
        forecast_seed, cfg, rng, enable_events=enable_events
    )

    if not kp_df.empty:
        kp_seed = kp_df["Kp"].dropna().tail(3).values
    else:
        kp_seed = omni_3h["Kp"].tail(3).values
    if len(kp_seed) < 3:
        kp_seed = np.pad(kp_seed, (3 - len(kp_seed), 0), constant_values=2.0)

    kp_forecasts = predict_scenario_kp_clipped(model, scenarios, kp_seed, kp_cap)
    kp_weighted = weighted_ensemble(kp_forecasts, cfg.scenario_weights)
    storm_prob = float(np.mean(kp_weighted >= 5.0))

    # NEW: skill backtest + NOAA official-forecast comparison.
    forecast_skill = evaluate_forecast_skill(omni_3h, model, cfg, rng, n_origins=skill_origins)
    noaa_kp = fetch_noaa_kp_forecast(cfg.noaa_timeout_s)
    noaa_comparison = compare_to_noaa_forecast(forecast_times, kp_weighted, noaa_kp)

    outputs = {
        "sources": {"noaa": noaa_source, "omni": omni_source},
        "counts": {
            "plasma_rows": len(plasma_df),
            "mag_rows": len(mag_df),
            "kp_rows": len(kp_df),
            "omni_3h_rows": len(omni_3h),
        },
        "metrics": metrics,
        "latest": {
            "time": latest_time,
            "predicted_kp": latest_pred,
            "observed_kp": latest_actual,
            "category": kp_label(latest_pred),
        },
        "forecast": {
            "times": list(forecast_times),
            "kp_by_scenario": kp_forecasts,
            "kp_weighted": kp_weighted,
            "mean_weighted_kp": float(np.mean(kp_weighted)),
            "peak_weighted_kp": float(np.max(kp_weighted)),
            "storm_probability_windows": storm_prob,
            "storm_chance_percent": float(100.0 * storm_prob),
            "forecast_seed_source": "noaa_recent" if len(noaa_3h) >= 4 else "omni_fallback",
        },
    }

    serialized = serialize_outputs_v8(
        outputs,
        forecast_skill=forecast_skill,
        noaa_comparison=noaa_comparison,
        disclaimer=DEFAULT_DISCLAIMER,
    )

    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)

    return outputs, serialized

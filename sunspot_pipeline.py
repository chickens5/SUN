# Welcome to the Sunspot Activity Prediction Pipeline!
#


## Quick test
#python sunspot_pipeline.py --no-plots

# Full output for React
#python sunspot_pipeline.py --no-plots --json-out sunspot_output.json

# Longer training window (back to SC14)
#python sunspot_pipeline.py --start-year 1902 --json-out sunspot_output.json

# This module fetches the International Sunspot Number (SSN v2.0) from SIDC Brussels,
# trains a gradient-boosted regression model to forecast monthly SSN up to 12 months ahead,
# and derives solar-cycle-phase features for integration into GGSP-7.0.
#
# ─── What do Sunspots have to do with the magnetosphere in regards to our GGSP? ────────────────────────────────────────────────────
# The solar cycle modulates the *background rate* of geomagnetic storms. Right now, we are heading towards a solar minimum.

# During the solar maximum (SSN ~ 150+)--> we have CMEs, flares, and co-rotating interaction regions (CIRs) that occur
# roughly 5× more often than the solar minimum (SSN < 30). 

# GGSP's scenario weights (Quiet 20% / Moderate 50% / Active 30%) were calibrated without solar-cycle awareness.
# Integrating a solar activity modifier shifts those weights dynamically — during SC25
# peak, the Active scenario deserves more probability mass; at solar minimum, Quiet should dominate.
#
# F10.7 (10.7 cm radio flux) has a near-linear relationship with SSN:
#   F10.7 ≈ 67 + 0.32 × SSN_smooth    (valid across SC17–SC25)
# F10.7 is the operational solar activity index used by space agencies for satellite
# drag prediction and ionospheric modelling — it's the standard "solar background" proxy.
#
# ─── DATA SOURCE ────────────────────────────────────────────────────────────────
# SIDC (Solar Influences Data Center), Royal Observatory of Belgium, Brussels.
# International Sunspot Number v2.0 — globally accepted standard since 2015 recalibration.
# URL: https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt
# Format per row:  YYYY  MM  YYYY.YYY  SSN  StdDev  Nobs  Provisional(0=final/1=provisional)
#
# ─── RUN ────────────────────────────────────────────────────────────────────────
#   python sunspot_pipeline.py                          # plots + console summary
#   python sunspot_pipeline.py --no-plots --json-out sunspot_output.json
#   python sunspot_pipeline.py --start-year 1900 --forecast-months 18
# ────────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HEADERS = {"User-Agent": "Mozilla/5.0 (sunspot-predictor; educational)"}
SIDC_MONTHLY_URL = "https://www.sidc.be/silso/DATA/SN_m_tot_V2.0.txt"

# Reference epoch for computing absolute month index (used in Fourier features).
# Using 1749-01 (start of SSN records) ensures the Fourier phase is absolute and
# consistent across all training runs — a model trained on 1950-2000 and applied in
# 2026 will still compute the correct heliomagnetic phase angle for today.
_EPOCH = pd.Timestamp("1749-01-15", tz="UTC")

# Feature column names — must stay in sync with build_features().
# Any caller constructing X arrays must match this ordered list exactly.
FEATURE_COLUMNS_SSN = [
    "ssn_lag_1m",   # 1-month lag  — short-term momentum
    "ssn_lag_2m",   # 2-month lag
    "ssn_lag_3m",   # 3-month lag
    "ssn_lag_6m",   # 6-month lag  — sub-annual context
    "ssn_lag_12m",  # 12-month lag — year-over-year cycle position
    "ssn_lag_24m",  # 24-month lag — 2-year memory (rise / decline phase)
    "ssn_lag_36m",  # 36-month lag — cycle phase anchor (3 years back)
    "ssn_roll3m",   # 3-month rolling mean  — noise-smoothed near-term
    "ssn_roll12m",  # 12-month rolling mean — proxy for official smoothed SSN
    "sin_11yr",     # Fourier fundamental (11-year cycle)
    "cos_11yr",     # Fourier fundamental (quadrature)
    "sin_5p5yr",    # Fourier 2nd harmonic (5.5-year half-cycle)
    "cos_5p5yr",    # Fourier 2nd harmonic (quadrature)
]


@dataclass(frozen=True)
class SunspotConfig:

    train_start_year: int = 1950
    # 1950 covers Solar Cycles 18–25, giving the model 6+ complete cycle examples.
    # The SSN series starts in 1749 but pre-1950 data has larger uncertainties and
    # fewer contributing observatories. SC18 (peak ~1947) is the oldest well-observed
    # cycle we include.

    forecast_months: int = 12
    # 12-month horizon is useful as a GGSP background signal. Beyond ~6 months,
    # solar cycle prediction is probabilistic due to the dynamo's intrinsic nonlinearity.
    # The pipeline outputs point estimates but they should be treated as central-tendency
    # forecasts — uncertainty grows roughly linearly with horizon.

    train_fraction: float = 0.85
    # With ~900 monthly samples post-1950, 0.85 holds out ~135 months (~SC24 descending
    # phase + SC25 rising) as the chronological test set — a meaningful real-world
    # validation period that covers a full weak cycle and the start of a strong one.

    n_estimators: int = 300
    # More estimators than GGSP (200) because the monthly SSN series is smoother and the
    # per-tree contribution is smaller. The lower learning rate (0.03 vs 0.05) compensates.

    max_depth: int = 3
    # Monthly SSN is a smoother, lower-noise signal than 3-hourly Kp. Depth 3 is
    # expressive enough to model amplitude variation across cycles without overfitting
    # on the ~900-sample training set.

    learning_rate: float = 0.03
    # Slower than GGSP (0.05) because we have fewer training samples and need the model
    # to generalise across cycles with very different amplitudes — SC19 (peak SSN ~190)
    # vs SC24 (peak SSN ~82). Slow learning helps the model find cycle-invariant patterns.

    subsample: float = 0.8
    # Stochastic subsampling prevents overfitting and is consistent with GGSP config.

    min_samples_leaf: int = 5

    random_state: int = 42

    timeout_s: int = 20

    solar_cycle_period_months: int = 132
    # 11 years × 12 months. The actual cycle length varies between ~9 and ~14 years
    # (mean ≈ 11.0 yr, median ≈ 10.7 yr). 132 months is used for the Fourier features;
    # the model learns amplitude variation on top of this fixed-period template.


# ─── DATA LOADING ───────────────────────────────────────────────────────────────

def _fetch_text(url: str, timeout: int, retries: int = 3) -> str:
    """Fetch plain text with retry/backoff — mirrors _fetch_json() in ggsp_pipeline_v7."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(0.5 * attempt)
    raise RuntimeError(f"Fetch failed after {retries} attempts for {url}: {last_error}")


def _parse_sidc_monthly(text: str) -> pd.DataFrame:
    """Parse SIDC SN_m_tot_V2.0.txt into a time-indexed DataFrame.

    SIDC format:  YYYY  MM  YYYY.YYY  SSN  StdDev  Nobs  Provisional
    SSN = -1.0 means the month is not yet published — those rows are dropped.
    Provisional rows (flag=1) are included; SIDC revises them rarely and they
    are the only source of the most recent 1-2 months.
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year  = int(parts[0])
            month = int(parts[1])
            ssn   = float(parts[3])
            if ssn < 0:
                continue   # not yet published
            rows.append({
                "time_tag": pd.Timestamp(year=year, month=month, day=15, tz="UTC"),
                "ssn": ssn,
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        raise RuntimeError("SIDC monthly file parsed to zero valid rows")
    df = pd.DataFrame(rows).set_index("time_tag").sort_index()
    return df


def load_sidc_data(config: SunspotConfig) -> Tuple[pd.DataFrame, str]:
    """Download the monthly SSN series from SIDC Brussels and clip to training window."""
    text = _fetch_text(SIDC_MONTHLY_URL, config.timeout_s)
    df = _parse_sidc_monthly(text)
    if len(df) < 120:
        raise RuntimeError(f"SIDC returned only {len(df)} rows — expected ≥120")
    start = pd.Timestamp(year=config.train_start_year, month=1, day=1, tz="UTC")
    df = df[df.index >= start]
    return df, "live_sidc"


# ─── FEATURE ENGINEERING ────────────────────────────────────────────────────────

def _month_index(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Convert timestamps to integer months since the fixed 1749-01 epoch.

    An absolute epoch means the Fourier features encode the true heliomagnetic
    phase regardless of which training window was used. If the epoch were relative
    to training start, applying the model in a future year would produce a shifted
    phase angle and degrade forecast accuracy.
    """
    delta = (
        (timestamps.year  - _EPOCH.year)  * 12
      + (timestamps.month - _EPOCH.month)
    )
    return np.asarray(delta, dtype=float)


def build_features(ssn_series: pd.Series, config: SunspotConfig) -> pd.DataFrame:
    """Build the ML feature matrix from a monthly SSN series.

    Feature groups
    ──────────────────────────────────────────────────────────────────────────
    Autoregressive lags (1-3 m)
        Short-term momentum. A rising sunspot count this month strongly
        predicts a continued rise next month — cycle ramps unfold over years.

    Medium lags (6, 12 m)
        Sub-annual and annual context. Lag-12 is particularly informative
        because cycles have partial year-on-year coherence during rise/decline.

    Long lags (24, 36 m)
        Cycle-phase positioning. If SSN was near zero 36 months ago and is
        now at 80, we are ~3 years into a new cycle. The model learns this.

    Rolling means (3 m, 12 m)
        Noise-smoothed activity level. The 12-month mean closely approximates
        the official 13-month smoothed SSN used to define cycle maxima/minima,
        giving the model an implicit phase estimate without hand-crafting one.

    Fourier features (11 yr + 5.5 yr harmonics)
        Encode the quasi-periodic oscillation explicitly. Without these, the
        GBR would need to discover the ~132-month period purely from lags,
        requiring very deep trees. The Fourier terms give the cycle shape for
        free and let the model focus on amplitude variation across cycles.
    ──────────────────────────────────────────────────────────────────────────
    """
    df = pd.DataFrame({"ssn": ssn_series.copy()})

    # ── Autoregressive and long-period lags ───────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 36]:
        df[f"ssn_lag_{lag}m"] = df["ssn"].shift(lag)

    # ── Rolling smoothed estimates ─────────────────────────────────────────
    df["ssn_roll3m"]  = df["ssn"].rolling(3,  min_periods=1).mean()
    df["ssn_roll12m"] = df["ssn"].rolling(12, min_periods=6).mean()

    # ── Fourier features (absolute heliomagnetic phase) ────────────────────
    t = _month_index(df.index)
    P = config.solar_cycle_period_months
    df["sin_11yr"]  = np.sin(2 * np.pi * t / P)
    df["cos_11yr"]  = np.cos(2 * np.pi * t / P)
    df["sin_5p5yr"] = np.sin(4 * np.pi * t / P)   # 2nd harmonic
    df["cos_5p5yr"] = np.cos(4 * np.pi * t / P)

    return df.dropna()


# ─── MODEL TRAINING ─────────────────────────────────────────────────────────────

def fit_and_evaluate_model(
    ssn_series: pd.Series,
    config: SunspotConfig,
) -> Tuple[Pipeline, dict, pd.DataFrame]:
    """Train a 1-step-ahead SSN regressor with a chronological train/test split.

    Returns the fitted sklearn Pipeline, performance metrics dict, and a
    test-period DataFrame (ssn_true vs ssn_pred) for plotting and inspection.

    One-step-ahead framing is chosen for consistency with the GGSP Kp model.
    For the 12-month forecast we apply it recursively (see forecast_ssn()).
    """
    feature_df = build_features(ssn_series, config)

    # Target: SSN one month ahead. Shift back by 1 and align with feature index.
    target = ssn_series.shift(-1).reindex(feature_df.index).dropna()
    feature_df = feature_df.loc[target.index]

    if len(feature_df) < 48:
        raise ValueError("Not enough data — need at least 48 months after lag construction.")

    split_idx = int(config.train_fraction * len(feature_df))
    split_idx = max(12, min(split_idx, len(feature_df) - 12))

    X = feature_df[FEATURE_COLUMNS_SSN].values
    y = target.values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            min_samples_leaf=config.min_samples_leaf,
            random_state=config.random_state,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_test), 0, None)  # SSN is always ≥ 0

    baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "mae":          float(mean_absolute_error(y_test, y_pred)),
        "r2":           float(r2_score(y_test, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "test_months":  int(len(y_test)),
    }

    test_frame = pd.DataFrame(
        {"ssn_true": y_test, "ssn_pred": y_pred},
        index=feature_df.index[split_idx:],
    )
    return model, metrics, test_frame


# ─── RECURSIVE FORECAST ──────────────────────────────────────────────────────────

def forecast_ssn(
    model: Pipeline,
    ssn_series: pd.Series,
    config: SunspotConfig,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """Recursively forecast SSN for the next `forecast_months` months.

    Each predicted value immediately feeds back as the lag input for subsequent
    steps — the same autoregressive pattern used for Kp lags in ggsp_pipeline_v7.
    This correctly propagates the model's own uncertainty forward (a rising forecast
    at step 3 will cause elevated lag features at step 4, consistent with cycle physics).

    Beyond ~6 months, this is better treated as a central-tendency trajectory
    than a precise point estimate. Prediction intervals are not computed here but
    could be added via quantile GBR or bootstrap ensembles.
    """
    # Seed with the last 36 months of observed data to populate all lag features.
    history = list(ssn_series.dropna().tail(36).values)
    last_time = ssn_series.dropna().index[-1]

    # Build forecast times at mid-month (day=15) to match the training data convention.
    forecast_times = pd.DatetimeIndex([
        pd.Timestamp(
            year=(last_time + pd.DateOffset(months=i + 1)).year,
            month=(last_time + pd.DateOffset(months=i + 1)).month,
            day=15,
            tz="UTC",
        )
        for i in range(config.forecast_months)
    ])

    # Build a combined timeline so _month_index returns the correct absolute Fourier
    # phase for each future month without referencing training-relative offsets.
    obs_tail_times = ssn_series.dropna().index[-36:]
    all_times = pd.DatetimeIndex(list(obs_tail_times) + list(forecast_times))
    t_all = _month_index(all_times)

    P = config.solar_cycle_period_months
    predictions: list[float] = []

    for step in range(config.forecast_months):
        # Build the feature vector manually to match FEATURE_COLUMNS_SSN exactly.
        h = history
        x = np.array([[
            h[-1],            # ssn_lag_1m
            h[-2],            # ssn_lag_2m
            h[-3],            # ssn_lag_3m
            h[-6],            # ssn_lag_6m
            h[-12],           # ssn_lag_12m
            h[-24],           # ssn_lag_24m
            h[-36],           # ssn_lag_36m
            np.mean(h[-3:]),  # ssn_roll3m
            np.mean(h[-12:]), # ssn_roll12m
            # Fourier phase at the forecast month (index 36 = first forecast step)
            np.sin(2 * np.pi * t_all[36 + step] / P),   # sin_11yr
            np.cos(2 * np.pi * t_all[36 + step] / P),   # cos_11yr
            np.sin(4 * np.pi * t_all[36 + step] / P),   # sin_5p5yr
            np.cos(4 * np.pi * t_all[36 + step] / P),   # cos_5p5yr
        ]])
        pred = float(np.clip(model.predict(x)[0], 0, None))
        predictions.append(pred)
        history.append(pred)   # predicted value becomes next step's lag-1m

    return forecast_times, np.array(predictions)


# ─── SOLAR CYCLE PHASE ───────────────────────────────────────────────────────────

def compute_cycle_phase(
    ssn_series: pd.Series,
    ssn_forecast: np.ndarray,
    forecast_times: pd.DatetimeIndex,
) -> dict:
    """Derive solar cycle phase metrics and the GGSP integration outputs.

    Returns:
    ────────────────────────────────────────────────────────────────────────
    current_smooth_ssn      13-month centred moving average (SIDC definition).
                            Extended using forecast values for the last 6 months
                            where centred smoothing would otherwise be undefined.
    f107_proxy              Estimated F10.7 solar flux index (sfu).
                            F10.7 ≈ 67 + 0.32 × SSN_smooth is an empirical fit
                            valid across Solar Cycles 17-25.
    ssn_normalized          (current_smooth - cycle_min) / (cycle_max - cycle_min)
                            normalised within Solar Cycle 25 (started Dec 2019).
                            0 = cycle minimum, 1 = highest observed so far this cycle.
    solar_activity_tier     Human-readable activity label.
    storm_rate_modifier     [0.2, 2.0] float for GGSP scenario weight adjustment.
                            Derived from the empirical ~5× storm-rate ratio between
                            solar max and min (Borovsky & Shprits 2017;
                            Richardson et al. 2000).
    ────────────────────────────────────────────────────────────────────────
    """
    # Extend the observed series with the first 6 forecast months so the centred
    # 13-month smoother can produce a value for the most recent observed months.
    extended = pd.concat([
        ssn_series,
        pd.Series(ssn_forecast[:6], index=forecast_times[:6]),
    ]).sort_index()
    ssn_smooth = extended.rolling(13, center=True, min_periods=7).mean().dropna()
    current_smooth = float(ssn_smooth.iloc[-1])

    # F10.7 proxy — the standard operational solar background index.
    f107_proxy = float(67.0 + 0.32 * current_smooth)

    # Activity tier thresholds approximate the quartiles of SC19-SC25 smoothed SSN.
    if current_smooth < 30:
        tier = "Minimum"
    elif current_smooth < 80:
        tier = "Rising / Declining"
    elif current_smooth < 140:
        tier = "Moderate Maximum"
    else:
        tier = "High Maximum"

    # Cycle normalisation within SC25 (official start: December 2019, smoothed min ≈ 1.8).
    # Using observed data from 2019 onward so the normalisation reflects the actual cycle
    # amplitude rather than a pre-assumed peak value.
    sc25_start = pd.Timestamp("2019-01-01", tz="UTC")
    sc25_smooth = ssn_smooth[ssn_smooth.index >= sc25_start]
    cycle_min = float(sc25_smooth.min()) if not sc25_smooth.empty else 0.0
    cycle_max = float(sc25_smooth.max()) if not sc25_smooth.empty else max(current_smooth, 1.0)
    denom = max(cycle_max - cycle_min, 1.0)
    ssn_normalized = float(np.clip((current_smooth - cycle_min) / denom, 0.0, 1.0))

    # GGSP storm_rate_modifier: maps [solar_min → solar_max] to [0.4 → 1.6].
    # At cycle minimum (ssn_normalized=0): modifier=0.4 — Active scenario weight halved.
    # At cycle maximum (ssn_normalized=1): modifier=1.6 — Active scenario weight 60% higher.
    # The GGSP caller multiplies its base Active weight (0.30) by this factor, then
    # renormalises all three scenario weights to sum to 1.0 (see _compute_adjusted_weights).
    storm_rate_modifier = float(np.clip(0.4 + 1.2 * ssn_normalized, 0.2, 2.0))

    return {
        "current_smooth_ssn":     round(current_smooth, 1),
        "f107_proxy":             round(f107_proxy, 1),
        "ssn_normalized":         round(ssn_normalized, 3),
        "solar_activity_tier":    tier,
        "storm_rate_modifier":    round(storm_rate_modifier, 3),
        "cycle_min_ssn":          round(cycle_min, 1),
        "cycle_max_observed_ssn": round(cycle_max, 1),
    }


def _compute_adjusted_weights(modifier: float) -> Dict[str, float]:
    """Compute GGSP scenario weights adjusted for current solar activity level.

    Base weights (Quiet=0.20, Moderate=0.50, Active=0.30) were calibrated for
    average cycle conditions. This function scales the Active weight by modifier
    and redistributes the remainder between Quiet and Moderate in their original
    2:5 ratio, so all three weights still sum to 1.0.

    Examples:
      modifier=0.4  (solar min): Quiet≈0.25, Moderate≈0.63, Active≈0.12
      modifier=1.0  (average):   Quiet≈0.20, Moderate≈0.50, Active≈0.30
      modifier=1.6  (solar max): Quiet≈0.15, Moderate≈0.37, Active≈0.48
    """
    base_quiet    = 0.20
    base_moderate = 0.50
    base_active   = 0.30

    active_adj  = base_active * modifier
    remaining   = 1.0 - active_adj
    # Redistribute remainder preserving the original Quiet:Moderate ratio (2:5).
    quiet_ratio = base_quiet    / (base_quiet + base_moderate)   # 0.2/0.7
    mod_ratio   = base_moderate / (base_quiet + base_moderate)   # 0.5/0.7
    quiet_adj   = remaining * quiet_ratio
    mod_adj     = remaining * mod_ratio

    total = quiet_adj + mod_adj + active_adj   # should be ~1.0 before rounding
    return {
        "Quiet":    round(quiet_adj  / total, 3),
        "Moderate": round(mod_adj    / total, 3),
        "Active":   round(active_adj / total, 3),
    }


# ─── PLOTTING ────────────────────────────────────────────────────────────────────

def plot_sunspot_results(
    ssn_series: pd.Series,
    test_frame: pd.DataFrame,
    forecast_times: pd.DatetimeIndex,
    ssn_forecast: np.ndarray,
    metrics: dict,
    cycle_info: dict,
) -> None:
    """Three-panel figure: full history, held-out test fit, and 12-month forecast."""
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    # ── Panel 1: Full historical SSN ──────────────────────────────────────
    smooth_hist = ssn_series.rolling(13, center=True, min_periods=7).mean()
    axes[0].plot(ssn_series.index, ssn_series.values,
                 color="#aaaaaa", lw=0.7, label="Monthly SSN (raw)")
    axes[0].plot(smooth_hist.index, smooth_hist.values,
                 color="#cc4125", lw=1.5, label="13-month smooth (SIDC definition)")
    axes[0].set_title("International Sunspot Number v2.0 — Full Historical Record")
    axes[0].set_ylabel("SSN")
    axes[0].legend(loc="upper left", fontsize=8)

    # ── Panel 2: Test-period fit ──────────────────────────────────────────
    axes[1].plot(test_frame.index, test_frame["ssn_true"],
                 color="#333333", lw=1.2, label="Observed (1-month ahead)")
    axes[1].plot(test_frame.index, test_frame["ssn_pred"],
                 color="#3d85c6", lw=1.2, label="Predicted")
    axes[1].set_title(
        f"Held-out Validation | MAE={metrics['mae']:.1f} SSN  |  "
        f"R²={metrics['r2']:.3f}  |  baseline MAE={metrics['baseline_mae']:.1f}"
    )
    axes[1].set_ylabel("SSN")
    axes[1].legend(loc="upper left", fontsize=8)

    # ── Panel 3: 12-month forecast ────────────────────────────────────────
    tail = ssn_series.tail(30)
    axes[2].plot(tail.index, tail.values, color="#333333", lw=1.2, label="Recent observed")
    axes[2].plot(forecast_times, ssn_forecast,
                 color="#e69138", lw=2, marker="o", ms=4, label="12-month forecast")
    axes[2].axhline(
        cycle_info["current_smooth_ssn"],
        color="red", ls="--", lw=0.8, alpha=0.7,
        label=f"Current smooth SSN: {cycle_info['current_smooth_ssn']:.0f}",
    )
    axes[2].set_title(
        f"12-Month SSN Forecast  |  Tier: {cycle_info['solar_activity_tier']}  |  "
        f"F10.7≈{cycle_info['f107_proxy']} sfu  |  "
        f"GGSP storm modifier: {cycle_info['storm_rate_modifier']:.2f}"
    )
    axes[2].set_ylabel("SSN")
    axes[2].legend(loc="upper left", fontsize=8)

    plt.tight_layout()


# ─── SERIALISATION ───────────────────────────────────────────────────────────────

def _serialize_outputs(outputs: dict) -> dict:
    """Convert pipeline outputs to JSON-serializable types for React consumption."""
    return {
        "source":      outputs["source"],
        "data_months": outputs["data_months"],
        "metrics":     outputs["metrics"],
        "current": {
            **{k: v for k, v in outputs["current"].items() if k != "time"},
            "time": outputs["current"]["time"].isoformat(),
        },
        "cycle_phase": outputs["cycle_phase"],
        "forecast": {
            "times":         [t.isoformat() for t in outputs["forecast"]["times"]],
            "ssn_predicted": [round(float(v), 1) for v in outputs["forecast"]["ssn_predicted"]],
        },
        "ggsp_integration": outputs["ggsp_integration"],
    }


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────────

def run_sunspot_pipeline(
    config: SunspotConfig | None = None,
    make_plots: bool = True,
    json_output_path: str | None = None,
) -> dict:
    """Orchestrate: fetch → features → train → forecast → cycle phase → GGSP weights."""
    cfg = config or SunspotConfig()

    ssn_df, source = load_sidc_data(cfg)
    ssn_series = ssn_df["ssn"]

    model, metrics, test_frame = fit_and_evaluate_model(ssn_series, cfg)
    forecast_times, ssn_forecast = forecast_ssn(model, ssn_series, cfg)
    cycle_info = compute_cycle_phase(ssn_series, ssn_forecast, forecast_times)

    current_ssn  = float(ssn_series.iloc[-1])
    current_time = ssn_series.index[-1]

    outputs = {
        "source":      source,
        "data_months": int(len(ssn_series)),
        "metrics":     metrics,
        "current": {
            "time":                current_time,
            "ssn_monthly":         round(current_ssn, 1),
            "ssn_smoothed":        cycle_info["current_smooth_ssn"],
            "f107_proxy":          cycle_info["f107_proxy"],
            "solar_activity_tier": cycle_info["solar_activity_tier"],
        },
        "cycle_phase": {
            "ssn_normalized":         cycle_info["ssn_normalized"],
            "cycle_min_ssn":          cycle_info["cycle_min_ssn"],
            "cycle_max_observed_ssn": cycle_info["cycle_max_observed_ssn"],
        },
        "forecast": {
            "times":         forecast_times,
            "ssn_predicted": ssn_forecast,
        },
        # ggsp_integration is the key output for GGSP-7.0 integration.
        # Pass storm_rate_modifier into PipelineConfig.scenario_weights at runtime,
        # or use recommended_scenario_weights directly to override the defaults.
        "ggsp_integration": {
            "storm_rate_modifier": cycle_info["storm_rate_modifier"],
            "f107_proxy":          cycle_info["f107_proxy"],
            "ssn_normalized":      cycle_info["ssn_normalized"],
            "recommended_scenario_weights": _compute_adjusted_weights(
                cycle_info["storm_rate_modifier"]
            ),
        },
    }

    if make_plots:
        plot_sunspot_results(
            ssn_series, test_frame, forecast_times, ssn_forecast, metrics, cycle_info
        )
        plt.show()

    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_outputs(outputs), f, indent=2)

    return outputs


# ─── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sunspot Activity Prediction Pipeline")
    parser.add_argument("--no-plots",        action="store_true",    help="Skip matplotlib plots")
    parser.add_argument("--start-year",      type=int, default=1950, help="Training data start year (default: 1950)")
    parser.add_argument("--forecast-months", type=int, default=12,   help="Months to forecast ahead (default: 12)")
    parser.add_argument("--json-out",        type=str, default=None, metavar="PATH",
                        help="Write results to a JSON file (e.g. sunspot_output.json)")
    args = parser.parse_args()

    config = SunspotConfig(
        train_start_year=args.start_year,
        forecast_months=args.forecast_months,
    )
    results = run_sunspot_pipeline(
        config=config,
        make_plots=not args.no_plots,
        json_output_path=args.json_out,
    )

    print("=== Sunspot Pipeline Summary ===")
    print(f"Source: {results['source']} | Training months: {results['data_months']}")
    m = results["metrics"]
    print(f"MAE={m['mae']:.1f} SSN  |  R²={m['r2']:.3f}  |  baseline MAE={m['baseline_mae']:.1f}  |  test months={m['test_months']}")
    cur = results["current"]
    print(
        f"Latest ({cur['time'].strftime('%Y-%m')}):  "
        f"SSN={cur['ssn_monthly']}  smoothed={cur['ssn_smoothed']}  "
        f"F10.7≈{cur['f107_proxy']} sfu  tier={cur['solar_activity_tier']}"
    )
    gi = results["ggsp_integration"]
    w  = gi["recommended_scenario_weights"]
    print(
        f"GGSP modifier={gi['storm_rate_modifier']:.3f}  |  "
        f"Adjusted weights → Quiet={w['Quiet']}  Moderate={w['Moderate']}  Active={w['Active']}"
    )
    fcast = results["forecast"]
    peak_idx = int(np.argmax(fcast["ssn_predicted"]))
    print(
        f"12-month forecast peak: SSN={fcast['ssn_predicted'][peak_idx]:.0f}  "
        f"at {fcast['times'][peak_idx].strftime('%Y-%m')}"
    )


if __name__ == "__main__":
    main()

# ggsp/features.py
#
# STAGE 3a — Aggregate live NOAA 1-minute feeds into the 14-feature 3h frame.
#
# This module is the bridge between raw NOAA sensor data and the format
# the trained model expects.  It needs to produce EXACTLY the same feature
# columns, in the SAME order, computed with the SAME formulas as omni_client.py
# does for the training data — otherwise the model will receive inputs it was
# never trained on and predictions will be garbage.
#
# Think of FEATURE_COLUMNS (defined in config.py) as a typed contract:
#   omni_client.py  → builds the training table that satisfies this contract
#   features.py     → builds the live inference row that satisfies this contract
#   forecast.py     → builds the ensemble batch that satisfies this contract
#   model.py        → trains on and predicts from data satisfying this contract
#
# If you ever add a feature to FEATURE_COLUMNS, you must update ALL FOUR files.
# The _EXPECTED_N_FEATURES assertion in model.py and forecast.py will catch
# any mismatch at runtime before it silently corrupts a prediction.
#
# Used by:  pipeline.py (Stage 3 of run_pipeline)
# Imports:  config.py, physics.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .physics import newell_coupling


def build_noaa_3h_features(
    plasma_df: pd.DataFrame,
    mag_df: pd.DataFrame,
    kp_df: pd.DataFrame | None = None,
    omni_kp_hist: "pd.Series | None" = None,
) -> pd.DataFrame:
    """STAGE 3a — Aggregate NOAA 1-min feeds into the 14-feature 3h frame.

    Parameters
    ----------
    plasma_df     : 1-min plasma data (speed, density) from noaa_client
    mag_df        : 1-min IMF data (by_gsm, bz_gsm, bt) from noaa_client
    kp_df         : 3-h Kp observations — used to build kp_lag_3h/6h/9h.
                    Pass None if not available (lags default to 2.0).
    omni_kp_hist  : OMNI Kp time series (pd.Series) — used to source kp_lag_27d.
                    The 7-day NOAA feed cannot reach back 27 days on its own.
                    Pass None to fall back to 2.0 (quiet-day default).

    Returns
    -------
    DataFrame with exactly the columns in FEATURE_COLUMNS.
    Empty DataFrame with those columns if there is no usable data.
    """
    # Inner join on time so we only keep rows where both instruments reported.
    merged = plasma_df.join(mag_df, how="inner")
    merged = merged[["speed", "density", "by_gsm", "bz_gsm", "bt"]].dropna()
    if merged.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Resample the 1-min data to 3-hour windows.
    # bz: both mean and min — min captures brief southward spikes within the window.
    noaa_3h = merged.resample("3h", label="left").agg({
        "speed":   "mean",
        "density": "mean",
        "by_gsm":  "mean",
        "bz_gsm":  ["mean", "min"],
        "bt":      "mean",
    })
    noaa_3h.columns = [
        "speed_mean", "density_mean", "by_mean",
        "bz_mean", "bz_min", "bt_mean",
    ]
    noaa_3h = noaa_3h.dropna()
    if noaa_3h.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Newell coupling using real By_GSM (same formula as omni_client.py).
    # Using the real By here is what fixed the systematically wrong clock angle
    # that the old constant-2.0-nT proxy introduced.
    noaa_3h["coupling_mean"] = newell_coupling(
        noaa_3h["speed_mean"], noaa_3h["bt_mean"],
        noaa_3h["by_mean"],    noaa_3h["bz_mean"],
    )

    # Dynamic ram pressure — same normalisation as omni_client.py.
    noaa_3h["p_dyn_mean"] = (
        noaa_3h["density_mean"] * (noaa_3h["speed_mean"] / 100.0) ** 2
    )

    # ── Kp lag features ────────────────────────────────────────────────────────
    # Kp is observed every 3 hours.  We resample to the same 3h grid, forward-fill
    # brief gaps, then shift to build the lookback features.
    # If no Kp data is available at all, we fall back to 2.0 (typical quiet Kp)
    # so inference still runs — the model handles quiet-day well.
    if kp_df is not None and not kp_df.empty:
        kp_3h = kp_df["Kp"].resample("3h", label="left").mean()
        noaa_3h = noaa_3h.join(kp_3h, how="left")
        noaa_3h["Kp"] = noaa_3h["Kp"].ffill()
        noaa_3h["kp_lag_3h"] = noaa_3h["Kp"].shift(1).ffill().fillna(2.0)
        noaa_3h["kp_lag_6h"] = noaa_3h["Kp"].shift(2).ffill().fillna(2.0)
        noaa_3h["kp_lag_9h"] = noaa_3h["Kp"].shift(3).ffill().fillna(2.0)
        noaa_3h = noaa_3h.drop(columns=["Kp"], errors="ignore")
    else:
        # Quiet-day neutral fill — model still runs without storm-memory context.
        noaa_3h["kp_lag_3h"] = 2.0
        noaa_3h["kp_lag_6h"] = 2.0
        noaa_3h["kp_lag_9h"] = 2.0

    # Rectified southward E-field (same formula as omni_client.py).
    noaa_3h["vbs"] = (
        noaa_3h["speed_mean"] * np.maximum(0.0, -noaa_3h["bz_mean"]) / 1000.0
    )

    # Russell–McPherron seasonal proxy (same formula as omni_client.py).
    _doy = noaa_3h.index.day_of_year.astype(float)
    noaa_3h["equinox_term"] = np.cos(4.0 * np.pi * (_doy - 80.0) / 365.25)

    # ── kp_lag_27d ─────────────────────────────────────────────────────────────
    # The 7-day NOAA feed cannot reach back 27 days, so we look up OMNI Kp at
    # (timestamp − 27 days) for each row.  We allow ±3h tolerance to handle
    # minor grid misalignment.  Falls back to 2.0 if omni_kp_hist is not passed
    # or the timestamp predates OMNI coverage.
    #
    # pipeline.py ensures omni_kp_hist is always passed (omni_3h["Kp"]) so this
    # fallback only activates if OMNI failed to load entirely.
    if omni_kp_hist is not None and not omni_kp_hist.empty:
        lag_times = noaa_3h.index - pd.Timedelta(days=27)
        kp_27d = omni_kp_hist.reindex(
            lag_times, method="nearest", tolerance=pd.Timedelta(hours=3)
        )
        kp_27d.index = noaa_3h.index
        noaa_3h["kp_lag_27d"] = kp_27d.fillna(2.0).values
    else:
        noaa_3h["kp_lag_27d"] = 2.0

    return noaa_3h

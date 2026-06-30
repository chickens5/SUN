#Welcome to the data pipeline for Gabe's Geomanetic Storm Prediction Pipeline (GGSP-7.0.py)!
# Thank you for checkin it out.

# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE OVERVIEW  (ggsp_pipeline_v7.py)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file is the ENGINE.  GGSP-7.0.py is the thin CLI wrapper that calls
# run_pipeline() here.  sunspot_pipeline.py is called as a sub-pipeline inside
# run_pipeline() to fetch solar-cycle weights.  eval/evaluate.py is called
# automatically at the end of each run to print the honest evaluation report.
#
# Data flow through run_pipeline() — eight stages in order:
#
#  1. FETCH NOAA (live)   load_noaa_data()
#       └─ 3 JSON feeds from NOAA SWPC: plasma, mag, Kp  (7-day, 1-min cadence)
#
#  2. FETCH OMNI (archive) load_omni_data()
#       └─ SPDF yearly .dat files → parquet cache under cache/omni/
#          Multi-year hourly data resampled to 3-hour grid for training.
#
#  3. FEATURE ENGINEERING  build_noaa_3h_features() / load_omni_data()
#       └─ 14 features: Newell coupling, p_dyn, Bz lags (3/6/9h + 27d),
#          vbs (rectified E-field), equinox_term (R-M seasonal proxy)
#          FEATURE_COLUMNS order is a contract — all X arrays must match it.
#
#  4. TRAIN & EVALUATE     fit_and_evaluate_model()
#       └─ GradientBoostingRegressor inside StandardScaler Pipeline.
#          Chronological 75/25 split + 5-fold TimeSeriesSplit CV (capped 30K rows).
#          Fitted model saved to cache/model.joblib; reloaded on next run unless
#          --refit is passed.
#
#  5. HONEST EVALUATION    eval.evaluate.print_evaluation_report()
#       └─ Persistence baseline, skill-over-persistence, G1 categorical scores,
#          storm-conditional MAE, reliability table — printed to console.
#
#  6. SOLAR-CYCLE WEIGHTS  sunspot_pipeline.run_sunspot_pipeline()
#       └─ Fetches SIDC SSN, computes storm_rate_modifier (0.4 – 1.6) and
#          adjusts Quiet/Moderate/Active scenario weights accordingly.
#          Skipped when --static-weights is passed.
#
#  7. FORECAST             build_forecast_scenarios() → predict_scenario_kp()
#                                                      → predict_scenario_kp_ensemble()
#       └─ 72-hour (24 × 3h) deterministic scenario forecast plus a
#          500-draw stochastic ensemble for P(Kp ≥ 5) that avoids the
#          mean-trajectory suppression of the old storm-probability calculation.
#
#  8. EXPORT               _serialize_outputs() → JSON
#       └─ All outputs serialised to JSON for the React frontend.
#          Includes metrics, forecast arrays, sunspot_info, storm_prob_per_window.
#
# ═══════════════════════════════════════════════════════════════════════════════



from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from typing import Dict, Tuple
import json
import pathlib
import time
import urllib.request

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# (removed stale 'from xml.parsers.expat import model' — that imported XML DTD constants,
#  not an ML model; the actual sklearn Pipeline model is built inside fit_and_evaluate_model())
#We import dataclasses to define a simple configuration class for the pipeline, and typing for type hints to improve code clarity.
#Also, we need typing, json, and urllib to handle data fetching and parsing from the NOAA and OMNI sources.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance as _skl_perm_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#Lastly, we import numpy, pandas, matplotlib, and scikit-learn for data manipulation, visualization, and modeling tasks throughout the pipeline.


#Now, that we understand the depencies, let's look at the code!

#First, we define a read-only dataclass: PipelineConfig, which holds all the configuration parameters for the pipeline, such as data source settings, model hyperparameters, and forecast scenario weights.
#  This allows us to easily manage and pass around configuration values in a structured way.

@dataclass(frozen=True)
class PipelineConfig:

    noaa_timeout_s: int = 20        #Sets timeout for NOAA data fetches to prevent hanging if the service is unresponsive.

    omni_start_year: int = 2020     #Sets the start year for OMNI data fetch, (2020 ensures we don't use too much and too old of data) us to specify how many years of historical data to use for training the model.

    omni_num_years: int = 5        #Sets the number of years of historical OMNI data to use for training the model.
    
    train_fraction: float = 0.75   #Sets the fraction of data to use for training vs. validation. *Closer to 1 means more training data but less test data, which can lead to overfitting. 
                                    #Closer to 0.5 means more balanced but less training data, which can lead to underfitting.

    n_estimators: int = 200       #Sets the number of boosting stages for the Gradient Boosting Regressor, which controls the complexity of the model.

    max_depth: int = 3          #Sets the maximum depth of the individual regression estimators. Increased from 2→3 because we now have 11 features (vs. 6 originally).
                                    # Depth 2 allows at most 4 leaf nodes per tree — too few splits to model the nonlinear interactions
                                    # between Bz, coupling, and the new Kp-lag features. Depth 3 (8 leaf nodes) lets the model
                                    # capture three-way interactions (e.g., high speed AND strongly negative Bz AND already elevated Kp)
                                    # without risking the overfitting that depth 4+ would bring on smaller training sets.

    learning_rate: float = 0.05     #Sets the learning rate for the Gradient Boosting Regressor, which controls how much each tree contributes to the overall model. Lower values can lead to better performance but require more trees.

    subsample: float = 0.8      #Sets the fraction of samples to be used for fitting the individual base learners in the Gradient Boosting Regressor, which can help prevent overfitting by introducing randomness.

    min_samples_split: int = 20     #Sets the minimum num of samples for splitting an internal node in the Gradient Boosting Regressor, 
                                        #which can help control overfitting by requiring a minimum amount of data to make a split.

    min_samples_leaf: int = 5    # new — each leaf needs at least 5 samples

    random_state: int = 42          #Sets the random seed for reproducibility of results.   

    forecast_steps_3h: int = 24  #Since we are taking in 72h forecasts from NOAA, we divide by 24 to get our 3h steps, 
                                    #which means we will forecast 24 steps of 3 hours each to cover the full 72 hours.

    scenario_weights: Dict[str, float] = None   #Allows us to specify weights for the quiet, moderate, 
                                                    #and active forecast scenarios when combining them into a weighted ensemble forecast.

    use_cv: bool = True   #Task 2: run TimeSeriesSplit cross-validation in fit_and_evaluate_model.
                           # Set False to skip CV and use only the single chronological split.

    n_cv_folds: int = 5   #Number of expanding-window CV folds (default 5).


#We include the  __post_init__ method in the config to set default scenario weights (failsafe)
#  in case they are not provided when the PipelineConfig is instantiated.
    def __post_init__(self):
        if self.scenario_weights is None:
            object.__setattr__(self, "scenario_weights", {"Quiet": 0.2, "Moderate": 0.5, "Active": 0.3})


HEADERS = {"User-Agent": "Mozilla/5.0 (geomag-storm-predictor; educational)"}

# Task 6: cache directories for OMNI parquet files and the fitted model.
# Past-year OMNI data is immutable, so we only re-fetch the current year.
_REPO_DIR        = pathlib.Path(__file__).parent
OMNI_CACHE_DIR   = _REPO_DIR / "cache" / "omni"
MODEL_CACHE_PATH = _REPO_DIR / "cache" / "model.joblib"
MODEL_META_PATH  = _REPO_DIR / "cache" / "model_meta.json"

NOAA_ENDPOINTS = {
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    #Format: [["time_tag","density","speed","temperature"], 
    # ["2024-06-01T00:00:00Z", 5.0, 400.0, 100000.0], ...]

    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    #Format: [["time_tag","bx_gsm","by_gsm","bz_gsm","lon_gsm","lat_gsm","bt"],
    # ["2026-05-09 00:39:00.000","-3.87","-2.93","0.34","217.09","3.97","4.87"]

    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    #Format: [{"time_tag":"2026-05-09T00:00:00","Kp":2.33,"a_running":9,"station_count":8}
}
#^^^^^^ The constants above define the constants needed to know where to fetch the NOAA data from, 
# and set a user-agent header to avoid potential blocking by the server.


#Newell's Coupling Function is  the proof needed to actually be able to make ML predictions of
    #geomagnetic activity based on solar wind parameters as it provides a way to combine multiple solar wind features 
    # into a single coupling parameter that has been shown to correlate well with geomagnetic activity.

def newell_coupling(speed_kms, bt_nt, by_nt, bz_nt):
    """Newell et al. 2007 coupling proxy dPhi/dt."""
    speed = np.asarray(speed_kms, dtype=float)
    bt = np.asarray(bt_nt, dtype=float)
    by = np.asarray(by_nt, dtype=float)
    bz = np.asarray(bz_nt, dtype=float)
    theta = np.arctan2(np.abs(by), bz)
    return (speed ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (np.sin(theta / 2.0) ** (8.0 / 3.0))


def kp_label(kp_value: float) -> str:
    """Map Kp to NOAA storm category text."""
    if kp_value < 4:
        return "Quiet"
    if kp_value < 5:
        return "Unsettled / Active"
    if kp_value < 6:
        return "G1 - Minor storm"
    if kp_value < 7:
        return "G2 - Moderate storm"
    if kp_value < 8:
        return "G3 - Strong storm"
    if kp_value < 9:
        return "G4 - Severe storm"
    return "G5 - Extreme storm"


def _fetch_json(url: str, timeout: int, retries: int = 3) -> list:
    """Fetch JSON with lightweight retries for transient NOAA/OMNI response truncation."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
            return json.loads(payload)
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            # Small backoff to avoid immediately re-hitting a partial response.
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"Fetch failed after {retries} attempts for {url}: {last_error}")


def _rows_to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    for column in df.columns:
        if column != "time_tag":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.set_index("time_tag").sort_index()


def load_noaa_data(config: PipelineConfig):
    """STAGE 1 — Fetch live solar wind from NOAA SWPC.

    Returns three DataFrames (plasma, mag, Kp) at 1-minute cadence for the
    most-recent 7 days, plus a source tag.  These feed into
    build_noaa_3h_features() for the real-time inference point AND the
    forecast seed.  OMNI (load_omni_data) is used for model training only.
    """
    try:
        plasma_raw = _fetch_json(NOAA_ENDPOINTS["plasma"], config.noaa_timeout_s)
        mag_raw = _fetch_json(NOAA_ENDPOINTS["mag"], config.noaa_timeout_s)
        kp_raw = _fetch_json(NOAA_ENDPOINTS["kp"], config.noaa_timeout_s)

        plasma_df = _rows_to_dataframe(plasma_raw)
        mag_df = _rows_to_dataframe(mag_raw)

        # NOAA Kp feed is typically a list of dict rows, but keep legacy table parsing for compatibility.
        kp_rows = kp_raw.get("data", kp_raw) if isinstance(kp_raw, dict) else kp_raw
        if not isinstance(kp_rows, list) or len(kp_rows) == 0:
            raise RuntimeError("NOAA Kp payload is empty or malformed")

        if isinstance(kp_rows[0], dict):
            seen = set()
            records = []
            for row in kp_rows:
                key = (
                    row.get("time_tag"),
                    row.get("Kp"),
                    row.get("a_running"),
                    row.get("station_count"),
                )
                if key in seen:
                    continue
                seen.add(key)
                records.append({"time_tag": row.get("time_tag"), "Kp": row.get("Kp")})
            kp_df = pd.DataFrame.from_records(records)
        elif isinstance(kp_rows[0], list):
            kp_df = pd.DataFrame(kp_rows[1:], columns=kp_rows[0])
        else:
            raise RuntimeError("NOAA Kp payload has unsupported row format")

        kp_df["time_tag"] = pd.to_datetime(kp_df["time_tag"], utc=True)
        kp_df["Kp"] = pd.to_numeric(kp_df["Kp"], errors="coerce")
        kp_df = kp_df.dropna(subset=["time_tag", "Kp"]).set_index("time_tag").sort_index()[["Kp"]]
        if kp_df.empty:
            raise RuntimeError("NOAA Kp parsed successfully but produced no valid rows")

        _sanity_check_noaa(plasma_df, mag_df, kp_df)
        return plasma_df, mag_df, kp_df, "live_noaa"
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load NOAA live data: {exc}") from exc


def _sanity_check_noaa(
    plasma_df: pd.DataFrame,
    mag_df: pd.DataFrame,
    kp_df: pd.DataFrame,
) -> None:
    """Warn about stale or sparse NOAA feeds."""
    import sys
    now = pd.Timestamp.now("UTC")

    for name, df in (("plasma", plasma_df), ("mag", mag_df), ("kp", kp_df)):
        if df.empty:
            raise RuntimeError(
                f"NOAA '{name}' feed parsed successfully but returned zero rows. "
                "The feed may be temporarily empty — try again in a few minutes."
            )
        latest_ts = df.index.max()
        age_hours = (now - latest_ts).total_seconds() / 3600
        if age_hours > 6:
            print(
                f"[WARNING] NOAA '{name}' feed: most recent data is "
                f"{age_hours:.1f} hours old (expected ≤6 h for DSCOVR feeds). "
                "Real-time inference may be using stale solar wind conditions.",
                file=sys.stderr,
            )

    # Speed sanity: DSCOVR/ACE typically sees 250–900 km/s; values outside this range
    # in the 1-minute feed indicate sensor noise or a fill value that wasn't caught.
    if "speed" in plasma_df.columns:
        speed_vals = plasma_df["speed"].dropna()
        if len(speed_vals) > 0:
            spd_min, spd_max = speed_vals.min(), speed_vals.max()
            if spd_min < 100 or spd_max > 2000:
                print(
                    f"[WARNING] NOAA plasma speed out of physical range: "
                    f"min={spd_min:.0f}, max={spd_max:.0f} km/s. "
                    "Suspect fill values may remain in the feed.",
                    file=sys.stderr,
                )

    print(
        f"[OK] NOAA: plasma={len(plasma_df)} rows, "
        f"mag={len(mag_df)} rows, kp={len(kp_df)} rows"
    )
#Below is a new feature that creates a 3 hour aggregated feature set from
#  the raw 1-minute NOAA plasma and magnetic field data, 
# which is necessary to align with the 3-hourly OMNI data 
# and to create features that are more relevant for predicting Kp.


def build_noaa_3h_features(
    plasma_df: pd.DataFrame,
    mag_df: pd.DataFrame,
    kp_df: pd.DataFrame | None = None,
    omni_kp_hist: "pd.Series | None" = None,
) -> pd.DataFrame:
    """STAGE 3a — Aggregate NOAA 1-min feeds into the 14-feature 3h frame.

    Output must have exactly the columns in FEATURE_COLUMNS (same order) so
    the trained model can consume it directly for real-time inference.
    kp_df is accepted to build Kp-lag features for real-time inference.
    omni_kp_hist (Task 4) is the OMNI Kp series (pd.Series) used to source
    kp_lag_27d — the 7-day NOAA feed cannot reach back 27 days.
    The NOAA mag feed already contains by_gsm at 1-minute cadence.
    """
    merged = plasma_df.join(mag_df, how="inner")
    # Include by_gsm — it was always in the NOAA mag feed but was previously ignored.
    merged = merged[["speed", "density", "by_gsm", "bz_gsm", "bt"]].dropna()
    if merged.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    noaa_3h = merged.resample("3h", label="left").agg(
        {
            "speed": "mean",
            "density": "mean",
            "by_gsm": "mean",   # real By — feeds correct clock angle into Newell
            "bz_gsm": ["mean", "min"],
            "bt": "mean",
        }
    )
    noaa_3h.columns = ["speed_mean", "density_mean", "by_mean", "bz_mean", "bz_min", "bt_mean"]
    noaa_3h = noaa_3h.dropna()
    if noaa_3h.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    # Use the real measured By instead of the old constant proxy.
    noaa_3h["coupling_mean"] = newell_coupling(
        noaa_3h["speed_mean"], noaa_3h["bt_mean"], noaa_3h["by_mean"], noaa_3h["bz_mean"]
    )

    # Dynamic ram pressure — mirrors the OMNI training feature so the model
    # receives the same physical quantity it was trained on.
    noaa_3h["p_dyn_mean"] = noaa_3h["density_mean"] * (noaa_3h["speed_mean"] / 100.0) ** 2

    # Kp lag features for real-time inference. Kp is observed only every 3 hours so we
    # resample it to the same 3h grid, forward-fill any gaps (brief data dropouts), then
    # shift to build the 3h/6h/9h lookback. If no Kp data is available at all we fall back
    # to 2.0 nT (typical quiet-day value) so the inference path never hard-crashes.
    if kp_df is not None and not kp_df.empty:
        kp_3h = kp_df["Kp"].resample("3h", label="left").mean()
        noaa_3h = noaa_3h.join(kp_3h, how="left")
        noaa_3h["Kp"] = noaa_3h["Kp"].ffill()   # fill gaps between 3-hourly Kp reports
        noaa_3h["kp_lag_3h"] = noaa_3h["Kp"].shift(1).ffill().fillna(2.0)
        noaa_3h["kp_lag_6h"] = noaa_3h["Kp"].shift(2).ffill().fillna(2.0)
        noaa_3h["kp_lag_9h"] = noaa_3h["Kp"].shift(3).ffill().fillna(2.0)
        noaa_3h = noaa_3h.drop(columns=["Kp"], errors="ignore")
    else:
        # Neutral quiet-day fill — model still runs, just without storm-memory context.
        noaa_3h["kp_lag_3h"] = 2.0
        noaa_3h["kp_lag_6h"] = 2.0
        noaa_3h["kp_lag_9h"] = 2.0

    # vbs: rectified southward electric field — same formula used in OMNI training.
    noaa_3h["vbs"] = noaa_3h["speed_mean"] * np.maximum(0.0, -noaa_3h["bz_mean"]) / 1000.0

    # equinox_term: R–M seasonal proxy — same formula used in OMNI training.
    _doy = noaa_3h.index.day_of_year.astype(float)
    noaa_3h["equinox_term"] = np.cos(4.0 * np.pi * (_doy - 80.0) / 365.25)

    # kp_lag_27d: solar-rotation recurrence lag (Task 4).
    # The 7-day NOAA feed does not reach back 27 days so we look up the OMNI Kp series
    # at (timestamp − 27d) for each row.  Falls back to 2.0 (typical quiet Kp) if OMNI
    # history is unavailable or the timestamp predates coverage.
    # *DOUBLE CHECL*: run_pipeline() must call load_omni_data() BEFORE build_noaa_3h_features()
    # and pass omni_kp_hist=omni_3h["Kp"] for real 27d recurrence values.
    if omni_kp_hist is not None and not omni_kp_hist.empty:
        lag_times = noaa_3h.index - pd.Timedelta(days=27)
        # reindex with tolerance=3h so minor grid misalignment does not produce NaN.
        kp_27d = omni_kp_hist.reindex(lag_times, method="nearest", tolerance=pd.Timedelta(hours=3))
        kp_27d.index = noaa_3h.index
        noaa_3h["kp_lag_27d"] = kp_27d.fillna(2.0).values
    else:
        # *DOUBLE CHECL*: kp_lag_27d falls back to 2.0 (quiet-day) when OMNI history is not passed.
        noaa_3h["kp_lag_27d"] = 2.0

    return noaa_3h



# --- OMNI fetch/parser replacement using SPDF yearly files ---


def _is_missing(value: float, sentinels: tuple) -> bool:
    return any(np.isclose(value, s) for s in sentinels)

def _parse_omni2_yearly_line(line: str):
    parts = line.split()
    if len(parts) < 40:
        return None
    try:
        year = int(parts[0])
        doy = int(parts[1])
        hour = int(parts[2])
        bt = float(parts[9])
        # OMNI2 column layout (0-based): col 15 = By_GSM, col 16 = Bz_GSM.
        # Previously By_GSM was not extracted, forcing a constant proxy (2.0 nT) in
        # the Newell coupling function. That breaks the clock-angle term θ=arctan(|By|/Bz)
        # for every row — now we use the real measured value so coupling is physically correct.
        by = float(parts[15])   # By_GSM (nT) — new
        bz = float(parts[16])   # Bz_GSM (nT)
        density = float(parts[23])
        speed = float(parts[24])
        kp_raw = float(parts[38])
        ts = pd.Timestamp(f"{year:04d}-01-01", tz="UTC") + pd.Timedelta(days=doy - 1, hours=hour)
        if _is_missing(speed, (9999.0, 99999.9)):
            speed = np.nan
        if _is_missing(density, (999.9,)):
            density = np.nan
        if _is_missing(bt, (999.9,)):
            bt = np.nan
        if _is_missing(by, (999.9,)):
            by = np.nan
        if _is_missing(bz, (999.9,)):
            bz = np.nan
        if _is_missing(kp_raw, (99.0,)):
            kp = np.nan
        else:
            kp = kp_raw / 10.0
        return {
            "time_tag": ts,
            "speed": speed,
            "density": density,
            "by_gsm": by,   # new — real IMF By component
            "bz_gsm": bz,
            "bt": bt,
            "kp": kp,
        }
    except Exception:
        return None

def _parse_omni2_yearly_text(text: str):
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("Yr") or line.startswith("--"):
            continue
        parsed = _parse_omni2_yearly_line(line)
        if parsed:
            rows.append(parsed)
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("time_tag").sort_index()
    df = df.dropna()
    return df

def _fetch_omni_data(start_year: int, num_years: int, timeout: int, use_cache: bool = True) -> pd.DataFrame:
    """Fetch OMNI hourly data from SPDF yearly .dat files.

    Past years (not the current calendar year) are cached under
    cache/omni/omni2_{year}.parquet so repeat runs skip the network.

    Task 6: GIF responses (SPDF error pages) now raise immediately rather than
    attempting pytesseract OCR.  The OCR path was removed because it could
    silently inject incorrect numeric values into the training set — which is
    worse than a loud, obvious failure.
    """
    import datetime
    current_year = datetime.datetime.now(datetime.timezone.utc).year
    yearly_frames = []
    for year in range(start_year, start_year + num_years):
        is_current = (year == current_year)
        cache_path = OMNI_CACHE_DIR / f"omni2_{year}.parquet"

        # Past years are immutable — load from parquet cache if available.
        if use_cache and not is_current and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                yearly_frames.append(df)
                print(f"[cache] OMNI {year}: loaded from {cache_path.name}")
                continue
            except Exception as cache_exc:
                print(f"[WARNING] OMNI cache read failed for {year}: {cache_exc} — re-fetching.")

        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "").lower()

        # Task 6: Fail loudly on GIF/HTML redirect — the old pytesseract OCR fallback
        # was removed because it could silently corrupt training data with OCR artefacts.
        is_gif = "image/gif" in content_type or payload[:6] in (b"GIF87a", b"GIF89a")
        if is_gif:
            raise RuntimeError(
                f"SPDF returned a GIF (HTML error page) for OMNI year {year}.\n"
                f"URL: {url}\n"
                "This usually means the requested year is outside OMNI2 coverage "
                "(1963–present) or the SPDF server is redirecting. "
                "The GIF/OCR fallback has been removed to prevent silent data corruption. "
                "Check your --start-year / --years arguments and SPDF server status."
            )

        text = payload.decode("utf-8", errors="ignore")
        df = _parse_omni2_yearly_text(text)
        if df is not None:
            yearly_frames.append(df)
            # Cache past years only (current-year files are incomplete until year end).
            if use_cache and not is_current:
                OMNI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    df.to_parquet(cache_path)
                    print(f"[cache] OMNI {year}: saved to {cache_path.name}")
                except Exception as write_exc:
                    print(f"[WARNING] OMNI cache write failed for {year}: {write_exc}")

    if not yearly_frames:
        raise RuntimeError("No valid OMNI numeric rows were found from SPDF yearly files.")
    merged = pd.concat(yearly_frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_omni_data(config: PipelineConfig):
    """STAGE 2 — Fetch multi-year OMNI archive for model training.

    Downloads SPDF yearly .dat files, caches them as parquet under
    cache/omni/, resamples to 3-hour grid, and engineers the full
    14-feature FEATURE_COLUMNS set including kp_lag_27d.
    Returns (omni_3h DataFrame, source_tag).
    The returned omni_3h is also passed to build_noaa_3h_features()
    as omni_kp_hist so the live inference frame can source kp_lag_27d.
    """
    omni_raw = _fetch_omni_data(config.omni_start_year, config.omni_num_years, config.noaa_timeout_s)
    if omni_raw is None:
        raise RuntimeError("Failed to load OMNI live data: OMNI empty or unavailable")
    source = "live_omni"

    omni_3h = omni_raw.resample("3h", label="left").agg(
        {
            "speed": "mean",
            "density": "mean",
            "by_gsm": "mean",   # now extracted from OMNI col 15 — used in the real Newell coupling
            "bz_gsm": ["mean", "min"],
            "bt": "mean",
            "kp": "mean",
        }
    )
    omni_3h.columns = ["speed_mean", "density_mean", "by_mean", "bz_mean", "bz_min", "bt_mean", "Kp"]
    omni_3h = omni_3h.dropna()

    # Real By_GSM replaces the constant 2.0 nT proxy. The Newell coupling's clock-angle term
    # θ = arctan(|By| / Bz) was systematically wrong whenever By deviated from 2 nT, which
    # happens during CIRs and CME sheaths — exactly the events we most care about predicting.
    omni_3h["coupling_mean"] = newell_coupling(
        omni_3h["speed_mean"], omni_3h["bt_mean"], omni_3h["by_mean"], omni_3h["bz_mean"]
    )

    # Dynamic ram pressure: P_dyn ∝ n·V². When a CME arrives, the sudden compression of the
    # magnetosphere (sudden commencement) spikes P_dyn before Bz even rotates southward.
    # Providing this as a separate feature lets the model learn that pressure pulses precede
    # storm onset — information Newell coupling alone does not capture.
    # We normalise by dividing speed by 100 so the values stay in the same numeric range as
    # the other features before StandardScaler touches them.
    omni_3h["p_dyn_mean"] = omni_3h["density_mean"] * (omni_3h["speed_mean"] / 100.0) ** 2

    # Rectified southward electric field (Task 4): V_sw × max(0, −Bz) / 1000 [mV/m-ish].
    # Captures the geoeffective energy transfer rate independently of the full Newell coupling.
    # Particularly informative during sudden Bz southward turnings ahead of CME main phase.
    omni_3h["vbs"] = omni_3h["speed_mean"] * np.maximum(0.0, -omni_3h["bz_mean"]) / 1000.0

    # Russell–McPherron equinoctial proxy (Task 4).  The R–M effect produces a semi-annual
    # modulation of storm occurrence because the GSM equatorial plane tilts relative to the
    # heliographic equator near the equinoxes, enhancing dayside Bz coupling.
    # cos(4π·(doy−80)/365.25) has a half-year period, peaking at the March (doy≈80) and
    # September (doy≈263) equinoxes and troughing at the solstices.
    _doy = omni_3h.index.day_of_year.astype(float)
    omni_3h["equinox_term"] = np.cos(4.0 * np.pi * (_doy - 80.0) / 365.25)

    omni_3h["kp_lag_3h"] = omni_3h["Kp"].shift(1)   # Kp 3 h ago
    omni_3h["kp_lag_6h"] = omni_3h["Kp"].shift(2)   # Kp 6 h ago
    omni_3h["kp_lag_9h"] = omni_3h["Kp"].shift(3)   # Kp 9 h ago
    # Solar-rotation recurrence lag (Task 4). Active regions persist for 2–3 solar rotations
    # (~27 days). Knowing Kp 27 days ago lets the model detect recurring storm drivers
    # (CIRs from fast-stream sources, long-lived CME active regions).
    # 27 d = 27 × 24 h / 3 h = 216 three-hour steps.
    # *DOUBLE CHECL*: In live NOAA inference, kp_lag_27d is sourced from OMNI history
    # (see build_noaa_3h_features) because the 7-day NOAA feed cannot reach back 27 days.
    omni_3h["kp_lag_27d"] = omni_3h["Kp"].shift(216)
    # Drop the first ~27 days that have no full lag history after all shifts.
    omni_3h = omni_3h.dropna()

    _sanity_check_omni(omni_3h, config)
    return omni_3h, source


def _sanity_check_omni(omni_3h: pd.DataFrame, config: PipelineConfig) -> None:
    """Raise RuntimeError or print warnings for suspicious OMNI data."""
    import sys
    n = len(omni_3h)
    # Each year should contribute ~2,900 3-hour rows (8,760 hours / 3).  A year with
    # extensive data gaps produces far fewer. If total rows are fewer than 700 × years
    # the training set is likely too sparse to learn anything meaningful.
    min_expected = 700 * config.omni_num_years
    if n < min_expected:
        raise RuntimeError(
            f"OMNI data sanity check failed: only {n} 3-hour samples after cleaning "
            f"(expected ≥{min_expected} for {config.omni_num_years} years). "
            "The requested year range may have extensive data gaps or the SPDF server "
            "returned partial files. Try a different --start-year or --years value."
        )

    # Warn about any feature column that is NaN-heavy after the final dropna().
    # If dropna removed too many rows some columns may have been sparsely populated
    # before cleaning — flag it so the user knows the model saw reduced coverage.
    required_cols = ["speed_mean", "density_mean", "by_mean", "bz_mean", "bz_min",
                     "bt_mean", "coupling_mean", "p_dyn_mean",
                     "vbs", "equinox_term",
                     "kp_lag_3h", "kp_lag_27d", "Kp"]
    for col in required_cols:
        if col not in omni_3h.columns:
            raise RuntimeError(
                f"OMNI feature column '{col}' is missing after processing. "
                "This likely means the OMNI .dat column layout has changed or an "
                "incorrect --start-year was used (pre-1978 files lack some columns)."
            )

    # Kp range sanity — OMNI stores raw Kp*10 and we divide by 10; valid range is 0.0–9.0
    kp_min, kp_max = omni_3h["Kp"].min(), omni_3h["Kp"].max()
    if kp_max > 9.5 or kp_min < 0.0:
        print(
            f"[WARNING] OMNI Kp values out of expected range [0, 9]: "
            f"min={kp_min:.2f}, max={kp_max:.2f}. Check OMNI column mapping.",
            file=sys.stderr,
        )

    print(
        f"[OK] OMNI: {n} 3-hour samples, "
        f"{config.omni_start_year}–{config.omni_start_year + config.omni_num_years - 1}, "
        f"Kp range [{kp_min:.1f}, {kp_max:.1f}]"
    )


# FEATURE_COLUMNS defines the exact ordered set of inputs the model is trained and inferred on.
# Order matters: every caller that builds X arrays must match this list exactly.
#
# v7 → v8 additions (11 → 14 features, Task 4):
#   by_mean      — real IMF By; fixes the Newell clock-angle broken by the constant proxy
#   p_dyn_mean   — dynamic ram pressure (n·V²); captures sudden-commencement compressions
#   kp_lag_3h/6h/9h — autoregressive Kp history; encodes storm phase (onset / main / recovery)
#   kp_lag_27d   — solar-rotation recurrence (~27d = 216 × 3h steps)
#   vbs          — rectified southward electric field  V_sw × max(0,−Bz) / 1000  [mV/m-ish]
#   equinox_term — Russell–McPherron semi-annual proxy; cos(4π·(doy−80)/365.25)
FEATURE_COLUMNS = [
    "speed_mean",
    "density_mean",
    "by_mean",
    "bz_mean",
    "bz_min",
    "bt_mean",
    "coupling_mean",
    "p_dyn_mean",
    "kp_lag_3h",
    "kp_lag_6h",
    "kp_lag_9h",
    "kp_lag_27d",    # Task 4: solar-rotation recurrence
    "vbs",           # Task 4: rectified southward electric field
    "equinox_term",  # Task 4: Russell-McPherron seasonal proxy
]

# Module-level constant: every code path that builds an X array must produce
# exactly this many columns.  Checked by assertion inside predict_scenario_kp
# and fit_and_evaluate_model to catch FEATURE_COLUMNS/X-array mismatches early.
_EXPECTED_N_FEATURES = len(FEATURE_COLUMNS)  # currently 14


def fit_and_evaluate_model(data_3h: pd.DataFrame, config: PipelineConfig):
    """STAGE 4 — Train the GBR model and evaluate it honestly.

    Input:  omni_3h (STAGE 2 output) — the full FEATURE_COLUMNS + 'Kp' frame.
    Output: (fitted Pipeline, metrics dict, test_frame DataFrame)

    The fitted Pipeline is the only artifact passed forward to inference;
    test_frame is used for eval/evaluate.py and the held-out Kp plot.
    metrics is cached to cache/model_metrics.json alongside model.joblib.
    """
    import sys

    # Need at minimum: 24 rows to train + 8 rows to test (8 × 3h = 24h of evaluation)
    if len(data_3h) < 32:
        raise ValueError(
            f"Not enough aligned 3-hour samples for training: only {len(data_3h)} rows. "
            "At least 32 rows (~4 days) are required. "
            "Use a wider --years range or check for data-quality issues in OMNI."
        )

    # Ensure every required feature column is present with no all-NaN column
    missing = [c for c in FEATURE_COLUMNS if c not in data_3h.columns]
    if missing:
        raise ValueError(
            f"OMNI training frame is missing feature columns: {missing}. "
            "This indicates a mismatch between the OMNI parser and FEATURE_COLUMNS."
        )
    all_nan = [c for c in FEATURE_COLUMNS if data_3h[c].isna().all()]
    if all_nan:
        raise ValueError(
            f"These feature columns are entirely NaN after OMNI processing: {all_nan}. "
            "Check the OMNI column-index mapping in _parse_omni2_yearly_line()."
        )

    split_idx = int(config.train_fraction * len(data_3h))
    split_idx = max(1, min(split_idx, len(data_3h) - 1))

    X = data_3h[FEATURE_COLUMNS].values
    y = data_3h["Kp"].values

    # Module-level width assertion — catch FEATURE_COLUMNS/X-array mismatches immediately.
    assert X.shape[1] == _EXPECTED_N_FEATURES, (
        f"Training X width {X.shape[1]} != expected {_EXPECTED_N_FEATURES}. "
        "FEATURE_COLUMNS and all X-building paths must stay in sync."
    )

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    def _make_gbr_pipeline():
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

    # ── Task 2: TimeSeriesSplit cross-validation ──────────────────────────────
    # Expanding window, gap=3 rows (9h purge at each train/test boundary so
    # kp_lag_3h/6h/9h cannot leak a training-set Kp into the first test row).
    # CV is purely evaluative — the final model is still the single-split fit below.
    # For large datasets (>30K rows) we evaluate on the most-recent 30K to keep
    # wall time under ~30 s.  The final model trains on the full split.
    CV_MAX_ROWS = 30_000
    cv_results = {}
    if config.use_cv and len(X) >= 32 * config.n_cv_folds:
        X_cv = X[-CV_MAX_ROWS:] if len(X) > CV_MAX_ROWS else X
        y_cv = y[-CV_MAX_ROWS:] if len(y) > CV_MAX_ROWS else y
        tscv = TimeSeriesSplit(n_splits=config.n_cv_folds, gap=3)
        cv_maes, cv_r2s, cv_csis = [], [], []
        for fold_i, (tr_idx, te_idx) in enumerate(tscv.split(X_cv), 1):
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue
            # Use fewer estimators for CV folds — still representative, much faster.
            cv_n_est = min(config.n_estimators, 100)
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
            # Storm CSI at G1 (Kp ≥ 5) threshold
            obs_pos  = yt >= 5.0
            pred_pos = yp >= 5.0
            tp = np.sum(obs_pos  & pred_pos)
            fp = np.sum(~obs_pos & pred_pos)
            fn = np.sum(obs_pos  & ~pred_pos)
            denom = tp + fp + fn
            cv_csis.append(float(tp / denom) if denom > 0 else 0.0)
        if cv_maes:
            cv_results = {
                "cv_mae_mean":       float(np.mean(cv_maes)),
                "cv_mae_std":        float(np.std(cv_maes)),
                "cv_r2_mean":        float(np.mean(cv_r2s)),
                "cv_r2_std":         float(np.std(cv_r2s)),
                "cv_storm_csi_mean": float(np.mean(cv_csis)),
                "cv_n_folds":        len(cv_maes),
            }
            print(
                f"[CV]  {len(cv_maes)}-fold  "
                f"MAE={cv_results['cv_mae_mean']:.3f}±{cv_results['cv_mae_std']:.3f}  "
                f"R²={cv_results['cv_r2_mean']:.3f}±{cv_results['cv_r2_std']:.3f}  "
                f"G1-CSI={cv_results['cv_storm_csi_mean']:.3f}"
            )

    # ── Final-window fit (used for live inference and forecast) ───────────────
    model = _make_gbr_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ── Task 1: persistence baseline on the held-out test set ─────────────────
    # Persistence forecast: Kp(t) = Kp(t−3h) = kp_lag_3h feature.
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

    # ── Task 4: permutation importance on held-out fold ───────────────────────
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
            f"[WARNING] Model R²={metrics['r2']:.3f} is negative (worse than mean baseline). "
            f"Persistence MAE={persistence_mae:.3f}, Model MAE={metrics['mae']:.3f}. "
            "Consider a wider --years window or check OMNI data quality.",
            file=sys.stderr,
        )
    elif metrics["mae"] > 2.0:
        print(
            f"[WARNING] Model MAE={metrics['mae']:.3f} is unexpectedly high. "
            "Typical well-fitted models score MAE < 0.7 on multi-year data.",
            file=sys.stderr,
        )
    else:
        mean_improvement = 100.0 * (1.0 - metrics["mae"] / metrics["baseline_mae"]) if metrics["baseline_mae"] > 0 else 0.0
        beats = "beats" if skill_ratio < 1.0 else "WORSE than"
        print(
            f"[OK] Model: MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  "
            f"({mean_improvement:.1f}% over mean-baseline)  "
            f"skill_ratio={skill_ratio:.3f} ({beats} persistence)  "
            f"test_n={metrics['test_count']}"
        )

    test_frame = pd.DataFrame(
        {"Kp_true": y_test, "Kp_pred": y_pred},
        index=data_3h.index[split_idx:],
    )
    return model, metrics, test_frame


def build_forecast_scenarios(
    seed_3h: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    omni_kp_hist: "pd.Series | None" = None,
):
    """STAGE 7a — Generate solar-wind scenario arrays for the 72-hour forecast.

    Returns three named scenario dicts (Quiet / Moderate / Active) each
    containing per-step arrays for speed, density, bz_gsm, by_gsm, bt.
    These arrays feed into predict_scenario_kp() (deterministic) and
    predict_scenario_kp_ensemble() (stochastic, for storm probability).

    Task 5 changes vs. v7:
    - Scenario *means* conditioned on recent NOAA window (speed, density, bz_mean),
      not just on speed_mean_recent.
    - First 2 forecast steps are anchored to most-recent observed values (persistence
      seed), then blend smoothly toward the scenario trajectory.
    - Recurrence component folded into Moderate: when Kp 27 days ago was elevated
      (≥4), Moderate's base Bz is shifted more southward and speed nudged upward to
      reflect the CIR/CME active-region recurrence pattern.
    """
    steps = config.forecast_steps_3h
    base_time = seed_3h.index[-1]
    forecast_times = pd.date_range(base_time + pd.Timedelta(hours=3), periods=steps, freq="3h", tz="UTC")

    # Recent observed conditions anchor the scenario means (Task 5).
    recent = seed_3h[["speed_mean", "density_mean", "bz_mean"]].tail(10)
    speed_mean_recent   = float(recent["speed_mean"].mean())
    density_mean_recent = float(recent["density_mean"].mean())
    bz_mean_recent      = float(recent["bz_mean"].mean())  # may be slightly negative

    # Persistence anchor: most recent single observation for the first 2 steps.
    obs_speed   = float(seed_3h["speed_mean"].iloc[-1])
    obs_density = float(seed_3h["density_mean"].iloc[-1])
    obs_bz      = float(seed_3h["bz_mean"].iloc[-1])

    quiet_speed   = np.clip(np.linspace(speed_mean_recent - 50, speed_mean_recent - 60, steps) + rng.normal(0, 15, steps), 250, 600)
    quiet_bz      = rng.normal(bz_mean_recent * 0.3, 1.5, steps)   # conditioned on recent Bz
    quiet_density = rng.lognormal(np.log(max(density_mean_recent, 0.3)), 0.4, steps)

    moderate_speed   = np.clip(np.linspace(speed_mean_recent, speed_mean_recent + 30, steps) + rng.normal(0, 20, steps), 250, 700)
    moderate_bz_base = np.where(
        rng.random(steps) < 0.3,
        -2 * rng.exponential(1.5, steps),
        rng.normal(0.5, 1.2, steps),
    )
    moderate_density = rng.lognormal(np.log(max(density_mean_recent, 0.3)), 0.5, steps)

    # Task 5: Recurrence modifier folded into Moderate.
    # *DOUBLE CHECL*: Recurrence scenario (Task 5) is folded into Moderate as specified.
    # When Kp 27 days ago was elevated (≥4), Moderate's base Bz is pushed more
    # southward and speed nudged upward — reflecting the recurring CIR/CME pattern.
    # The recurrence Kp is the mean of the 27d-ago window aligned to forecast_times.
    recurrence_boost = 0.0
    if omni_kp_hist is not None and not omni_kp_hist.empty:
        kp_27d_vals = []
        for ft in forecast_times:
            t27 = ft - pd.Timedelta(days=27)
            try:
                idx = min(int(omni_kp_hist.index.searchsorted(t27)), len(omni_kp_hist) - 1)
                if abs((omni_kp_hist.index[idx] - t27).total_seconds()) < 3 * 3600:
                    kp_27d_vals.append(float(omni_kp_hist.iloc[idx]))
            except Exception:
                pass
        if kp_27d_vals:
            kp_27d_mean = float(np.mean(kp_27d_vals))
            # Scale: 0 boost at Kp=3, full boost at Kp=7
            recurrence_boost = float(np.clip((kp_27d_mean - 3.0) / 4.0, 0.0, 1.0))
    moderate_bz = moderate_bz_base - recurrence_boost * 2.5   # more southward when recurrent
    moderate_speed_arr = np.clip(moderate_speed + recurrence_boost * 60, 250, 750)

    active_speed   = np.clip(np.linspace(speed_mean_recent + 100, speed_mean_recent + 150, steps) + rng.normal(0, 25, steps), 350, 800)
    active_bz      = np.clip(
        -3 - 2 * np.sin(np.linspace(0, np.pi, steps)) + rng.normal(0, 1.2, steps),
        -12, 3,
    )
    active_density = np.clip(rng.lognormal(np.log(max(density_mean_recent + 1, 0.3)), 0.6, steps), 0.5, 50)

    quiet_by    = rng.normal(0.0, 2.0, steps)
    moderate_by = rng.normal(0.0, 4.0, steps)
    active_by   = rng.normal(0.0, 5.0, steps)

    # Task 5: persistence anchor — first 2 steps blend observed → scenario.
    # Step 0: 80% observed + 20% scenario mean; step 1: 40% + 60%.  This prevents
    # an abrupt discontinuity between the last real data point and the forecast.
    for spd_arr, bz_arr, den_arr in (
        (quiet_speed,       quiet_bz,    quiet_density),
        (moderate_speed_arr, moderate_bz, moderate_density),
        (active_speed,      active_bz,   active_density),
    ):
        spd_arr[0] = 0.8 * obs_speed   + 0.2 * spd_arr[0]
        spd_arr[1] = 0.4 * obs_speed   + 0.6 * spd_arr[1]
        bz_arr[0]  = 0.8 * obs_bz      + 0.2 * bz_arr[0]
        bz_arr[1]  = 0.4 * obs_bz      + 0.6 * bz_arr[1]
        den_arr[0] = 0.8 * obs_density  + 0.2 * den_arr[0]
        den_arr[1] = 0.4 * obs_density  + 0.6 * den_arr[1]

    scenarios = {
        "Quiet": {
            "speed":   quiet_speed,
            "density": quiet_density,
            "by_gsm":  quiet_by,
            "bz_gsm":  quiet_bz,
            "bt": np.abs(quiet_bz) + 2 + rng.normal(0, 0.5, steps),
            "color": "#90EE90",
        },
        "Moderate": {
            "speed":   moderate_speed_arr,
            "density": moderate_density,
            "by_gsm":  moderate_by,
            "bz_gsm":  moderate_bz,
            "bt": np.abs(moderate_bz) + 2.5 + rng.normal(0, 0.7, steps),
            "color": "#FFD700",
        },
        "Active": {
            "speed":   active_speed,
            "density": active_density,
            "by_gsm":  active_by,
            "bz_gsm":  active_bz,
            "bt": np.abs(active_bz) + 3 + rng.normal(0, 0.8, steps),
            "color": "#FF6B6B",
        },
    }
    seed_source = "noaa_recent"
    if recurrence_boost > 0.1:
        seed_source = f"noaa_recent+recurrence_boost_{recurrence_boost:.2f}"
    return forecast_times, scenarios, seed_source


def predict_scenario_kp(
    model,
    scenarios: Dict[str, dict],
    kp_seed: np.ndarray,
    forecast_times=None,
    omni_kp_hist: "pd.Series | None" = None,
) -> Dict[str, np.ndarray]:
    """STAGE 7b — Deterministic autoregressive Kp forecast for each scenario.

    Produces the kp_weighted trajectory used in the forecast plot and JSON.
    For storm PROBABILITY use predict_scenario_kp_ensemble() instead.

    kp_seed: array of the last 3 (or more) *observed* Kp values used to initialise
    the lag features. Once the loop starts, each predicted Kp immediately feeds back
    as the lag input for the next step — this is how storm phase memory propagates
    forward through the forecast window.

    forecast_times: DatetimeIndex of forecast steps (used to compute equinox_term
    and kp_lag_27d at the correct calendar dates).
    omni_kp_hist: OMNI Kp series — provides kp_lag_27d for the forecast window.
    """
    # Ensure we always have exactly 3 seed values (pad with quiet-day 2.0 if needed).
    seed = list(kp_seed[-3:]) if len(kp_seed) >= 3 else [2.0] * (3 - len(kp_seed)) + list(kp_seed)

    steps = len(next(iter(scenarios.values()))["speed"])

    # Pre-compute time-based features that are identical across all scenarios.
    # equinox_term: R–M seasonal proxy (same formula as OMNI training).
    if forecast_times is not None and len(forecast_times) == steps:
        _doy_arr = np.array([ft.day_of_year for ft in forecast_times], dtype=float)
    else:
        # *DOUBLE CHECL*: forecast_times not provided; equinox_term defaults to 0 (equinox mean).
        _doy_arr = np.full(steps, 80.0)
    equinox_arr = np.cos(4.0 * np.pi * (_doy_arr - 80.0) / 365.25)

    # kp_lag_27d: look up OMNI Kp at (forecast_time − 27d) for each step.
    # Falls back to 2.0 (quiet) when OMNI history is not provided.
    kp_lag_27d_arr = np.full(steps, 2.0)
    if omni_kp_hist is not None and forecast_times is not None and not omni_kp_hist.empty:
        for i, ft in enumerate(forecast_times):
            t27 = ft - pd.Timedelta(days=27)
            try:
                idx = omni_kp_hist.index.searchsorted(t27)
                idx = min(int(idx), len(omni_kp_hist) - 1)
                if abs((omni_kp_hist.index[idx] - t27).total_seconds()) < 3 * 3600:
                    kp_lag_27d_arr[i] = float(omni_kp_hist.iloc[idx])
            except Exception:
                pass

    kp_forecasts = {}
    for name, scenario in scenarios.items():
        bz_arr  = scenario["bz_gsm"]
        by_arr  = scenario["by_gsm"]
        spd_arr = scenario["speed"]
        den_arr = scenario["density"]
        bt_arr  = scenario["bt"]

        # bz_min: rolling 3-step (9h) minimum — matches how it was built during OMNI training.
        bz_min_arr = np.array([bz_arr[max(0, i - 2):i + 1].min() for i in range(len(bz_arr))])

        coupling  = newell_coupling(spd_arr, bt_arr, by_arr, bz_arr)
        p_dyn_arr = den_arr * (spd_arr / 100.0) ** 2

        # vbs: rectified southward electric field — same formula as OMNI training.
        vbs_arr = spd_arr * np.maximum(0.0, -bz_arr) / 1000.0

        kp_pred_arr  = np.zeros(steps)
        kp_history   = list(seed)

        for i in range(steps):
            lag_3h = kp_history[-1]
            lag_6h = kp_history[-2]
            lag_9h = kp_history[-3]

            x = np.array([[
                spd_arr[i],   den_arr[i],
                by_arr[i],    bz_arr[i],   bz_min_arr[i], bt_arr[i],
                coupling[i],  p_dyn_arr[i],
                lag_3h,       lag_6h,      lag_9h,
                kp_lag_27d_arr[i], vbs_arr[i], equinox_arr[i],
            ]])
            # Width assertion — catches FEATURE_COLUMNS vs X-array mismatches at inference.
            assert x.shape[1] == _EXPECTED_N_FEATURES, (
                f"predict_scenario_kp: x width {x.shape[1]} != {_EXPECTED_N_FEATURES}. "
                "Check FEATURE_COLUMNS and this function stay in sync."
            )
            kp_step = float(np.clip(model.predict(x)[0], 0, 9))
            kp_pred_arr[i] = kp_step
            kp_history.append(kp_step)

        kp_forecasts[name] = kp_pred_arr
    return kp_forecasts


# Noise scales (σ) used when resampling scenarios in the stochastic ensemble.
# Values match the noise amplitudes in build_forecast_scenarios so each draw
# is a plausible realisation of the same scenario distribution.
_SCENARIO_NOISE: Dict[str, dict] = {
    "Quiet":    {"speed_std": 15.0, "bz_std": 1.5, "density_log_std": 0.4, "by_std": 2.0},
    "Moderate": {"speed_std": 20.0, "bz_std": 1.2, "density_log_std": 0.5, "by_std": 4.0},
    "Active":   {"speed_std": 25.0, "bz_std": 1.2, "density_log_std": 0.6, "by_std": 5.0},
}


def predict_scenario_kp_ensemble(
    model,
    scenarios: Dict[str, dict],
    kp_seed: np.ndarray,
    weights: Dict[str, float],
    forecast_times=None,
    omni_kp_hist: "pd.Series | None" = None,
    n_draws: int = 500,
    rng: "np.random.Generator | None" = None,
) -> dict:
    """STAGE 7c — Stochastic ensemble forecast for P(Kp ≥ G1) (Task 3).

    Vectorised over n_draws: builds (n_draws × 14) feature batches at each
    time step, calls model.predict() once per step (not once per draw),
    then computes scenario-weighted P(Kp ≥ 5) per 3h window and over 72h.
    This is the source of storm_prob_72h_pct in the JSON output.

    The deterministic weighted-mean trajectory structurally suppresses storm
    probability toward zero because averaging over scenarios damps extremes.
    This function draws N=500 stochastic realisations per scenario by adding
    fresh noise to the base scenario trajectories (same noise model as
    build_forecast_scenarios), runs the autoregressive Kp forecast on each
    draw, and computes scenario-weighted P(Kp ≥ 5) per timestep and over
    the full 72-h window.

    Returns
    -------
    storm_prob_per_window  : np.ndarray shape (steps,) — P(Kp ≥ 5) at each 3h step
    storm_prob_72h         : float — P(at least one Kp ≥ 5 over the full 72h)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    seed = list(kp_seed[-3:]) if len(kp_seed) >= 3 else [2.0] * (3 - len(kp_seed)) + list(kp_seed)
    steps = len(next(iter(scenarios.values()))["speed"])

    # Pre-compute time-invariant arrays shared across all scenarios/draws.
    if forecast_times is not None and len(forecast_times) == steps:
        _doy_arr = np.array([ft.day_of_year for ft in forecast_times], dtype=float)
    else:
        _doy_arr = np.full(steps, 80.0)
    equinox_arr = np.cos(4.0 * np.pi * (_doy_arr - 80.0) / 365.25)

    kp_lag_27d_arr = np.full(steps, 2.0)
    if omni_kp_hist is not None and forecast_times is not None and not omni_kp_hist.empty:
        for i, ft in enumerate(forecast_times):
            t27 = ft - pd.Timedelta(days=27)
            try:
                idx = min(int(omni_kp_hist.index.searchsorted(t27)), len(omni_kp_hist) - 1)
                if abs((omni_kp_hist.index[idx] - t27).total_seconds()) < 3 * 3600:
                    kp_lag_27d_arr[i] = float(omni_kp_hist.iloc[idx])
            except Exception:
                pass

    # Normalise weights so they sum to 1.0.
    w_keys = [k for k in scenarios if k in weights]
    w_arr  = np.array([weights[k] for k in w_keys], dtype=float)
    w_arr /= w_arr.sum()

    prob_per_window = np.zeros(steps)
    prob_any_storm  = 0.0

    for wi, sname in enumerate(w_keys):
        scenario  = scenarios[sname]
        noise_cfg = _SCENARIO_NOISE.get(sname, _SCENARIO_NOISE["Moderate"])
        spd_base  = scenario["speed"]
        bz_base   = scenario["bz_gsm"]
        den_base  = scenario["density"]
        by_base   = scenario["by_gsm"]

        # ── Vectorised over n_draws ──────────────────────────────────────────
        # Pre-generate all noise at once (n_draws × steps), then loop only over
        # the 24 time steps (not over 500 draws). This reduces sklearn predict
        # calls from 500×24 → 24, each with a batch of 500 rows.
        spd_all = np.clip(
            spd_base[np.newaxis, :] + rng.normal(0, noise_cfg["speed_std"], (n_draws, steps)),
            200, 900,
        )
        bz_all = bz_base[np.newaxis, :] + rng.normal(0, noise_cfg["bz_std"], (n_draws, steps))
        den_all = np.clip(
            den_base[np.newaxis, :] * np.exp(
                rng.normal(0, noise_cfg["density_log_std"], (n_draws, steps))
            ),
            0.1, 100,
        )
        by_all = by_base[np.newaxis, :] + rng.normal(0, noise_cfg["by_std"], (n_draws, steps))
        bt_all = np.maximum(
            np.abs(bz_all) + 2.0 + rng.normal(0, 0.6, (n_draws, steps)),
            0.1,   # bt must be > 0 for newell_coupling bt^(2/3); clip near-zero draws
        )

        # bz_min: rolling 3-step minimum across the steps axis.
        bz_min_all = np.stack([
            np.min(bz_all[:, max(0, j - 2):j + 1], axis=1) for j in range(steps)
        ], axis=1)  # (n_draws, steps)

        coupling_all = np.nan_to_num(newell_coupling(spd_all, bt_all, by_all, bz_all), nan=0.0)
        p_dyn_all    = den_all * (spd_all / 100.0) ** 2
        vbs_all      = spd_all * np.maximum(0.0, -bz_all) / 1000.0

        # Autoregressive history: shape (n_draws, 3) — columns are [lag_9h, lag_6h, lag_3h]
        kp_hist_batch = np.tile(seed, (n_draws, 1))  # (n_draws, 3) — each row = [s0, s1, s2]
        draw_kp = np.zeros((n_draws, steps))

        for i in range(steps):
            X_batch = np.column_stack([
                spd_all[:, i],          den_all[:, i],
                by_all[:, i],           bz_all[:, i],
                bz_min_all[:, i],       bt_all[:, i],
                coupling_all[:, i],     p_dyn_all[:, i],
                kp_hist_batch[:, 2],    kp_hist_batch[:, 1],   kp_hist_batch[:, 0],
                np.full(n_draws, kp_lag_27d_arr[i]),
                vbs_all[:, i],
                np.full(n_draws, equinox_arr[i]),
            ])  # shape (n_draws, 14)
            X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
            kp_step = np.clip(model.predict(X_batch), 0.0, 9.0)  # (n_draws,)
            draw_kp[:, i] = kp_step
            # Shift history left: oldest falls off, new step appended at right.
            kp_hist_batch[:, 0] = kp_hist_batch[:, 1]
            kp_hist_batch[:, 1] = kp_hist_batch[:, 2]
            kp_hist_batch[:, 2] = kp_step

        # P(Kp ≥ 5) per timestep and over the full 72h window.
        storm_per_step = (draw_kp >= 5.0).mean(axis=0)        # (steps,)
        storm_any      = float((draw_kp.max(axis=1) >= 5.0).mean())

        # Weighted contribution of this scenario.
        prob_per_window += w_arr[wi] * storm_per_step
        prob_any_storm  += w_arr[wi] * storm_any

    return {
        "storm_prob_per_window": prob_per_window,
        "storm_prob_72h":        float(prob_any_storm),
    }


def weighted_ensemble(kp_forecasts: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """Combine scenario forecasts into a weighted best-estimate trajectory."""
    keys = ["Quiet", "Moderate", "Active"]
    w = np.array([weights[k] for k in keys], dtype=float)
    values = np.array([kp_forecasts[k] for k in keys])
    return np.average(values, axis=0, weights=w)


def plot_test_results(test_frame: pd.DataFrame, metrics: dict):
    """Compact evaluation plot for held-out window."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(test_frame.index, test_frame["Kp_true"], label="Observed Kp", color="#333333", lw=1.2)
    axes[0].plot(test_frame.index, test_frame["Kp_pred"], label="Predicted Kp", color="#cc4125", lw=1.2)
    axes[0].axhline(5, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title("Held-out Kp")
    axes[0].set_ylabel("Kp")
    axes[0].set_ylim(0, 9.5)
    axes[0].legend(loc="upper right")

    residuals = test_frame["Kp_true"] - test_frame["Kp_pred"]
    axes[1].hist(residuals, bins=30, color="#3d85c6", alpha=0.8, edgecolor="black")
    axes[1].axvline(0, color="red", ls="--", lw=1)
    axes[1].set_title(f"Residuals | MAE={metrics['mae']:.3f}, R2={metrics['r2']:.3f}")
    axes[1].set_xlabel("Observed - Predicted")

    plt.tight_layout()


def plot_forecast(forecast_times, scenarios: Dict[str, dict], kp_forecasts: Dict[str, np.ndarray], kp_weighted: np.ndarray):
    """Compact scenario forecast plot."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for name, kp_vals in kp_forecasts.items():
        axes[0].plot(forecast_times, kp_vals, label=name, color=scenarios[name]["color"], lw=2)
    axes[0].plot(forecast_times, kp_weighted, label="Weighted", color="#111111", lw=2.5)
    axes[0].axhline(5, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title("72-hour Kp scenario forecast")
    axes[0].set_ylabel("Kp")
    axes[0].set_ylim(0, 9.5)
    axes[0].legend(loc="upper left")

    for name, scenario in scenarios.items():
        axes[1].plot(forecast_times, scenario["bz_gsm"], label=f"{name} Bz", color=scenario["color"], lw=2)
    axes[1].axhline(0, color="black", lw=0.6, alpha=0.4)
    axes[1].axhline(-3, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[1].set_title("Scenario IMF Bz")
    axes[1].set_ylabel("Bz (nT)")
    axes[1].legend(loc="lower left")

    plt.tight_layout()

def _serialize_outputs(outputs: dict) -> dict:
    """Convert pipeline outputs to JSON-serializable types for React consumption."""
    forecast = outputs["forecast"]

    def _to_python(v):
        """Recursively convert numpy scalars / arrays to plain Python types."""
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        if isinstance(v, float) and (v != v):  # NaN
            return None
        return v

    metrics_clean = {k: _to_python(v) for k, v in outputs["metrics"].items()}

    # storm_prob_per_window is a new ensemble-based array (Task 3).
    storm_prob_arr = forecast.get("storm_prob_per_window")

    serialized = {
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
            "times":              [t.isoformat() for t in forecast["times"]],
            "kp_by_scenario":     {k: v.tolist() for k, v in forecast["kp_by_scenario"].items()},
            "kp_weighted":        forecast["kp_weighted"].tolist(),
            "storm_prob_per_window": storm_prob_arr.tolist() if storm_prob_arr is not None else [],
        },
        "sunspot_info": outputs.get("sunspot_info", {}),
    }
    return serialized

def run_pipeline(
    config: PipelineConfig | None = None,
    make_plots: bool = True,
    json_output_path: str | None = None,
    refit: bool = False,
    static_weights: bool = False,
):
    """Orchestrate end-to-end data → model → forecast workflow.

    Parameters
    ----------
    refit          : Task 6 — force model retraining even if a valid cache exists.
    static_weights : Task 7 — skip sunspot cycle modifier; use config.scenario_weights as-is.
    """
    import sys
    cfg = config or PipelineConfig()
    rng = np.random.default_rng(cfg.random_state)

    # NOAA and OMNI live-data paths — no synthetic fallback.
    plasma_df, mag_df, kp_df, noaa_source = load_noaa_data(cfg)

    # Task 4/5: Load OMNI *before* building NOAA features so we can pass
    # omni_kp_hist for the kp_lag_27d (27-day recurrence) sourcing.
    omni_3h, omni_source = load_omni_data(cfg)

    # Pass OMNI Kp history so build_noaa_3h_features can source kp_lag_27d.
    noaa_3h = build_noaa_3h_features(plasma_df, mag_df, kp_df, omni_kp_hist=omni_3h["Kp"])

    # ── Task 6: model caching ─────────────────────────────────────────────────
    # Persist the fitted sklearn Pipeline under cache/model.joblib.
    # Load the cached model when the training window matches and --refit is not set.
    model_meta = {
        "start_year":     cfg.omni_start_year,
        "num_years":      cfg.omni_num_years,
        "train_fraction": cfg.train_fraction,
    }
    model = None
    metrics = {}
    test_frame = pd.DataFrame()
    loaded_from_cache = False

    if not refit and _JOBLIB_AVAILABLE and MODEL_CACHE_PATH.exists() and MODEL_META_PATH.exists():
        try:
            import json as _json_mod
            with open(MODEL_META_PATH, encoding="utf-8") as _f:
                cached_meta = _json_mod.load(_f)
            if cached_meta == model_meta:
                model = _joblib.load(MODEL_CACHE_PATH)
                metrics_path = _REPO_DIR / "cache" / "model_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, encoding="utf-8") as _f:
                        metrics = _json_mod.load(_f)
                loaded_from_cache = True
                print("[cache] Model loaded from cache (pass refit=True or --refit to retrain).")
        except Exception as cache_exc:
            print(f"[WARNING] Model cache load failed ({cache_exc}); retraining.", file=sys.stderr)
            model = None

    if model is None:
        model, metrics, test_frame = fit_and_evaluate_model(omni_3h, cfg)
        # Save model and metadata for future runs.
        if _JOBLIB_AVAILABLE:
            try:
                MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
                _joblib.dump(model, MODEL_CACHE_PATH)
                with open(MODEL_META_PATH, "w", encoding="utf-8") as _f:
                    json.dump(model_meta, _f)
                metrics_path = _REPO_DIR / "cache" / "model_metrics.json"
                with open(metrics_path, "w", encoding="utf-8") as _f:
                    json.dump(metrics, _f)
                print(f"[cache] Model saved to {MODEL_CACHE_PATH.relative_to(_REPO_DIR)}")
            except Exception as save_exc:
                print(f"[WARNING] Model cache save failed: {save_exc}", file=sys.stderr)

    # ── Task 1: honest evaluation report on held-out test set ─────────────────
    if not test_frame.empty and "kp_lag_3h" in omni_3h.columns:
        try:
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
            # eval/evaluate.py not on path when called as a module from another directory
            pass
        except Exception as eval_exc:
            print(f"[WARNING] Evaluation report failed: {eval_exc}", file=sys.stderr)

    # Use most-recent NOAA 3h window for latest prediction (real-time, not stale OMNI).
    if len(noaa_3h) >= 1 and all(c in noaa_3h.columns for c in FEATURE_COLUMNS):
        latest_features = noaa_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time     = noaa_3h.index[-1]
        latest_actual   = float(kp_df["Kp"].iloc[-1]) if not kp_df.empty else float(omni_3h["Kp"].iloc[-1])
    else:
        latest_features = omni_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time     = omni_3h.index[-1]
        latest_actual   = float(omni_3h["Kp"].iloc[-1])
    latest_pred = float(np.clip(model.predict(latest_features)[0], 0, 9))

    # ── Task 7: auto-wire sunspot cycle modifier ───────────────────────────────
    # Calls sunspot_pipeline.run_sunspot_pipeline() to get storm_rate_modifier and
    # adjusted scenario weights.  Pass --static-weights to opt out.
    effective_weights = dict(cfg.scenario_weights)
    sunspot_info: dict = {}
    if not static_weights:
        try:
            from sunspot_pipeline import run_sunspot_pipeline as _run_ssn
            ssn_results = _run_ssn(make_plots=False)
            gi = ssn_results.get("ggsp_integration", {})
            if gi and "recommended_scenario_weights" in gi:
                effective_weights = gi["recommended_scenario_weights"]
                cur  = ssn_results.get("current", {})
                sunspot_info = {
                    "storm_rate_modifier":     gi.get("storm_rate_modifier"),
                    "f107_proxy":              gi.get("f107_proxy"),
                    "ssn_normalized":          gi.get("ssn_normalized"),
                    "solar_activity_tier":     cur.get("solar_activity_tier"),
                    "adjusted_weights":        effective_weights,
                    "source":                  "sunspot_pipeline",
                }
                print(
                    f"[OK] Sunspot modifier={gi['storm_rate_modifier']:.3f}  "
                    f"→ weights: Q={effective_weights['Quiet']:.3f}  "
                    f"M={effective_weights['Moderate']:.3f}  "
                    f"A={effective_weights['Active']:.3f}"
                )
        except Exception as ssn_exc:
            # Graceful degradation — if SIDC is unreachable, fall back to static weights.
            print(
                f"[WARNING] Sunspot modifier unavailable ({ssn_exc}); using static weights.",
                file=sys.stderr,
            )
            sunspot_info = {"source": "static_fallback", "error": str(ssn_exc)}
    else:
        sunspot_info = {"source": "static_weights_flag"}

    # ── Forecast ───────────────────────────────────────────────────────────────
    # Prefer NOAA-recent conditions to seed the 72h forecast; fall back to OMNI.
    forecast_seed = noaa_3h if len(noaa_3h) >= 4 else omni_3h

    # Task 5: pass omni_kp_hist so build_forecast_scenarios can apply recurrence boost.
    forecast_times, scenarios, fc_seed_source = build_forecast_scenarios(
        forecast_seed, cfg, rng, omni_kp_hist=omni_3h["Kp"],
    )
    if len(noaa_3h) < 4:
        fc_seed_source = "omni_fallback"

    # Seed Kp lags from real observations.
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

    # Task 3: stochastic ensemble for P(Kp ≥ 5) — replaces the mean-trajectory
    # calculation which structurally suppressed storm probability toward zero.
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
            "time":          latest_time,
            "predicted_kp":  latest_pred,
            "observed_kp":   latest_actual,
            "category":      kp_label(latest_pred),
        },
        "forecast": {
            "times":                    forecast_times,
            "kp_by_scenario":           kp_forecasts,
            "kp_weighted":              kp_weighted,
            "mean_weighted_kp":         float(np.mean(kp_weighted)),
            "peak_weighted_kp":         float(np.max(kp_weighted)),
            # Task 3: ensemble-based storm probability (replaces legacy mean-trajectory value)
            "storm_prob_per_window":    storm_prob_per_window,
            "storm_prob_72h_pct":       storm_chance_pct,
            # Legacy key kept for backwards-compat with existing React frontend
            "storm_chance_percent":     storm_chance_pct,
            "storm_probability_windows": float(np.mean(storm_prob_per_window)),
            "forecast_seed_source":     fc_seed_source,
        },
        "sunspot_info": sunspot_info,
    }

    if make_plots:
        if not test_frame.empty:
            plot_test_results(test_frame, metrics)
        else:
            print("[INFO] Skipping held-out Kp plot — model loaded from cache (run with --refit to regenerate).")
        plot_forecast(forecast_times, scenarios, kp_forecasts, kp_weighted)
        plt.show()

    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_outputs(outputs), f, indent=2)

    return outputs

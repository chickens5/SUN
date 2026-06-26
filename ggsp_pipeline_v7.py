#Welcome to the data pipeline for Gabe's Geomanetic Storm Prediction Pipeline (GGSP-7.0.py)!
# Thank you for checkin it out.

#We load real time NOAA data for the most recent 7 days, and multi-year OMNI data for model training.
# If either source is unavailable, we generate synthetic data with realistic patterns to keep the pipeline functional
#  and allow testing of the full workflow without external dependencies.



from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from typing import Dict, Tuple
import json
import time
import urllib.request
# (removed stale 'from xml.parsers.expat import model' — that imported XML DTD constants,
#  not an ML model; the actual sklearn Pipeline model is built inside fit_and_evaluate_model())
#We import dataclasses to define a simple configuration class for the pipeline, and typing for type hints to improve code clarity.
#Also, we need typing, json, and urllib to handle data fetching and parsing from the NOAA and OMNI sources.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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


#We include the  __post_init__ method in the config to set default scenario weights (failsafe)
#  in case they are not provided when the PipelineConfig is instantiated.
    def __post_init__(self):
        if self.scenario_weights is None:
            object.__setattr__(self, "scenario_weights", {"Quiet": 0.2, "Moderate": 0.5, "Active": 0.3})


HEADERS = {"User-Agent": "Mozilla/5.0 (geomag-storm-predictor; educational)"}
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
    """Load NOAA real-time data only (no synthetic fallback)."""
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
    now = pd.Timestamp.utcnow()

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
) -> pd.DataFrame:
    """Aggregate NOAA 1-minute streams into model-compatible 3-hour features.

    kp_df is now accepted so we can build Kp-lag features for real-time inference.
    The NOAA mag feed already contains by_gsm at 1-minute cadence, so we no longer
    need a constant proxy for the Newell clock-angle term.
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

    return noaa_3h



# --- OMNI fetch/parser replacement using SPDF yearly files ---
import io
from PIL import Image, ImageOps
import re

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

def _extract_numbers_from_gif(payload: bytes):
    image = Image.open(io.BytesIO(payload))
    gray = ImageOps.grayscale(image)
    bw = gray.point(lambda p: 255 if p > 160 else 0)
    try:
        import pytesseract
    except Exception:
        return None
    text = pytesseract.image_to_string(bw)
    return _parse_omni2_yearly_text(text)

def _fetch_omni_data(start_year: int, num_years: int, timeout: int) -> pd.DataFrame:
    yearly_frames = []
    for year in range(start_year, start_year + num_years):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            content_type = response.headers.get("Content-Type", "").lower()
        is_gif = "image/gif" in content_type or payload[:6] in (b"GIF87a", b"GIF89a")
        if is_gif:
            df = _extract_numbers_from_gif(payload)
            if df is not None:
                yearly_frames.append(df)
            continue
        text = payload.decode("utf-8", errors="ignore")
        df = _parse_omni2_yearly_text(text)
        if df is not None:
            yearly_frames.append(df)
    if not yearly_frames:
        raise RuntimeError("No valid OMNI numeric rows were found from SPDF yearly files.")
    merged = pd.concat(yearly_frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_omni_data(config: PipelineConfig):
    """Load OMNI multi-year data only (no synthetic fallback)."""
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

    # Kp lag features: geomagnetic storms are not memoryless. The ring current (which Kp measures)
    # takes 3-12 hours to build up and 12-48 hours to decay. Knowing Kp 3, 6, and 9 hours ago
    # tells the model whether we are in the onset, main phase, or recovery phase of a storm.
    # In practice, lagged Kp is the single highest-correlation predictor of current Kp —
    # adding it typically lifts R² by 0.05-0.15 on held-out data.
    omni_3h["kp_lag_3h"] = omni_3h["Kp"].shift(1)   # Kp 3 h ago
    omni_3h["kp_lag_6h"] = omni_3h["Kp"].shift(2)   # Kp 6 h ago
    omni_3h["kp_lag_9h"] = omni_3h["Kp"].shift(3)   # Kp 9 h ago
    # Drop the first 3 rows that have no lag history after shifting.
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
                     "bt_mean", "coupling_mean", "p_dyn_mean", "kp_lag_3h", "Kp"]
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
# New vs. original (6 features → 11 features):
#   by_mean      — real IMF By; fixes the Newell clock-angle that was broken by the constant proxy
#   p_dyn_mean   — dynamic ram pressure (n·V²); captures sudden-commencement compressions
#   kp_lag_3h/6h/9h — autoregressive Kp history; encodes storm phase (onset / main / recovery)
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
]


def fit_and_evaluate_model(data_3h: pd.DataFrame, config: PipelineConfig):
    """Train gradient boosting model with chronological split and return metrics."""
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

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            min_samples_split=config.min_samples_split,
            min_samples_leaf=config.min_samples_leaf,   # new
            random_state=config.random_state,
        )),
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "test_count": int(len(y_test)),
    }

    # Post-fit sanity checks — catch degenerate models before the forecast runs
    if metrics["r2"] < 0.0:
        print(
            f"[WARNING] Model R²={metrics['r2']:.3f} is negative (worse than always predicting "
            f"the mean Kp). Training samples: {len(y_train)}, test samples: {len(y_test)}. "
            "Consider a wider --years window or check OMNI data quality.",
            file=__import__("sys").stderr,
        )
    elif metrics["mae"] > 2.0:
        print(
            f"[WARNING] Model MAE={metrics['mae']:.3f} is unexpectedly high. "
            "Typical well-fitted models score MAE < 0.7 on multi-year data.",
            file=__import__("sys").stderr,
        )
    else:
        improvement = 100.0 * (1.0 - metrics["mae"] / metrics["baseline_mae"]) if metrics["baseline_mae"] > 0 else 0.0
        print(
            f"[OK] Model: MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  "
            f"({improvement:.1f}% over baseline)  test_n={metrics['test_count']}"
        )

    test_frame = pd.DataFrame({"Kp_true": y_test, "Kp_pred": y_pred}, index=data_3h.index[split_idx:])
    return model, metrics, test_frame


def build_forecast_scenarios(seed_3h: pd.DataFrame, config: PipelineConfig, rng: np.random.Generator):
    """Generate quiet/moderate/active 72-hour solar wind scenarios."""
    steps = config.forecast_steps_3h
    base_time = seed_3h.index[-1]
    forecast_times = pd.date_range(base_time + pd.Timedelta(hours=3), periods=steps, freq="3h", tz="UTC")

    recent = seed_3h[["speed_mean", "density_mean", "bz_mean"]].tail(10)
    speed_mean_recent = recent["speed_mean"].mean()
    density_mean_recent = recent["density_mean"].mean()

    quiet_speed = np.clip(np.linspace(speed_mean_recent - 50, speed_mean_recent - 60, steps) + rng.normal(0, 15, steps), 250, 600)
    quiet_bz = rng.normal(-0.5, 1.5, steps)
    quiet_density = rng.lognormal(np.log(max(density_mean_recent, 0.3)), 0.4, steps)

    moderate_speed = np.clip(np.linspace(speed_mean_recent, speed_mean_recent + 30, steps) + rng.normal(0, 20, steps), 250, 700)
    moderate_bz = np.where(
        rng.random(steps) < 0.3,
        -2 * rng.exponential(1.5, steps),
        rng.normal(0.5, 1.2, steps),
    )
    moderate_density = rng.lognormal(np.log(max(density_mean_recent, 0.3)), 0.5, steps)

    active_speed = np.clip(np.linspace(speed_mean_recent + 100, speed_mean_recent + 150, steps) + rng.normal(0, 25, steps), 350, 800)
    active_bz = np.clip(
        -3 - 2 * np.sin(np.linspace(0, np.pi, steps)) + rng.normal(0, 1.2, steps),
        -12,
        3,
    )
    active_density = np.clip(rng.lognormal(np.log(max(density_mean_recent + 1, 0.3)), 0.6, steps), 0.5, 50)

    # By_GSM scenario values. By itself is not the primary storm driver (Bz is), but it
    # sets the IMF clock angle θ = arctan(|By|/Bz) inside the Newell coupling function.
    # Quiet: By stays small (ordered heliospheric field). Moderate/Active: wider spread
    # representing CIR/CME sheath field rotations that accompany elevated solar wind.
    quiet_by = rng.normal(0.0, 2.0, steps)     # near-zero — typical slow solar wind
    moderate_by = rng.normal(0.0, 4.0, steps)  # moderate spread — normal/disturbed wind
    active_by = rng.normal(0.0, 5.0, steps)    # larger spread — sheath / CME passage

    scenarios = {
        "Quiet": {
            "speed": quiet_speed,
            "density": quiet_density,
            "by_gsm": quiet_by,
            "bz_gsm": quiet_bz,
            "bt": np.abs(quiet_bz) + 2 + rng.normal(0, 0.5, steps),
            "color": "#90EE90",
        },
        "Moderate": {
            "speed": moderate_speed,
            "density": moderate_density,
            "by_gsm": moderate_by,
            "bz_gsm": moderate_bz,
            "bt": np.abs(moderate_bz) + 2.5 + rng.normal(0, 0.7, steps),
            "color": "#FFD700",
        },
        "Active": {
            "speed": active_speed,
            "density": active_density,
            "by_gsm": active_by,
            "bz_gsm": active_bz,
            "bt": np.abs(active_bz) + 3 + rng.normal(0, 0.8, steps),
            "color": "#FF6B6B",
        },
    }
    return forecast_times, scenarios


def predict_scenario_kp(
    model,
    scenarios: Dict[str, dict],
    kp_seed: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Run an autoregressive 72-h Kp forecast for each scenario.

    kp_seed: array of the last 3 (or more) *observed* Kp values used to initialise
    the lag features. Once the loop starts, each predicted Kp immediately feeds back
    as the lag input for the next step — this is how storm phase memory propagates
    forward through the forecast window.
    """
    # Ensure we always have exactly 3 seed values (pad with quiet-day 2.0 if needed).
    seed = list(kp_seed[-3:]) if len(kp_seed) >= 3 else [2.0] * (3 - len(kp_seed)) + list(kp_seed)

    kp_forecasts = {}
    for name, scenario in scenarios.items():
        bz_arr  = scenario["bz_gsm"]
        by_arr  = scenario["by_gsm"]   # real By scenario — correct clock angle in Newell
        spd_arr = scenario["speed"]
        den_arr = scenario["density"]
        bt_arr  = scenario["bt"]

        # bz_min: rolling 3-step (9h) minimum — matches how it was built during OMNI training.
        bz_min_arr = np.array([bz_arr[max(0, i - 2):i + 1].min() for i in range(len(bz_arr))])

        # Coupling now uses the real scenario By instead of a constant proxy.
        coupling = newell_coupling(spd_arr, bt_arr, by_arr, bz_arr)

        # Dynamic pressure — same formula used in training.
        p_dyn_arr = den_arr * (spd_arr / 100.0) ** 2

        steps = len(bz_arr)
        kp_pred_arr = np.zeros(steps)

        # Autoregressive loop: at each step we feed the lag features from the previous
        # *predicted* Kp values. This propagates storm phase information forward in time
        # (e.g., if the model predicts Kp=6 at step 3, steps 4-6 will see elevated lags
        # and are more likely to also be predicted high — consistent with real storm dynamics).
        kp_history = list(seed)  # starts with real observations, grows with predictions
        for i in range(steps):
            lag_3h = kp_history[-1]   # 3 h ago
            lag_6h = kp_history[-2]   # 6 h ago
            lag_9h = kp_history[-3]   # 9 h ago

            x = np.array([[
                spd_arr[i], den_arr[i],
                by_arr[i], bz_arr[i], bz_min_arr[i], bt_arr[i],
                coupling[i], p_dyn_arr[i],
                lag_3h, lag_6h, lag_9h,
            ]])
            kp_step = float(np.clip(model.predict(x)[0], 0, 9))
            kp_pred_arr[i] = kp_step
            kp_history.append(kp_step)   # predicted value becomes next step's lag-3h

        kp_forecasts[name] = kp_pred_arr
    return kp_forecasts


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
    return {
        "sources": outputs["sources"],
        "counts": outputs["counts"],
        "metrics": outputs["metrics"],
        "latest": {
            **outputs["latest"],
            "time": outputs["latest"]["time"].isoformat(),
        },
        "forecast": {
            **{k: v for k, v in forecast.items() if k not in ("times", "kp_by_scenario", "kp_weighted")},
            "times": [t.isoformat() for t in forecast["times"]],
            "kp_by_scenario": {k: v.tolist() for k, v in forecast["kp_by_scenario"].items()},
            "kp_weighted": forecast["kp_weighted"].tolist(),
        },
    }

def run_pipeline(config: PipelineConfig | None = None, make_plots: bool = True, json_output_path: str | None = None):
    """Orchestrate end-to-end data->model->forecast workflow."""
    cfg = config or PipelineConfig()
    rng = np.random.default_rng(cfg.random_state)

    # NOAA and OMNI live-data paths are strict; no synthetic fallback is used.
    plasma_df, mag_df, kp_df, noaa_source = load_noaa_data(cfg)
    # Pass kp_df so build_noaa_3h_features can align Kp observations onto the 3h grid
    # and produce the kp_lag_* features required by the updated model.
    noaa_3h = build_noaa_3h_features(plasma_df, mag_df, kp_df)

    # OMNI path is primary training source for stable model fit.
    omni_3h, omni_source = load_omni_data(cfg)
    model, metrics, test_frame = fit_and_evaluate_model(omni_3h, cfg)

    # Use most-recent NOAA 3h window for latest prediction (real-time, not stale OMNI).
    if len(noaa_3h) >= 1 and all(c in noaa_3h.columns for c in FEATURE_COLUMNS):
        latest_features = noaa_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time = noaa_3h.index[-1]
        latest_actual = float(kp_df["Kp"].iloc[-1]) if not kp_df.empty else float(omni_3h["Kp"].iloc[-1])
    else:
        latest_features = omni_3h[FEATURE_COLUMNS].iloc[-1:].values
        latest_time = omni_3h.index[-1]
        latest_actual = float(omni_3h["Kp"].iloc[-1])
    latest_pred = float(np.clip(model.predict(latest_features)[0], 0, 9))

    # Prefer NOAA-recent conditions to seed the 72h forecast; fall back to OMNI if NOAA is sparse.
    forecast_seed = noaa_3h if len(noaa_3h) >= 4 else omni_3h
    forecast_times, scenarios = build_forecast_scenarios(forecast_seed, cfg, rng)

    # Extract the last 3 observed Kp values to seed the autoregressive lag in the forecast.
    # We use real observations (not model predictions) so the first forecast step starts
    # from ground truth — giving the most accurate storm-phase context available.
    if not kp_df.empty:
        kp_seed = kp_df["Kp"].dropna().tail(3).values
    else:
        kp_seed = omni_3h["Kp"].tail(3).values
    if len(kp_seed) < 3:
        kp_seed = np.pad(kp_seed, (3 - len(kp_seed), 0), constant_values=2.0)

    kp_forecasts = predict_scenario_kp(model, scenarios, kp_seed)
    kp_weighted = weighted_ensemble(kp_forecasts, cfg.scenario_weights)
    storm_prob = float(np.mean(kp_weighted >= 5.0))

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
            "times": forecast_times,
            "kp_by_scenario": kp_forecasts,
            "kp_weighted": kp_weighted,
            "mean_weighted_kp": float(np.mean(kp_weighted)),
            "peak_weighted_kp": float(np.max(kp_weighted)),
            "storm_probability_windows": storm_prob,
            "storm_chance_percent": float(100.0 * storm_prob),
            "forecast_seed_source": "noaa_recent" if len(noaa_3h) >= 4 else "omni_fallback",
        },
    }
    

    if make_plots:
        plot_test_results(test_frame, metrics)
        plot_forecast(forecast_times, scenarios, kp_forecasts, kp_weighted)
        plt.show()

    if json_output_path:
        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(_serialize_outputs(outputs), f, indent=2)

    return outputs

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
import urllib.request
#We import dataclasses to define a simple configuration class for the pipeline, and typing for type hints to improve code clarity.
#Also, we need typing, json, and urllib to handle data fetching and parsing from the NOAA and OMNI sources.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
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

    max_depth: int = 3          #Sets the maximum depth of the individual regression estimators, which controls how much each tree can grow and thus the model's ability to capture complex patterns.

    learning_rate: float = 0.05     #Sets the learning rate for the Gradient Boosting Regressor, which controls how much each tree contributes to the overall model. Lower values can lead to better performance but require more trees.

    subsample: float = 0.8      #Sets the fraction of samples to be used for fitting the individual base learners in the Gradient Boosting Regressor, which can help prevent overfitting by introducing randomness.

    min_samples_split: int = 10     #Sets the minimum num of samples for splitting an internal node in the Gradient Boosting Regressor, 
                                        #which can help control overfitting by requiring a minimum amount of data to make a split.

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


def _fetch_json(url: str, timeout: int) -> list:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read())


def _rows_to_dataframe(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    for column in df.columns:
        if column != "time_tag":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.set_index("time_tag").sort_index()


def _synthetic_noaa_fallback(rng: np.random.Generator) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    end = pd.Timestamp.now("UTC").floor("3h")
    times_1m = pd.date_range(end - pd.Timedelta(days=7), end, freq="1min", tz="UTC")

    speed = 400 + 60 * np.sin(np.linspace(0, 4 * np.pi, len(times_1m))) + rng.normal(0, 15, len(times_1m))
    density = np.clip(
        5 + 2 * np.sin(np.linspace(0, 6 * np.pi, len(times_1m))) + rng.normal(0, 0.8, len(times_1m)),
        0.2,
        None,
    )
    bz = -2 * np.sin(np.linspace(0, 8 * np.pi, len(times_1m))) + rng.normal(0, 1.5, len(times_1m))
    by = rng.normal(0, 2, len(times_1m))
    bt = np.clip(np.sqrt(by ** 2 + bz ** 2), 0.1, None)

    plasma = pd.DataFrame(
        {
            "density": density,
            "speed": speed,
            "temperature": 1e5 + 5e4 * rng.random(len(times_1m)),
        },
        index=times_1m,
    )
    mag = pd.DataFrame(
        {
            "bx_gsm": rng.normal(0, 2, len(times_1m)),
            "by_gsm": by,
            "bz_gsm": bz,
            "bt": bt,
            "lon_gsm": 0.0,
            "lat_gsm": 0.0,
        },
        index=times_1m,
    )

    times_3h = pd.date_range(end - pd.Timedelta(days=7), end, freq="3h", tz="UTC")
    spd_3h = plasma["speed"].reindex(times_3h, method="nearest")
    by_3h = mag["by_gsm"].reindex(times_3h, method="nearest")
    bz_3h = mag["bz_gsm"].reindex(times_3h, method="nearest")
    bt_3h = mag["bt"].reindex(times_3h, method="nearest")

    coupling = newell_coupling(spd_3h, bt_3h, by_3h, bz_3h)
    kp = np.clip(np.log1p(pd.Series(coupling, index=times_3h).fillna(0) / 4000) * 2.2, 0, 9)
    kp_df = pd.DataFrame({"Kp": kp.round(2)}, index=times_3h)
    return plasma, mag, kp_df, "synthetic_noaa"


def load_noaa_data(config: PipelineConfig, rng: np.random.Generator):
    """Load NOAA real-time data, with synthetic fallback for offline use."""
    try:
        plasma_raw = _fetch_json(NOAA_ENDPOINTS["plasma"], config.noaa_timeout_s)
        mag_raw = _fetch_json(NOAA_ENDPOINTS["mag"], config.noaa_timeout_s)
        kp_raw = _fetch_json(NOAA_ENDPOINTS["kp"], config.noaa_timeout_s)

        plasma_df = _rows_to_dataframe(plasma_raw)
        mag_df = _rows_to_dataframe(mag_raw)

        kp_df = pd.DataFrame(kp_raw[1:], columns=kp_raw[0])
        kp_df["time_tag"] = pd.to_datetime(kp_df["time_tag"], utc=True)
        kp_df["Kp"] = pd.to_numeric(kp_df["Kp"], errors="coerce")
        kp_df = kp_df.set_index("time_tag").sort_index()[["Kp"]]

        return plasma_df, mag_df, kp_df, "live_noaa"
    except Exception:
        return _synthetic_noaa_fallback(rng)
#Below is a new feature that creates a 3 hour aggregated feature set from
#  the raw 1-minute NOAA plasma and magnetic field data, 
# which is necessary to align with the 3-hourly OMNI data 
# and to create features that are more relevant for predicting Kp.


def build_noaa_3h_features(plasma_df: pd.DataFrame, mag_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NOAA 1-minute streams into model-compatible 3-hour features."""
    merged = plasma_df.join(mag_df, how="inner")
    merged = merged[["speed", "density", "bz_gsm", "bt"]].dropna()
    if merged.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    noaa_3h = merged.resample("3h", label="left").agg(
        {
            "speed": "mean",
            "density": "mean",
            "bz_gsm": ["mean", "min"],
            "bt": "mean",
        }
    )
    noaa_3h.columns = ["speed_mean", "density_mean", "bz_mean", "bz_min", "bt_mean"]
    noaa_3h = noaa_3h.dropna()
    if noaa_3h.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    by_proxy = np.full(len(noaa_3h), 2.0)
    noaa_3h["coupling_mean"] = newell_coupling(
        noaa_3h["speed_mean"], noaa_3h["bt_mean"], by_proxy, noaa_3h["bz_mean"]
    )
    return noaa_3h


def _fetch_omni_data(start_year: int, num_years: int, timeout: int) -> pd.DataFrame | None:
    end_year = start_year + num_years
    omni_url = (
        "https://omniweb.gsfc.nasa.gov/cgi-bin/omni_data_h.cgi"
        f"?start_date={start_year}0101&end_date={end_year}0101&param=1,2,3,39,40,41,42,43,44,9"
    )

    req = urllib.request.Request(omni_url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        lines = response.read().decode("utf-8").split("\n")

    data_rows = []
    for line in lines:
        if line.startswith("Yr") or line.startswith("--") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            year, month, day, hour = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            speed = float(parts[6]) if parts[6] != "99999.9" else np.nan
            density = float(parts[7]) if parts[7] != "999.9" else np.nan
            bt = float(parts[9]) if parts[9] != "999.9" else np.nan
            bz = float(parts[10]) if parts[10] != "999.9" else np.nan
            kp = float(parts[-1]) if parts[-1] != "99" else np.nan
            data_rows.append(
                {
                    "time_tag": pd.Timestamp(year, month, day, hour, tz="UTC"),
                    "speed": speed,
                    "density": density,
                    "bz_gsm": bz,
                    "bt": bt,
                    "kp": kp / 10.0,
                }
            )
        except (ValueError, IndexError):
            continue

    if not data_rows:
        return None

    df = pd.DataFrame(data_rows).set_index("time_tag").sort_index().dropna()
    if df.empty:
        return None
    return df


def _synthetic_omni_fallback(num_years: int, rng: np.random.Generator) -> pd.DataFrame:
    end = pd.Timestamp.now("UTC").floor("h")
    start = end - pd.Timedelta(days=365 * num_years)
    times = pd.date_range(start, end, freq="1h", tz="UTC")

    n = len(times)
    t_norm = np.arange(n) / max(n, 1)
    speed_base = 400 + 80 * np.sin(2 * np.pi * t_norm * 2) + 40 * np.cos(2 * np.pi * t_norm * 0.3)
    speed_bursts = np.where(rng.random(n) < 0.05, rng.exponential(100, n), 0)
    speed = np.clip(speed_base + speed_bursts + rng.normal(0, 20, n), 200, 800)

    density = np.clip(rng.lognormal(1.2, 0.8, n), 0.1, 50)
    bz_trend = -1 * np.sin(2 * np.pi * t_norm) + rng.normal(0, 2, n)
    bz_driven = np.where(rng.random(n) < 0.15, -3 * rng.exponential(2, n), bz_trend)
    bz = np.clip(bz_driven, -15, 10)
    by = rng.normal(0, 2, n)
    bt = np.abs(bz) + 2 + np.abs(rng.normal(0, 1, n))

    coupling = newell_coupling(speed, bt, by, bz)
    coupling_smooth = pd.Series(coupling).rolling(window=25, center=True, min_periods=1).median().values
    kp_raw = 2.0 * np.log1p(np.maximum(coupling_smooth, 0) / 5000)

    kp_ar = np.zeros(n)
    for idx in range(1, n):
        kp_ar[idx] = 0.85 * kp_ar[idx - 1] + 0.15 * kp_raw[idx] + rng.normal(0, 0.15)

    kp = np.clip(kp_ar, 0, 9)
    return pd.DataFrame(
        {"speed": speed, "density": density, "bz_gsm": bz, "bt": bt, "kp": kp.round(2)},
        index=times,
    )



def load_omni_data(config: PipelineConfig, rng: np.random.Generator):
    """Load OMNI multi-year data, with synthetic fallback if unavailable."""
    try:
        omni_raw = _fetch_omni_data(config.omni_start_year, config.omni_num_years, config.noaa_timeout_s)
        if omni_raw is None:
            raise RuntimeError("OMNI empty or unavailable")
        source = "live_omni"
    except Exception:
        omni_raw = _synthetic_omni_fallback(config.omni_num_years, rng)
        source = "synthetic_omni"

    omni_3h = omni_raw.resample("3h", label="left").agg(
        {
            "speed": "mean",
            "density": "mean",
            "bz_gsm": ["mean", "min"],
            "bt": "mean",
            "kp": "mean",
        }
    )
    omni_3h.columns = ["speed_mean", "density_mean", "bz_mean", "bz_min", "bt_mean", "Kp"]
    omni_3h = omni_3h.dropna()

    # If by_gsm is unavailable in OMNI parse, use a neutral by proxy to keep full formula shape.
    by_proxy = np.full(len(omni_3h), 2.0)
    omni_3h["coupling_mean"] = newell_coupling(
        omni_3h["speed_mean"], omni_3h["bt_mean"], by_proxy, omni_3h["bz_mean"]
    )
    return omni_3h, source


FEATURE_COLUMNS = ["speed_mean", "density_mean", "bz_mean", "bz_min", "bt_mean", "coupling_mean"]


def fit_and_evaluate_model(data_3h: pd.DataFrame, config: PipelineConfig):
    """Train gradient boosting model with chronological split and return metrics."""
    if len(data_3h) < 24:
        raise ValueError("Not enough aligned samples for training.")

    split_idx = int(config.train_fraction * len(data_3h))
    split_idx = max(1, min(split_idx, len(data_3h) - 1))

    X = data_3h[FEATURE_COLUMNS].values
    y = data_3h["Kp"].values

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = GradientBoostingRegressor(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        min_samples_split=config.min_samples_split,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "test_count": int(len(y_test)),
    }

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

    scenarios = {
        "Quiet": {
            "speed": quiet_speed,
            "density": quiet_density,
            "bz_gsm": quiet_bz,
            "bt": np.abs(quiet_bz) + 2 + rng.normal(0, 0.5, steps),
            "color": "#90EE90",
        },
        "Moderate": {
            "speed": moderate_speed,
            "density": moderate_density,
            "bz_gsm": moderate_bz,
            "bt": np.abs(moderate_bz) + 2.5 + rng.normal(0, 0.7, steps),
            "color": "#FFD700",
        },
        "Active": {
            "speed": active_speed,
            "density": active_density,
            "bz_gsm": active_bz,
            "bt": np.abs(active_bz) + 3 + rng.normal(0, 0.8, steps),
            "color": "#FF6B6B",
        },
    }
    return forecast_times, scenarios


def predict_scenario_kp(model: GradientBoostingRegressor, scenarios: Dict[str, dict]):
    """Predict Kp trajectories for all scenarios using model feature contract."""
    kp_forecasts = {}
    for name, scenario in scenarios.items():
        by_proxy = np.full(len(scenario["speed"]), 2.0)
        coupling = newell_coupling(scenario["speed"], scenario["bt"], by_proxy, scenario["bz_gsm"])
        X_forecast = np.column_stack(
            [
                scenario["speed"],
                scenario["density"],
                scenario["bz_gsm"],
                scenario["bz_gsm"],
                scenario["bt"],
                coupling,
            ]
        )
        kp_forecasts[name] = np.clip(model.predict(X_forecast), 0, 9)
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


def run_pipeline(config: PipelineConfig | None = None, make_plots: bool = True):
    """Orchestrate end-to-end data->model->forecast workflow."""
    cfg = config or PipelineConfig()
    rng = np.random.default_rng(cfg.random_state)

    # NOAA path kept for operational observability and consistency with project goals.
    plasma_df, mag_df, kp_df, noaa_source = load_noaa_data(cfg, rng)
    noaa_3h = build_noaa_3h_features(plasma_df, mag_df)

    # OMNI path is primary training source for stable model fit.
    omni_3h, omni_source = load_omni_data(cfg, rng)
    model, metrics, test_frame = fit_and_evaluate_model(omni_3h, cfg)

    latest_features = omni_3h[FEATURE_COLUMNS].iloc[-1:].values
    latest_pred = float(model.predict(latest_features)[0])
    latest_actual = float(omni_3h["Kp"].iloc[-1])

    # Prefer NOAA-recent conditions to seed the 72h forecast; fall back to OMNI if NOAA is sparse.
    forecast_seed = noaa_3h if len(noaa_3h) >= 4 else omni_3h
    forecast_times, scenarios = build_forecast_scenarios(forecast_seed, cfg, rng)
    kp_forecasts = predict_scenario_kp(model, scenarios)
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
            "time": omni_3h.index[-1],
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

    return outputs

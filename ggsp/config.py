# ggsp/config.py
#
# This is the SHARED FOUNDATION of the whole pipeline.
# Every other module in this package imports from here — so think of it as
# the single source of truth for "what does the pipeline know about itself?"
#
# Why a frozen dataclass for PipelineConfig?
#   A frozen dataclass means the config can't be accidentally mutated mid-run.
#   Once you build a PipelineConfig with your CLI flags, it stays constant all
#   the way through data fetch → training → forecast → export.  This makes the
#   pipeline reproducible: same config always produces same results (modulo live
#   NOAA data, which changes every minute).
#
# Three categories of things live here:
#   1. PipelineConfig  — all tunable knobs (hyperparams, data window, CV settings)
#   2. FEATURE_COLUMNS — the ordered list of ML inputs (a contract between all X-builders)
#   3. Cache paths     — where model.joblib / omni parquet files live on disk

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict

# _REPO_DIR points to the SUN/ workspace root — one level above this file (ggsp/).
# All cache paths are anchored here so they work regardless of which directory
# you launch Python from.
_REPO_DIR        = pathlib.Path(__file__).parent.parent
OMNI_CACHE_DIR   = _REPO_DIR / "cache" / "omni"
MODEL_CACHE_PATH = _REPO_DIR / "cache" / "model.joblib"
MODEL_META_PATH  = _REPO_DIR / "cache" / "model_meta.json"

# User-Agent header: some servers block requests with no user-agent string.
# We identify ourselves as an educational tool so we're not mistaken for a bot.
HEADERS = {"User-Agent": "Mozilla/5.0 (geomag-storm-predictor; educational)"}


# ── PipelineConfig ─────────────────────────────────────────────────────────────
#
# All pipeline parameters in one place.  frozen=True prevents accidental mutation
# after construction — treat this object as read-only once you've built it.
#
# Student note: if you want to experiment with different model hyperparameters,
# change the defaults here OR pass keyword arguments when constructing:
#   cfg = PipelineConfig(n_estimators=300, max_depth=4)
# Never hard-code numbers inside functions — always go through config.

@dataclass(frozen=True)
class PipelineConfig:

    # ── Data source settings ───────────────────────────────────────────────────
    noaa_timeout_s: int = 20
    # Timeout (seconds) for NOAA live JSON fetches.  20 s is generous for the
    # small JSON payloads (~few KB); increase only if you're on a very slow link.

    omni_start_year: int = 2020
    # First calendar year of OMNI training data to fetch from SPDF.
    # OMNI2 coverage starts in 1963.  Longer windows = more storm events = better
    # G1 CSI statistics, but also slower training and larger cache.

    omni_num_years: int = 5
    # How many consecutive years to load starting at omni_start_year.
    # Rule of thumb: need ≥8 years for statistically reliable G1-CSI estimates
    # (≥200 storm events in the 25% test split).  5 years is the fast-iteration default.

    train_fraction: float = 0.75
    # Chronological train/test split.  0.75 means the first 75% of OMNI rows
    # train the model and the last 25% are the held-out evaluation window.
    # Never shuffle — time series data MUST be split in time order to avoid leakage.

    # ── GBR hyperparameters ────────────────────────────────────────────────────
    n_estimators: int = 200
    # Number of boosting trees.  More = slower but potentially more accurate.
    # CV folds use min(n_estimators, 100) to keep wall time reasonable.

    max_depth: int = 3
    # Max depth of each individual tree.  Depth 3 → 8 leaf nodes per tree.
    # This lets the model capture three-way interactions (e.g., high speed AND
    # negative Bz AND already elevated Kp) without the overfitting that depth 4+
    # would bring on smaller training windows.

    learning_rate: float = 0.05
    # Shrinkage factor applied to each new tree's contribution.  Lower values
    # slow learning but generally produce better generalisation — compensate by
    # increasing n_estimators if you lower this.

    subsample: float = 0.8
    # Fraction of training rows used to fit each tree (stochastic gradient
    # boosting).  0.8 introduces useful variance reduction without sacrificing
    # too much of the training signal.

    min_samples_split: int = 20
    # A node must have at least this many samples to be eligible for splitting.
    # Higher values make trees shallower and more conservative — good regulariser
    # when the training set is small.

    min_samples_leaf: int = 5
    # Each leaf must contain at least this many samples.
    # Prevents the model from memorising tiny clusters in the training data.

    random_state: int = 42
    # Seeds numpy and sklearn random number generators for reproducibility.
    # Change this to check that results are not overly seed-sensitive.

    # ── Forecast settings ──────────────────────────────────────────────────────
    forecast_steps_3h: int = 24
    # Number of 3-hour autoregressive steps in the forecast window.
    # 24 steps × 3 h = 72 hours — matches the standard NOAA storm warning horizon.

    scenario_weights: Dict[str, float] = None
    # Prior probabilities for the Quiet / Moderate / Active solar-wind scenarios.
    # These are overridden by sunspot_pipeline when --static-weights is NOT set.
    # None triggers the __post_init__ default of {Quiet:0.2, Moderate:0.5, Active:0.3}.

    # ── Cross-validation settings ──────────────────────────────────────────────
    use_cv: bool = True
    # When True, model.py runs 5-fold TimeSeriesSplit CV and reports mean±std of
    # MAE / R² / G1-CSI across folds.  Set False to skip CV (saves ~30 s on 35yr data).

    n_cv_folds: int = 5
    # Number of expanding-window folds.  Each fold trains on all data up to a
    # cutpoint and tests on the next segment.  gap=3 rows (9h) prevents the
    # autoregressive Kp lags from leaking training-set Kp into the first test row.

    def __post_init__(self):
        # Provide the default scenario weights if none were given.
        # We use object.__setattr__ because the dataclass is frozen — this is the
        # recommended pattern for setting mutable defaults on frozen dataclasses.
        if self.scenario_weights is None:
            object.__setattr__(self, "scenario_weights", {"Quiet": 0.2, "Moderate": 0.5, "Active": 0.3})


# ── FEATURE_COLUMNS ─────────────────────────────────────────────────────────────
#
# This list IS A CONTRACT between three separate modules:
#   omni_client.py  — builds the training DataFrame with exactly these columns
#   features.py     — builds the real-time inference frame with exactly these columns
#   forecast.py     — builds X_batch with exactly these columns (in this order)
#   model.py        — asserts X.shape[1] == _EXPECTED_N_FEATURES at train + predict
#
# If you add or remove a feature, update this list AND every module that builds
# an X array.  The _EXPECTED_N_FEATURES assertion will catch any mismatch loudly
# at runtime so you never silently predict with the wrong feature set.
#
# Why these 14?  (Added in v7 → v8 refactor, see eval/REPORT.md for impact)
#   speed_mean / density_mean / bz_mean / bz_min / bt_mean
#       Core solar wind parameters.  bz_min captures the worst southward Bz in
#       the 9h window — critical for storm onset because a brief dip triggers
#       ring current injection even if the mean Bz looks benign.
#   by_mean
#       Real IMF By replaces the old constant 2.0 nT proxy.  Fixes the Newell
#       clock-angle θ = arctan(|By|/Bz) which was systematically wrong before.
#   coupling_mean
#       Newell et al. 2007: dΦ/dt ∝ V^(4/3) · Bt^(2/3) · sin^(8/3)(θ/2).
#       The most physically grounded predictor of magnetospheric energy input.
#   p_dyn_mean
#       Dynamic ram pressure n·V².  Sudden pressure pulses compress the
#       magnetosphere (sudden commencement) before the CME's southward Bz arrives.
#   kp_lag_3h / kp_lag_6h / kp_lag_9h
#       Autoregressive Kp memory.  Encodes storm phase: rising/peak/decay.
#       Without these, the model can't distinguish whether a high Kp is new or
#       the tail of a storm that started 9 hours ago.
#   kp_lag_27d
#       Solar-rotation recurrence: active regions persist 2–3 rotations (~27d).
#       Knowing Kp 27 days ago flags recurrent CIR/CME storm drivers.
#   vbs
#       Rectified southward electric field: V_sw × max(0, –Bz) / 1000.
#       Captures geoeffective energy transfer independently of Newell coupling.
#   equinox_term
#       Russell–McPherron semi-annual proxy: cos(4π·(doy–80)/365.25).
#       Storms are ~50% more frequent near equinoxes — this lets the model
#       learn that seasonal modulation without seeing the calendar date directly.

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
    "kp_lag_27d",    # solar-rotation recurrence (27d = 216 × 3h shifts)
    "vbs",           # rectified southward E-field
    "equinox_term",  # Russell-McPherron seasonal proxy
]

# Module-level constant checked by assertion in model.py and forecast.py.
# If you see an assertion error mentioning this constant, FEATURE_COLUMNS and
# an X-building function have gotten out of sync — fix FEATURE_COLUMNS first.
_EXPECTED_N_FEATURES = len(FEATURE_COLUMNS)  # 14

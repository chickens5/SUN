# ggsp/forecast.py
#
# STAGE 7 — Build scenario trajectories and compute storm probability.
#
# Three functions, three jobs:
#
#   build_forecast_scenarios()
#       Generates solar-wind parameter arrays (speed, density, Bz, By, Bt)
#       for three physical scenarios: Quiet, Moderate, Active.
#       Conditioned on the most-recent NOAA observations so the forecast
#       doesn't start from an arbitrary baseline.
#
#   predict_scenario_kp()
#       Runs a deterministic autoregressive Kp forecast for each scenario.
#       "Autoregressive" means each predicted Kp feeds back as the lag feature
#       for the next step — storm memory propagates forward in time.
#       Produces the smooth trajectory lines on the forecast plot.
#
#   predict_scenario_kp_ensemble()
#       Runs 500 stochastic realisations of each scenario (adding noise to the
#       base solar-wind arrays) and computes scenario-weighted P(Kp ≥ 5).
#       THIS is the storm probability number — not the deterministic trajectory.
#       The deterministic mean trajectory structurally suppresses P(storm) toward
#       zero because averaging damps extremes.  The ensemble avoids that bias.
#
# Statistical note on n_draws:
#   With P(storm) = 20%, SE = sqrt(0.2 × 0.8 / 500) ≈ 1.8% — well-resolved.
#   For rare events (P < 2%), 500 draws are not enough: SE ≈ 0.3% but the
#   absolute uncertainty on P dominates.  We warn when this happens so you
#   know to treat the low probability as an upper bound rather than a precise
#   estimate.  Increasing n_draws to 5000 would reduce SE to ~0.03% but takes
#   10× longer.
#
# Used by:  pipeline.py (Stage 7 of run_pipeline)
# Imports:  config.py, physics.py

from __future__ import annotations

import sys
from typing import Dict

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, PipelineConfig, _EXPECTED_N_FEATURES
from .physics import newell_coupling


# ── Scenario noise scales ─────────────────────────────────────────────────────
#
# These σ values define how much randomness gets added to each base scenario
# array in the stochastic ensemble.  They match the noise amplitudes used in
# build_forecast_scenarios() so each draw is a plausible realisation of the
# same distribution.
#
# Quiet has narrower distributions (solar wind is more predictable during
# calm conditions).  Active has wider distributions (CME arrival timing
# and Bz southward rotation are harder to predict).
_SCENARIO_NOISE: Dict[str, dict] = {
    "Quiet":    {"speed_std": 15.0, "bz_std": 1.5, "density_log_std": 0.4, "by_std": 2.0},
    "Moderate": {"speed_std": 20.0, "bz_std": 1.2, "density_log_std": 0.5, "by_std": 4.0},
    "Active":   {"speed_std": 25.0, "bz_std": 1.2, "density_log_std": 0.6, "by_std": 5.0},
}


def build_forecast_scenarios(
    seed_3h: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    omni_kp_hist: "pd.Series | None" = None,
):
    """STAGE 7a — Generate three 72-hour solar-wind scenario trajectories.

    Returns (forecast_times, scenarios, seed_source_str) where scenarios is:
      {"Quiet": {...arrays...}, "Moderate": {...arrays...}, "Active": {...arrays...}}
    Each scenario dict contains per-step 1D arrays for speed, density, bz_gsm,
    by_gsm, bt, and a plot colour string.

    Design choices (Task 5):
    1. Scenario MEANS are conditioned on the most-recent NOAA window (tail 10 rows)
       so the forecast starts from realistic current conditions, not a fixed climatology.
    2. The first 2 steps use a 80%/40% persistence anchor — blending the most-recent
       observed values into the scenario trajectory to avoid a sharp discontinuity.
    3. Recurrence boost in Moderate: if Kp was elevated 27 days ago (≥4), the moderate
       scenario shifts Bz more southward and speed slightly higher.  This reflects the
       CIR/CME recurring active-region pattern.
    """
    steps = config.forecast_steps_3h
    base_time = seed_3h.index[-1]
    forecast_times = pd.date_range(
        base_time + pd.Timedelta(hours=3),
        periods=steps, freq="3h", tz="UTC",
    )

    # Anchor scenario means to recent observations (tail 10 rows ≈ last 30 h).
    recent = seed_3h[["speed_mean", "density_mean", "bz_mean"]].tail(10)
    spd_r  = float(recent["speed_mean"].mean())
    den_r  = float(recent["density_mean"].mean())
    bz_r   = float(recent["bz_mean"].mean())

    # Most-recent single observation for the persistence anchor (steps 0 and 1).
    obs_spd = float(seed_3h["speed_mean"].iloc[-1])
    obs_den = float(seed_3h["density_mean"].iloc[-1])
    obs_bz  = float(seed_3h["bz_mean"].iloc[-1])

    # ── Quiet scenario ─────────────────────────────────────────────────────────
    quiet_speed   = np.clip(
        np.linspace(spd_r - 50, spd_r - 60, steps) + rng.normal(0, 15, steps), 250, 600
    )
    quiet_bz      = rng.normal(bz_r * 0.3, 1.5, steps)
    quiet_density = rng.lognormal(np.log(max(den_r, 0.3)), 0.4, steps)
    quiet_by      = rng.normal(0.0, 2.0, steps)

    # ── Moderate scenario ──────────────────────────────────────────────────────
    mod_speed = np.clip(
        np.linspace(spd_r, spd_r + 30, steps) + rng.normal(0, 20, steps), 250, 700
    )
    mod_bz_base = np.where(
        rng.random(steps) < 0.3,
        -2 * rng.exponential(1.5, steps),
        rng.normal(0.5, 1.2, steps),
    )
    mod_density = rng.lognormal(np.log(max(den_r, 0.3)), 0.5, steps)
    mod_by      = rng.normal(0.0, 4.0, steps)

    # Recurrence boost: look up OMNI Kp 27 days before each forecast step.
    # Scale from 0 (no boost at Kp_27d = 3) to 1 (full boost at Kp_27d = 7).
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
            recurrence_boost = float(np.clip(
                (float(np.mean(kp_27d_vals)) - 3.0) / 4.0, 0.0, 1.0
            ))

    mod_bz    = mod_bz_base - recurrence_boost * 2.5
    mod_speed = np.clip(mod_speed + recurrence_boost * 60, 250, 750)

    # ── Active scenario ────────────────────────────────────────────────────────
    act_speed   = np.clip(
        np.linspace(spd_r + 100, spd_r + 150, steps) + rng.normal(0, 25, steps), 350, 800
    )
    act_bz = np.clip(
        -3 - 2 * np.sin(np.linspace(0, np.pi, steps)) + rng.normal(0, 1.2, steps),
        -12, 3,
    )
    act_density = np.clip(
        rng.lognormal(np.log(max(den_r + 1, 0.3)), 0.6, steps), 0.5, 50
    )
    act_by = rng.normal(0.0, 5.0, steps)

    # ── Persistence anchor — first 2 steps ────────────────────────────────────
    # Step 0: 80% observed + 20% scenario.
    # Step 1: 40% observed + 60% scenario.
    # This prevents an abrupt jump from the last real data point into the forecast.
    for spd_arr, bz_arr, den_arr in (
        (quiet_speed, quiet_bz,  quiet_density),
        (mod_speed,   mod_bz,    mod_density),
        (act_speed,   act_bz,    act_density),
    ):
        spd_arr[0] = 0.8 * obs_spd + 0.2 * spd_arr[0]
        spd_arr[1] = 0.4 * obs_spd + 0.6 * spd_arr[1]
        bz_arr[0]  = 0.8 * obs_bz  + 0.2 * bz_arr[0]
        bz_arr[1]  = 0.4 * obs_bz  + 0.6 * bz_arr[1]
        den_arr[0] = 0.8 * obs_den + 0.2 * den_arr[0]
        den_arr[1] = 0.4 * obs_den + 0.6 * den_arr[1]

    scenarios = {
        "Quiet": {
            "speed":   quiet_speed,
            "density": quiet_density,
            "by_gsm":  quiet_by,
            "bz_gsm":  quiet_bz,
            "bt": np.abs(quiet_bz) + 2   + rng.normal(0, 0.5, steps),
            "color": "#90EE90",
        },
        "Moderate": {
            "speed":   mod_speed,
            "density": mod_density,
            "by_gsm":  mod_by,
            "bz_gsm":  mod_bz,
            "bt": np.abs(mod_bz) + 2.5  + rng.normal(0, 0.7, steps),
            "color": "#FFD700",
        },
        "Active": {
            "speed":   act_speed,
            "density": act_density,
            "by_gsm":  act_by,
            "bz_gsm":  act_bz,
            "bt": np.abs(act_bz) + 3    + rng.normal(0, 0.8, steps),
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

    Each step predicts Kp one 3h period ahead, then feeds that prediction back
    as the lag input for the next step.  This is how storm-phase memory (onset →
    main phase → recovery) propagates forward through the 72h window.

    For storm PROBABILITY, use predict_scenario_kp_ensemble() — the deterministic
    weighted mean trajectory suppresses extremes and gives an artificially low
    storm probability.

    Returns
    -------
    kp_forecasts : {"Quiet": array(24,), "Moderate": array(24,), "Active": array(24,)}
    """
    seed  = (list(kp_seed[-3:]) if len(kp_seed) >= 3
             else [2.0] * (3 - len(kp_seed)) + list(kp_seed))
    steps = len(next(iter(scenarios.values()))["speed"])

    # Pre-compute time-based features shared across all scenarios.
    if forecast_times is not None and len(forecast_times) == steps:
        _doy_arr = np.array([ft.day_of_year for ft in forecast_times], dtype=float)
    else:
        _doy_arr = np.full(steps, 80.0)   # equinox default if times not provided
    equinox_arr = np.cos(4.0 * np.pi * (_doy_arr - 80.0) / 365.25)

    # kp_lag_27d at each forecast step — sourced from OMNI history.
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

    kp_forecasts = {}
    for name, scenario in scenarios.items():
        bz_arr  = scenario["bz_gsm"]
        by_arr  = scenario["by_gsm"]
        spd_arr = scenario["speed"]
        den_arr = scenario["density"]
        bt_arr  = scenario["bt"]

        # bz_min: rolling 9h minimum (same definition as training).
        bz_min_arr = np.array([bz_arr[max(0, i - 2):i + 1].min() for i in range(len(bz_arr))])

        coupling  = newell_coupling(spd_arr, bt_arr, by_arr, bz_arr)
        p_dyn_arr = den_arr * (spd_arr / 100.0) ** 2
        vbs_arr   = spd_arr * np.maximum(0.0, -bz_arr) / 1000.0

        kp_pred_arr = np.zeros(steps)
        kp_history  = list(seed)

        for i in range(steps):
            lag_3h = kp_history[-1]
            lag_6h = kp_history[-2]
            lag_9h = kp_history[-3]

            x = np.array([[
                spd_arr[i],    den_arr[i],
                by_arr[i],     bz_arr[i],     bz_min_arr[i], bt_arr[i],
                coupling[i],   p_dyn_arr[i],
                lag_3h,        lag_6h,        lag_9h,
                kp_lag_27d_arr[i], vbs_arr[i], equinox_arr[i],
            ]])
            assert x.shape[1] == _EXPECTED_N_FEATURES, (
                f"predict_scenario_kp: x width {x.shape[1]} != {_EXPECTED_N_FEATURES}. "
                "FEATURE_COLUMNS and this function must stay in sync."
            )
            kp_step = float(np.clip(model.predict(x)[0], 0, 9))
            kp_pred_arr[i] = kp_step
            kp_history.append(kp_step)

        kp_forecasts[name] = kp_pred_arr
    return kp_forecasts


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
    """STAGE 7c — Stochastic ensemble for P(Kp ≥ G1) over the 72h window.

    Vectorised over n_draws so that each of the 24 time steps runs ONE
    sklearn predict call on an (n_draws × 14) batch — 24 calls total instead
    of the naive 500 × 24 = 12,000 calls.

    Returns
    -------
    storm_prob_per_window  : np.ndarray (24,) — P(Kp ≥ 5) at each 3h step
    storm_prob_72h         : float — P(at least one step has Kp ≥ 5)

    Statistical note on n_draws:
      P(storm) ≈ 20%  →  SE ≈ 1.8%  →  well-resolved with 500 draws.
      P(storm) < 2%   →  SE ≈ 0.3%  →  500 draws may be insufficient for
      a precise estimate — treat the result as an upper bound.  A warning
      is printed when this condition is detected.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    seed  = (list(kp_seed[-3:]) if len(kp_seed) >= 3
             else [2.0] * (3 - len(kp_seed)) + list(kp_seed))
    steps = len(next(iter(scenarios.values()))["speed"])

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

    # Normalise scenario weights to sum to 1.0.
    w_keys = [k for k in scenarios if k in weights]
    w_arr  = np.array([weights[k] for k in w_keys], dtype=float)
    w_arr /= w_arr.sum()

    prob_per_window = np.zeros(steps)
    prob_any_storm  = 0.0

    for wi, sname in enumerate(w_keys):
        sc        = scenarios[sname]
        noise_cfg = _SCENARIO_NOISE.get(sname, _SCENARIO_NOISE["Moderate"])

        # Pre-generate ALL noise at once: (n_draws × steps) arrays.
        # Looping over steps (24 iterations) instead of draws (500) keeps this fast.
        spd_all = np.clip(
            sc["speed"][np.newaxis, :] + rng.normal(0, noise_cfg["speed_std"], (n_draws, steps)),
            200, 900,
        )
        bz_all  = sc["bz_gsm"][np.newaxis, :] + rng.normal(0, noise_cfg["bz_std"], (n_draws, steps))
        den_all = np.clip(
            sc["density"][np.newaxis, :] * np.exp(
                rng.normal(0, noise_cfg["density_log_std"], (n_draws, steps))
            ),
            0.1, 100,
        )
        by_all = sc["by_gsm"][np.newaxis, :] + rng.normal(0, noise_cfg["by_std"], (n_draws, steps))
        # bt must be > 0 for bt^(2/3) in Newell coupling — clip near-zero draws.
        bt_all = np.maximum(
            np.abs(bz_all) + 2.0 + rng.normal(0, 0.6, (n_draws, steps)),
            0.1,
        )

        # Rolling 3-step bz_min across the steps axis.
        bz_min_all = np.stack(
            [np.min(bz_all[:, max(0, j - 2):j + 1], axis=1) for j in range(steps)],
            axis=1,
        )  # (n_draws, steps)

        coupling_all = np.nan_to_num(newell_coupling(spd_all, bt_all, by_all, bz_all), nan=0.0)
        p_dyn_all    = den_all * (spd_all / 100.0) ** 2
        vbs_all      = spd_all * np.maximum(0.0, -bz_all) / 1000.0

        # Autoregressive Kp history — one row per draw.
        # Column order: [kp_9h, kp_6h, kp_3h]  (oldest → most recent)
        kp_hist = np.tile(seed, (n_draws, 1))   # (n_draws, 3)
        draw_kp = np.zeros((n_draws, steps))

        for i in range(steps):
            X_batch = np.column_stack([
                spd_all[:, i],     den_all[:, i],
                by_all[:, i],      bz_all[:, i],
                bz_min_all[:, i],  bt_all[:, i],
                coupling_all[:, i], p_dyn_all[:, i],
                kp_hist[:, 2],     kp_hist[:, 1],  kp_hist[:, 0],
                np.full(n_draws, kp_lag_27d_arr[i]),
                vbs_all[:, i],
                np.full(n_draws, equinox_arr[i]),
            ])  # (n_draws, 14)
            X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
            kp_step = np.clip(model.predict(X_batch), 0.0, 9.0)  # (n_draws,)
            draw_kp[:, i] = kp_step
            # Shift autoregressive history left: oldest drops off, new step appended.
            kp_hist[:, 0] = kp_hist[:, 1]
            kp_hist[:, 1] = kp_hist[:, 2]
            kp_hist[:, 2] = kp_step

        storm_per_step = (draw_kp >= 5.0).mean(axis=0)              # (steps,)
        storm_any      = float((draw_kp.max(axis=1) >= 5.0).mean()) # scalar

        prob_per_window += w_arr[wi] * storm_per_step
        prob_any_storm  += w_arr[wi] * storm_any

    # ── Statistical adequacy warning ───────────────────────────────────────────
    # SE = sqrt(P × (1-P) / n_draws).  If P < 2%, even 500 draws give a relative
    # uncertainty that makes the estimate hard to trust.
    if prob_any_storm < 0.02 and prob_any_storm > 0.0:
        se = (prob_any_storm * (1 - prob_any_storm) / n_draws) ** 0.5
        print(
            f"[INFO] Very low storm probability ({100*prob_any_storm:.1f}%). "
            f"With n_draws={n_draws}, SE ≈ {100*se:.2f}%. "
            "Treat this as an upper bound rather than a precise estimate. "
            "Increase n_draws (e.g., 5000) for better resolution at low probabilities.",
            file=sys.stderr,
        )

    return {
        "storm_prob_per_window": prob_per_window,
        "storm_prob_72h":        float(prob_any_storm),
    }


def weighted_ensemble(
    kp_forecasts: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """Combine scenario forecasts into a single weighted best-estimate trajectory.

    This is the smooth line labelled "Weighted" on the forecast plot.
    It is NOT used for storm probability — see predict_scenario_kp_ensemble().
    """
    keys   = ["Quiet", "Moderate", "Active"]
    w      = np.array([weights[k] for k in keys], dtype=float)
    values = np.array([kp_forecasts[k] for k in keys])
    return np.average(values, axis=0, weights=w)

# ggsp/viz.py
#
# Matplotlib visualisations for GGSP results.
#
# Kept in its own file because:
#   1. Plots are optional (--no-plots runs the full pipeline without them).
#   2. Matplotlib is a heavyweight import we don't want in every module.
#   3. Separating plotting from computation makes it easy to add new plot types
#      or swap out matplotlib for another library later.
#
# Both functions create figures but do NOT call plt.show() — that is done
# by pipeline.py after both figures are created, so they appear together.
#
# Used by:  pipeline.py (optional Stage 8 of run_pipeline)
# Imports:  nothing from this package — only matplotlib

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_test_results(test_frame: pd.DataFrame, metrics: dict) -> None:
    """Two-panel evaluation plot for the chronological held-out test window.

    Left panel:  Observed vs. predicted Kp over the test period.
    Right panel: Histogram of residuals (Observed – Predicted).

    Why these two?
      The time-series panel shows whether the model tracks storm onset and
      recovery, and how often it misses peaks above the G1 threshold (red dashed).
      The residual histogram tells you if the errors are symmetric (unbiased)
      or skewed.  A rightward skew means the model systematically under-predicts
      storm peaks — a common problem for regression models on imbalanced data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(test_frame.index, test_frame["Kp_true"],
                 label="Observed Kp", color="#333333", lw=1.2)
    axes[0].plot(test_frame.index, test_frame["Kp_pred"],
                 label="Predicted Kp", color="#cc4125", lw=1.2)
    axes[0].axhline(5, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title("Held-out Kp (test set)")
    axes[0].set_ylabel("Kp")
    axes[0].set_ylim(0, 9.5)
    axes[0].legend(loc="upper right")

    residuals = test_frame["Kp_true"] - test_frame["Kp_pred"]
    axes[1].hist(residuals, bins=30, color="#3d85c6", alpha=0.8, edgecolor="black")
    axes[1].axvline(0, color="red", ls="--", lw=1)
    axes[1].set_title(
        f"Residuals  |  MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}"
    )
    axes[1].set_xlabel("Observed − Predicted")

    plt.tight_layout()


def plot_forecast(
    forecast_times,
    scenarios: Dict[str, dict],
    kp_forecasts: Dict[str, np.ndarray],
    kp_weighted: np.ndarray,
) -> None:
    """Two-panel 72-hour scenario forecast plot.

    Top panel:    Kp trajectories for each scenario + weighted best estimate.
    Bottom panel: IMF Bz trajectories for each scenario.

    Reading the plot:
      The three scenario lines (Quiet / Moderate / Active) bracket the plausible
      range of outcomes.  The black "Weighted" line is the scenario-probability-
      weighted best estimate — use this for the headline Kp number.
      The Bz panel is important context: a Bz that goes and stays southward (< −3 nT)
      is the key driver of storm intensification.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for name, kp_vals in kp_forecasts.items():
        axes[0].plot(forecast_times, kp_vals,
                     label=name, color=scenarios[name]["color"], lw=2)
    axes[0].plot(forecast_times, kp_weighted,
                 label="Weighted", color="#111111", lw=2.5)
    axes[0].axhline(5, color="red", ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title("72-hour Kp scenario forecast")
    axes[0].set_ylabel("Kp")
    axes[0].set_ylim(0, 9.5)
    axes[0].legend(loc="upper left")

    for name, scenario in scenarios.items():
        axes[1].plot(forecast_times, scenario["bz_gsm"],
                     label=f"{name} Bz", color=scenario["color"], lw=2)
    axes[1].axhline(0,  color="black", lw=0.6, alpha=0.4)
    axes[1].axhline(-3, color="red",   ls="--", lw=0.8, alpha=0.5)
    axes[1].set_title("Scenario IMF Bz")
    axes[1].set_ylabel("Bz (nT)")
    axes[1].legend(loc="lower left")

    plt.tight_layout()

"""eval/evaluate.py — Honest evaluation harness for GGSP-7.0.

ARCHITECTURE NOTE
─────────────────
This module is called in two ways:

  1. Automatically inside run_pipeline() (ggsp_pipeline_v7.py) after
     fit_and_evaluate_model() returns — always printed to console.

  2. Standalone CLI:
       python eval/evaluate.py --json path/to/output.json
     Reads a previously saved GGSP JSON to reprint metrics without re-running
     the full pipeline.

It has NO side-effects on the model or forecast — read-only evaluation only.

Reports (printed every run — acceptance criterion):

  • Persistence baseline    : predict Kp(t) = Kp(t−3h).  MAE and R².
  • Skill over persistence  : model_MAE / persistence_MAE  and ΔR².
                               This is the headline metric — beats-persistence?
  • Storm-conditional MAE   : MAE only on rows where observed Kp ≥ 5.
  • Categorical scores       : POD, FAR, CSI at the G1 (Kp ≥ 5) threshold.
  • Reliability summary      : binned predicted-Kp → mean observed-Kp (text table).

Acceptance criterion
──────────────────────────────────────────────────────────────────────
Call print_evaluation_report() at least once per run (done in GGSP-7.0.py
after fit_and_evaluate_model returns).  The report always includes:
  - persistence MAE
  - skill_ratio  (model_MAE / persistence_MAE)
  - beats_persistence verdict

Standalone usage
──────────────────────────────────────────────────────────────────────
  python eval/evaluate.py --json path/to/output.json
  # reads kp_weighted (deterministic forecast) and actual Kp from JSON
  # to exercise the report functions on saved run data.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


# ── Core metric functions ──────────────────────────────────────────────────────


def diebold_mariano_test(
    model_errors: np.ndarray,
    persistence_errors: np.ndarray,
) -> dict:
    """Diebold-Mariano (1995) test: is the model significantly more accurate than persistence?

    Why do we need this?
      Reporting skill_ratio = 0.699 tells you the model's MAE is 30% lower than
      persistence, but it doesn't tell you whether that improvement is statistically
      significant or just a lucky run.  The DM test gives you a p-value.

    How it works:
      1. Form the loss differential: d(t) = e1(t)^2 - e2(t)^2
         where e1 = model errors, e2 = persistence errors.
      2. Test H0: E[d] = 0 (equal forecast accuracy).
         H1 (one-sided): E[d] < 0 (model is MORE accurate than persistence).
      3. Compute the DM statistic using a Newey-West HAC variance with n_lags=3
         (= 9h of 3h data) to account for autocorrelation in d(t).
      4. Under H0, DM ~ N(0,1) asymptotically.

    With N ≈ 3,000+ test rows (5yr window), the normal approximation is excellent.

    Parameters
    ----------
    model_errors      : model residuals on the test set  (y_true - y_pred)
    persistence_errors: persistence residuals on the same set (y_true - kp_lag_3h)

    Returns
    -------
    dict with dm_statistic, p_value, significant_p05, verdict
    """
    from scipy import stats as _stats

    # Loss differential using squared errors (standard DM choice).
    d = model_errors ** 2 - persistence_errors ** 2
    n = len(d)
    if n < 10:
        return {
            "dm_statistic": float("nan"),
            "p_value":       float("nan"),
            "n":             n,
            "significant_p05": False,
            "verdict": "insufficient data for DM test (n < 10)",
        }

    d_mean = float(np.mean(d))

    # Newey-West HAC variance: accounts for autocorrelation in d(t).
    # We use n_lags = 3 to cover the 3h/6h/9h Kp autocorrelation range.
    # The Bartlett kernel weight w_j = 1 - j/(n_lags+1) downweights distant lags.
    n_lags  = 3
    gamma_0 = float(np.var(d, ddof=1))
    hac_var = gamma_0
    for lag in range(1, n_lags + 1):
        gamma_lag = float(np.mean(
            (d[lag:] - d_mean) * (d[:-lag] - d_mean)
        ))
        bartlett_weight = 1.0 - lag / (n_lags + 1)
        hac_var += 2.0 * bartlett_weight * gamma_lag
    hac_var = max(hac_var, 1e-12)   # prevent division by zero

    dm_stat = d_mean / (hac_var / n) ** 0.5

    # One-sided p-value: P(Z < DM).  Negative DM means model MAE < persistence MAE.
    p_value = float(_stats.norm.cdf(dm_stat))
    significant = bool(p_value < 0.05)

    return {
        "dm_statistic":    round(float(dm_stat), 4),
        "p_value":         round(p_value, 4),
        "n":               n,
        "significant_p05": significant,
        "verdict": (
            "model is significantly more accurate than persistence (p<0.05)"
            if significant else
            "improvement over persistence is NOT significant at p=0.05"
        ),
    }


def persistence_baseline(y_true: np.ndarray, kp_lag: np.ndarray) -> dict:
    """Persistence: predict Kp(t) = Kp(t−3h).

    Parameters
    ----------
    y_true  : observed Kp values on the test set  (shape N,)
    kp_lag  : kp_lag_3h — the Kp value one step (3 h) earlier, same shape.
              In the training frame this is the 'kp_lag_3h' feature column.

    Returns
    -------
    dict with keys: mae, r2, n
    """
    mask = ~(np.isnan(y_true) | np.isnan(kp_lag))
    yt = y_true[mask]
    yp = kp_lag[mask]
    if len(yt) == 0:
        return {"mae": float("nan"), "r2": float("nan"), "n": 0}
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "r2":  float(r2_score(yt, yp)),
        "n":   int(len(yt)),
    }


def skill_over_persistence(
    model_mae: float,
    persistence_mae: float,
    model_r2: float,
    persistence_r2: float,
) -> dict:
    """Skill scores relative to the persistence baseline.

    skill_ratio < 1.0  →  model beats persistence (lower MAE is better)
    delta_r2    > 0.0  →  model beats persistence (higher R² is better)
    """
    ratio    = model_mae / persistence_mae if persistence_mae > 0 else float("nan")
    delta_r2 = model_r2 - persistence_r2
    return {
        "model_mae":        round(model_mae,      4),
        "persistence_mae":  round(persistence_mae, 4),
        "skill_ratio":      round(ratio,           4),   # headline
        "delta_r2":         round(delta_r2,        4),
        "beats_persistence": bool(ratio < 1.0),
    }


def storm_conditional_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 5.0,
) -> dict:
    """MAE computed only on rows where observed Kp ≥ threshold (G1+ storm events)."""
    mask     = y_true >= threshold
    n_storm  = int(mask.sum())
    if n_storm == 0:
        return {"mae_storm": float("nan"), "n_storm": 0, "threshold": threshold}
    return {
        "mae_storm": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "n_storm":   n_storm,
        "threshold": threshold,
    }


def categorical_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 5.0,
) -> dict:
    """2×2 contingency table and categorical verification scores at the G1 threshold.

    POD (Probability of Detection) = TP / (TP + FN)
        — fraction of actual storms that were forecast
    FAR (False Alarm Ratio)        = FP / (TP + FP)
        — fraction of storm forecasts that were false alarms
    CSI (Critical Success Index)   = TP / (TP + FP + FN)
        — threat score; penalises both misses and false alarms
    """
    obs_pos  = y_true >= threshold
    pred_pos = y_pred >= threshold
    TP = int(np.sum( obs_pos &  pred_pos))
    FP = int(np.sum(~obs_pos &  pred_pos))
    FN = int(np.sum( obs_pos & ~pred_pos))
    TN = int(np.sum(~obs_pos & ~pred_pos))
    POD = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FAR = FP / (TP + FP) if (TP + FP) > 0 else 0.0
    CSI = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "POD": round(POD, 4),
        "FAR": round(FAR, 4),
        "CSI": round(CSI, 4),
        "threshold": threshold,
    }


def reliability_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 9,
) -> str:
    """Reliability (calibration) table — text output, no plot needed.

    Bins predicted Kp into equal-width intervals [0, 9] and reports
    mean observed Kp inside each bin.  A perfectly calibrated model
    shows a 1:1 relationship between predicted and observed bin means.
    """
    bins = np.linspace(0, 9, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append((
            f"{lo:.1f}–{hi:.1f}",
            n,
            round(float(y_pred[mask].mean()), 2),
            round(float(y_true[mask].mean()), 2),
        ))

    lines = [
        "",
        "  ── Reliability (predicted Kp bin → mean observed Kp) ─────────",
        f"  {'Bin':>9}  {'N':>7}  {'Mean pred':>10}  {'Mean obs':>10}",
        "  " + "-" * 47,
    ]
    for row in rows:
        lines.append(f"  {row[0]:>9}  {row[1]:>7}  {row[2]:>10.2f}  {row[3]:>10.2f}")
    lines.append("")
    return "\n".join(lines)


# ── Master report printer ──────────────────────────────────────────────────────


def print_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kp_lag_3h: np.ndarray,
    model_metrics: dict,
    file=None,
) -> dict:
    """Print the full honest evaluation report.

    Parameters
    ----------
    y_true        : observed Kp on the test set
    y_pred        : model predictions on the test set
    kp_lag_3h     : kp_lag_3h feature column values on the test set
                    (= Kp one 3-hour step earlier — the persistence forecast)
    model_metrics : dict with at minimum 'mae', 'r2', 'baseline_mae'
                    (as returned by fit_and_evaluate_model)

    Returns
    -------
    dict with keys: persistence, skill, storm, categorical
    """
    if file is None:
        file = sys.stdout

    persist = persistence_baseline(y_true, kp_lag_3h)
    skill   = skill_over_persistence(
        model_metrics["mae"], persist["mae"],
        model_metrics["r2"],  persist["r2"],
    )
    storm = storm_conditional_mae(y_true, y_pred)
    cats  = categorical_scores(y_true, y_pred)
    reltab = reliability_table(y_true, y_pred)

    # Diebold-Mariano significance test (new in ggsp refactor).
    model_errs       = y_true - y_pred
    persistence_errs = y_true - kp_lag_3h
    dm = diebold_mariano_test(model_errs, persistence_errs)

    verdict = (
        "YES ✓ model beats persistence"
        if skill["beats_persistence"]
        else "NO  ✗ persistence is better than model"
    )

    print("", file=file)
    print("  ═══════════════════════════════════════════════════════════", file=file)
    print("    GGSP Honest Evaluation Report", file=file)
    print("  ═══════════════════════════════════════════════════════════", file=file)
    print(f"  Test samples        : {persist['n']}", file=file)
    print("", file=file)
    print("  ── Baseline Comparison ──────────────────────────────────────", file=file)
    print(f"  Mean-climatology MAE : {model_metrics.get('baseline_mae', float('nan')):.4f}", file=file)
    print(f"  Persistence MAE      : {persist['mae']:.4f}  (R²={persist['r2']:.4f})", file=file)
    print(f"  Model MAE            : {skill['model_mae']:.4f}  (R²={model_metrics['r2']:.4f})", file=file)
    print("", file=file)
    print("  ── Skill over Persistence  (PRIMARY HEADLINE) ───────────────", file=file)
    print(f"  Skill ratio          : {skill['skill_ratio']:.4f}  (< 1 = model beats persistence)", file=file)
    print(f"  ΔR²                  : {skill['delta_r2']:+.4f}  (> 0 = model beats persistence)", file=file)
    print(f"  Verdict              : {verdict}", file=file)
    print("", file=file)
    print("  ── Diebold-Mariano Significance Test ────────────────────────", file=file)
    print(f"  DM statistic         : {dm['dm_statistic']:.4f}  (negative = model better)", file=file)
    print(f"  p-value (one-sided)  : {dm['p_value']:.4f}", file=file)
    print(f"  n (test rows)        : {dm['n']}", file=file)
    print(f"  Result               : {dm['verdict']}", file=file)
    print("", file=file)
    print("  ── Storm-Event Metrics (Kp ≥ 5) ────────────────────────────", file=file)
    if storm["n_storm"] == 0:
        print("  (no storm events in the test window — G1+ metrics unavailable)", file=file)
    else:
        print(f"  Storm rows           : {storm['n_storm']}", file=file)
        print(f"  Storm-conditional MAE: {storm['mae_storm']:.4f}", file=file)
    print("", file=file)
    print("  ── G1 Categorical Scores (Kp ≥ 5 threshold) ────────────────", file=file)
    print(f"  Contingency  TP={cats['TP']}  FP={cats['FP']}  FN={cats['FN']}  TN={cats['TN']}", file=file)
    print(f"  POD={cats['POD']:.3f}   FAR={cats['FAR']:.3f}   CSI={cats['CSI']:.3f}", file=file)
    print(reltab, file=file)
    print("  ═══════════════════════════════════════════════════════════", file=file)
    print("", file=file)

    return {
        "persistence": persist,
        "skill":       skill,
        "storm":       storm,
        "categorical": cats,
        "diebold_mariano": dm,
    }


# ── Cross-validation report helper ────────────────────────────────────────────


def print_cv_summary(cv_metrics: dict, file=None) -> None:
    """Pretty-print the cross-validation metrics dict from fit_and_evaluate_model."""
    if file is None:
        file = sys.stdout
    mae_mean = cv_metrics.get("cv_mae_mean", float("nan"))
    mae_std  = cv_metrics.get("cv_mae_std",  float("nan"))
    r2_mean  = cv_metrics.get("cv_r2_mean",  float("nan"))
    r2_std   = cv_metrics.get("cv_r2_std",   float("nan"))
    csi_mean = cv_metrics.get("cv_storm_csi_mean", float("nan"))
    n_folds  = cv_metrics.get("cv_n_folds", "?")
    print("", file=file)
    print(f"  [CV]  {n_folds}-fold TimeSeriesSplit (gap=3 rows):", file=file)
    print(f"        MAE  = {mae_mean:.4f} ± {mae_std:.4f}", file=file)
    print(f"        R²   = {r2_mean:.4f} ± {r2_std:.4f}", file=file)
    print(f"        G1-CSI = {csi_mean:.4f}", file=file)
    print("", file=file)


# ── CLI (read a saved JSON and produce report) ─────────────────────────────────


def _cli_from_json(json_path: str) -> None:
    """Produce an evaluation report from a saved GGSP JSON output file.

    This exercises the report functions on run data without needing to
    re-run the full pipeline — useful for post-hoc analysis.
    Requires that the JSON was produced with the updated pipeline that
    includes 'persistence_mae' and 'persistence_r2' in metrics.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    if "persistence_mae" not in metrics:
        print(
            "[WARNING] JSON does not contain 'persistence_mae'. "
            "Re-run the pipeline with the updated GGSP-7.0.py to get full metrics.",
            file=sys.stderr,
        )
        print(f"\nAvailable metrics: {list(metrics.keys())}")
        return

    print(f"\nEvaluation from saved JSON: {json_path}")
    print(f"Training window   : OMNI rows = {data.get('counts', {}).get('omni_3h_rows', '?')}")
    print(f"Test set size     : {metrics.get('test_count', '?')} rows")
    print(f"\nModel MAE         : {metrics.get('mae', float('nan')):.4f}")
    print(f"Model R²          : {metrics.get('r2', float('nan')):.4f}")
    print(f"Mean-clim MAE     : {metrics.get('baseline_mae', float('nan')):.4f}")
    print(f"Persistence MAE   : {metrics.get('persistence_mae', float('nan')):.4f}")
    print(f"Persistence R²    : {metrics.get('persistence_r2', float('nan')):.4f}")

    if metrics.get("persistence_mae", 0) > 0:
        ratio = metrics["mae"] / metrics["persistence_mae"]
        delta_r2 = metrics["r2"] - metrics.get("persistence_r2", 0.0)
        beats = ratio < 1.0
        print(f"\nSkill ratio       : {ratio:.4f}  ({'BEATS persistence' if beats else 'WORSE than persistence'})")
        print(f"ΔR²               : {delta_r2:+.4f}")

    # Cross-validation
    if "cv_mae_mean" in metrics:
        print(f"\nCV MAE            : {metrics['cv_mae_mean']:.4f} ± {metrics['cv_mae_std']:.4f}")
        print(f"CV R²             : {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"CV G1-CSI         : {metrics.get('cv_storm_csi_mean', float('nan')):.4f}")

    forecast = data.get("forecast", {})
    storm_72h = forecast.get("storm_prob_72h_pct")
    if storm_72h is not None:
        print(f"\nStorm P(72h)      : {storm_72h:.1f}%  (ensemble method)")
    else:
        storm_chance = forecast.get("storm_chance_percent")
        if storm_chance is not None:
            print(f"\nStorm chance      : {storm_chance:.1f}%  (legacy mean-trajectory method)")


def main():
    parser = argparse.ArgumentParser(
        description="GGSP evaluation report from saved JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python eval/evaluate.py --json ggsp_output.json",
    )
    parser.add_argument(
        "--json", type=str, required=True, metavar="PATH",
        help="Path to a GGSP JSON output file (produced with --json-out).",
    )
    args = parser.parse_args()
    _cli_from_json(args.json)


if __name__ == "__main__":
    main()

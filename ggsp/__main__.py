# ggsp/__main__.py
#
# CLI entry point for the ggsp package.
#
# Run as:
#   python -m ggsp [options]          ← recommended
#   python GGSP-7.0.py [options]      ← legacy wrapper (still works)
#
# This file is identical in functionality to the original GGSP-7.0.py.
# Moving it here means the pipeline can be invoked as a proper Python
# package (`python -m ggsp`) in addition to running the top-level script.
#
# Architecture note:
#   This file does NO ML work.  Its only jobs are:
#     1. Parse command-line arguments (argparse)
#     2. Validate argument values (validate_args)
#     3. Build a PipelineConfig from the validated arguments
#     4. Call run_pipeline() and print a human-readable summary
#   Everything else lives in the ggsp submodules imported below.

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone

from .config import PipelineConfig
from .pipeline import run_pipeline

# OMNI2 coverage spans 1963–present.  Requesting years before 1963 returns
# empty .dat files.  The current-year file is also excluded by default because
# it's still being written hourly and the trailing partial year skews training.
OMNI_FIRST_YEAR    = 1963
CURRENT_YEAR       = datetime.now(timezone.utc).year
OMNI_LAST_SAFE_YEAR = CURRENT_YEAR - 1


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable error strings.

    An empty list means all checks passed and the pipeline can start.
    Warnings are printed to stderr but do not block the run.
    Hard errors (start year out of range, bad split fraction, etc.) are
    returned in the list and cause the CLI to exit before touching the network.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── start-year range ──────────────────────────────────────────────────────
    if args.start_year < OMNI_FIRST_YEAR:
        errors.append(
            f"--start-year {args.start_year} is before OMNI2 coverage begins ({OMNI_FIRST_YEAR})."
        )
    if args.start_year > OMNI_LAST_SAFE_YEAR:
        errors.append(
            f"--start-year {args.start_year} is in the future or the current year. "
            f"Latest safe start year is {OMNI_LAST_SAFE_YEAR}."
        )

    # ── years count ───────────────────────────────────────────────────────────
    if args.years < 1:
        errors.append(f"--years must be at least 1 (got {args.years}).")

    end_year = args.start_year + args.years - 1
    if args.start_year >= OMNI_FIRST_YEAR and end_year > OMNI_LAST_SAFE_YEAR:
        warnings.append(
            f"--start-year {args.start_year} + --years {args.years} ends at {end_year}, "
            f"but OMNI2 is only confirmed through {OMNI_LAST_SAFE_YEAR}. "
            f"Years beyond {OMNI_LAST_SAFE_YEAR} may be missing or partial."
        )

    if 1 <= args.years < 2:
        warnings.append(
            f"--years {args.years} gives ~2,900 3-hour OMNI samples. "
            "Recommend --years 3+ for stable training."
        )

    # G1-CSI statistical reliability advisory (from omni_client._sanity_check_omni).
    if args.years < 8:
        warnings.append(
            f"--years {args.years} may give <200 G1+ events in the test set. "
            "G1-CSI 95% CI will be ~±0.13.  Use --years ≥ 8 for tighter bounds."
        )

    # ── train-frac ────────────────────────────────────────────────────────────
    if not (0.5 <= args.train_frac <= 0.95):
        errors.append(
            f"--train-frac {args.train_frac} is outside [0.50, 0.95]. "
            "Below 0.50 leaves too few training samples; above 0.95 leaves too few test samples."
        )

    # ── json-out path ─────────────────────────────────────────────────────────
    if args.json_out is not None:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if not os.path.isdir(out_dir):
            errors.append(
                f"--json-out directory does not exist: '{out_dir}'. "
                "Create the directory first."
            )
        if not args.json_out.endswith(".json"):
            warnings.append(
                f"--json-out '{args.json_out}' has no .json extension. "
                "The React frontend expects a .json file."
            )

    for w in warnings:
        print(f"[WARNING] {w}", file=sys.stderr)

    return errors


def main() -> None:
    """Parse CLI arguments, validate, build config, run pipeline, print summary."""

    parser = argparse.ArgumentParser(
        description="Geomagnetic Storm Prediction Pipeline (GGSP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m ggsp                                           # defaults: 2020, 5 yr\n"
            "  python -m ggsp --start-year 1990 --years 35             # SC22–SC25\n"
            "  python -m ggsp --no-plots --json-out ggsp_output.json   # headless + React\n"
            "  python -m ggsp --refit --static-weights                 # force retrain\n"
            f"\nOMNI data range: {OMNI_FIRST_YEAR}–{OMNI_LAST_SAFE_YEAR}"
        ),
    )
    parser.add_argument("--no-plots",     action="store_true",
                        help="Skip matplotlib plots (headless / CI mode).")
    parser.add_argument("--start-year",   type=int,   default=2020,
                        help=f"First OMNI training year ({OMNI_FIRST_YEAR}–{OMNI_LAST_SAFE_YEAR}, default 2020).")
    parser.add_argument("--years",        type=int,   default=5,
                        help="Number of consecutive OMNI years (default 5).")
    parser.add_argument("--train-frac",   type=float, default=0.75,
                        help="Chronological train/test split fraction (0.50–0.95, default 0.75).")
    parser.add_argument("--json-out",     type=str,   default=None, metavar="PATH",
                        help="Write results JSON for the React frontend.")
    parser.add_argument("--refit",        action="store_true",
                        help="Force model retraining even if cache/model.joblib is valid.")
    parser.add_argument("--static-weights", action="store_true",
                        help="Skip sunspot modifier; use fixed Q=0.20 / M=0.50 / A=0.30 weights.")
    args = parser.parse_args()

    # ── Pre-run validation ────────────────────────────────────────────────────
    errors = validate_args(args)
    if errors:
        print("\n[ERROR] Pipeline cannot start — fix the following issues:\n", file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}", file=sys.stderr)
        print(f"\nRun 'python -m ggsp --help' for usage.\n", file=sys.stderr)
        sys.exit(1)

    # ── Configuration summary ─────────────────────────────────────────────────
    end_year = args.start_year + args.years - 1
    print("=" * 57)
    print("   GGSP — Geomagnetic Storm Prediction Pipeline")
    print("=" * 57)
    print(f"  OMNI training window : {args.start_year}–{end_year}  ({args.years} yr)")
    print(f"  Train/test split     : {args.train_frac:.0%} / {1 - args.train_frac:.0%}")
    print(f"  Plots                : {'off (headless)' if args.no_plots else 'on'}")
    print(f"  JSON output          : {args.json_out or 'none'}")
    print("=" * 57)
    print()

    # ── Build config and run ──────────────────────────────────────────────────
    config = PipelineConfig(
        omni_start_year=args.start_year,
        omni_num_years=args.years,
        train_fraction=args.train_frac,
    )

    try:
        results = run_pipeline(
            config=config,
            make_plots=not args.no_plots,
            json_output_path=args.json_out,
            refit=args.refit,
            static_weights=args.static_weights,
        )
    except RuntimeError as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        print(
            "\nCommon causes:\n"
            "  - NOAA SWPC feeds are temporarily unreachable\n"
            "  - SPDF OMNI server timeout — try a shorter --years window\n"
            "  - Requested OMNI years return empty .dat files\n",
            file=sys.stderr,
        )
        sys.exit(2)

    # ── Results summary ───────────────────────────────────────────────────────
    print("=== GGSP Pipeline Summary ===")
    print(f"NOAA source: {results['sources']['noaa']}")
    print(f"OMNI source: {results['sources']['omni']}")
    counts = results["counts"]
    print(
        f"Rows: plasma={counts['plasma_rows']}, mag={counts['mag_rows']}, "
        f"kp={counts['kp_rows']}, omni_3h={counts['omni_3h_rows']}"
    )

    metrics = results["metrics"]
    persistence_mae = metrics.get("persistence_mae", float("nan"))
    skill_ratio     = metrics["mae"] / persistence_mae if persistence_mae > 0 else float("nan")
    beats           = "beats persistence" if skill_ratio < 1.0 else "WORSE than persistence"
    improvement     = (
        100.0 * (1.0 - metrics["mae"] / metrics["baseline_mae"])
        if metrics.get("baseline_mae", 0) > 0 else 0.0
    )
    print(f"Model  : MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}")
    print(
        f"Baselines: mean-clim MAE={metrics['baseline_mae']:.3f} (+{improvement:.1f}% over mean)  "
        f"persistence MAE={persistence_mae:.3f}  skill_ratio={skill_ratio:.3f} ({beats})"
    )
    if metrics.get("cv_mae_mean") is not None:
        ci = metrics.get("cv_mae_ci95", float("nan"))
        print(
            f"CV     : MAE={metrics['cv_mae_mean']:.3f}±{metrics['cv_mae_std']:.3f}"
            f"  (95% CI ±{ci:.3f})  "
            f"R²={metrics['cv_r2_mean']:.3f}±{metrics['cv_r2_std']:.3f}  "
            f"G1-CSI={metrics.get('cv_storm_csi_mean', float('nan')):.3f}  "
            f"({metrics.get('cv_n_folds', '?')} folds)"
        )

    if metrics["r2"] < 0.0:
        print("[WARNING] R² is negative — model is worse than always predicting the mean.", file=sys.stderr)
    elif metrics["mae"] > 2.0:
        print(f"[WARNING] MAE={metrics['mae']:.3f} is unexpectedly high (>2.0 Kp).", file=sys.stderr)

    latest = results["latest"]
    print(
        f"Latest : {latest['time']}  predicted Kp={latest['predicted_kp']:.2f}  "
        f"observed Kp={latest['observed_kp']:.2f}  [{latest['category']}]"
    )

    forecast = results["forecast"]
    storm_72h = forecast.get("storm_prob_72h_pct", forecast.get("storm_chance_percent", 0.0))
    print(
        f"72h    : mean Kp={forecast['mean_weighted_kp']:.2f}  "
        f"peak Kp={forecast['peak_weighted_kp']:.2f}  "
        f"P(storm 72h)={storm_72h:.1f}%  "
        f"seed={forecast['forecast_seed_source']}"
    )

    ssn = results.get("sunspot_info", {})
    if ssn.get("source") == "sunspot_pipeline":
        w = ssn.get("adjusted_weights", {})
        print(
            f"Sunspot: modifier={ssn.get('storm_rate_modifier', '?'):.3f}  "
            f"tier={ssn.get('solar_activity_tier', '?')}  "
            f"weights Q={w.get('Quiet', '?'):.3f} "
            f"M={w.get('Moderate', '?'):.3f} "
            f"A={w.get('Active', '?'):.3f}"
        )

    if args.json_out:
        print(f"JSON   : written to {os.path.abspath(args.json_out)}")


if __name__ == "__main__":
    main()

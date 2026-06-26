#~Gabe J~ | UMSL 26' | Computer Science | 

#Welcome to GGSP-7.0.py!  

#Good Day to you kind person.

#~ Gabe | chickens5

#UMSL | Class of 2026 | Computer Science


#This python file, generated from our notebook, helps us predict the **planetary Kp index (global measure of geomagnetic storm intensity)**  
# from real-time solar wind observations--from NOAA and the OmniDataset, using a gradient-boosted regression model trained on multi-year data.

        #We don't use any other models because the interactions involved with solar wind and the magnetosphere
            #are highly nonlinear, which means our model is able to capture complex relationships between solar wind parameters and Kp.
            # 
            # Newell's coupling function in our ML application provides the hierarchical structure of the solar wind-magnetosphere coupling,
            #  which is crucial for understanding and predicting space weather phenomena. 

            # By incorporating this function into our model, we can capture the complex interactions between solar wind parameters 
            # #and their impact on Earth's magnetosphere, leading to more accurate forecasts of geomagnetic storms and their potential 
            # effects on satellite operations, communication systems, and power grids. This approach not only enhances our predictive
            #  capabilities but also contributes to the broader scientific understanding of space weather dynamics.
            
#Okay, lets get to the code:

#Open terminal and create virtual enviornment to ensure your machine doesn't have any dependency conflicts

#  python -m venv venv      

#    .\venv\Scripts\activate

#    pip install numpy pandas matplotlib scikit-learn scipyg

#          :p bleeeeeggghhhh



# We predict the **planetary Kp index (global measure of geomagnetic storm intensity)**  from real-time solar wind observations
    #--from NOAA and the OmniDataset, using a gradient-boosted regression model trained on multi-year data.
        #We don't use any other models because the interactions involved with solar wind and the magnetosphere
            #are highly nonlinear, which means our model is able to capture complex relationships between solar wind parameters and Kp.

# **Why this matters.** Kp drives auroras, GPS errors, satellite drag, and power-grid risk.
#  Carrignton level events haven't happened in over a century, but who knows what's to come? 

# NOAA SWPC publishes the upstream solar wind in near-real-time from the DSCOVR spacecraft (~1.5 million km sunward of Earth),
#  giving roughly 30–60 minutes of warning before the plasma actually hits the magnetosphere. 

        # That gap is exactly where ML lives.


# 
# **Approach.** Pull three live JSON feeds from NOAA SWPC,
#  align them on a common time grid (UTC), engineer a handful of physics-motivated features (mainly Newells function + basic solar wind parameters),
#    and train a gradient-boosted regressor to predict Kp.
# 
# **Data sources** (all public, no API key):
# - `plasma-7-day.json` — solar wind density, speed, temperature
# - `mag-7-day.json` — interplanetary magnetic field components (Bx, By, **Bz**, Bt) in GSM coordinates
# - `noaa-planetary-k-index.json` — observed Kp, our target



 #           <o
#               > 3
#            <o



    # *Important*


    # In CLI, we can use the --no-plots flag for fast headless execution, 
    # which will skip generating matplotlib plots. 


        #AND


#   python GGSP-7.0.py --start-year 2018 --years 7 --train-frac 0.8

    # --> allows you to specify the start year and number of years for OMNI data,
        # as well as the fraction of data to use for training. 
        # After running the pipeline, it prints a summary of the results,
        #  including data sources, row counts, performance metrics, and forecasts.

# All the requirements are in requirements.txt, and (*AFTER CREATING A VIRTUAL ENVIRONMENT*)
#  where you can install them with: pip install -r requirements.txt 

#We import __future_ annotations to enable postponed evaluation of type hints,
#  which can help with forward references and improve code readability.

#argparse is used to parse command-line arguments, allowing us to customize the pipeline's behavior without modifying the code.


from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from ggsp_pipeline_v7 import PipelineConfig, run_pipeline      #Importing Pipelineconfig(run_pipeline()) from ggsp_pipeline.py

# OMNI2 data begins in 1963; requesting earlier years returns empty .dat files
OMNI_FIRST_YEAR = 1963
# Leave at least the current partial year as uncommitted data (fetching the present year
# mid-run can produce partial rows that skew the training chronology)
CURRENT_YEAR = datetime.utcnow().year
OMNI_LAST_SAFE_YEAR = CURRENT_YEAR - 1


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return a list of human-readable error strings; empty list means all checks passed."""
    errors: list[str] = []
    warnings: list[str] = []

    # ── start-year range ──────────────────────────────────────────────────────
    if args.start_year < OMNI_FIRST_YEAR:
        errors.append(
            f"--start-year {args.start_year} is before OMNI2 coverage begins ({OMNI_FIRST_YEAR}). "
            f"The SPDF server has no data for years prior to {OMNI_FIRST_YEAR}."
        )
    if args.start_year > OMNI_LAST_SAFE_YEAR:
        errors.append(
            f"--start-year {args.start_year} is in the future or the current year. "
            f"Latest safe start year is {OMNI_LAST_SAFE_YEAR} (current year data may be incomplete)."
        )

    # ── years count ───────────────────────────────────────────────────────────
    if args.years < 1:
        errors.append(f"--years must be at least 1 (got {args.years}).")

    end_year = args.start_year + args.years - 1
    if args.start_year >= OMNI_FIRST_YEAR and end_year > OMNI_LAST_SAFE_YEAR:
        # Trim silently would be confusing; warn and let the pipeline attempt it —
        # SPDF will just return short/empty files for future years
        warnings.append(
            f"--start-year {args.start_year} + --years {args.years} extends to {end_year}, "
            f"but OMNI2 data is only confirmed through {OMNI_LAST_SAFE_YEAR}. "
            f"Years beyond {OMNI_LAST_SAFE_YEAR} may be missing or partial — "
            f"consider reducing --years to {OMNI_LAST_SAFE_YEAR - args.start_year + 1}."
        )

    # ── too few years = weak model ─────────────────────────────────────────────
    if 1 <= args.years < 2:
        warnings.append(
            f"--years {args.years} gives only ~2,900 3-hour OMNI samples (1 year). "
            "The model may underfit on a single solar-rotation cycle. "
            "Recommend --years 3 or more for stable training."
        )

    # ── train-frac ────────────────────────────────────────────────────────────
    if not (0.5 <= args.train_frac <= 0.95):
        errors.append(
            f"--train-frac {args.train_frac} is outside the accepted range [0.50, 0.95]. "
            "Values below 0.50 leave too few training samples; above 0.95 leaves too few test samples "
            "for meaningful evaluation."
        )

    # ── json-out path ─────────────────────────────────────────────────────────
    if args.json_out is not None:
        out_dir = os.path.dirname(os.path.abspath(args.json_out))
        if not os.path.isdir(out_dir):
            errors.append(
                f"--json-out directory does not exist: '{out_dir}'. "
                "Create the directory first or use a path inside an existing folder."
            )
        if not args.json_out.endswith(".json"):
            warnings.append(
                f"--json-out '{args.json_out}' has no .json extension. "
                "The React frontend expects a .json file — consider adding the extension."
            )

    # Print warnings (non-fatal) before returning errors
    for w in warnings:
        print(f"[WARNING] {w}", file=sys.stderr)

    return errors


def main():

    parser = argparse.ArgumentParser(
        description="Geomagnetic Storm Prediction Pipeline (GGSP-7.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python GGSP-7.0.py                                        # default: 2020, 5 years\n"
            "  python GGSP-7.0.py --start-year 1990 --years 32           # SC22–SC24 training window\n"
            "  python GGSP-7.0.py --no-plots --json-out ggsp_output.json # headless + React export\n"
            f"\nOMNI data range: {OMNI_FIRST_YEAR}–{OMNI_LAST_SAFE_YEAR}"
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots (headless mode)")
    parser.add_argument(
        "--start-year", type=int, default=2020,
        help=f"First year of OMNI training data (must be {OMNI_FIRST_YEAR}–{OMNI_LAST_SAFE_YEAR}, default: 2020)",
    )
    parser.add_argument(
        "--years", type=int, default=5,
        help="Number of consecutive OMNI years to load (default: 5)",
    )
    parser.add_argument(
        "--train-frac", type=float, default=0.75,
        help="Fraction of OMNI data used for training vs. test (0.50–0.95, default: 0.75)",
    )
    parser.add_argument(
        "--json-out", type=str, default=None, metavar="PATH",
        help="Write pipeline results to a JSON file for React consumption (e.g. ggsp_output.json)",
    )
    args = parser.parse_args()

    # ── Pre-run validation ────────────────────────────────────────────────────
    errors = validate_args(args)
    if errors:
        print("\n[ERROR] Pipeline cannot start — fix the following issues:\n", file=sys.stderr)
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}", file=sys.stderr)
        print(
            f"\nRun 'python GGSP-7.0.py --help' for usage.\n"
            f"Valid OMNI years: {OMNI_FIRST_YEAR}–{OMNI_LAST_SAFE_YEAR}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Configuration summary ─────────────────────────────────────────────────
    end_year = args.start_year + args.years - 1
    print("=" * 55)
    print("  GGSP-7.0 — Geomagnetic Storm Prediction Pipeline")
    print("=" * 55)
    print(f"  OMNI training window : {args.start_year}–{end_year}  ({args.years} yr)")
    print(f"  Train/test split     : {args.train_frac:.0%} / {1 - args.train_frac:.0%}  (chronological)")
    print(f"  Plots                : {'off (headless)' if args.no_plots else 'on'}")
    print(f"  JSON output          : {args.json_out or 'none'}")
    print("=" * 55)
    print()

    # ── Build config and run ──────────────────────────────────────────────────
    config = PipelineConfig(
        omni_start_year=args.start_year,
        omni_num_years=args.years,
        train_fraction=args.train_frac,
    )

    try:
        results = run_pipeline(config=config, make_plots=not args.no_plots, json_output_path=args.json_out)
    except RuntimeError as exc:
        # run_pipeline raises RuntimeError for recoverable data-fetch problems;
        # print a clean message rather than a full traceback
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        print(
            "\nCommon causes:\n"
            "  - NOAA SWPC feeds are temporarily unreachable (try again in a few minutes)\n"
            "  - SPDF OMNI server timeout — try --years with a shorter window\n"
            "  - Requested OMNI years return empty .dat files (check year range above)\n",
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
    improvement = 100.0 * (1.0 - metrics["mae"] / metrics["baseline_mae"]) if metrics["baseline_mae"] > 0 else 0.0
    print(
        f"Model  : MAE={metrics['mae']:.3f}  R²={metrics['r2']:.3f}  "
        f"(baseline MAE={metrics['baseline_mae']:.3f}, {improvement:.1f}% improvement)"
    )

    # Warn if the model looks degenerate
    if metrics["r2"] < 0.0:
        print(
            "[WARNING] R² is negative — the model is worse than always predicting the mean. "
            "Try a wider training window (--years 5+) or check for data-quality issues.",
            file=sys.stderr,
        )
    elif metrics["mae"] > 2.0:
        print(
            f"[WARNING] MAE={metrics['mae']:.3f} is unexpectedly high (>2.0 Kp). "
            "Model may be poorly fitted — consider a wider training window.",
            file=sys.stderr,
        )

    latest = results["latest"]
    print(
        f"Latest : {latest['time']}  predicted Kp={latest['predicted_kp']:.2f}  "
        f"observed Kp={latest['observed_kp']:.2f}  [{latest['category']}]"
    )

    forecast = results["forecast"]
    print(
        f"72h    : mean Kp={forecast['mean_weighted_kp']:.2f}  "
        f"peak Kp={forecast['peak_weighted_kp']:.2f}  "
        f"storm chance={forecast['storm_chance_percent']:.1f}%  "
        f"seed={forecast['forecast_seed_source']}"
    )

    if args.json_out:
        print(f"JSON   : written to {os.path.abspath(args.json_out)}")


if __name__ == "__main__":
    main()
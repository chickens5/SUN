#~Gabe J~ | UMSL 26' | Computer Science | 

#Welcome to GGSP-7.0.py!  

 #           <o
#               > 3
#            <o


    #This is the entry point for the GGSP data pipeline, which is where we perform all the data processing, modeling, and forecasting steps.

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

from ggsp_pipeline_v7 import PipelineConfig, run_pipeline      #Importing Pipelineconfig(run_pipeline()) from ggsp_pipeline.py


def main():

    #First, we set up CLI parser arguments to allow different configurations when running the pipeline.

    parser = argparse.ArgumentParser(description="Geomagnetic Storm Prediction Pipeline")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots")
    parser.add_argument("--start-year", type=int, default=2020, help="OMNI start year")
    parser.add_argument("--years", type=int, default=5, help="Number of OMNI years to use")
    parser.add_argument("--train-frac", type=float, default=0.75, help="Chronological train split fraction")
    args = parser.parse_args()

    #If you decide to use CLI, this is where the PipelineConfig class is instantiated with OMNI parameters.
    config = PipelineConfig(
        omni_start_year=args.start_year,
        omni_num_years=args.years,
        train_fraction=args.train_frac,
    )

    #Then, the pipeline is executed with the specified configuration and plot settings.

    results = run_pipeline(config=config, make_plots=not args.no_plots)

    print("=== GGSP Pipeline Summary ===")
    print(f"NOAA source: {results['sources']['noaa']}")
    print(f"OMNI source: {results['sources']['omni']}")
    print(
        "Rows: "
        f"plasma={results['counts']['plasma_rows']}, "
        f"mag={results['counts']['mag_rows']}, "
        f"kp={results['counts']['kp_rows']}, "
        f"omni_3h={results['counts']['omni_3h_rows']}"
    )

    metrics = results["metrics"]
    print(f"MAE={metrics['mae']:.3f} | R2={metrics['r2']:.3f} | baseline_MAE={metrics['baseline_mae']:.3f}")

    latest = results["latest"]
    print(
        f"Latest window {latest['time']}: predicted Kp={latest['predicted_kp']:.2f}, "
        f"observed Kp={latest['observed_kp']:.2f}, category={latest['category']}"
    )

    forecast = results["forecast"]
    print(
        "72h weighted forecast: "
        f"mean Kp={forecast['mean_weighted_kp']:.2f}, "
        f"peak Kp={forecast['peak_weighted_kp']:.2f}, "
        f"storm-window probability={forecast['storm_chance_percent']:.1f}%, "
        f"seed={forecast['forecast_seed_source']}"
    )


if __name__ == "__main__":
    main()

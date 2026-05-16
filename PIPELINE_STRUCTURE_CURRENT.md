# GGSP Current Prediction Structure

## File Roles

- `ggsp_pipeline.py`: Core pipeline module (data, features, model, forecast, plots).
- `GGSP-7.0.py`: Thin runner that builds config, runs pipeline, prints summary.

## End-to-End Prediction Flow

1. Configuration
- `PipelineConfig` stores all tunables (data windows, train split, model hyperparameters, forecast horizon, scenario weights).

2. Data Ingestion
- `load_noaa_data(...)` tries live NOAA feeds first.
- If NOAA fails, `_synthetic_noaa_fallback(...)` creates physically plausible data with the same schema.
- `load_omni_data(...)` loads multi-year OMNI data for stable model training.
- If OMNI fails, `_synthetic_omni_fallback(...)` generates realistic synthetic multi-year data.

3. Physics Feature Engineering
- `newell_coupling(speed_kms, bt_nt, by_nt, bz_nt)` is the single canonical coupling implementation.

- OMNI data is resampled to 3-hour cadence to match Kp windows.

- The model feature set is standardized as:
  - `speed_mean`
  - `density_mean`
  - `bz_mean`
  - `bz_min`
  - `bt_mean`
  - `coupling_mean`

4. Model Build and Evaluation
- `fit_and_evaluate_model(...)` performs chronological split (no shuffle), trains `GradientBoostingRegressor`, and reports:
  - MAE - 
  - R2
  - Baseline MAE

5. Current-State Prediction and Forecast
- `run_pipeline(...)` predicts the latest Kp from the latest feature row.
- `build_forecast_scenarios(...)` creates Quiet/Moderate/Active 72-hour trajectories.
- `predict_scenario_kp(...)` predicts Kp for each scenario.
- `weighted_ensemble(...)` combines scenario outputs into one best-estimate path.

## What Was Consolidated

The current structure removes redundant logic by design:
- One `newell_coupling` function reused everywhere.
- One `kp_label` mapping function.
- One feature contract (`FEATURE_COLUMNS`) for training and forecasting.
- One orchestrator (`run_pipeline`) that returns structured outputs.
- One CLI entrypoint (`GGSP-7.0.py`) instead of repeating pipeline logic inline.

## Inline Commenting Plan (Where to Add)

To make the code self-explanatory for readers, add inline comments in this order:

1. Class-level comments
- `PipelineConfig`: explain each parameter category (data, model, forecast).

2. Function-level comments
- Data loaders: explain fallback rationale and returned schema.
- Feature functions: explain physical meaning and units.
- Model functions: explain split strategy and metric choice.
- Forecast functions: explain scenario assumptions and weighting.

3. Library-call comments (only where non-obvious)
- `urllib.request.urlopen`: network timeout behavior and failure path.
- `pd.to_datetime(..., utc=True)`: timezone normalization.
- `resample('3h')`: Kp cadence alignment rationale.
- `GradientBoostingRegressor(...)`: why these defaults are selected.
- `np.clip(...)`: physical bounds enforcement.


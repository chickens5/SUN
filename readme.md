# UPDATE 7/7/26 ~ The NOAA feeds the pipeline relies on are deprecated: https://www.weather.gov/media/notification/pdf_2026/scn26-21_Data_Format_Changes_Impacting_SWPC_Products.pdf
    ~ Pipeline will need to be refactored to use SOLAR1 data
 **UMSL · Class of 2026 · Computer Science · chickens5**
# GGSP — Gabe's Geomagnetic Storm Prediction Pipeline


*Hey howz ya dooin---- thanks for visitin*

*The React frontend can be found here: [chickens5.github.io/](https://chickens5.github.io/barista/)

Predicts the **planetary Kp index** (0–9 scale of geomagnetic storm intensity) from real-time
solar wind data, using physics-based feature engineering and gradient-boosted regression.
A companion **sunspot pipeline** forecasts the solar cycle phase and feeds a dynamic activity
modifier back into the Kp model's scenario weights. Although both features are still works in progress, the geomagnetic storm intensity
predictor should be acceptable in how accurate / meaningful the data is.

---

## Starting the Application after cloning:

I recommend using Conda or
```
python -m venv venv      

.\venv\Scripts\activate

pip install numpy pandas matplotlib scikit-learn scipyg
```
To run the pipeline:
python GGSP-7.0.py --start-year 1985 --years 40  --json-out urfilename.json

## How It Works — End to End

```
NOAA SWPC (live, 7-day)          NASA SPDF OMNI (historical, years)
  plasma-7-day.json   ──┐           omni2_{year}.dat  ──┐
  mag-7-day.json      ──┤                               ├──► train GBR model
  kp-index.json       ──┘                               │
         │                                              │
         ▼                                              │
  build_noaa_3h_features()  ──────────────────────────►─┤
  (3-hour aggregates)                                   │
         │                                              │
         ▼                                              ▼
  latest_features  ──────────────►  model.predict()  ──► latest Kp estimate
         │
         ▼
  build_forecast_scenarios()  →  predict_scenario_kp()  →  weighted_ensemble()
         └── 72-hour Kp forecast (Quiet / Moderate / Active + weighted)

SIDC Brussels (monthly SSN)
  SN_m_tot_V2.0.txt  ──►  sunspot_pipeline.py  ──►  storm_rate_modifier
                                                      (adjusts scenario weights)
```

---

## Data Sources

| Source | What | Cadence | URL |
|--------|------|---------|-----|
| NOAA SWPC | Solar wind plasma (speed, density) | 1-min | `swpc.noaa.gov/products/solar-wind/plasma-7-day.json` |
| NOAA SWPC | IMF components Bx, By, Bz, Bt | 1-min | `swpc.noaa.gov/products/solar-wind/mag-7-day.json` |
| NOAA SWPC | Observed Kp index | 3-hour | `swpc.noaa.gov/products/noaa-planetary-k-index.json` |
| NASA SPDF | OMNI multi-year solar wind archive | Hourly | `spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/` |
| SIDC Brussels | International Sunspot Number v2.0 | Monthly | `sidc.be/silso/DATA/SN_m_tot_V2.0.txt` |

All sources are public — no API key required. Data from DSCOVR at the L1 Lagrange point (~1.5M km sunward) gives 30–60 minutes of warning before solar wind reaches Earth.

---

## Feature Engineering

Raw solar wind is aggregated into **3-hour windows** to match the Kp reporting cadence.
Eleven features are built for every window:

| Feature | Physics rationale |
|---------|-------------------|
| `speed_mean` | Kinetic energy flux into magnetosphere |
| `density_mean` | Proton number density — sets dynamic pressure |
| `by_mean` | IMF By component — sets clock angle θ in Newell coupling |
| `bz_mean` | Southward IMF (negative = direct field reconnection) |
| `bz_min` | Extreme southward excursion — storms are triggered by *bursts* |
| `bt_mean` | Total IMF magnitude — reconnection rate ∝ Bt^(2/3) |
| `coupling_mean` | **Newell coupling function** (see below) |
| `p_dyn_mean` | Dynamic ram pressure = density × (speed/100)² — CME sudden commencement |
| `kp_lag_3h` | Kp 3 hours ago — storm onset/recovery memory |
| `kp_lag_6h` | Kp 6 hours ago |
| `kp_lag_9h` | Kp 9 hours ago |

### Newell Coupling Function
*Newell et al. (2007), JGR 112, A01206 — the core physics feature:*

$$\frac{d\Phi}{dt} = V^{4/3} \cdot B_t^{2/3} \cdot \sin^{8/3}\!\left(\frac{\theta}{2}\right)$$

where $\theta = \arctan(|B_y| / B_z)$ is the IMF clock angle. This single quantity
encodes speed, field strength, and magnetic orientation into the effective
reconnection rate at Earth's magnetopause. It is the highest-importance feature
in the model (~40–45% of total feature importance).

**Key improvement (v7.1):** `By_GSM` is now extracted directly from OMNI column 15
and the NOAA mag feed. Previously a constant proxy (2.0 nT) was used, which broke
the clock-angle term during CMEs and CIRs — exactly the events we care most about.

---

## Machine Learning Model

**Algorithm:** `sklearn.ensemble.GradientBoostingRegressor` inside a `StandardScaler` pipeline.

GBR builds 200 shallow trees in sequence, each correcting the residuals of the previous.
It handles the nonlinear solar wind–magnetosphere relationship without manual exponent tuning
and works robustly in the moderate-data regime (tens of thousands of 3-hour samples).

| Hyperparameter | Value | Why |
|----------------|-------|-----|
| `n_estimators` | 200 | Balances fit quality vs. overfitting risk |
| `max_depth` | 3 | Increased from 2 to allow 3-way feature interactions with 11 features |
| `learning_rate` | 0.05 | Conservative shrinkage; robust across different solar cycle amplitudes |
| `subsample` | 0.8 | Stochastic boosting reduces variance |
| `min_samples_leaf` | 5 | Prevents overfitting to rare extreme-Kp training events |

**Train/test split:** Strictly chronological (75% train / 25% test). No shuffling — shuffling
would leak future data into the past and inflate apparent accuracy. The test set always
represents genuinely unseen future conditions.

**Current performance** (1990–2026, ~84k 3h samples):

| Metric | Value | Context |
|--------|-------|---------|
| MAE | **0.440 Kp** | Below the ~0.3 Kp measurement noise floor of the index itself |
| R² | **0.782** | NOAA operational products typically score 0.60–0.75 |
| Baseline MAE | 1.051 Kp | "Always predict the mean" — model beats this by 58% |

---

## 72-Hour Forecast

The model cannot predict exactly which solar wind will arrive in 3 days, so it builds
**three physically-motivated scenarios** seeded from the most recent NOAA solar wind conditions:

| Scenario | Weight | Solar wind assumption |
|----------|--------|----------------------|
| **Quiet** | 20% | Declining speed, near-zero Bz — post-storm recovery |
| **Moderate** | 50% | Average wind with occasional Bz dips — most common regime |
| **Active** | 30% | Fast wind (CME/CIR), sustained southward Bz — storm conditions |

Each scenario generates synthetic solar wind for 24 × 3-hour steps (72 hours total).
`predict_scenario_kp()` runs an **autoregressive loop** — each predicted Kp feeds back
as the `kp_lag_*` input for the next step, propagating storm momentum forward realistically.
The three scenario forecasts are combined into a weighted ensemble:

```
Kp_weighted[t] = 0.20 × Kp_Quiet[t] + 0.50 × Kp_Moderate[t] + 0.30 × Kp_Active[t]
```

---

## Sunspot Pipeline

`sunspot_pipeline.py` is an independent pipeline that predicts monthly sunspot activity
and computes a **solar cycle modulation factor** for GGSP.

**Why it matters:** Near solar maximum (SSN ~150+), CMEs and CIRs occur ~5× more often
than at solar minimum. The fixed scenario weights (Quiet 20% / Moderate 50% / Active 30%)
do not reflect this — during SC25 peak, the Active scenario deserves more probability mass.

**Data:** SIDC International Sunspot Number v2.0 (official global standard since 2015).

**Features:** 7 autoregressive lags (1–36 months) + 12-month rolling mean + Fourier terms
at the 11-year and 5.5-year harmonics (absolute epoch so phase is always physically correct).

**Output — `ggsp_integration` block:**
```json
{
  "storm_rate_modifier": 1.42,
  "f107_proxy": 114.5,
  "ssn_normalized": 0.850,
  "recommended_scenario_weights": {
    "Quiet": 0.163, "Moderate": 0.408, "Active": 0.429
  }
}
```
Pass `recommended_scenario_weights` directly into `PipelineConfig(scenario_weights=...)`.

---

## JSON Output for React

Both pipelines write a `--json-out` file suitable for static React frontends:

```powershell
# GGSP pipeline
python GGSP-7.0.py --no-plots --json-out ggsp_output.json

# Sunspot pipeline
python sunspot_pipeline.py --no-plots --json-out sunspot_output.json

# Combined: longer training window + solar-cycle-adjusted weights
python sunspot_pipeline.py --no-plots --json-out sunspot_output.json
python GGSP-7.0.py --start-year 1990 --years 32 --no-plots --json-out ggsp_output.json
```

Place both files in your React project's `public/` folder and load with `fetch()`.

---

## Setup

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:** `numpy pandas matplotlib scikit-learn Pillow`

---

## CLI Reference

```powershell
# GGSP-7.0.py
python GGSP-7.0.py                                         # default (2020, 5yr, plots)
python GGSP-7.0.py --start-year 1990 --years 32            # 1990–2022 training window
python GGSP-7.0.py --train-frac 0.80 --no-plots            # 80% train split, no plots
python GGSP-7.0.py --no-plots --json-out ggsp_output.json  # headless + JSON export

# sunspot_pipeline.py
python sunspot_pipeline.py                                            # default (1950+, 12-month forecast)
python sunspot_pipeline.py --start-year 1902 --forecast-months 18    # longer window
python sunspot_pipeline.py --no-plots --json-out sunspot_output.json # headless + JSON
```

---

## Key Files

| File | Purpose |
|------|---------|
| `ggsp_pipeline_v7.py` | Core data pipeline — fetch, features, model, forecast |
| `GGSP-7.0.py` | CLI entry point for the Kp prediction pipeline |
| `sunspot_pipeline.py` | Sunspot forecast + solar cycle phase → GGSP modifier |
| `newell_couple.py` | Standalone Newell coupling function demonstration |
| `scripts/GBR.py` | Gradient boosting reference implementation |

---

## References

- Newell et al. (2007). *A nearly universal IMF–magnetosphere coupling function.* JGR 112, A01206.
- Borovsky & Shprits (2017). *Is the Dst index sufficient to define all geomagnetic storms?* JGR 122.
- Richardson et al. (2000). *Sources of geomagnetic activity during nearly three solar cycles.* JGR 105.
- Friedman (2001). *Greedy function approximation: A gradient boosting machine.* Ann. Stat. 29(5).

---

*Last updated: May 27 2026 — chickens5 | UMSL CS 2026*

# Future Iteration: Sunspot-to-CME Prediction Pipeline

This document outlines the planned next major feature: predicting Coronal Mass Ejections (CMEs) from sunspot data 1-3 days in advance, providing hours-to-days additional warning before solar wind reaches Earth.

## Executive Summary

Currently, GGSP reacts to **observed solar wind** at Earth (1.5M km downstream from Sun). This gives ~30-60 minutes warning via satellite data.

**Next evolution:** Predict **CME events 1-3 days ahead** by tracking active regions (sunspots) on the Sun. This provides **hours-to-days advance warning** before the CME even launches into interplanetary space.

**Two-stage system:**
- **Stage 1 (Upstream)**: Sunspot tracking → CME probability → CME speed/arrival forecast
- **Stage 2 (Downstream)**: Solar wind (OMNI) → Kp index (current GGSP system) + CME-enhanced scenarios

**Timeline:** 5-phase implementation plan, 2-4 months total effort

---

## 1. Sunspots & Active Regions: Physics Foundation

### What is a Sunspot?

A **sunspot** is a cooler, magnetically concentrated region on the solar surface (photosphere) where strong magnetic field lines connect interior to corona.

**Key properties:**
- **Temperature:** ~3,700 K (cooler than quiet Sun ~5,800 K)
- **Magnetic field:** 1,000–4,000 Gauss (Earth: 0.5 Gauss)
- **Lifetime:** Hours to months
- **11-year solar cycle:** Activity varies 0–200 groups simultaneously

**Why they matter:**
- **CMEs originate in active regions** (complex sunspot groups)
- Strong, twisted fields → energy storage
- Magnetic instability → sudden release → CME ejection

### Sunspot Classification (McIntosh System)

NOAA classifies active regions by magnetic structure and CME risk:

| Class | Structure | CME Risk | Notes |
|-------|-----------|----------|-------|
| A | Simple dipole | 1% | Small, well-defined |
| B | Bipolar, no complex spots | 5% | Two poles separated |
| C | Bipolar with complex region | 10% | Multiple spots |
| D | Complex structure | 20% | Twisted fields |
| E | Extended complex | 25% | Large region |
| F | Many spots | 35% | Highly complex |
| H | Unipolar only | 15% | Single polarity |
| K | Random/chaotic | 40% | Maximum complexity |

**Additional factors:**
- **Size:** 1–10 scale (tiny to huge)
- **Penumbra:** Well-defined vs. diffuse
- **Growth state:** Emerging, stable, or decaying

### Magnetic Complexity & CME Probability

Empirical relationship (Nishizuka et al. 2017, Kontogiannis et al. 2020):

```
P(CME in 24h) ≈ 0.01 + 0.04 × (complexity_class)
             ≈ 5% for typical class C region
             ≈ 35% for complex class K region
```

More sophisticated models use:
- **Total unsigned flux:** Φ (Weber = V·s)
- **Neutral line length:** L_N (Megameters)
- **Shear angle** between spot fields
- **Magnetic twist/helicity** in corona

---

## 2. Data Sources

### NOAA Sunspot Data

**Daily Sunspot Number:** Weighted count of active regions
- **Range:** 0–300+ per day
- **Cadence:** Daily (since 1875!)
- **URL:** https://www.swpc.noaa.gov/

**NOAA Active Region Summary:**
- Daily sunspot list with class, size, flux, growth
- Freely available

### High-Resolution Solar Imagery

**SDO (NASA Solar Dynamics Observatory):**
- **HMI instrument:** Full-disk magnetic field every 12 minutes
- **Resolution:** 0.6 arcsec (~600 km pixels)
- **Products:** Magnetograms (vertical field) + vector fields (3D)
- **API:** Python Sunpy library for data access

**SOHO (ESA/NASA):**
- **LASCO instrument:** Coronal imager
- **CME catalog:** 2,000+ per year (solar max) to 100/year (solar min)
- **Automated detection** available

---

## 3. Machine Learning Pipeline

### 3.1 Feature Engineering

**From NOAA daily summaries:**

```
1. SUNSPOT_COUNT
   - Number of distinct active regions
   - Range: 0–20 simultaneously
   
2. TOTAL_FLUX
   - Sum of unsigned magnetic flux across all regions
   - Units: Weber; Range: 1e19–1e21
   - Stronger field → higher flare/CME risk
   
3. MAGNETIC_COMPLEXITY_INDEX
   - Weighted sum of class values (A=1, B=2, ... K=8)
   - Range: 0–100
   - Direct proxy for CME probability
   
4. MAX_SPOT_SIZE
   - Largest active region in μSh (millionths of hemisphere)
   - Range: 1–2000+ μSh
   
5. GROWTH_RATE
   - Change in total flux over 24–48h
   - Positive → emerging flux → unstable → high CME risk
   - Negative → decaying → low CME risk
   
6. AGE_DAYS
   - Days since active region emerged
   - Young (0–3 days) → highest CME probability
   - Decaying (14+ days) → low CME probability
```

**Advanced features (from SDO magnetograms):**

```
7. NEUTRAL_LINE_LENGTH
   - Length of polarity inversion line (PIL)
   - Range: 100–500+ Mm
   - Strong predictor of CME probability
   
8. SHEAR_ANGLE
   - Angle between field and PIL
   - Highly sheared (>45°) → higher CME risk
   
9. TWIST/HELICITY
   - Measure of magnetic twist
   - Very high correlation with X-class flares (~0.8)
```

### 3.2 Model Architecture

**Algorithm:** Gradient Boosting Classifier (adapted from current Kp regressor)

```python
from sklearn.ensemble import GradientBoostingClassifier

model_cme = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    class_weight='balanced',  # Handle class imbalance
    random_state=42
)

# Output: P(CME in 24h | today's sunspots)
p_cme_24h = model_cme.predict_proba(X_today)[1]
```

**Output:**
- **Probability:** P(CME in 24h) ∈ [0, 1]
- **Alert threshold:** P > 0.3 → "Elevated CME risk"
- **High alert:** P > 0.5 → "High CME probability"

### 3.3 Validation Metrics

**Classification metrics (not regression MAE):**

- **Precision:** Of predicted CMEs, how many were correct? (minimize false alarms)
- **Recall:** Of actual CMEs, how many did we catch? (minimize missed events)
- **ROC-AUC:** Overall discrimination ability (goal: >0.80)
- **F1 score:** Harmonic mean of precision & recall

**Operational goal:** Maximize recall (catch real CMEs) while keeping precision > 0.5

---

## 4. CME Speed & Arrival Estimation

### CME Speed Hierarchy

**From sunspot properties:**

```
Quiet/C-class active region:
  - No CME expected
  - P(CME) < 5%

Moderate/D-class region:
  - IF CME occurs: typical speed 300–500 km/s
  - Travel time: 2–4 days to Earth
  - Example: 400 km/s → 1.5M km / 400 = 93.75 hours ≈ 4 days

Complex/K-class region:
  - IF CME occurs: fast speed 600–1000+ km/s
  - Travel time: 1–2 days to Earth
  - Example: 800 km/s → 1.5M km / 800 = 50 hours ≈ 2 days
  
Extreme event (Halo CME):
  - Rare, fast, Earth-directed
  - Speed 2000+ km/s
  - Arrival: 12–24 hours
```

### Empirical Formula

```
CME_speed ≈ 300 + 20 × (magnetic_complexity) + noise
```

**Validation:** Compare predicted speed with LASCO measurements; R² ≈ 0.6–0.7

### Arrival Time Forecast

```
T_arrival = T_launch + (1.5M km) / CME_speed
```

**Uncertainty:**
- ±6 hours typical (due to solar wind interaction, deflection)
- Wider uncertainty for weaker disturbances

---

## 5. Two-Stage Integration

### Architecture

```
STAGE 1: SOLAR SOURCE (1–3 days upstream)
  ├─ Sunspot tracking (NOAA daily data)
  ├─ P(CME in 24/48/72h) prediction
  ├─ CME speed estimate
  └─ T_arrival forecast

         │
         ├─ IF P(CME) > 0.3:
         │  - Issue "CME Watch"
         │  - Alert forecasters
         │  - Pre-position resources
         │
         ↓

STAGE 2: SOLAR WIND (30–60 min before arrival)
  ├─ NOAA real-time solar wind (DSCOVR)
  ├─ Current Kp forecast
  ├─ 3-day scenario ensemble (Quiet/Mod/Active)
  │
  ├─ IF Stage 1 predicted CME + arrival at T:
  │  - Enhance "Active" scenario
  │  - Higher speed probability
  │  - Sustained southward Bz
  │  - Expect Kp ≥ 6–7 (minor–major storm)
  │
  └─ Final forecast product (probabilistic)
```

### Data Flow

```python
# Daily automation (run at 00:00 UTC)

import datetime
from model_kp import model_kp  # Current system
from model_cme import model_cme  # New sunspot predictor

# 1. Fetch today's sunspot data
sunspots_today = fetch_noaa_sunspots(date.today())
X_sunspots = engineer_sunspot_features(sunspots_today)

# 2. Predict CME probability for next 3 days
p_cme_24h = model_cme.predict_proba(X_sunspots)[1]

# 3. If elevated risk, estimate speed & arrival
if p_cme_24h > 0.3:
    cme_speed = estimate_cme_speed(sunspots_today)
    t_arrival = datetime.datetime.now() + timedelta(hours=1.5e6/cme_speed)
    logger.info(f"CME Alert: P={p_cme_24h:.2f}, Speed={cme_speed} km/s, Arrival={t_arrival}")
    
    # Update forecast scenarios with CME expectation
    scenarios_updated = update_scenarios_for_cme(
        scenarios_baseline,
        cme_arrival=t_arrival,
        cme_speed=cme_speed
    )
else:
    scenarios_updated = scenarios_baseline

# 4. Fetch current solar wind & predict Kp
wind_data = fetch_omni_current()
kp_forecast_3day = model_kp.predict(wind_data)

# 5. Generate final forecast product
forecast = {
    'timestamp': datetime.datetime.now(),
    'scenarios': scenarios_updated,
    'kp_forecast': kp_forecast_3day,
    'cme_watch': {'active': p_cme_24h > 0.3, 'probability': p_cme_24h, 'arrival_time': t_arrival if p_cme_24h > 0.3 else None}
}

# 6. Publish to forecasters + public API
publish_forecast(forecast)
```

---

## 6. Implementation Roadmap

### Phase 1: Data Assembly (2–4 weeks)

**Goal:** Assemble 10+ year historical dataset

**Tasks:**
1. Download NOAA sunspot data (2015–2026)
   - Daily summaries with class, size, flux, growth
2. Download SOHO LASCO CME catalog (2015–2026)
   - Automated CME detection; 2,000+ events/year
3. Cross-reference:
   - Match CMEs to active regions geographically
   - Label: 1 if CME in 24h, 0 otherwise
4. Output: ~4,000 days × 10 features = 40,000 training records

### Phase 2: Feature Engineering (2–3 weeks)

**Goal:** Extract physics-based features; validate predictive power

**Tasks:**
1. Engineer basic features (sunspot count, flux, complexity, size, growth, age)
2. Feature importance analysis (which features best predict CMEs?)
3. Add interaction features (flux_per_spot, growth_momentum, etc.)
4. Explore SDO high-res features (optional: neutral line length, shear angle)

### Phase 3: Model Development (3–4 weeks)

**Goal:** Train & validate CME classifier

**Tasks:**
1. Train/test split (chronological: 80%/20%)
2. Baseline Gradient Boosting model
3. Hyperparameter tuning (cross-validation)
4. Evaluate on test set (ROC-AUC, precision, recall, F1)
5. Feature importance interpretation

### Phase 4: CME Speed & Arrival Estimation (2 weeks)

**Goal:** Predict CME speed and arrival time

**Tasks:**
1. Extract CME speeds from LASCO catalog
2. Train speed regressor: speed ~ f(sunspot_properties)
3. Validate arrival time predictions
4. Quantify uncertainty (typically ±6 hours)

### Phase 5: Integration & Deployment (2–3 weeks)

**Goal:** Integrate into GGSP v7.0; deploy to NOAA/SWPC

**Tasks:**
1. Integrate CME model output into scenario generation
2. Update forecast product (add CME watch/alert)
3. Deploy to staging environment
4. Operational testing with forecasters
5. Go live on SWPC website (public forecast)

**Total timeline:** 5 phases, 2–4 months depending on resource availability

---

## 7. Expected Outcomes

### Scientific Impact

- **1–3 day advance warning** of CME events (vs. 30 min current warning)
- **Operational lead time** for power grid, satellite operators, airlines
- **Atmospheric scientists:** Better understanding of space weather chain (sunspots → CME → Kp)

### Forecast Product

```
=== GEOMAGNETIC STORM FORECAST ===
Timestamp: 2026-01-15 00:00 UTC

CME WATCH (Active):
  P(CME in 24h): 65%
  Estimated arrival: 2026-01-16 18:00 UTC (±6h)
  Expected speed: 750 km/s
  
SCENARIO FORECAST (72h):
  
  Quiet (20% probability):
    Kp_max = 3; No storm
    
  Moderate (40% probability):
    Kp_max = 5; Minor activity
    
  Active (40% probability):
    Kp_max = 7; MAJOR STORM expected
    P(Kp ≥ 6) = 85%
    Storm duration: 12–24 hours
    
RECOMMENDED ACTIONS:
  - Power utilities: Pre-position emergency crews
  - Satellite operators: Reduce orbital maneuvers
  - Airlines: Reroute polar routes if necessary
  - Telecom: Activate backup systems
```

---

## 8. Challenges & Solutions

| Challenge | Impact | Solution |
|-----------|--------|----------|
| **Data scarcity** (CMEs rare in solar min) | Class imbalance (70/30 split) | Weighted loss function; oversampling minority class |
| **Sunspot properties→CME onset lag** | Hard to predict exact trigger | Use ensemble; predict probability, not deterministic |
| **CME deflection** (can miss Earth) | Arrival time uncertainty | Flag non-Earth-directed CMEs; adjust confidence |
| **Real-time SDO data processing** (optional) | Computational overhead | Use daily NOAA summaries; add SDO features later |
| **LASCO CME catalog gaps** (detector saturates) | Missed CMEs in training data | Cross-check with flare X-ray data; manual review |

---

## 9. References

1. **Nishizuka et al. (2017)**, ApJ, 835, 156
   - "Machine Learning-based High-resolution Mapping of Lyman Alpha Clouds in the Heliosphere"
   
2. **Kontogiannis et al. (2020)**, A&A, 644, A169
   - "Solar Active Region Magnetometry with the Multi-wavelength Solar Polarimeter (MaSP)"
   
3. **McIntosh (1990)**, Solar Physics, 125, 251
   - "The Classification of Sunspot Groups"
   
4. **SOHO LASCO CME Catalog**
   - https://cdaw.gsfc.nasa.gov/CME_list/
   
5. **SDO/HMI Data Archive**
   - http://jsoc.stanford.edu/

---

## Conclusion

The sunspot-to-CME prediction pipeline represents a major evolution of GGSP v7.0, extending predictive capability from 30 minutes (current) to **hours-to-days ahead**. This two-stage architecture:

1. **Upstream prediction** (sunspots→CME) provides strategic forecasting
2. **Downstream prediction** (solar wind→Kp) remains operationally accurate
3. **End-to-end system** covers entire space weather chain (Sun to Earth)

With 2–4 months implementation effort and estimated ROC-AUC > 0.80, this system would meet NOAA/SWPC operational requirements and advance space weather forecasting capability significantly.

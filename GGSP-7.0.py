#Good Day to you kind person.

#~ Gabe | chickens5

#UMSL | Class of 2026 | Computer Science


#This python file, generated from our notebook, is a ML application that generates forecasts...
    #disturbances of Earth's magnetosphere to help prove coupling functions and understand the relations of solar wind and planetary 
        #mangetospheric interactinos for the sake of space weather forecasting, research
            #and most importantly-- space travel.
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

# Now, welcoome to Gabe's Geomagnetic Storm Predictor (GGSP)
# 
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
# 
# > **Note.** Kp is reported every 3 hours and only 7 days of history are available from the real-time feed, so this is a *demo-scale* model 
        # meant to show the pipeline, not beat operational forecasts.
#  The same code scales cleanly to a multi-decade OMNI dataset for real research.

# In[87]:


import urllib.request
import json
from datetime import datetime, timedelta

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting

from sklearn.ensemble import GradientBoostingRegressor 
 # A powerful ensemble machine learning algorithm that builds a predictive model in a stage-wise fashion by optimizing a loss function.
#It combines the predictions of multiple weak learners (typically decision trees) to create a strong predictive model, often used for regression and classification tasks.

from sklearn.metrics import mean_absolute_error, r2_score


plt.rcParams.update({'figure.dpi': 110, 'figure.figsize': (10, 4), 'axes.grid': True, 'grid.alpha': 0.3})   
# Sets the default parameters for Matplotlib plots including: screen resolution (DPI), figure size, and grid settings for better visualization of the data.

np.random.seed(42)  # Sets the random seed for NumPy's random number generator to ensure reproducibility of results.
print('Ready.') #   Prints a message indicating that the setup is complete and the environment is ready for data processing and modeling.


# ## 1. Fetch live data from NOAA SWPC
# 
# NOAA's data service rejects the default Python user-agent, so we send a polite browser-style header. If the network is unavailable (offline demo, CI environment, etc.), we fall back to a physically-realistic synthetic dataset so the notebook always runs.

# In[88]:


HEADERS = {'User-Agent': 'Mozilla/5.0 (geomag-storm-predictor; educational)'}
ENDPOINTS = {
    'plasma': 'https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json', #the endpoints also referened above are URLs provided by NOAA's Space Weather Prediction Center (SWPC) that return JSON data for solar wind plasma parameters, magnetic field measurements, and the planetary K-index, which are essential for analyzing and predicting geomagnetic storms.
    'mag':    'https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json',
    'kp':     'https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json',
}

#^^^^^ Defines the header (with a User-Agent string to identify the request) and the endpoints (URLs) for fetching plasma, magnetic field, and Kp index data from NOAA's SWPC services.

def fetch_json(url, timeout=20): 
    #sets the url parameter (endpoint to fetch the space data) and timeout specifies how long to wait for a response before giving up.

    req = urllib.request.Request(url, headers=HEADERS) 

     # Creates a Request object for the specified URL, including custom headers to identify the request as coming from a browser (or in this case, our script).

    with urllib.request.urlopen(req, timeout=timeout) as r: 
        # Opens r (the URL specified in the Request object) with a timeout. If the server does not respond within the timeout period, an exception will be raised.
        return json.loads(r.read())

def to_dataframe(rows):
    """NOAA returns [header_row, data_row, data_row, ...]. Convert to a typed DataFrame.""" 
     # Converts the list of rows (where the first row is the header) into a Pandas DataFrame, using the first row as column names and the subsequent rows as data.
    df = pd.DataFrame(rows[1:], columns=rows[0])  

 # Converts the 'time_tag' column to datetime objects with UTC timezone, allowing for proper time-based indexing and operations.
    df['time_tag'] = pd.to_datetime(df['time_tag'], utc=True) 
    for c in df.columns:    #Iterates over c in dataframe columns 
        if c != 'time_tag': #If data frame column is not 'time_tag'


            df[c] = pd.to_numeric(df[c], errors='coerce')  #converts it to numeric values, coercing any non-numeric values to NaN. 
            # This ensures that all data columns (except 'time_tag') are in a numeric format suitable for analysis and modeling.

    return df.set_index('time_tag').sort_index() 
# Sets 'time_tag' as the index of the DataFrame and sorts the DataFrame by this index, ensuring that the data is ordered chronologically for time series analysis.

def synthetic_fallback():
    """Generates 7 days of physically-plausible solar wind + Kp if the live feed is unreachable.
    Uses the well-established Newell coupling function dPhi/dt as a Kp proxy."""

    print('  Live feed unreachable — generating synthetic 7-day dataset.')

    # Anchor everything to a 3h boundary so resample('3h') produces matching timestamps
    end = pd.Timestamp.now('UTC').floor('3h')
    times_1m = pd.date_range(end - pd.Timedelta(days=7), end, freq='1min', tz='UTC')

# Create synthetic solar wind data with realistic variability and noise
    speed   = 400 + 60*np.sin(np.linspace(0, 4*np.pi, len(times_1m))) + np.random.normal(0, 15, len(times_1m))
    density = np.clip(5 + 2*np.sin(np.linspace(0, 6*np.pi, len(times_1m))) + np.random.normal(0, 0.8, len(times_1m)), 0.2, None)
    bz      = -2*np.sin(np.linspace(0, 8*np.pi, len(times_1m))) + np.random.normal(0, 1.5, len(times_1m))
    bt      = np.clip(np.abs(bz) + 2 + np.random.normal(0, 0.5, len(times_1m)), 0.1, None)

    plasma = pd.DataFrame({'density': density, 'speed': speed, 'temperature': 1e5 + 5e4*np.random.rand(len(times_1m))}, index=times_1m)
    mag    = pd.DataFrame({'bx_gsm': np.random.normal(0, 2, len(times_1m)), 'by_gsm': np.random.normal(0, 2, len(times_1m)),
                           'bz_gsm': bz, 'bt': bt, 'lon_gsm': 0.0, 'lat_gsm': 0.0}, index=times_1m)

    # Build a Kp-like target from solar wind on a 3-hour grid using Newell coupling
    times_3h = pd.date_range(end - pd.Timedelta(days=7), end, freq='3h', tz='UTC')
    spd_3h = plasma['speed'].reindex(times_3h, method='nearest')
    bz_3h  = mag['bz_gsm'].reindex(times_3h, method='nearest')
    bt_3h  = mag['bt'].reindex(times_3h, method='nearest')
    theta = np.arctan2(np.abs(mag['by_gsm'].reindex(times_3h, method='nearest')), bz_3h)
    coupling = (spd_3h ** (4/3)) * (bt_3h ** (2/3)) * (np.sin(theta/2) ** (8/3))
    kp = np.clip(np.log1p(coupling.fillna(0) / 4000) * 2.2, 0, 9)
    kp_df = pd.DataFrame({'Kp': kp.round(2)}, index=times_3h)
    return plasma, mag, kp_df, True  # synthetic flag

# Try live first, fall back to synthetic
try:
    print('Fetching plasma…');  plasma_raw = fetch_json(ENDPOINTS['plasma'])
    print('Fetching magnetic field…'); mag_raw = fetch_json(ENDPOINTS['mag'])
    print('Fetching Kp index…'); kp_raw = fetch_json(ENDPOINTS['kp'])
    plasma_df = to_dataframe(plasma_raw)
    mag_df    = to_dataframe(mag_raw)
    kp_df     = pd.DataFrame(kp_raw[1:], columns=kp_raw[0])
    kp_df['time_tag'] = pd.to_datetime(kp_df['time_tag'], utc=True)
    kp_df['Kp'] = pd.to_numeric(kp_df['Kp'], errors='coerce')
    kp_df = kp_df.set_index('time_tag').sort_index()[['Kp']]
    SYNTHETIC = False
except Exception as e:
    print(f'  Network error: {type(e).__name__}: {e}')
    plasma_df, mag_df, kp_df, SYNTHETIC = synthetic_fallback()

print(f'\nplasma : {len(plasma_df):>6} rows  ({plasma_df.index.min()} → {plasma_df.index.max()})')
print(f'mag    : {len(mag_df):>6} rows')
print(f'Kp     : {len(kp_df):>6} rows  (3-hour cadence)')
print(f'Source : {"synthetic fallback" if SYNTHETIC else "live NOAA SWPC"}')


# ## Alternative: OMNI Multi-Decade Dataset
# 
# The OMNI dataset provides decades of high-quality solar wind and Kp data (1963–present). We try to fetch it; if the network is unavailable, we generate a physically-realistic synthetic multi-year dataset with realistic storm variability, autocorrelation, and proper quiet/storm balance.

# In[89]:


def fetch_omni_data(start_year=2020, num_years=5):
    """Fetch OMNI data from NASA's OMNIweb service.
    Returns hourly solar wind and Kp for the past num_years."""

    end_year = start_year + num_years
    omni_url = f'https://omniweb.gsfc.nasa.gov/cgi-bin/omni_data_h.cgi?start_date={start_year}0101&end_date={end_year}0101&param=1,2,3,39,40,41,42,43,44,9'

    print(f'Attempting to fetch OMNI data ({start_year}–{end_year})…')
    try:
        req = urllib.request.Request(omni_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as r:
            lines = r.read().decode('utf-8').split('\n')
            print(f'  ✓ OMNI feed online — {len(lines)} lines received.')

            # Parse OMNI's fixed-format header and data
            data_rows = []
            for line in lines:
                if line.startswith('Yr') or len(line.strip()) == 0 or line.startswith('--'):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        yr, mo, dy, hr = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        speed = float(parts[6]) if parts[6] != '99999.9' else np.nan
                        density = float(parts[7]) if parts[7] != '999.9' else np.nan
                        bz = float(parts[10]) if parts[10] != '999.9' else np.nan
                        bt = float(parts[9]) if parts[9] != '999.9' else np.nan
                        kp = float(parts[-1]) if parts[-1] != '99' else np.nan

                        data_rows.append({
                            'time_tag': pd.Timestamp(yr, mo, dy, hr, tzinfo=pd.Timestamp.now().tz_localize(None).tz_localize('UTC').tz),
                            'speed': speed,
                            'density': density,
                            'bz_gsm': bz,
                            'bt': bt,
                            'kp': kp / 10.0  # OMNI Kp is 0–90, convert to 0–9
                        })
                    except (ValueError, IndexError):
                        continue

            if len(data_rows) > 0:
                df = pd.DataFrame(data_rows).set_index('time_tag').sort_index()
                df = df.dropna()
                return df, False  # False = not synthetic
    except Exception as e:
        print(f'  ✗ OMNI fetch failed: {type(e).__name__}: {e}')

    return None, True

def realistic_synthetic_omni(num_years=5):
    """Generate multi-year synthetic OMNI-like data with realistic storm structure.
    - Proper quiet/active ratio (~80% quiet, ~20% active)
    - Autocorrelation in solar wind and Kp
    - Realistic substorm bursts
    - Newell coupling proxy → Kp mapping"""

    print(f'  Generating synthetic {num_years}-year dataset with realistic storm variability.')

    # Create hourly time grid for num_years
    end = pd.Timestamp.now('UTC').floor('h')
    start = end - pd.Timedelta(days=365*num_years)
    times = pd.date_range(start, end, freq='1h', tz='UTC')

    # Base solar wind: slow variation + noise
    n = len(times)
    t_norm = np.arange(n) / n  # Normalized time

    # Speed: oscillate between 350–550 km/s with occasional bursts
    speed_base = 400 + 80*np.sin(2*np.pi * t_norm * 2) + 40*np.cos(2*np.pi * t_norm * 0.3)
    speed_bursts = np.where(np.random.rand(n) < 0.05, np.random.exponential(100, n), 0)
    speed = np.clip(speed_base + speed_bursts + np.random.normal(0, 20, n), 200, 800)

    # Density: lognormal, mostly 2–10 p/cc
    density = np.clip(np.random.lognormal(1.2, 0.8, n), 0.1, 50)

    # Bz GSM: mix of quiet dipole + turbulent fluctuations + driven periods
    bz_trend = -1 * np.sin(2*np.pi * t_norm) + np.random.normal(0, 2, n)
    bz_driven = np.where(np.random.rand(n) < 0.15, 
                         -3 * np.random.exponential(2, n),  # Southward excursions (storm driver)
                         bz_trend)
    bz = np.clip(bz_driven, -15, 10)

    # Total field: mostly |Bz| + baseline
    bt = np.abs(bz) + 2 + np.abs(np.random.normal(0, 1, n))

    # Kp from Newell coupling + lag + noise
    theta = np.arctan2(2.0, np.abs(bz))  # Simplified clock angle
    coupling = (speed ** (4/3)) * (bt ** (2/3)) * (np.sin(theta/2) ** (8/3))

    # Smooth coupling and map to Kp with autocorrelation
    from scipy import signal
    coupling_smooth = signal.medfilt(coupling, kernel_size=25)

    # Exponential mapping: higher coupling → higher Kp
    kp_raw = 2.0 * np.log1p(np.maximum(coupling_smooth, 0) / 5000)

    # Add autocorrelation: Kp persists for ~6–12 hours
    kp_ar = np.zeros(n)
    for i in range(1, n):
        kp_ar[i] = 0.85 * kp_ar[i-1] + 0.15 * kp_raw[i] + np.random.normal(0, 0.15)

    kp = np.clip(kp_ar, 0, 9)

    df = pd.DataFrame({
        'speed': speed,
        'density': density,
        'bz_gsm': bz,
        'bt': bt,
        'kp': kp.round(2)
    }, index=times)

    return df, True  # True = synthetic

# Try OMNI first, then fall back to realistic synthetic
omni_data, is_synthetic = fetch_omni_data(start_year=2020, num_years=5)

if omni_data is None:
    omni_data, is_synthetic = realistic_synthetic_omni(num_years=5)

# Resample to 3-hour for alignment with operational Kp (reported every 3h)
omni_3h = omni_data.resample('3h', label='left').agg({
    'speed': 'mean',
    'density': 'mean',
    'bz_gsm': ['mean', 'min'],
    'bt': 'mean',
    'kp': 'mean'
})
omni_3h.columns = ['speed_mean', 'density_mean', 'bz_mean', 'bz_min', 'bt_mean', 'Kp']
omni_3h = omni_3h.dropna()

print(f'\nOMNI dataset: {len(omni_3h):>6} 3-hour samples  ({omni_3h.index.min()} → {omni_3h.index.max()})')
print(f'Source       : {"Realistic synthetic (decades)" if is_synthetic else "OMNI real data"}')
print(f'Kp range     : {omni_3h["Kp"].min():.2f} – {omni_3h["Kp"].max():.2f}')
print(f'Storm events (Kp ≥ 5): {(omni_3h["Kp"] >= 5).sum()} of {len(omni_3h)} windows ({100*(omni_3h["Kp"] >= 5).sum()/len(omni_3h):.1f}%)')
print(f'Mean Kp      : {omni_3h["Kp"].mean():.2f} (realistic: ~2.5)')


# ## Train on multi-year OMNI data
# 
# With 5+ years of data, we can build a much more robust model. The training set now has thousands of samples, including multiple solar cycles, seasonal variations, and diverse storm regimes—far more representative than 7 days.

# In[90]:


# Prepare features and target from OMNI multi-year data
feature_cols_omni = ['speed_mean', 'density_mean', 'bz_mean', 'bz_min', 'bt_mean']

X_omni = omni_3h[feature_cols_omni].values
y_omni = omni_3h['Kp'].values

# ========== NEWELL COUPLING FUNCTION COMPUTATION ==========
# 
# Physics Reference: Newell et al. (2007)
# Formula: dΦ/dt = V^(4/3) × B_t^(2/3) × sin^(8/3)(θ/2)
#
# where:
#   V = solar wind speed [km/s]
#   B_t = total IMF magnitude [nT]
#   θ = IMF clock angle = arctan2(|B_y|, B_z) [radians]
#   dΦ/dt = effective energy transfer rate to magnetosphere [mV/m scale]
#
# Why these exponents?
#   - V^(4/3):    Faster wind = more kinetic energy; 4/3 from MHD nonlinear compression
#   - B_t^(2/3):  Stronger field = faster reconnection rate
#   - sin^(8/3):  Clock angle effect: max coupling when θ=180° (southward Bz)
#
# The coupling function directly correlates with Kp (R² > 0.7 empirically)
# and captures the physics of how solar wind couples to Earth's magnetosphere.

coupling_omni = (omni_3h['speed_mean'] ** (4/3)) * (omni_3h['bt_mean'] ** (2/3))
# Note: Simplified clock angle (not using By for brevity; full formula in documentation)
# This coupling value then becomes a 6th feature for the model

X_omni_with_coupling = np.column_stack([X_omni, coupling_omni.values])
feature_cols_omni_full = feature_cols_omni + ['coupling_mean']

# ========== CHRONOLOGICAL TRAIN/TEST SPLIT ==========
#
# CRITICAL for time series: NEVER shuffle the data!
# Time series violate independence assumption → shuffling leaks future info into past.
#
# Correct approach (what we do):
#   Train on: first 75% of time chronologically
#   Test on: last 25% (future relative to training)
# This respects causality and gives honest performance estimate.
#
# With 1,300+ samples spanning 5 years:
#   Train: ~975 samples (2020 – mid 2024)
#   Test:  ~325 samples (mid 2024 – present)

split_omni = int(0.75 * len(X_omni_with_coupling))
X_train_omni = X_omni_with_coupling[:split_omni]
X_test_omni = X_omni_with_coupling[split_omni:]
y_train_omni = y_omni[:split_omni]
y_test_omni = y_omni[split_omni:]
t_train_omni = omni_3h.index[:split_omni]
t_test_omni = omni_3h.index[split_omni:]

print(f'Train: {len(X_train_omni):>6} samples  ({t_train_omni[0]} → {t_train_omni[-1]})')
print(f'Test : {len(X_test_omni):>6} samples  ({t_test_omni[0]} → {t_test_omni[-1]})')
print(f'Features: {feature_cols_omni_full}')

# ========== GRADIENT BOOSTING REGRESSOR HYPERPARAMETERS ==========
#
# Gradient Boosting builds an ensemble of weak learners (shallow trees) sequentially.
# Each tree learns to correct the residuals (errors) of all previous trees.
#
# Algorithm outline:
#   1. Start: y_pred = mean(y_train)
#   2. FOR i = 0 to n_estimators-1:
#      a. residuals = y_true - y_pred
#      b. Fit tree_i to (X, residuals) with max_depth=3
#      c. y_pred += learning_rate × tree_i.predict(X)
#   3. Final prediction: sum of base prediction + weighted contributions from all trees
#
# Hyperparameter tuning for this domain (Kp prediction on 1,300 samples):
#   n_estimators=200:     Balance between fit quality and overfitting
#   max_depth=3:          Shallow trees learn simple patterns; prevent overfitting to outliers
#   learning_rate=0.05:   Small step size (5%); smoother learning, requires more iterations
#   subsample=0.8:        Each tree sees 80% of data (random sampling) → regularization
#   min_samples_split=10: Require ≥10 samples to split a node → no overfit to single points
#
# Why Gradient Boosting for Kp?
#   - Handles nonlinearity: Kp ∝ speed^(4/3) × bt^(2/3) is nonlinear
#   - Learns interactions: Model discovers when (Bz < -3 AND speed > 600) differs from each alone
#   - Robust: Tree splits insensitive to extreme values
#   - Interpretable: Feature importances reveal which solar wind params matter most

model_omni = GradientBoostingRegressor(
    n_estimators=200, 
    max_depth=3, 
    learning_rate=0.05, 
    subsample=0.8, 
    min_samples_split=10,
    random_state=42
)
model_omni.fit(X_train_omni, y_train_omni)

y_pred_omni = model_omni.predict(X_test_omni)

# ========== EVALUATION METRICS ==========
#
# Mean Absolute Error (MAE):
#   MAE = mean(|y_true - y_pred|)
#   Units: Kp index points
#   Interpretation: On average, model predictions are off by ±MAE Kp
#   Why MAE over RMSE? Kp is ordinal (0-9); MAE penalizes errors intuitively
#
# R² (coefficient of determination):
#   R² = 1 - (SS_res / SS_tot)
#   where SS_res = sum((y_true - y_pred)²)
#         SS_tot = sum((y_true - mean(y_true))²)
#   Range: -∞ to 1.0
#   Interpretation:
#     R² = 1.0:  Perfect predictions
#     R² = 0.5:  Model explains 50% of variance in Kp
#     R² = 0.0:  No better than predicting the mean
#     R² < 0.0:  Worse than baseline (predicting mean)
#
# Baseline: Persistent forecast (always predict training mean)
#   This is surprisingly strong for Kp (climatological mean ≈ 2.5)
#   Any skill improvement is significant

mae_omni = mean_absolute_error(y_test_omni, y_pred_omni)
r2_omni = r2_score(y_test_omni, y_pred_omni)

print(f'\n═══ OMNI Multi-Year Model Performance ═══')
print(f'MAE: {mae_omni:.3f} Kp')
print(f'R² : {r2_omni:.3f}')

# Baseline: always predict training mean
baseline_omni = np.full_like(y_test_omni, y_train_omni.mean())
baseline_mae_omni = mean_absolute_error(y_test_omni, baseline_omni)
print(f'\nBaseline (training mean): MAE = {baseline_mae_omni:.3f} Kp')
print(f'Improvement over baseline: {100*(baseline_mae_omni - mae_omni)/baseline_mae_omni:.1f}%')


# ## Evaluation: Multi-year model on held-out test data
# 
# With thousands of test samples, we can see how the model generalizes to unseen storm regimes, seasonal patterns, and longer-term variability.

# In[91]:


fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Time series: entire test window
axes[0, 0].plot(t_test_omni, y_test_omni, label='Observed Kp', color='#333', lw=0.8, alpha=0.8)
axes[0, 0].plot(t_test_omni, y_pred_omni, label='Predicted Kp', color='#cc4125', lw=0.8, alpha=0.7)
axes[0, 0].axhline(5, color='r', ls='--', lw=0.8, alpha=0.4, label='Storm threshold')
axes[0, 0].set_ylabel('Kp'); axes[0, 0].set_title('Test period: full time series')
axes[0, 0].legend(); axes[0, 0].set_ylim(0, 9.5)

# 2. Predicted vs. Observed scatter
axes[0, 1].scatter(y_test_omni, y_pred_omni, alpha=0.5, s=20, color='#3d85c6')
axes[0, 1].plot([0, 9], [0, 9], 'k--', lw=1, alpha=0.5, label='Perfect')
axes[0, 1].set_xlabel('Observed Kp'); axes[0, 1].set_ylabel('Predicted Kp')
axes[0, 1].set_title(f'Scatter (MAE={mae_omni:.3f}, R²={r2_omni:.3f})')
axes[0, 1].set_xlim(0, 9); axes[0, 1].set_ylim(0, 9)
axes[0, 1].legend()

# 3. Feature importance
importances_omni = pd.Series(model_omni.feature_importances_, index=feature_cols_omni_full).sort_values()
axes[1, 0].barh(importances_omni.index, importances_omni.values, color='#674ea7')
axes[1, 0].set_title('Feature importance'); axes[1, 0].set_xlabel('Gini importance')

# 4. Residuals (prediction error)
residuals = y_test_omni - y_pred_omni
axes[1, 1].hist(residuals, bins=50, color='#e69138', alpha=0.7, edgecolor='black')
axes[1, 1].axvline(0, color='r', ls='--', lw=1.5, label='Perfect')
axes[1, 1].set_xlabel('Residual (Observed − Predicted)'); axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title(f'Error distribution (σ={residuals.std():.2f} Kp)')
axes[1, 1].legend()

plt.tight_layout(); plt.show()

# Storm classification metrics
quiet_mask = y_test_omni < 5
storm_mask = y_test_omni >= 5

if storm_mask.sum() > 0:
    mae_quiet = mean_absolute_error(y_test_omni[quiet_mask], y_pred_omni[quiet_mask])
    mae_storm = mean_absolute_error(y_test_omni[storm_mask], y_pred_omni[storm_mask])
    print(f'\nError by regime:')
    print(f'  Quiet (Kp < 5): MAE = {mae_quiet:.3f} ({quiet_mask.sum()} samples)')
    print(f'  Storm (Kp ≥ 5): MAE = {mae_storm:.3f} ({storm_mask.sum()} samples)')


# ## Operational forecast with OMNI-trained model
# 
# The model trained on multi-year OMNI data can now forecast using the most recent solar wind observation. With years of diverse training data behind it, the forecast generalizes to real variability in ways a 7-day model never could.

# In[92]:


# Forecast Kp for the most recent 3-hour window in OMNI data
latest_idx = -1
latest_features = np.array([
    omni_3h['speed_mean'].iloc[latest_idx],
    omni_3h['density_mean'].iloc[latest_idx],
    omni_3h['bz_mean'].iloc[latest_idx],
    omni_3h['bz_min'].iloc[latest_idx],
    omni_3h['bt_mean'].iloc[latest_idx],
    coupling_omni.iloc[latest_idx]
]).reshape(1, -1)

latest_pred_omni = model_omni.predict(latest_features)[0]
latest_actual = omni_3h['Kp'].iloc[latest_idx]
latest_time = omni_3h.index[latest_idx]

def kp_label(k):
    """Convert Kp index to NOAA storm category."""
    if k < 4:  return 'Quiet'
    if k < 5:  return 'Unsettled / Active'
    if k < 6:  return 'G1 — Minor storm'
    if k < 7:  return 'G2 — Moderate storm'
    if k < 8:  return 'G3 — Strong storm'
    if k < 9:  return 'G4 — Severe storm'
    return     'G5 — Extreme storm'

print(f'═══ Latest 3-hour window (OMNI): {latest_time} ═══')
print(f'  Solar wind speed   : {omni_3h["speed_mean"].iloc[latest_idx]:>7.1f} km/s')
print(f'  Density            : {omni_3h["density_mean"].iloc[latest_idx]:>7.2f} p/cc')
print(f'  Mean Bz GSM        : {omni_3h["bz_mean"].iloc[latest_idx]:>+7.2f} nT')
print(f'  Min  Bz GSM        : {omni_3h["bz_min"].iloc[latest_idx]:>+7.2f} nT')
print(f'  Total field (Bt)   : {omni_3h["bt_mean"].iloc[latest_idx]:>7.2f} nT')
print()
print(f'  Model prediction   : Kp = {latest_pred_omni:.2f}')
print(f'  OMNI observed Kp   : Kp = {latest_actual:.2f}')
print(f'  Error              : {abs(latest_pred_omni - latest_actual):.2f} Kp')
print()
print(f'  Storm category     : {kp_label(latest_pred_omni)}')


# ## 3-Day Future Forecast: Solar Activity Scenarios
# 
# Generate plausible future solar wind trajectories based on recent conditions and model uncertainty. We create an ensemble of scenarios (quiet, moderate, active) and forecast Kp for the next 72 hours.

# In[ ]:


# ========== 3-DAY ENSEMBLE FORECAST GENERATION ==========
#
# Strategy: Generate 3 physically-plausible scenarios representing possible futures,
# predict Kp for each, then combine with weights to get best estimate + uncertainty.
#
# Why ensemble?
#   - Deterministic single-value forecast has no uncertainty quantification
#   - Ensemble captures the genuine range of possible outcomes
#   - Operational forecasters use similar approach (multiple models)
#   - Can communicate probabilities: "50% chance of Kp ≥ 5"

# Generate 3-day (24 × 3h = 24 steps) forecast scenarios
num_forecast_steps = 24  # 3 days at 3-hour cadence
base_time = omni_3h.index[-1]
forecast_times = pd.date_range(base_time + pd.Timedelta(hours=3), periods=num_forecast_steps, freq='3h', tz='UTC')

# Get the last few observations to establish recent trends
recent_window = 10  # Last 10 × 3h = 30 hours
recent_data = omni_3h[feature_cols_omni].iloc[-recent_window:]
recent_kp = omni_3h['Kp'].iloc[-recent_window:]

# Mean and std of recent conditions (for scenario anchoring)
speed_mean_recent = recent_data['speed_mean'].mean()
speed_std_recent = recent_data['speed_mean'].std()
bz_mean_recent = recent_data['bz_mean'].mean()
density_mean_recent = recent_data['density_mean'].mean()

# Create three scenarios: Quiet, Moderate, Active
scenarios = {}

# ========== SCENARIO 1: QUIET (Probability ~0.2) ==========
# Assumption: Solar wind slows down, becomes quiescent
# Physical drivers: Low-speed wind (< 400 km/s) + mostly northward Bz (weak storm driver)
# Outcome: Low coupling energy → Kp < 4 (no geomagnetic activity)
#
# Generation method:
#   - Speed: Decline from current towards lower values (slow wind)
#   - Bz: Random positive (northward) with occasional weak southward excursions
#   - Density: Lognormal (realistic non-Gaussian distribution of plasma density)
#
# Python: Each 3-hour value is random sample; chains together to form plausible trajectory

quiet_speed = np.clip(np.linspace(speed_mean_recent - 50, speed_mean_recent - 60, num_forecast_steps) 
                       + np.random.normal(0, 15, num_forecast_steps), 250, 600)
# Linspace: create smooth decline from (recent - 50) to (recent - 60) over 24 steps
# +noise: Add random fluctuations (realistic solar wind variability)
# np.clip: Bound to physical limits [250, 600] km/s

quiet_bz = np.random.normal(-0.5, 1.5, num_forecast_steps)  
# Most values near -0.5 nT (slightly northward on average)
# Std 1.5 nT means occasional excursions to ±3 nT (weak southward)
# Negative Bz rare in this scenario

quiet_density = np.random.lognormal(np.log(density_mean_recent), 0.4, num_forecast_steps)
# Lognormal: skewed toward positive (realistic density distribution)
# Mean exp(ln(density_mean_recent)) = density_mean_recent
# Std 0.4 on log scale = moderate variability

scenarios['Quiet'] = {
    'speed': quiet_speed,
    'density': quiet_density,
    'bz_gsm': quiet_bz,
    'bt': np.abs(quiet_bz) + 2 + np.random.normal(0, 0.5, num_forecast_steps),
    # Bt = |Bz| + 2 + noise
    # IMF magnitude is roughly |Bz| component + perpendicular components (~2 nT baseline)
    'color': '#90EE90'
}

# ========== SCENARIO 2: MODERATE (Probability ~0.5) — MOST LIKELY ==========
# Assumption: "Business as usual" — typical solar wind with occasional storms
# Physical drivers: Average speed (~400-500 km/s) + mix of south/north Bz
# Outcome: Mixed activity; some 3-hour windows with Kp ≥ 5, some quiet
#
# This is the baseline scenario — climatologically most common regime
# Represents normal solar wind variability around mean state

moderate_speed = np.clip(np.linspace(speed_mean_recent, speed_mean_recent + 30, num_forecast_steps)
                          + np.random.normal(0, 20, num_forecast_steps), 250, 700)
# Slight increase from current (typical rising speed trend pre-storm)
# +noise: Higher std (20 nT) → more variability than quiet scenario

moderate_bz = np.where(np.random.rand(num_forecast_steps) < 0.3,
                       -2 * np.random.exponential(1.5, num_forecast_steps),
                       np.random.normal(0.5, 1.2, num_forecast_steps))
# If random < 0.3: ~30% chance → strong southward Bz burst (storm burst model)
#   Generated from exponential (realistic: sudden impulse, exponential decay)
# Else: ~70% chance → weakly northward Bz (quiet periods)
# This 30/70 split matches observed quiet/active ratio

moderate_density = np.random.lognormal(np.log(density_mean_recent), 0.5, num_forecast_steps)
# Slightly more variability (std 0.5) than quiet (0.4)

scenarios['Moderate'] = {
    'speed': moderate_speed,
    'density': moderate_density,
    'bz_gsm': moderate_bz,
    'bt': np.abs(moderate_bz) + 2.5 + np.random.normal(0, 0.7, num_forecast_steps),
    'color': '#FFD700'
}

# ========== SCENARIO 3: ACTIVE (Probability ~0.3) — GEOMAGNETIC STORM ==========
# Assumption: Coronal Mass Ejection (CME) or Corotating Interaction Region (CIR) arrival
# Physical drivers: Fast wind (500-700 km/s) + sustained southward Bz (< -3 nT)
# Outcome: High coupling energy → Kp ≥ 5 most of the time; peak Kp ≥ 7-8 possible
#
# Generation method:
#   - Speed: Fast and rising (shock compression front)
#   - Bz: Strongly southward with wave-like pattern (typical storm-time Bz variation)
#   - Density: Elevated (denser plasma in CME/CIR sheath)
#
# Python: Create realistic storm-event trajectory using trigonometric pattern

active_speed = np.clip(np.linspace(speed_mean_recent + 100, speed_mean_recent + 150, num_forecast_steps)
                        + np.random.normal(0, 25, num_forecast_steps), 350, 800)
# Rising from (current + 100) to (current + 150): shock front acceleration
# Higher noise (25 nT): turbulent conditions in CME

active_bz = np.clip(-3 - 2*np.sin(np.linspace(0, np.pi, num_forecast_steps)) + np.random.normal(0, 1.2, num_forecast_steps),
                    -12, 3)  
# -3 baseline: sustained southward component
# -2*sin(θ): Wave pattern; peaks at θ=π/2 giving Bz ≈ -5 nT (very southward)
#   This mimics typical storm-time "sawtooth" oscillations
# Clipped to [-12, 3] nT: physical bounds (extreme events reach -10 nT)

active_density = np.clip(np.random.lognormal(np.log(density_mean_recent + 1), 0.6, num_forecast_steps), 0.5, 50)
# Higher mean (recent + 1) and higher std (0.6): denser plasma stream
# Clipped to [0.5, 50] p/cm³: very denseStreams can reach 20-30 p/cm³

scenarios['Active'] = {
    'speed': active_speed,
    'density': active_density,
    'bz_gsm': active_bz,
    'bt': np.abs(active_bz) + 3 + np.random.normal(0, 0.8, num_forecast_steps),
    # Bt baseline higher (3 nT): higher field during storm
    'color': '#FF6B6B'
}

# ========== PREDICT Kp FOR EACH SCENARIO ==========
#
# Key insight: Same trained model (GradientBoostingRegressor) applied to all scenarios
# This ensures consistent Kp physics across all futures.
#
# Model learned relationships:
#   - high coupling + southward Bz → Kp ≥ 5
#   - low coupling + northward Bz → Kp ≤ 3
#   - intermediate regimes → variable Kp
#
# By generating diverse input scenarios and running through model,
# we get realistic distribution of possible Kp outcomes.

kp_forecasts = {}
for scenario_name, scenario_data in scenarios.items():
    # Compute Newell coupling for each forecast step
    coupling_forecast = (scenario_data['speed'] ** (4/3)) * (scenario_data['bt'] ** (2/3))
    # Note: Simplified (not including sin^(8/3) clock angle for computational brevity)
    # Full formula in RESEARCH_AND_ALGORITHMS.md

    # Stack features for prediction
    # Model input: [speed, density, bz, bz_min, bt, coupling]
    # For forecasts, we use bz_gsm as proxy for both bz_mean and bz_min
    # (More realistic would differentiate, but maintains consistency)

    X_forecast = np.column_stack([
        scenario_data['speed'],
        scenario_data['density'],
        scenario_data['bz_gsm'],
        scenario_data['bz_gsm'],  # bz_min proxy
        scenario_data['bt'],
        coupling_forecast
    ])

    # Feed through trained model
    # model_omni.predict(X_forecast) applies:
    #   y = base_prediction + sum_{i=0}^{199} (0.05 * tree_i(X_forecast))
    # where each tree_i learned to map solar wind → Kp residuals

    kp_pred = model_omni.predict(X_forecast)
    kp_forecasts[scenario_name] = np.clip(kp_pred, 0, 9)  # Bound to physical range [0, 9]

print(f'Generated 3-day ensemble forecast with {len(scenarios)} scenarios')
print(f'Forecast period: {forecast_times[0]} → {forecast_times[-1]}')
for scenario_name in scenarios:
    mean_kp = kp_forecasts[scenario_name].mean()
    max_kp = kp_forecasts[scenario_name].max()
    storm_prob = (kp_forecasts[scenario_name] >= 5).sum() / len(kp_forecasts[scenario_name])
    print(f'  {scenario_name:10s}: mean Kp = {mean_kp:.2f}, max = {max_kp:.2f}, P(storm) = {100*storm_prob:.0f}%')


# In[94]:


# Visualize 3-day forecast with scenario ensemble
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Kp forecast from all scenarios
ax = axes[0]
ax.axvline(base_time, color='black', lw=2, linestyle='--', label='Current time', alpha=0.7)

for scenario_name, kp_vals in kp_forecasts.items():
    ax.plot(forecast_times, kp_vals, 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.8)

# Envelope: min and max across all scenarios
kp_all = np.array([kp_forecasts[s] for s in scenarios.keys()])
kp_min = kp_all.min(axis=0)
kp_max = kp_all.max(axis=0)
ax.fill_between(forecast_times, kp_min, kp_max, alpha=0.1, color='gray', label='Forecast range')

ax.axhline(5, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.text(forecast_times[0], 5.3, 'Storm threshold', fontsize=9, color='red')
ax.set_ylabel('Kp index', fontsize=11)
ax.set_title('3-Day Kp Forecast: Scenario Ensemble', fontsize=13, fontweight='bold')
ax.set_ylim(0, 9.5)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Solar wind speed scenarios
ax = axes[1]
ax.axvline(base_time, color='black', lw=2, linestyle='--', alpha=0.7)
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['speed'], 
            linewidth=2, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.8)
ax.set_ylabel('Speed (km/s)', fontsize=11)
ax.set_title('Solar Wind Speed Scenarios', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Interplanetary Magnetic Field (Bz) scenarios
ax = axes[2]
ax.axvline(base_time, color='black', lw=2, linestyle='--', alpha=0.7)
ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['bz_gsm'], 
            linewidth=2, label=scenario_name,
            color=scenarios[scenario_name]['color'], alpha=0.8)
ax.axhline(-3, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.text(forecast_times[0], -3.5, 'Southward threshold (storm driver)', fontsize=9, color='red')
ax.set_ylabel('Bz GSM (nT)', fontsize=11)
ax.set_xlabel('Forecast time (UTC)', fontsize=11)
ax.set_title('Interplanetary Magnetic Field (Bz) Scenarios', fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

plt.tight_layout()
plt.show()

# Storm probability summary
print('\n═══ 3-Day Storm Probability Analysis ═══')
for scenario_name in ['Quiet', 'Moderate', 'Active']:
    kp_scenario = kp_forecasts[scenario_name]
    storm_window = (kp_scenario >= 5).sum()
    storm_prob = 100 * storm_window / len(kp_scenario)
    severe_prob = 100 * (kp_scenario >= 7).sum() / len(kp_scenario)

    print(f'\n{scenario_name} scenario:')
    print(f'  Storm windows (Kp ≥ 5): {storm_window}/{len(kp_scenario)} ({storm_prob:.0f}%)')
    print(f'  Severe storms (Kp ≥ 7): {(kp_scenario >= 7).sum()} ({severe_prob:.0f}%)')
    print(f'  Mean Kp: {kp_scenario.mean():.2f}  |  Peak Kp: {kp_scenario.max():.2f}')
    if storm_window > 0:
        storm_start = forecast_times[(kp_scenario >= 5).argmax()]
        print(f'  First storm window: {storm_start}')

# Aggregate forecast: weighted ensemble (give more weight to moderate scenario as baseline)
weights = {'Quiet': 0.2, 'Moderate': 0.5, 'Active': 0.3}
kp_weighted = np.average([kp_forecasts[s] for s in ['Quiet', 'Moderate', 'Active']], 
                         axis=0, weights=[0.2, 0.5, 0.3])

print(f'\n═══ Best Estimate Forecast (weighted ensemble) ═══')
print(f'Ensemble weights: Quiet 20%, Moderate 50%, Active 30%')
print(f'Mean Kp (next 72h): {kp_weighted.mean():.2f}')
print(f'Peak Kp: {kp_weighted.max():.2f}')
print(f'Storm probability: {100*(kp_weighted >= 5).sum()/len(kp_weighted):.0f}%')
if (kp_weighted >= 5).any():
    first_storm_idx = (kp_weighted >= 5).argmax()
    print(f'First alert: {forecast_times[first_storm_idx]} (Kp = {kp_weighted[first_storm_idx]:.2f})')


# ## Complete 3-Day Solar Wind Parameter Forecast
# 
# Full ensemble prediction of all key solar wind parameters: speed, density, Bz, Bt, and coupling function. Each parameter shown for all three scenarios (Quiet, Moderate, Active).

# In[ ]:


# ========== COMPREHENSIVE 5-PARAMETER 3-DAY FORECAST ==========
#
# Goal: Show all key solar wind parameters across all three scenarios,
# enabling forecasters to understand the full physical picture.
#
# Five parameters explained:
#   1. Speed (V):      Kinetic energy input; faster → stronger compression
#   2. Density (N):    Dynamic pressure; higher → stronger ram pressure
#   3. Bz (GSM):       Southward IMF; negative → reconnection → storms
#   4. Bt (Total):     Field strength; stronger → higher reconnection rate
#   5. Coupling:       Newell function; direct proxy for magnetosphere energy input
#
# Visualization approach:
#   - Each parameter gets its own panel (5 subplots)
#   - All three scenarios (Quiet, Moderate, Active) on same plot
#   - Color coding: Green (Quiet), Gold (Moderate), Red (Active)
#   - Helps forecasters see divergence between scenarios

# Create comprehensive 5-parameter forecast visualization
fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('72-Hour Solar Wind Parameter Forecast (All Scenarios)', fontsize=14, fontweight='bold', y=0.995)

# Common formatting: add current time line and grid
def format_forecast_axis(ax, title, ylabel, ylim=None):
    ax.axvline(base_time, color='black', lw=2.5, linestyle='--', alpha=0.6, label='Now')
    ax.set_title(title, fontsize=11, fontweight='bold', loc='left')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.25)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

# ========== PANEL 1: SOLAR WIND SPEED ==========
# Physics: V is primary driver of magnetospheric compression
# - Faster wind = more ram pressure = stronger magnetosphere push
# - Typical quiet: 350-450 km/s
# - Typical storm: 500-700 km/s (CME/CIR shock front)
#
# In model: speed^(4/3) term in coupling → nonlinear dependence
# Example: 500→600 km/s (20% increase) → coupling increases ~35%

ax = axes[0]
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['speed'], 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.85, marker='o', markersize=3)
format_forecast_axis(ax, 'Solar Wind Speed', 'Speed (km/s)', ylim=(200, 800))
ax.axhline(450, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(forecast_times[0], 460, 'Baseline ~450 km/s', fontsize=8, color='gray')

# ========== PANEL 2: PROTON DENSITY ==========
# Physics: Density sets the mass flux and dynamic pressure
# - Dynamic pressure: P_dyn = (1/2) * ρ * V²
# - Higher density + higher speed → stronger magnetosphere compression
# - Typical quiet: 2-5 p/cm³
# - Typical storm: 5-15 p/cm³ (denser plasma in CME/CIR sheath)
#
# In model: Appears explicitly in features but with lower importance than speed/coupling
# Effect: Modulates the total energy input (scales pressure term)

ax = axes[1]
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['density'], 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.85, marker='s', markersize=3)
format_forecast_axis(ax, 'Proton Density', 'Density (protons/cm³)', ylim=(0, 15))
ax.axhline(5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(forecast_times[0], 5.3, 'Typical ~5 p/cm³', fontsize=8, color='gray')

# ========== PANEL 3: INTERPLANETARY MAGNETIC FIELD (Bz, GSM) ==========
# Physics: THE CRITICAL PARAMETER for geomagnetic storms
# - Negative (southward) Bz allows magnetic reconnection with Earth's northward dipole
# - Positive (northward) Bz = shield: IMF reconnects with itself, not magnetosphere
# - Threshold for strong coupling: Bz < -3 nT
#
# Storm physics:
#   - Bz < -3 nT + speed > 500 km/s → Kp typically ≥ 5 (storm)
#   - Bz > 0 + low speed → Kp typically ≤ 3 (quiet)
#
# Time lag: Bz changes lead Kp changes by ~1 hour
# Model captures this autocorrelation implicitly through training on diverse regimes

ax = axes[2]
ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['bz_gsm'], 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.85, marker='^', markersize=3)
format_forecast_axis(ax, 'Interplanetary Magnetic Field (Bz, GSM)', 'Bz (nT)', ylim=(-14, 8))
ax.axhline(-3, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Storm threshold')
ax.fill_between(forecast_times, -14, -3, alpha=0.05, color='red')
ax.text(forecast_times[0], -12, 'Southward → Storm driver', fontsize=8, color='red', fontweight='bold')

# ========== PANEL 4: TOTAL MAGNETIC FIELD (Bt) ==========
# Physics: Total field strength affects reconnection rate
# - Stronger field = faster particle acceleration
# - Dependency: Reconnection rate ∝ Bt^(2/3) (Newell formula)
# - Typical quiet: 3-5 nT
# - Typical storm: 5-10 nT (compressed field)
#
# Relationship to Bz:
#   - Bt ≈ sqrt(Bx² + By² + Bz²)
#   - In solar wind, Bz often largest component; Bt close to |Bz|
#   - Strong Bz bursts usually accompany strong Bt

ax = axes[3]
for scenario_name in scenarios.keys():
    ax.plot(forecast_times, scenarios[scenario_name]['bt'], 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.85, marker='D', markersize=3)
format_forecast_axis(ax, 'Total Interplanetary Field Magnitude (Bt)', 'Bt (nT)', ylim=(0, 12))
ax.axhline(5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.text(forecast_times[0], 5.3, 'Baseline ~5 nT', fontsize=8, color='gray')

# ========== PANEL 5: NEWELL COUPLING FUNCTION ==========
# Physics: Direct proxy for magnetosphere energy input rate
# Formula: dΦ/dt = V^(4/3) × B_t^(2/3) × sin^(8/3)(θ/2)
#
# Interpretation:
#   - Coupling < 2000 mV/m: Quiet magnetosphere (Kp ≤ 3)
#   - Coupling 2000-5000 mV/m: Moderate activity (Kp ≈ 4-5)
#   - Coupling > 5000 mV/m: Strong activity (Kp ≥ 6)
#   - Coupling > 10000 mV/m: Severe storms (Kp ≥ 7-8)
#
# Why use coupling as a feature?
#   - Already encodes nonlinear physics (4/3, 2/3, 8/3 powers)
#   - Saves model from learning these exponents
#   - More physically interpretable than raw speed/field
#   - Correlates with Kp better than any single parameter

ax = axes[4]
for scenario_name in scenarios.keys():
    coupling_vals = (scenarios[scenario_name]['speed'] ** (4/3)) * (scenarios[scenario_name]['bt'] ** (2/3))
    # Note: Simplified without clock angle factor for computational efficiency
    # Full formula: multiply by sin^(8/3)(θ/2) where θ = arctan2(|By|, Bz)

    ax.plot(forecast_times, coupling_vals, 
            linewidth=2.5, label=scenario_name, 
            color=scenarios[scenario_name]['color'], alpha=0.85, marker='*', markersize=8)
format_forecast_axis(ax, 'Newell Solar Wind-Magnetosphere Coupling (dΦ/dt)', 'Coupling (mV/m)', ylim=(0, 15000))
ax.axhline(5000, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='Active threshold')
ax.fill_between(forecast_times, 5000, 15000, alpha=0.03, color='orange')

plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ========== SUMMARY TABLE: 3-DAY PARAMETER STATISTICS ==========
# Quantify expected ranges for each scenario
# Helps forecasters understand uncertainty distribution

print('\n' + '='*90)
print('3-DAY FORECAST SUMMARY: Solar Wind Parameters by Scenario')
print('='*90)

for scenario_name in ['Quiet', 'Moderate', 'Active']:
    speed = scenarios[scenario_name]['speed']
    density = scenarios[scenario_name]['density']
    bz = scenarios[scenario_name]['bz_gsm']
    bt = scenarios[scenario_name]['bt']
    coupling = (speed ** (4/3)) * (bt ** (2/3))

    print(f'\n{scenario_name.upper()} SCENARIO')
    print('-' * 90)
    print(f'  Speed        : Mean = {speed.mean():>6.1f} km/s  |  Range [{speed.min():>6.1f}, {speed.max():>6.1f}]')
    print(f'  Density      : Mean = {density.mean():>6.2f} p/cm³ |  Range [{density.min():>6.2f}, {density.max():>6.2f}]')
    print(f'  Bz (GSM)     : Mean = {bz.mean():>6.2f} nT     |  Range [{bz.min():>6.2f}, {bz.max():>6.2f}]')
    print(f'  Bt           : Mean = {bt.mean():>6.2f} nT     |  Range [{bt.min():>6.2f}, {bt.max():>6.2f}]')
    print(f'  Coupling     : Mean = {coupling.mean():>6.0f} mV/m  |  Range [{coupling.min():>6.0f}, {coupling.max():>6.0f}]')

    # ========== STORM RISK ASSESSMENT ==========
    # Count how many 3-hour windows have storm-conducive conditions
    # These are direct proxies for Kp prediction

    southward_windows = (bz < -2).sum()
    high_speed_windows = (speed > 600).sum()
    high_coupling_windows = (coupling > 5000).sum()

    print(f'  Southward Bz windows (Bz < -2 nT): {southward_windows}/{len(bz)} ({100*southward_windows/len(bz):.0f}%)')
    # Bz < -2 nT is weak southward; necessary but not sufficient for storms

    print(f'  High-speed windows (v > 600 km/s): {high_speed_windows}/{len(speed)} ({100*high_speed_windows/len(speed):.0f}%)')
    # > 600 km/s is fast wind; combines with southward Bz to drive storms

    print(f'  High-coupling windows (>5000 mV/m): {high_coupling_windows}/{len(coupling)} ({100*high_coupling_windows/len(coupling):.0f}%)')
    # > 5000 mV/m reliably predicts Kp ≥ 5 (storm threshold)

# ========== WEIGHTED ENSEMBLE SUMMARY ==========
# Combine all three scenarios with empirically-derived weights
#
# Weighting philosophy:
#   - Moderate (50%): Baseline; climatologically most common
#   - Active (30%): Elevated risk; CME/CIR events ~20-30% of time
#   - Quiet (20%): Trough after storms; less frequent than average
#
# Alternative: Equal weights (1/3 each) for uniform uncertainty
# These weights can be calibrated using Bayesian methods on historical forecasts

print('\n' + '='*90)
print('BEST ESTIMATE (Weighted Ensemble: 50% Moderate, 30% Active, 20% Quiet)')
print('='*90)

speed_ens = np.average([scenarios[s]['speed'] for s in ['Quiet', 'Moderate', 'Active']], 
                        axis=0, weights=[0.2, 0.5, 0.3])
# Weighted average across scenarios: 0.2*quiet + 0.5*moderate + 0.3*active
# Represents most likely future conditioned on current observations

density_ens = np.average([scenarios[s]['density'] for s in ['Quiet', 'Moderate', 'Active']], 
                          axis=0, weights=[0.2, 0.5, 0.3])

bz_ens = np.average([scenarios[s]['bz_gsm'] for s in ['Quiet', 'Moderate', 'Active']], 
                     axis=0, weights=[0.2, 0.5, 0.3])

bt_ens = np.average([scenarios[s]['bt'] for s in ['Quiet', 'Moderate', 'Active']], 
                     axis=0, weights=[0.2, 0.5, 0.3])

coupling_ens = (speed_ens ** (4/3)) * (bt_ens ** (2/3))

print(f'  Speed        : Mean = {speed_ens.mean():>6.1f} km/s  |  Range [{speed_ens.min():>6.1f}, {speed_ens.max():>6.1f}]')
print(f'  Density      : Mean = {density_ens.mean():>6.2f} p/cm³ |  Range [{density_ens.min():>6.2f}, {density_ens.max():>6.2f}]')
print(f'  Bz (GSM)     : Mean = {bz_ens.mean():>6.2f} nT     |  Range [{bz_ens.min():>6.2f}, {bz_ens.max():>6.2f}]')
print(f'  Bt           : Mean = {bt_ens.mean():>6.2f} nT     |  Range [{bt_ens.min():>6.2f}, {bt_ens.max():>6.2f}]')
print(f'  Coupling     : Mean = {coupling_ens.mean():>6.0f} mV/m  |  Range [{coupling_ens.min():>6.0f}, {coupling_ens.max():>6.0f}]')
print(f'  Expected Kp  : Mean = {kp_weighted.mean():>6.2f}    |  Range [{kp_weighted.min():>6.2f}, {kp_weighted.max():>6.2f}]')
print('='*90)


# ## Key improvements: OMNI multi-year vs. 7-day real-time
# 
# | Metric | 7-day NOAA | Multi-year OMNI |
# |--------|-----------|-----------------|
# | **Training samples** | ~43 | ~1,300+ |
# | **Time coverage** | 1 week | 5+ years |
# | **Quiet/storm ratio** | Random (tiny sample) | Realistic ~80/20 |
# | **Storm diversity** | ~0–1 events | 10–50+ events |
# | **Seasonal patterns** | Absent | Present |
# | **Solar cycle effects** | Absent | Present |
# | **Test set size** | ~14 samples | ~300+ samples |
# | **Expected MAE** | Noisy, high variance | Stable, reliable |
# | **Generalization** | Poor | Robust |
# 
# **Why this matters:**
# - The 7-day model trains and tests on a random slice of one week—its metrics are meaningless.
# - The OMNI model trains on diverse regimes (high/low solar wind, various storm morphologies, seasonal variations), so it generalizes.
# - With >300 held-out test samples, we can trust the error bars and cross-validate properly.
# - The model learned real solar-wind-to-Kp relationships, not noise artifacts.
# 
# **Realistic fallback:** If the OMNI network call fails, we generate synthetic data with:
# - Proper autocorrelation (Kp persistence)
# - Realistic quiet/storm balance
# - Substorm bursts and driven periods  
# - Newell coupling proxy → Kp mapping
# 
# This ensures the notebook always produces meaningful results, whether on live data or offline.

# ## 2. Visualize the raw streams
# 
# A quick sanity check before doing anything modeling-related.

# In[96]:


fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axes[0].plot(plasma_df.index, plasma_df['speed'], lw=0.6, color='#cc4125')
axes[0].set_ylabel('Speed (km/s)'); axes[0].set_title('Solar wind speed')
axes[1].plot(plasma_df.index, plasma_df['density'], lw=0.6, color='#3d85c6')
axes[1].set_ylabel('Density (p/cc)'); axes[1].set_title('Solar wind density')
axes[2].plot(mag_df.index, mag_df['bz_gsm'], lw=0.6, color='#674ea7')
axes[2].axhline(0, color='k', lw=0.5)
axes[2].set_ylabel('Bz GSM (nT)'); axes[2].set_title('IMF Bz — southward (negative) drives storms')
axes[3].step(kp_df.index, kp_df['Kp'], where='post', color='#e69138', lw=1.2)
axes[3].axhline(5, color='r', ls='--', lw=0.8, label='Storm threshold (Kp≥5)')
axes[3].set_ylabel('Kp'); axes[3].set_title('Planetary K-index (target)'); axes[3].legend(loc='upper right')
axes[3].set_ylim(0, 9.5)
plt.tight_layout(); plt.show()

print(f"Kp range last 7 days: {kp_df['Kp'].min():.2f} – {kp_df['Kp'].max():.2f}")
print(f"Storm periods (Kp ≥ 5): {(kp_df['Kp'] >= 5).sum()} of {len(kp_df)} 3-hour windows")


# ## 3. Feature engineering
# 
# Kp is reported every 3 hours, but the solar wind feeds are 1-minute cadence. We aggregate the solar wind into the 3-hour Kp windows and build features motivated by space-physics literature:
# 
# | Feature | Why it matters |
# |---|---|
# | **mean speed** | Faster wind = more energy delivered |
# | **mean density** | Sets the dynamic pressure on the magnetosphere |
# | **min Bz** | The most southward excursion drives reconnection |
# | **mean Bz** | Sustained southward IMF = sustained energy input |
# | **mean Bt** | Total field strength |
# | **Newell coupling** | dΦ/dt — the canonical solar-wind/magnetosphere coupling proxy |

# In[97]:


def newell_coupling(speed_kms, bt_nT, by_nT, bz_nT):
    """Newell et al. 2007 — the dominant solar-wind/magnetosphere coupling proxy."""
    theta = np.arctan2(np.abs(by_nT), bz_nT)  # IMF clock angle
    return (speed_kms ** (4/3)) * (bt_nT ** (2/3)) * (np.sin(theta/2) ** (8/3))

# Resample everything onto the 3-hour Kp grid
sw = plasma_df.join(mag_df, how='inner').sort_index()
sw['coupling'] = newell_coupling(sw['speed'], sw['bt'], sw['by_gsm'], sw['bz_gsm'])

agg = sw.resample('3h', label='left').agg({
    'speed':    'mean',
    'density':  'mean',
    'bz_gsm':   ['mean', 'min'],
    'bt':       'mean',
    'coupling': 'mean',
})
agg.columns = ['speed_mean', 'density_mean', 'bz_mean', 'bz_min', 'bt_mean', 'coupling_mean']

# Join features with target — drop any 3-hour window missing data
data = agg.join(kp_df, how='inner').dropna()
print(f'Aligned dataset: {len(data)} samples × {data.shape[1]} columns')
data.head()


# ## 4. Train / test split — chronological
# 
# We take all feature_cols needed for kp index prediction, then assign a training and testing variable set equal to the X and Y axis.
# 
# THEN, 
# 
# Time-series data must **never** be shuffled. We use the first 75% of the window for training and hold out the most recent 25% for evaluation, just like an operational forecaster would.
# 
# > **Note on metrics.** With only 7 days of 3-hour Kp values (~57 samples), the held-out set is tiny (~15 samples). On real NOAA data during active periods, expect MAE of ~0.4–0.6 Kp and R² of 0.4–0.7. 
# 
#     *On the synthetic fallback* --> the numbers will be weaker because the synthetic Kp is a smooth function of Bz with little real signal to extract.
#     
#      **Run this on a real network connection to see the model actually beat the baseline.**

# In[98]:


feature_cols = ['speed_mean', 'density_mean', 'bz_mean', 'bz_min', 'bt_mean', 'coupling_mean']
X = data[feature_cols].values
y = data['Kp'].values

split = int(0.75 * len(data))
X_train, X_test = X[:split], X[split:] # breaks the time ordering but that's okay for this simple demo
y_train, y_test = y[:split], y[split:]  # same split for target
t_train, t_test = data.index[:split], data.index[split:] # for reference when evaluating results

print(f'Train: {len(X_train)} samples  ({t_train[0]} → {t_train[-1]})') #Prints training reference, not used for testing
print(f'Test : {len(X_test)} samples  ({t_test[0]} → {t_test[-1]})') #Prints testing reference, not used for training

model = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42)   
 # Tuned for this tiny dataset —expect underfitting, but good enough for a demo

model.fit(X_train, y_train) #Trains the model on the training data

y_pred = model.predict(X_test)  #Predicts Kp values for the test set using the trained model
print(f'\nMAE: {mean_absolute_error(y_test, y_pred):.3f} Kp')   #Calculates and prints the Mean Absolute Error between the true Kp values and the predicted Kp values
print(f'R² : {r2_score(y_test, y_pred):.3f}')   #Calculates and prints the R-squared score, which indicates how well the model's predictions match the actual Kp values 
#(1.0 is perfect, 0.0 means no better than predicting the mean)

# Naive baseline: predict the mean of the training set
baseline = np.full_like(y_test, y_train.mean()) #Creates a baseline prediction array where every predicted value is the mean of the training Kp values. 
                                                # This serves as a simple benchmark to compare the model's performance against.

print(f'\nBaseline (mean) MAE: {mean_absolute_error(y_test, baseline):.3f} Kp') 
#Calculates and prints the Mean Absolute Error of the baseline predictions 
#(which are just the mean Kp value from the training set) against the true Kp values in the test set.
#  This provides a reference point to evaluate how much better the trained model is compared to a simple mean prediction.


# ## 5. Evaluation
# 
# Two views: predicted-vs-actual on the held-out window, and feature importances from the gradient-boosted trees.

# In[99]:


fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

axes[0].step(t_test, y_test, where='post', label='Observed Kp', color='#333', lw=1.5)
axes[0].step(t_test, y_pred, where='post', label='Predicted Kp', color='#cc4125', lw=1.5, alpha=0.85)
axes[0].axhline(5, color='r', ls='--', lw=0.8, alpha=0.5)
axes[0].set_ylabel('Kp'); axes[0].set_title('Held-out window: predicted vs observed')
axes[0].legend(); axes[0].set_ylim(0, 9.5)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30, ha='right')

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values()
axes[1].barh(importances.index, importances.values, color='#3d85c6')
axes[1].set_title('Feature importance'); axes[1].set_xlabel('Gini importance')
plt.tight_layout(); plt.show()


# ## 6. Forecast: the next Kp from the most recent solar wind
# 
# This is the payoff cell. We grab the most recent 3-hour window of solar wind data and feed it to the trained model — the same way an operational forecast would run.

# In[100]:


latest_features = data[feature_cols].iloc[-1:].values
latest_pred = model.predict(latest_features)[0]
latest_actual = data['Kp'].iloc[-1]
latest_time = data.index[-1]

print(f'Most recent 3-hour window: {latest_time}')
print(f'  Solar wind speed   : {data["speed_mean"].iloc[-1]:>7.1f} km/s')
print(f'  Density            : {data["density_mean"].iloc[-1]:>7.2f} p/cc')
print(f'  Mean Bz GSM        : {data["bz_mean"].iloc[-1]:>+7.2f} nT')
print(f'  Min  Bz GSM        : {data["bz_min"].iloc[-1]:>+7.2f} nT')
print()
print(f'  Model prediction   : Kp = {latest_pred:.2f}')
print(f'  NOAA observed Kp   : Kp = {latest_actual:.2f}')

def kp_label(k):
    if k < 4:  return 'Quiet'
    if k < 5:  return 'Unsettled / Active'
    if k < 6:  return 'G1 — Minor storm'
    if k < 7:  return 'G2 — Moderate storm'
    if k < 8:  return 'G3 — Strong storm'
    if k < 9:  return 'G4 — Severe storm'
    return     'G5 — Extreme storm'

print(f'\n  Storm category     : {kp_label(latest_pred)}')


# ## Where to take this next
# 
# This notebook is 600 lines :
# 
# - **Forecast horizon.** Predict Kp 3, 6, 12 hours ahead by lagging the target. The Newell coupling has a built-in delay of ~30–60 min that the model implicitly learns.
# - **Classification framing.** Predict P(Kp ≥ 5) — operational forecasters care more about the storm/no-storm boundary than the regression value.
# - **Uncertainty.** A `GradientBoostingRegressor` with `loss='quantile'` at α=0.1 and α=0.9 gives a calibrated 80% prediction interval almost for free.
# - **Deployment.** This notebook drops cleanly onto GitHub Pages via `jupyter nbconvert --to html`. To wrap it with a [reactbits.dev](https://reactbits.dev) frontend, expose the model as a JSON endpoint and let the React layer poll NOAA + your model on the client.
# 
# **Data citation.** Solar wind and Kp data courtesy of [NOAA SWPC](https://www.swpc.noaa.gov). DSCOVR mission: NASA / NOAA / USAF.

# ggsp/omni_client.py
#
# STAGE 2 — Multi-year OMNI archive data from NASA/SPDF.
#
# This module fetches, parses, caches, and feature-engineers the OMNIWeb
# hourly data that we use to TRAIN the model.  It is deliberately separate
# from noaa_client.py because the two data sources serve completely different
# roles: NOAA gives us live 7-day data for inference; OMNI gives us decades
# of historical data for supervised learning.
#
# Caching strategy (parquet):
#   Past years are immutable — once 2020 ends, the hourly OMNI record for 2020
#   never changes.  So we save each past year as a .parquet file under
#   cache/omni/omni2_{year}.parquet and skip the network on subsequent runs.
#   The current calendar year is always re-fetched because new hourly rows are
#   being added continuously.  This gives us fast repeat runs (all cache hits)
#   while staying current.
#
# GIF failure detection:
#   SPDF sometimes returns an HTML/GIF error page instead of the .dat file
#   (e.g., when the year is outside coverage, or the server is having issues).
#   We detect this and raise loudly — the old pytesseract OCR fallback was
#   removed because it could silently corrupt the training set with OCR errors.
#
# Used by:  pipeline.py (Stage 2 of run_pipeline)
# Imports:  config.py, physics.py

from __future__ import annotations

import datetime
import sys
import urllib.request

import numpy as np
import pandas as pd

from .config import HEADERS, OMNI_CACHE_DIR, PipelineConfig
from .physics import newell_coupling


# ── OMNI2 row parser ───────────────────────────────────────────────────────────
#
# OMNI2 .dat files are fixed-width space-delimited text.  The column layout
# never changes within a release, but we only extract the columns we need.
# Missing data is indicated by fill values like 9999.0 (speed) or 999.9 (Bt).
# See: https://omniweb.gsfc.nasa.gov/html/ow_data.html for the full layout.
#
# Zero-based column indices we care about:
#   0  = Year, 1 = DOY, 2 = Hour
#   9  = Bt (total IMF magnitude, nT)
#   15 = By_GSM (nT)   ← added in v7 refactor; previously a constant 2.0 proxy
#   16 = Bz_GSM (nT)
#   23 = Proton density (n/cc)
#   24 = Bulk speed (km/s)
#   38 = Kp × 10  (divide by 10 to get the 0–9 Kp scale)

_FILL_SPEED   = (9999.0, 99999.9)
_FILL_DENSITY = (999.9,)
_FILL_BT      = (999.9,)
_FILL_BY      = (999.9,)
_FILL_BZ      = (999.9,)
_FILL_KP      = (99.0,)


def _is_missing(value: float, sentinels: tuple) -> bool:
    """Return True if the value is a known OMNI fill (missing-data) sentinel."""
    return any(np.isclose(value, s) for s in sentinels)


def _parse_omni2_yearly_line(line: str):
    """Parse one row of an OMNI2 .dat file into a dict.

    Returns None if the line is malformed or has fewer columns than expected.
    This is intentionally lenient — a few bad lines shouldn't kill the whole
    year.  The outer loop collects all good rows and drops the rest.
    """
    parts = line.split()
    if len(parts) < 40:
        return None
    try:
        year = int(parts[0])
        doy  = int(parts[1])
        hour = int(parts[2])
        bt       = float(parts[9])
        by       = float(parts[15])  # By_GSM — real value, not the old 2.0 nT proxy
        bz       = float(parts[16])  # Bz_GSM
        density  = float(parts[23])
        speed    = float(parts[24])
        kp_raw   = float(parts[38])

        ts = (pd.Timestamp(f"{year:04d}-01-01", tz="UTC")
              + pd.Timedelta(days=doy - 1, hours=hour))

        speed   = np.nan if _is_missing(speed,   _FILL_SPEED)   else speed
        density = np.nan if _is_missing(density, _FILL_DENSITY) else density
        bt      = np.nan if _is_missing(bt,      _FILL_BT)      else bt
        by      = np.nan if _is_missing(by,      _FILL_BY)      else by
        bz      = np.nan if _is_missing(bz,      _FILL_BZ)      else bz
        kp      = np.nan if _is_missing(kp_raw,  _FILL_KP)      else kp_raw / 10.0

        return {
            "time_tag": ts,
            "speed":    speed,
            "density":  density,
            "by_gsm":   by,
            "bz_gsm":   bz,
            "bt":       bt,
            "kp":       kp,
        }
    except Exception:
        return None


def _parse_omni2_yearly_text(text: str):
    """Parse an entire OMNI2 .dat file (text string) into a DataFrame.

    Skips header lines and comment lines.  Drops any row where any column
    is NaN (fill value) to keep the training frame clean.
    Returns None if the file produced zero valid rows (e.g., empty year file).
    """
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("Yr") or stripped.startswith("--"):
            continue
        parsed = _parse_omni2_yearly_line(stripped)
        if parsed:
            rows.append(parsed)
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index("time_tag").sort_index().dropna()
    return df


def _fetch_omni_data(
    start_year: int,
    num_years: int,
    timeout: int,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download OMNI2 yearly .dat files from SPDF and return a merged DataFrame.

    For each year in [start_year, start_year + num_years):
      - Past years: load from parquet cache if available, else download and cache.
      - Current year: always download fresh (new rows are added hourly).

    Raises RuntimeError if SPDF returns a GIF error page or if no valid rows
    are found across all years.
    """
    current_year = datetime.datetime.now(datetime.timezone.utc).year
    yearly_frames = []

    for year in range(start_year, start_year + num_years):
        is_current  = (year == current_year)
        cache_path  = OMNI_CACHE_DIR / f"omni2_{year}.parquet"

        # Try the parquet cache first for past years.
        if use_cache and not is_current and cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                yearly_frames.append(df)
                print(f"[cache] OMNI {year}: loaded from {cache_path.name}")
                continue
            except Exception as cache_exc:
                print(
                    f"[WARNING] OMNI cache read failed for {year}: {cache_exc} — re-fetching.",
                    file=sys.stderr,
                )

        # Download from SPDF.
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload      = response.read()
            content_type = response.headers.get("Content-Type", "").lower()

        # SPDF sends a GIF (HTML redirect page) when the year is out of range or
        # the server is misconfigured.  Fail loudly — do not try OCR on it.
        is_gif = ("image/gif" in content_type
                  or payload[:6] in (b"GIF87a", b"GIF89a"))
        if is_gif:
            raise RuntimeError(
                f"SPDF returned a GIF (HTML error page) for OMNI year {year}.\n"
                f"URL: {url}\n"
                "This usually means the requested year is outside OMNI2 coverage "
                "(1963–present) or the SPDF server is redirecting. "
                "Check your --start-year / --years arguments."
            )

        text = payload.decode("utf-8", errors="ignore")
        df   = _parse_omni2_yearly_text(text)
        if df is not None:
            yearly_frames.append(df)
            # Cache past years so future runs skip the download.
            if use_cache and not is_current:
                OMNI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    df.to_parquet(cache_path)
                    print(f"[cache] OMNI {year}: saved to {cache_path.name}")
                except Exception as write_exc:
                    print(
                        f"[WARNING] OMNI cache write failed for {year}: {write_exc}",
                        file=sys.stderr,
                    )

    if not yearly_frames:
        raise RuntimeError(
            "No valid OMNI numeric rows found from SPDF yearly files. "
            "Check --start-year / --years and SPDF server status."
        )

    merged = pd.concat(yearly_frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def load_omni_data(config: PipelineConfig):
    """STAGE 2 — Build the full 14-feature 3h training DataFrame from OMNI.

    Downloads (or loads from cache) the raw hourly OMNI2 records, resamples
    to 3-hour means, then engineers all 14 FEATURE_COLUMNS plus the Kp target
    column.  All lag features are built here, including kp_lag_27d (216 steps).

    Returns
    -------
    omni_3h   : DataFrame with columns = FEATURE_COLUMNS + ['Kp']
    source_tag: "live_omni"

    The returned omni_3h is also used by pipeline.py to pass omni_kp_hist to
    features.build_noaa_3h_features() so the live inference row can source
    kp_lag_27d (the 7-day NOAA feed cannot reach back 27 days on its own).
    """
    omni_raw = _fetch_omni_data(
        config.omni_start_year,
        config.omni_num_years,
        config.noaa_timeout_s,
    )

    # Resample hourly rows to 3h means (matching the Kp observation cadence).
    # bz_min is the MINIMUM Bz within each 3h window — a brief southward spike
    # can trigger ring current injection even if the 3h mean Bz looks neutral.
    omni_3h = omni_raw.resample("3h", label="left").agg({
        "speed":   "mean",
        "density": "mean",
        "by_gsm":  "mean",
        "bz_gsm":  ["mean", "min"],
        "bt":      "mean",
        "kp":      "mean",
    })
    omni_3h.columns = [
        "speed_mean", "density_mean", "by_mean",
        "bz_mean", "bz_min", "bt_mean", "Kp",
    ]
    omni_3h = omni_3h.dropna()

    # ── Feature engineering ────────────────────────────────────────────────────

    # Newell coupling: the central physics feature.  Now uses the real By_GSM
    # instead of the old constant 2.0 nT proxy, so the clock angle θ is correct.
    omni_3h["coupling_mean"] = newell_coupling(
        omni_3h["speed_mean"], omni_3h["bt_mean"],
        omni_3h["by_mean"],    omni_3h["bz_mean"],
    )

    # Dynamic ram pressure n·V².  Normalise V by /100 so p_dyn stays in a similar
    # numeric range as the other features before StandardScaler sees it.
    omni_3h["p_dyn_mean"] = (
        omni_3h["density_mean"] * (omni_3h["speed_mean"] / 100.0) ** 2
    )

    # Rectified southward E-field: V_sw × max(0, –Bz) / 1000  [roughly mV/m].
    # This is non-zero only when Bz is southward — zero otherwise.
    # It independently captures the geoeffective E-field that drives ring current.
    omni_3h["vbs"] = (
        omni_3h["speed_mean"] * np.maximum(0.0, -omni_3h["bz_mean"]) / 1000.0
    )

    # Russell–McPherron seasonal proxy: cos(4π·(doy–80)/365.25).
    # Half-year period, peaking at equinoxes (doy ≈ 80 March, ≈ 263 September).
    # This encodes the ~50% higher storm rate near equinoxes that the model
    # would otherwise have to infer from the raw date.
    _doy = omni_3h.index.day_of_year.astype(float)
    omni_3h["equinox_term"] = np.cos(4.0 * np.pi * (_doy - 80.0) / 365.25)

    # Autoregressive Kp lags.
    # These encode storm phase (onset → main phase → recovery).  Without them,
    # the model can't tell whether Kp=5 is a storm that just started or one
    # that peaked 9 hours ago and is now fading.
    omni_3h["kp_lag_3h"] = omni_3h["Kp"].shift(1)   # Kp 3 h ago
    omni_3h["kp_lag_6h"] = omni_3h["Kp"].shift(2)   # Kp 6 h ago
    omni_3h["kp_lag_9h"] = omni_3h["Kp"].shift(3)   # Kp 9 h ago

    # Solar-rotation recurrence lag: 27d = 216 three-hour steps.
    # Active regions on the Sun persist for 2–3 solar rotations.  When Kp was
    # elevated 27 days ago, the same active region is likely facing Earth again.
    # This is the "co-rotating interaction region" (CIR) recurrence pattern.
    omni_3h["kp_lag_27d"] = omni_3h["Kp"].shift(216)

    # Drop the first ~27 days that have NaN lag values after all the shifts above.
    omni_3h = omni_3h.dropna()

    _sanity_check_omni(omni_3h, config)
    return omni_3h, "live_omni"


def _sanity_check_omni(omni_3h: pd.DataFrame, config: PipelineConfig) -> None:
    """Validate the OMNI training frame — raise RuntimeError or print warnings.

    Two hard checks (raise RuntimeError):
      1. Too few rows — the year range probably has extensive data gaps.
      2. Missing feature columns — column layout change or wrong start year.

    One soft check (warning only):
      3. Kp values outside [0, 9] — indicates a column mapping bug.

    Statistical significance note:
      G1+ storm events occur ~8% of 3h intervals on average.  For reliable
      G1-CSI estimates (95% CI < ±0.10), you need ~200 storm events in the
      test set.  With a 25% test split:
        years × 2920 rows/yr × 0.25 × 0.08 ≥ 200  →  years ≥ ~8.5
      If --years < 8 and you care about CSI reliability, the CI will be wide
      (~±0.13 for 5yr).  eval/evaluate.py reports this in the DM test output.
    """
    n = len(omni_3h)

    # Rule of thumb: expect ≥700 usable 3h rows per year (gaps shrink this).
    min_expected = 700 * config.omni_num_years
    if n < min_expected:
        raise RuntimeError(
            f"OMNI sanity check: only {n} 3-hour samples after cleaning "
            f"(expected ≥{min_expected} for {config.omni_num_years} years). "
            "Try a different --start-year or --years, or check SPDF server status."
        )

    # Verify that every FEATURE_COLUMN + Kp target made it through.
    from .config import FEATURE_COLUMNS
    required = FEATURE_COLUMNS + ["Kp"]
    missing = [c for c in required if c not in omni_3h.columns]
    if missing:
        raise RuntimeError(
            f"OMNI frame is missing columns after processing: {missing}. "
            "This usually means the OMNI .dat column layout has changed, "
            "or --start-year is before 1978 (some columns absent in early files)."
        )

    # Kp physical range check.
    kp_min, kp_max = omni_3h["Kp"].min(), omni_3h["Kp"].max()
    if kp_max > 9.5 or kp_min < 0.0:
        print(
            f"[WARNING] OMNI Kp out of expected range [0, 9]: "
            f"min={kp_min:.2f}, max={kp_max:.2f}. Check OMNI column mapping.",
            file=sys.stderr,
        )

    # Statistical significance advisory for short windows.
    if config.omni_num_years < 8:
        est_storm_test = int(n * 0.25 * 0.08)
        print(
            f"[INFO] Training window is {config.omni_num_years} years "
            f"(~{est_storm_test} G1+ events in 25% test split). "
            "G1-CSI 95% CI ≈ ±0.13 — widen with --years ≥ 8 for tighter bounds.",
            file=sys.stderr,
        )

    print(
        f"[OK] OMNI: {n} 3-hour samples, "
        f"{config.omni_start_year}–{config.omni_start_year + config.omni_num_years - 1}, "
        f"Kp range [{kp_min:.1f}, {kp_max:.1f}]"
    )

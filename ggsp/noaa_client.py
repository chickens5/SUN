# ggsp/noaa_client.py
#
# STAGE 1 — Live solar-wind data from NOAA SWPC.
#
# This module is responsible for ONE thing: fetching the three real-time
# JSON feeds that NOAA publishes from the DSCOVR spacecraft and turning them
# into clean pandas DataFrames ready for feature engineering.
#
# Why three separate feeds?
#   NOAA publishes plasma data (speed, density, temperature) and magnetic field
#   data (Bx, By, Bz, Bt) on separate endpoints because they come from different
#   instruments on DSCOVR.  Kp is derived on the ground from a global network of
#   magnetometers and published separately.  We join them all on a UTC time grid.
#
# No synthetic fallback — if NOAA is unreachable, we raise loudly rather than
# silently training or forecasting on fake data.  Bad data in = bad forecast out.
#
# Used by:  pipeline.py (Stage 1 of run_pipeline)
# Imports:  config.py (for PipelineConfig, HEADERS)

from __future__ import annotations

import json
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

from .config import HEADERS, PipelineConfig

# ── NOAA SWPC live endpoint URLs ───────────────────────────────────────────────
#
# All three are public, no API key required.  DSCOVR sits at the L1 Lagrange
# point ~1.5 million km sunward of Earth, giving 30–60 min warning before the
# solar wind hits the magnetosphere.
#
# Plasma feed format: [["time_tag","density","speed","temperature"], [row], ...]
# Mag feed format:    [["time_tag","bx_gsm","by_gsm","bz_gsm","lon_gsm","lat_gsm","bt"], ...]
# Kp feed format:     [{"time_tag":..., "Kp":..., "a_running":..., "station_count":...}, ...]

NOAA_ENDPOINTS = {
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "mag":    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "kp":     "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
}


def _fetch_json(url: str, timeout: int, retries: int = 3) -> list:
    """Fetch a NOAA JSON endpoint with simple retry logic.

    NOAA feeds occasionally return truncated responses (especially during high
    solar activity when telemetry volume spikes).  We retry up to 3 times with
    a short backoff before giving up.  If all attempts fail we raise RuntimeError
    so the caller knows something is wrong, rather than silently using empty data.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
            return json.loads(payload)
        except (json.JSONDecodeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            # Short backoff: 0.5 s on first retry, 1.0 s on second.
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"Fetch failed after {retries} attempts for {url}: {last_error}")


def _rows_to_dataframe(rows: list) -> pd.DataFrame:
    """Convert a NOAA table-format JSON list into a time-indexed DataFrame.

    The NOAA plasma and mag feeds use a header row followed by data rows —
    just like a CSV in a list.  Kp uses a different format (handled separately).
    All non-time columns are coerced to float; NOAA sometimes returns strings.
    """
    df = pd.DataFrame(rows[1:], columns=rows[0])
    df["time_tag"] = pd.to_datetime(df["time_tag"], utc=True)
    for col in df.columns:
        if col != "time_tag":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.set_index("time_tag").sort_index()


def load_noaa_data(config: PipelineConfig):
    """STAGE 1 — Fetch the three live NOAA SWPC feeds and return clean DataFrames.

    Returns
    -------
    plasma_df  : 1-min cadence solar wind plasma (speed, density, temperature)
    mag_df     : 1-min cadence IMF (bx_gsm, by_gsm, bz_gsm, bt)
    kp_df      : 3-h Kp index observations (observed target for the last 7 days)
    source_tag : "live_noaa" — passed through to the JSON output for provenance

    These feed into features.build_noaa_3h_features() to build the real-time
    inference row AND the forecast seed.  OMNI (omni_client) is used for training.
    """
    try:
        plasma_raw = _fetch_json(NOAA_ENDPOINTS["plasma"], config.noaa_timeout_s)
        mag_raw    = _fetch_json(NOAA_ENDPOINTS["mag"],    config.noaa_timeout_s)
        kp_raw     = _fetch_json(NOAA_ENDPOINTS["kp"],     config.noaa_timeout_s)

        plasma_df = _rows_to_dataframe(plasma_raw)
        mag_df    = _rows_to_dataframe(mag_raw)

        # The Kp feed sometimes comes as a list-of-dicts instead of a table.
        # We handle both formats so we don't break if NOAA changes their schema.
        kp_rows = kp_raw.get("data", kp_raw) if isinstance(kp_raw, dict) else kp_raw
        if not isinstance(kp_rows, list) or len(kp_rows) == 0:
            raise RuntimeError("NOAA Kp payload is empty or malformed")

        if isinstance(kp_rows[0], dict):
            # Dict-of-rows format — deduplicate by (time_tag, Kp) key.
            seen = set()
            records = []
            for row in kp_rows:
                key = (row.get("time_tag"), row.get("Kp"),
                       row.get("a_running"), row.get("station_count"))
                if key in seen:
                    continue
                seen.add(key)
                records.append({"time_tag": row.get("time_tag"), "Kp": row.get("Kp")})
            kp_df = pd.DataFrame.from_records(records)
        elif isinstance(kp_rows[0], list):
            # Table format (same as plasma/mag).
            kp_df = pd.DataFrame(kp_rows[1:], columns=kp_rows[0])
        else:
            raise RuntimeError("NOAA Kp payload has unsupported row format")

        kp_df["time_tag"] = pd.to_datetime(kp_df["time_tag"], utc=True)
        kp_df["Kp"] = pd.to_numeric(kp_df["Kp"], errors="coerce")
        kp_df = (kp_df.dropna(subset=["time_tag", "Kp"])
                      .set_index("time_tag").sort_index()[["Kp"]])
        if kp_df.empty:
            raise RuntimeError("NOAA Kp parsed successfully but produced no valid rows")

        _sanity_check_noaa(plasma_df, mag_df, kp_df)
        return plasma_df, mag_df, kp_df, "live_noaa"

    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to load NOAA live data: {exc}") from exc


def _sanity_check_noaa(
    plasma_df: pd.DataFrame,
    mag_df: pd.DataFrame,
    kp_df: pd.DataFrame,
) -> None:
    """Warn about stale or physically implausible NOAA feed values.

    This is a defensive check — it doesn't stop the pipeline, it just prints
    [WARNING] messages so you know something looks off.  If the data is so bad
    that the feed is actually empty, we raise RuntimeError (that's a hard stop).

    Staleness threshold: DSCOVR telemetry nominally arrives within minutes.
    If the newest data point is >6 hours old, something is wrong upstream
    (spacecraft safehold, ground station outage, etc.).
    """
    now = pd.Timestamp.now("UTC")
    for name, df in (("plasma", plasma_df), ("mag", mag_df), ("kp", kp_df)):
        if df.empty:
            raise RuntimeError(
                f"NOAA '{name}' feed parsed successfully but returned zero rows. "
                "The feed may be temporarily empty — try again in a few minutes."
            )
        age_hours = (now - df.index.max()).total_seconds() / 3600
        if age_hours > 6:
            print(
                f"[WARNING] NOAA '{name}' feed: most recent data is "
                f"{age_hours:.1f} hours old (expected ≤6 h for DSCOVR feeds). "
                "Real-time inference may be using stale solar wind conditions.",
                file=sys.stderr,
            )

    # Physical speed range for DSCOVR/ACE: 250–900 km/s during normal operations.
    # Values below 100 or above 2000 almost certainly indicate a fill value or
    # sensor glitch that slipped through the NOAA QC filter.
    if "speed" in plasma_df.columns:
        spd = plasma_df["speed"].dropna()
        if len(spd) > 0 and (spd.min() < 100 or spd.max() > 2000):
            print(
                f"[WARNING] NOAA plasma speed out of physical range "
                f"[{spd.min():.0f}, {spd.max():.0f}] km/s — "
                "suspect fill values in the feed.",
                file=sys.stderr,
            )

    print(
        f"[OK] NOAA: plasma={len(plasma_df)} rows, "
        f"mag={len(mag_df)} rows, kp={len(kp_df)} rows"
    )

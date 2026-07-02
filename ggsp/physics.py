# ggsp/physics.py
#
# Pure physics functions — no I/O, no ML, no state.
# These can be imported and called from anywhere without side-effects.
#
# Why a separate file for just two functions?
#   newell_coupling() is called in THREE different modules:
#     omni_client.py  (building the training feature)
#     features.py     (building the live NOAA feature)
#     forecast.py     (building the ensemble batch features)
#   If the formula lived in any one of those, the others would need to import
#   it from a "wrong" place.  Putting shared math here keeps the dependency
#   graph clean: everything can import physics.py without creating a cycle.
#
# DO NOT CHANGE the Newell formula — it is the physics of solar wind coupling.
# Any modification would invalidate comparisons with published literature.

from __future__ import annotations

import numpy as np


def newell_coupling(speed_kms, bt_nt, by_nt, bz_nt):
    """Newell et al. 2007 coupling proxy dPhi/dt.

    This is the headline feature of GGSP — the physical quantity that
    measures how much energy the solar wind is pumping into the magnetosphere
    at each moment.  It combines speed, total field strength, and the IMF
    clock angle into a single value that correlates strongly with Kp.

    The formula:
        dPhi/dt = V^(4/3) * Bt^(2/3) * sin^(8/3)(theta/2)
    where theta = arctan2(|By|, Bz) is the IMF clock angle in the GSM YZ plane.

    When theta = 0  (Bz northward, By = 0) → sin(0) = 0 → no coupling.
    When theta = pi (Bz southward, By = 0) → sin(pi/2) = 1 → maximum coupling.
    The By term rotates the effective coupling away from the pure-southward case.

    All array inputs are broadcast together so you can pass scalars or arrays
    of any matching shape — the function returns the same shape as the input.
    """
    speed = np.asarray(speed_kms, dtype=float)
    bt    = np.asarray(bt_nt,    dtype=float)
    by    = np.asarray(by_nt,    dtype=float)
    bz    = np.asarray(bz_nt,    dtype=float)
    theta = np.arctan2(np.abs(by), bz)
    return (speed ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (np.sin(theta / 2.0) ** (8.0 / 3.0))


def kp_label(kp_value: float) -> str:
    """Map a numeric Kp value to the NOAA storm category string.

    NOAA uses the G-scale: G1 (minor) through G5 (extreme).
    We display this in the CLI summary and in the JSON output consumed
    by the React frontend to give users a plain-language storm level.

    Kp < 4    → Quiet (no storm)
    Kp 4–4.9  → Unsettled / Active (elevated but not storm-level)
    Kp 5–5.9  → G1 Minor  (the threshold used for G1-CSI evaluation)
    Kp 6–6.9  → G2 Moderate
    Kp 7–7.9  → G3 Strong
    Kp 8–8.9  → G4 Severe
    Kp ≥ 9    → G5 Extreme (Carrington-class territory)
    """
    if kp_value < 4:
        return "Quiet"
    if kp_value < 5:
        return "Unsettled / Active"
    if kp_value < 6:
        return "G1 - Minor storm"
    if kp_value < 7:
        return "G2 - Moderate storm"
    if kp_value < 8:
        return "G3 - Strong storm"
    if kp_value < 9:
        return "G4 - Severe storm"
    return "G5 - Extreme storm"

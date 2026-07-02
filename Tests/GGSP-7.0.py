#~Gabe J~ | UMSL 26' | Computer Science | 

#Welcome to GGSP-7.0.py!  

#Good Day to you kind person.


# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE OVERVIEW  (GGSP-7.0.py)
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file is now a THIN WRAPPER around the ggsp/ package.
#
# The actual CLI code lives in  ggsp/__main__.py  so the pipeline can be
# invoked two equivalent ways:
#
#   python GGSP-7.0.py [options]    ← this file (legacy / familiar path)
#   python -m ggsp    [options]     ← package invocation (recommended)
#
# Both call exactly the same main() function — no behaviour difference.
# All argument parsing, validation, and results printing is in ggsp/__main__.py.
#
# File roles in the project:
#
#   GGSP-7.0.py            ← YOU ARE HERE: legacy entry-point wrapper
#   ggsp/__main__.py       ← CLI: arg parsing, validation, summary print
#   ggsp/pipeline.py       ← Orchestrator: wires all 8 pipeline stages
#   ggsp/config.py         ← Shared config + FEATURE_COLUMNS contract
#   ggsp/physics.py        ← Newell coupling, kp_label (pure math)
#   ggsp/noaa_client.py    ← Stage 1: live NOAA SWPC fetch
#   ggsp/omni_client.py    ← Stage 2: OMNI archive fetch + caching
#   ggsp/features.py       ← Stage 3: NOAA 1-min → 14-feature 3h frame
#   ggsp/model.py          ← Stage 4: GBR training + CV + 95% CI
#   ggsp/forecast.py       ← Stage 7: scenario forecast + ensemble P(storm)
#   ggsp/viz.py            ← Optional: matplotlib plots
#   sunspot_pipeline.py    ← Stage 6: solar-cycle weight modifier
#   eval/evaluate.py       ← Stage 5: persistence baseline + DM test
#
# ═══════════════════════════════════════════════════════════════════════════════
"""
import sys
import os

# Ensure the SUN/ directory is on the path so `import ggsp` works when this
# file is double-clicked or run from a different working directory.
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from ggsp.__main__ import main

if __name__ == "__main__":
    main()

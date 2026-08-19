# ggsp/__init__.py
#
# Package initialiser for the ggsp module.
#
# This file makes `import ggsp` work and re-exports the two symbols that
# external code most commonly needs:
#   - PipelineConfig  (to build a run config)
#   - run_pipeline    (to execute the full pipeline)
#
# All other symbols live in their respective submodules.  Import them
# directly when you need them:
#   from ggsp.physics import newell_coupling
#   from ggsp.model import fit_and_evaluate_model
#   from ggsp.forecast import predict_scenario_kp_ensemble
#
# Backwards compatibility:
#   The old monolith was ggsp_pipeline_v7.py.  Code that does
#   `from ggsp_pipeline_v7 import PipelineConfig, run_pipeline`
#   still works — ggsp_pipeline_v7.py now imports from here.

from .config import PipelineConfig          # noqa: F401  (re-export)
from .pipeline import run_pipeline          # noqa: F401  (re-export)

__all__ = ["PipelineConfig", "run_pipeline"]

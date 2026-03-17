"""
patchOTDA — Optimal Transport domain adaptation for patch-clamp electrophysiology.

Provides tools for aligning experimental datasets that suffer from batch effects
(different labs, temperatures, solutions, etc.) using optimal transport methods.
"""

__version__ = "0.2.0"

from .domainAdapt import (
    PatchClampOTDA,
    metrics,
    PENALTY_SENTINEL,
    TIMEOUT,
)
try:
    from .datasets import MMS_DATA
except Exception:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Could not load MMS_DATA (possibly a pandas version mismatch). "
        "patchOTDA.MMS_DATA will be unavailable."
    )
    MMS_DATA = None

try:
    from .utils import (
        EXAMPLE_DATA_,
        REF_DATA_,
        VISp_MET_nodes,
        VISp_T_nodes,
        HICLASS_METHOD,
        MODELS,
        CLASS_MODELS,
        select_by_col,
        not_select_by_col,
        filter_MMS,
        param_grid_from_dict,
        find_outlier_idxs,
    )
except Exception:
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "Could not load patchOTDA.utils (optional dependencies may be missing). "
        "Utility functions and model registries will be unavailable."
    )

__all__ = [
    "PatchClampOTDA",
    "metrics",
    "MMS_DATA",
    "PENALTY_SENTINEL",
    "TIMEOUT",
    "EXAMPLE_DATA_",
    "REF_DATA_",
    "VISp_MET_nodes",
    "VISp_T_nodes",
    "HICLASS_METHOD",
    "MODELS",
    "CLASS_MODELS",
    "select_by_col",
    "not_select_by_col",
    "filter_MMS",
    "param_grid_from_dict",
    "find_outlier_idxs",
    "__version__",
]

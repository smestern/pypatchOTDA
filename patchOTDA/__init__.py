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

__all__ = [
    "PatchClampOTDA",
    "metrics",
    "MMS_DATA",
    "PENALTY_SENTINEL",
    "TIMEOUT",
    "__version__",
]

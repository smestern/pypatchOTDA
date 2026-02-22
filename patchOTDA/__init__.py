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
from .datasets import MMS_DATA

__all__ = [
    "PatchClampOTDA",
    "metrics",
    "MMS_DATA",
    "PENALTY_SENTINEL",
    "TIMEOUT",
    "__version__",
]

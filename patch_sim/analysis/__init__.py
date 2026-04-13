"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    fi_curve: F-I curve construction from current clamp multi-sweep results.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
"""

from .ap_metrics import (
    APAnalysisResult,
    SpikeMetrics,
    analyze_aps,
    analyze_aps_from_result,
)
from .fi_curve import FIAnalysisResult, FIPoint, analyze_fi, compute_fi_point
from .gv_curve import (
    BoltzmannFit,
    GVAnalysisResult,
    GVPoint,
    boltzmann,
    compute_gv,
)
from .iv_curve import IVAnalysisResult, IVPoint, analyze_iv, compute_iv_point

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "analyze_fi",
    "analyze_iv",
    "boltzmann",
    "compute_fi_point",
    "compute_gv",
    "compute_iv_point",
    "APAnalysisResult",
    "BoltzmannFit",
    "FIAnalysisResult",
    "FIPoint",
    "GVAnalysisResult",
    "GVPoint",
    "IVAnalysisResult",
    "IVPoint",
    "SpikeMetrics",
]

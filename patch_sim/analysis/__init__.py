"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
"""

from .ap_metrics import (
    APAnalysisResult,
    SpikeMetrics,
    analyze_aps,
    analyze_aps_from_result,
)
from .iv_curve import IVAnalysisResult, IVPoint, analyze_iv, compute_iv_point

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "analyze_iv",
    "compute_iv_point",
    "APAnalysisResult",
    "IVAnalysisResult",
    "IVPoint",
    "SpikeMetrics",
]

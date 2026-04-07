"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
    results: Shared data structures returned by analysis functions.
"""

from .ap_metrics import analyze_aps, analyze_aps_from_result
from .iv_curve import analyze_iv, compute_iv_point
from .results import APAnalysisResult, IVAnalysisResult, IVPoint, SpikeMetrics

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

"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    results: Shared data structures returned by analysis functions.
"""

from .ap_metrics import analyze_aps, analyze_aps_from_result
from .results import APAnalysisResult, SpikeMetrics

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "APAnalysisResult",
    "SpikeMetrics",
]

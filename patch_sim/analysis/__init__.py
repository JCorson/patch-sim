"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    fi_curve: F-I curve construction from current clamp multi-sweep results.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
    sfa: Spike-frequency adaptation curves and adaptation index.
"""

from .ap_metrics import (
    APAnalysisResult,
    SpikeMetrics,
    analyze_aps,
    analyze_aps_from_result,
)
from .fi_curve import (
    FIAnalysisResult,
    FIPoint,
    analyze_fi,
    compute_fi_point,
    estimate_rheobase,
)
from .gv_curve import (
    BoltzmannFit,
    GVAnalysisResult,
    GVPoint,
    boltzmann,
    compute_gv,
)
from .iv_curve import IVAnalysisResult, IVPoint, analyze_iv, compute_iv_point
from .sfa import SFAAnalysisResult, SFACurve, analyze_sfa, compute_sfa

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "analyze_fi",
    "analyze_iv",
    "analyze_sfa",
    "boltzmann",
    "compute_fi_point",
    "estimate_rheobase",
    "compute_gv",
    "compute_iv_point",
    "compute_sfa",
    "APAnalysisResult",
    "BoltzmannFit",
    "FIAnalysisResult",
    "FIPoint",
    "GVAnalysisResult",
    "GVPoint",
    "IVAnalysisResult",
    "IVPoint",
    "SFAAnalysisResult",
    "SFACurve",
    "SpikeMetrics",
]

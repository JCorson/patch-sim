"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    derivatives: General-purpose derivative utilities (e.g. dV/dt).
    fi_curve: F-I curve construction from current clamp multi-sweep results.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
    membrane_test: Dedicated membrane test for passive property characterisation.
    passive_properties: Passive membrane property extraction (R_in, τₘ, Cₘ).
    sfa: Spike-frequency adaptation curves and adaptation index.
"""

from .ap_metrics import (
    APAnalysisResult,
    SpikeMetrics,
    analyze_aps,
    analyze_aps_from_result,
)
from .derivatives import compute_dvdt
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
from .membrane_test import (
    MEMBRANE_TEST_CURRENT,
    MEMBRANE_TEST_POST_MS,
    MEMBRANE_TEST_PRE_MS,
    MEMBRANE_TEST_STEP_MS,
    run_membrane_test,
)
from .passive_properties import (
    PassiveProperties,
    analyze_passive_from_result,
    analyze_passive_properties,
    is_subthreshold,
)
from .sfa import SFAAnalysisResult, SFACurve, analyze_sfa, compute_sfa

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "analyze_fi",
    "analyze_iv",
    "analyze_passive_from_result",
    "analyze_passive_properties",
    "analyze_sfa",
    "boltzmann",
    "compute_dvdt",
    "compute_fi_point",
    "compute_gv",
    "compute_iv_point",
    "compute_sfa",
    "estimate_rheobase",
    "is_subthreshold",
    "run_membrane_test",
    "APAnalysisResult",
    "BoltzmannFit",
    "FIAnalysisResult",
    "FIPoint",
    "GVAnalysisResult",
    "GVPoint",
    "IVAnalysisResult",
    "IVPoint",
    "MEMBRANE_TEST_CURRENT",
    "MEMBRANE_TEST_POST_MS",
    "MEMBRANE_TEST_PRE_MS",
    "MEMBRANE_TEST_STEP_MS",
    "PassiveProperties",
    "SFAAnalysisResult",
    "SFACurve",
    "SpikeMetrics",
]

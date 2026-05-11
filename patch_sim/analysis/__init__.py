"""Analysis subpackage for patch_sim.

Provides tools for extracting quantitative metrics from simulation results.

Modules:
    ap_metrics: Action potential detection and metric extraction.
    burst_metrics: Burst detection and burst-metric extraction.
    calcium_transients: Calcium transient detection and decay-τ fitting.
    derivatives: General-purpose derivative utilities (e.g. dV/dt).
    fi_curve: F-I curve construction from current clamp multi-sweep results.
    hyperpolarization: Sag and rebound analysis from hyperpolarization sweeps.
    impedance: Impedance-profile analysis from a chirp current-clamp run.
    iv_curve: I-V curve construction from voltage clamp multi-sweep results.
    membrane_test: Dedicated membrane test for passive property characterisation.
    passive_properties: Passive membrane property extraction (R_in, τₘ, Cₘ).
    sfa: Spike-frequency adaptation curves and adaptation index.
    tau_v: Activation/inactivation time-constant fits from voltage-clamp sweeps.
"""

from .ap_metrics import (
    APAnalysisResult,
    SpikeMetrics,
    analyze_aps,
    analyze_aps_from_result,
)
from .burst_metrics import (
    BurstAnalysisResult,
    BurstMetrics,
    analyze_bursts,
    analyze_bursts_from_result,
)
from .calcium_transients import (
    CalciumTransient,
    CalciumTransientAnalysisResult,
    analyze_calcium_transients,
    analyze_calcium_transients_from_result,
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
from .hyperpolarization import (
    HyperpolarizationAnalysisResult,
    SagPoint,
    analyze_hyperpolarization,
    compute_sag_point,
)
from .impedance import ImpedanceProfile, analyze_impedance
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
    analyze_passive_properties,
    density_to_absolute_c_m,
    density_to_absolute_r_in,
    is_subthreshold,
)
from .sfa import SFAAnalysisResult, SFACurve, analyze_sfa, compute_sfa
from .tau_v import (
    DoubleExponentialFit,
    ExponentialFit,
    TauVAnalysisResult,
    TauVPoint,
    analyze_tau_v,
    compute_tau_v_point,
    double_exp_decay,
    single_exp_decay,
    single_exp_rise,
)

__all__ = [
    "analyze_aps",
    "analyze_aps_from_result",
    "analyze_bursts",
    "analyze_bursts_from_result",
    "analyze_calcium_transients",
    "analyze_calcium_transients_from_result",
    "analyze_fi",
    "analyze_hyperpolarization",
    "analyze_impedance",
    "analyze_iv",
    "analyze_passive_properties",
    "analyze_sfa",
    "analyze_tau_v",
    "boltzmann",
    "compute_dvdt",
    "compute_fi_point",
    "compute_gv",
    "compute_iv_point",
    "compute_sag_point",
    "compute_sfa",
    "compute_tau_v_point",
    "density_to_absolute_c_m",
    "density_to_absolute_r_in",
    "double_exp_decay",
    "estimate_rheobase",
    "is_subthreshold",
    "run_membrane_test",
    "single_exp_decay",
    "single_exp_rise",
    "APAnalysisResult",
    "BoltzmannFit",
    "BurstAnalysisResult",
    "BurstMetrics",
    "CalciumTransient",
    "CalciumTransientAnalysisResult",
    "DoubleExponentialFit",
    "ExponentialFit",
    "FIAnalysisResult",
    "FIPoint",
    "HyperpolarizationAnalysisResult",
    "ImpedanceProfile",
    "SagPoint",
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
    "TauVAnalysisResult",
    "TauVPoint",
]

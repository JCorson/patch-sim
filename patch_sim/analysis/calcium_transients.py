"""Calcium transient detection and metric extraction.

Provides :func:`analyze_calcium_transients` for extracting peak [Ca²⁺]ᵢ,
time-to-peak, and decay time constant from an intracellular calcium
concentration trace, and :func:`analyze_calcium_transients_from_result` as
a convenience wrapper for :class:`~patch_sim.clamp_simulations.SimulationResult`
structured arrays.

The internal Ca²⁺ concentration in :class:`~patch_sim.calcium.CalciumDynamics`
is stored in mM.  The analysis functions in this module accept input in mM
and report all output values in µM, matching the convention used in
calcium-imaging experiments.

Data classes:
    CalciumTransient: Per-transient measurement record.
    CalciumTransientAnalysisResult: Aggregated calcium-transient analysis.
"""

# Needed so that the SimulationResult annotation in the *_from_result wrapper
# is not evaluated eagerly at import time — SimulationResult is only imported
# under TYPE_CHECKING to avoid a circular dependency.
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

if TYPE_CHECKING:
    from patch_sim.clamp_simulations import SimulationResult

logger = logging.getLogger(__name__)

#: Conversion factor from mM to µM.
_MM_TO_UM: float = 1000.0

#: Default prominence for peak detection in µM.  At rest, [Ca²⁺]ᵢ is typically
#: 0.05–0.1 µM; AP-evoked transients reach 0.3–2 µM.  A 0.05 µM prominence
#: rejects baseline noise and slow drift while accepting genuine transients.
_DEFAULT_PROMINENCE_UM: float = 0.05

#: Default minimum separation between transients in ms.
_DEFAULT_MIN_SEPARATION_MS: float = 5.0

#: Default initial guess for the decay time constant (ms).
_TAU_INITIAL_GUESS_MS: float = 50.0

#: Maximum decay-fit window (ms).  Caps the fit window so that very long
#: traces with slow drift do not distort the single-exponential fit.
_MAX_DECAY_WINDOW_MS: float = 500.0

#: Minimum number of samples required to attempt a decay fit.
_MIN_DECAY_SAMPLES: int = 5

#: Onset is defined as the first sample (walking back from the peak) that
#: falls below baseline + this fraction of the peak-to-baseline amplitude.
_ONSET_AMPLITUDE_FRACTION: float = 0.1

#: Pre-onset window (ms) used to compute a per-transient baseline.
_BASELINE_WINDOW_MS: float = 10.0

#: Fraction of the trace used as the global pre-stimulus baseline.
_GLOBAL_BASELINE_FRACTION: float = 0.05


@dataclasses.dataclass
class CalciumTransient:
    """Metrics for a single detected calcium transient.

    Attributes:
        index: Zero-based transient number within the trace.
        onset_time: Time at which the rise from baseline begins (ms).
        peak_time: Time of peak [Ca²⁺]ᵢ (ms).
        peak_concentration: Peak [Ca²⁺]ᵢ (µM).
        baseline_concentration: Per-transient baseline [Ca²⁺]ᵢ (µM), measured
            in a small window before ``onset_time``.
        amplitude: ``peak_concentration - baseline_concentration`` (µM).
        time_to_peak: Duration from onset to peak (ms).
        decay_tau: Decay time constant from the exponential fit (ms), or
            ``None`` when both the fit and the 1/e-fallback estimate fail.
        decay_fit_converged: ``True`` when :func:`scipy.optimize.curve_fit`
            converged; ``False`` when a fallback estimate (or ``None``) was
            used.
    """

    index: int
    onset_time: float
    peak_time: float
    peak_concentration: float
    baseline_concentration: float
    amplitude: float
    time_to_peak: float
    decay_tau: float | None
    decay_fit_converged: bool


@dataclasses.dataclass
class CalciumTransientAnalysisResult:
    """Complete calcium-transient analysis of a [Ca²⁺]ᵢ trace.

    Attributes:
        transient_count: Number of detected transients.
        transients: Per-transient metrics.
        baseline_concentration: Global pre-stimulus mean [Ca²⁺]ᵢ (µM), or
            ``None`` when the trace is empty.
        mean_peak_concentration: Mean peak [Ca²⁺]ᵢ across detected transients
            (µM), or ``None`` when no transients were detected.
        mean_time_to_peak: Mean time-to-peak across detected transients (ms),
            or ``None`` when no transients were detected.
        mean_decay_tau: Mean decay τ across transients with a usable τ (ms),
            or ``None`` when no transient yielded a usable τ.
        mean_amplitude: Mean ``peak - baseline`` across transients (µM), or
            ``None`` when no transients were detected.
    """

    transient_count: int
    transients: list[CalciumTransient]
    baseline_concentration: float | None
    mean_peak_concentration: float | None
    mean_time_to_peak: float | None
    mean_decay_tau: float | None
    mean_amplitude: float | None


def _decay_model(t: np.ndarray, a: float, tau: float, baseline: float) -> np.ndarray:
    """Single-exponential model for the falling phase of a calcium transient.

    Models the decay as ``[Ca](t) = a · exp(-t / tau) + baseline`` where
    ``t`` is time measured from the peak.

    Args:
        t: Time array relative to the peak (ms).
        a: Amplitude of the exponential component (µM); equals
            ``peak - baseline`` at ``t = 0``.
        tau: Decay time constant (ms); must be positive.
        baseline: Asymptotic [Ca²⁺]ᵢ (µM).

    Returns:
        Predicted [Ca²⁺]ᵢ array in µM.
    """
    return a * np.exp(-t / tau) + baseline


def _fallback_tau(
    t_decay: np.ndarray,
    ca_decay_uM: np.ndarray,
    peak_uM: float,
    baseline_uM: float,
) -> float | None:
    """Estimate τ from the 1/e-deflection crossing when curve_fit fails.

    Returns the time at which the trace has decayed to
    ``baseline + (1/e) · (peak - baseline)`` measured from the peak.

    Args:
        t_decay: Time array for the decay window relative to the peak (ms).
        ca_decay_uM: [Ca²⁺]ᵢ array for the decay window (µM).
        peak_uM: Peak [Ca²⁺]ᵢ (µM).
        baseline_uM: Per-transient baseline [Ca²⁺]ᵢ (µM).

    Returns:
        Estimated decay τ in ms, or ``None`` when the trace never reaches
        the 1/e level within the window.
    """
    amplitude = peak_uM - baseline_uM
    if amplitude <= 0.0:
        return None

    target = baseline_uM + amplitude / np.e
    crossings = np.where(ca_decay_uM <= target)[0]
    if len(crossings) == 0:
        return None
    return float(t_decay[crossings[0]])


def _fit_decay(
    t_decay: np.ndarray,
    ca_decay_uM: np.ndarray,
    peak_uM: float,
    baseline_uM: float,
) -> tuple[float | None, bool]:
    """Fit a single-exponential decay to the falling phase of a transient.

    Args:
        t_decay: Time array for the decay window relative to the peak (ms).
        ca_decay_uM: [Ca²⁺]ᵢ array for the decay window (µM).
        peak_uM: Peak [Ca²⁺]ᵢ (µM).
        baseline_uM: Per-transient baseline [Ca²⁺]ᵢ (µM); used as the fixed
            asymptote of the exponential model.

    Returns:
        Tuple ``(tau, converged)`` where ``tau`` is the decay τ in ms (or
        ``None`` when both the fit and the 1/e fallback fail) and
        ``converged`` is ``True`` only when :func:`scipy.optimize.curve_fit`
        succeeded with a positive τ.
    """
    if len(t_decay) < _MIN_DECAY_SAMPLES:
        return _fallback_tau(t_decay, ca_decay_uM, peak_uM, baseline_uM), False

    a0 = peak_uM - baseline_uM

    def model_fixed_baseline(t: np.ndarray, a: float, tau: float) -> np.ndarray:
        """Single-exponential decay with the per-transient baseline fixed.

        Args:
            t: Time relative to the peak (ms).
            a: Amplitude (µM).
            tau: Decay τ (ms).

        Returns:
            Predicted [Ca²⁺]ᵢ array in µM.
        """
        return _decay_model(t, a, tau, baseline_uM)

    try:
        popt, _ = curve_fit(
            model_fixed_baseline,
            t_decay,
            ca_decay_uM,
            p0=[a0, _TAU_INITIAL_GUESS_MS],
            bounds=([0.0, 1e-3], [np.inf, np.inf]),
            maxfev=5000,
        )
        tau = float(popt[1])
        if tau <= 0.0 or not np.isfinite(tau):
            raise RuntimeError("Non-positive or non-finite tau from curve_fit")
        return tau, True
    except (RuntimeError, ValueError) as exc:
        logger.debug("Calcium decay exponential fit failed: %s", exc)
        return _fallback_tau(t_decay, ca_decay_uM, peak_uM, baseline_uM), False


def _find_onset(
    ca_uM: np.ndarray,
    peak_idx: int,
    baseline_uM: float,
    search_floor_idx: int,
) -> int:
    """Walk backward from ``peak_idx`` to locate the rise onset.

    The onset is the first index (searching backward) at which the trace
    falls below ``baseline + _ONSET_AMPLITUDE_FRACTION · (peak - baseline)``.

    Args:
        ca_uM: [Ca²⁺]ᵢ array (µM).
        peak_idx: Index of the peak.
        baseline_uM: Per-transient baseline (µM).
        search_floor_idx: Smallest index allowed for the onset (inclusive);
            the search will not walk further back than this.

    Returns:
        Index of the rise onset.  Falls back to ``search_floor_idx`` when no
        sample below the threshold is found.
    """
    peak_uM = float(ca_uM[peak_idx])
    threshold = baseline_uM + _ONSET_AMPLITUDE_FRACTION * (peak_uM - baseline_uM)
    for i in range(peak_idx, search_floor_idx - 1, -1):
        if ca_uM[i] <= threshold:
            return i
    return search_floor_idx


def _empty_result(
    baseline_concentration: float | None = None,
) -> CalciumTransientAnalysisResult:
    """Return a result representing a trace with no detected transients.

    Args:
        baseline_concentration: Global pre-stimulus baseline (µM) when the
            trace was long enough to compute one; ``None`` otherwise.

    Returns:
        A :class:`CalciumTransientAnalysisResult` with ``transient_count`` of
        0 and all per-transient summary fields set to ``None``.
    """
    return CalciumTransientAnalysisResult(
        transient_count=0,
        transients=[],
        baseline_concentration=baseline_concentration,
        mean_peak_concentration=None,
        mean_time_to_peak=None,
        mean_decay_tau=None,
        mean_amplitude=None,
    )


def analyze_calcium_transients(
    time: np.ndarray,
    ca_i: np.ndarray,
    *,
    prominence_uM: float = _DEFAULT_PROMINENCE_UM,
    min_separation_ms: float = _DEFAULT_MIN_SEPARATION_MS,
    max_decay_window_ms: float = _MAX_DECAY_WINDOW_MS,
) -> CalciumTransientAnalysisResult:
    """Detect calcium transients and extract peak, time-to-peak, and decay τ.

    Transients are detected with :func:`scipy.signal.find_peaks` using a
    prominence criterion expressed in µM; this is robust to slow baseline
    drift and does not require an absolute height threshold.  For each
    detected peak the function locates the rise onset, computes a
    per-transient baseline from a short pre-onset window, and fits a
    single-exponential decay to the falling phase using
    :func:`scipy.optimize.curve_fit`.

    Args:
        time: Time array in ms, shape ``(N,)``.
        ca_i: Intracellular [Ca²⁺] in **mM**, shape ``(N,)``, same length as
            ``time``.  Output values are reported in µM.
        prominence_uM: Minimum peak prominence in µM.  Defaults to 0.05 µM
            (50 nM), which sits well above typical resting fluctuations and
            below typical AP-evoked transient amplitudes.
        min_separation_ms: Minimum spacing between detected peaks in ms.
        max_decay_window_ms: Cap on the decay-fit window in ms.

    Returns:
        A :class:`CalciumTransientAnalysisResult` with per-transient and
        aggregate metrics in µM and ms.
    """
    time = np.asarray(time, dtype=float)
    ca_i = np.asarray(ca_i, dtype=float)
    n = len(time)

    if n < 2:
        return _empty_result()

    ca_uM = ca_i * _MM_TO_UM

    dt_arr = np.diff(time)
    dt_ms = float(np.median(dt_arr)) if len(dt_arr) > 0 else 0.0
    if dt_ms <= 0.0:
        return _empty_result()

    # --- Global baseline (pre-stimulus mean of the first _GLOBAL_BASELINE_FRACTION) ---
    baseline_samples = max(1, int(_GLOBAL_BASELINE_FRACTION * n))
    global_baseline_uM = float(np.mean(ca_uM[:baseline_samples]))

    # --- Detect peaks ---
    distance = max(1, int(round(min_separation_ms / dt_ms)))
    peak_indices, properties = find_peaks(
        ca_uM, prominence=prominence_uM, distance=distance
    )

    if len(peak_indices) == 0:
        return _empty_result(baseline_concentration=global_baseline_uM)

    # `left_bases` from find_peaks is the lowest point to the left of each
    # peak that bounds the prominence — i.e. the inter-transient trough.
    # We use it as an anchor for both onset detection and per-transient
    # baseline estimation; both are kept independent of `prev_peak_idx`
    # so closely-spaced transients are still characterised correctly.
    left_bases = np.asarray(properties["left_bases"], dtype=int)

    baseline_window_samples = max(1, int(round(_BASELINE_WINDOW_MS / dt_ms)))
    max_decay_samples = max(_MIN_DECAY_SAMPLES, int(round(max_decay_window_ms / dt_ms)))

    transients: list[CalciumTransient] = []

    for i, peak_idx in enumerate(peak_indices):
        peak_idx_int = int(peak_idx)
        peak_uM = float(ca_uM[peak_idx_int])
        left_base = int(left_bases[i])

        # --- Per-transient baseline: short window ending at left_base
        #     (the inter-transient trough), so the rising phase never
        #     contaminates the baseline estimate. ---
        baseline_start = max(0, left_base - baseline_window_samples + 1)
        baseline_end = left_base + 1  # exclusive
        if baseline_end <= baseline_start:
            baseline_uM = float(ca_uM[left_base])
        else:
            baseline_uM = float(np.mean(ca_uM[baseline_start:baseline_end]))

        # --- Onset (walk back from peak to first sample below threshold) ---
        onset_idx = _find_onset(ca_uM, peak_idx_int, baseline_uM, left_base)

        # --- Decay fit window: peak → next onset / end / cap ---
        if i + 1 < len(peak_indices):
            decay_end_idx = int(peak_indices[i + 1])
        else:
            decay_end_idx = n
        decay_end_idx = min(decay_end_idx, peak_idx_int + max_decay_samples)
        decay_end_idx = min(decay_end_idx, n)

        decay_slice = slice(peak_idx_int, decay_end_idx)
        t_decay = time[decay_slice] - time[peak_idx_int]
        ca_decay_uM = ca_uM[decay_slice]

        decay_tau, fit_converged = _fit_decay(
            t_decay, ca_decay_uM, peak_uM, baseline_uM
        )

        peak_time = float(time[peak_idx_int])
        onset_time = float(time[onset_idx])

        transients.append(
            CalciumTransient(
                index=len(transients),
                onset_time=onset_time,
                peak_time=peak_time,
                peak_concentration=peak_uM,
                baseline_concentration=baseline_uM,
                amplitude=peak_uM - baseline_uM,
                time_to_peak=peak_time - onset_time,
                decay_tau=decay_tau,
                decay_fit_converged=fit_converged,
            )
        )

    mean_peak = float(np.mean([t.peak_concentration for t in transients]))
    mean_ttp = float(np.mean([t.time_to_peak for t in transients]))
    mean_amp = float(np.mean([t.amplitude for t in transients]))

    tau_values = [t.decay_tau for t in transients if t.decay_tau is not None]
    mean_tau: float | None = float(np.mean(tau_values)) if tau_values else None

    return CalciumTransientAnalysisResult(
        transient_count=len(transients),
        transients=transients,
        baseline_concentration=global_baseline_uM,
        mean_peak_concentration=mean_peak,
        mean_time_to_peak=mean_ttp,
        mean_decay_tau=mean_tau,
        mean_amplitude=mean_amp,
    )


def analyze_calcium_transients_from_result(
    result: SimulationResult,
    *,
    prominence_uM: float = _DEFAULT_PROMINENCE_UM,
    min_separation_ms: float = _DEFAULT_MIN_SEPARATION_MS,
    max_decay_window_ms: float = _MAX_DECAY_WINDOW_MS,
) -> CalciumTransientAnalysisResult | None:
    """Extract calcium transient metrics from a :class:`SimulationResult`.

    Convenience wrapper around :func:`analyze_calcium_transients` that pulls
    the ``"time"`` and ``"ca_i"`` fields from the structured array.  Returns
    ``None`` when ``ca_i`` is absent (i.e. calcium dynamics were not active
    during the simulation).

    Args:
        result: A structured NumPy array returned by
            :func:`~patch_sim.simulate_current_clamp` or its voltage-clamp
            equivalents.
        prominence_uM: Minimum peak prominence in µM.
        min_separation_ms: Minimum spacing between detected peaks in ms.
        max_decay_window_ms: Cap on the decay-fit window in ms.

    Returns:
        A :class:`CalciumTransientAnalysisResult` when ``ca_i`` is present,
        otherwise ``None``.
    """
    field_names = result.dtype.names
    if field_names is None or "ca_i" not in field_names:
        return None

    return analyze_calcium_transients(
        result["time"],
        result["ca_i"],
        prominence_uM=prominence_uM,
        min_separation_ms=min_separation_ms,
        max_decay_window_ms=max_decay_window_ms,
    )

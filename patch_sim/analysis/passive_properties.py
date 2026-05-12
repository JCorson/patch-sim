"""Passive membrane property extraction from subthreshold current clamp steps.

Provides :func:`analyze_passive_properties` for computing input resistance and
membrane time constant from a subthreshold voltage response.

Data classes:
    PassiveProperties: Passive membrane property measurements.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy.optimize import curve_fit

from patch_sim.analysis.ap_metrics import analyze_aps

logger = logging.getLogger(__name__)

#: Fraction of the stimulus window used as the steady-state measurement region.
_SS_FRACTION: float = 0.2

#: Fraction of the stimulus window used as the exponential fit window.
_FIT_FRACTION: float = 0.6

#: Maximum duration of the exponential fit window in ms.  The window is
#: capped at this value so that slow gating-variable relaxations (which occur
#: on longer timescales) do not distort the single-exponential fit of the fast
#: membrane capacitance charging transient.
_MAX_FIT_WINDOW_MS: float = 25.0

#: Minimum stimulus duration (ms) required to extract passive properties.
_MIN_STIM_DURATION_MS: float = 2.0

#: Default initial guess for the membrane time constant (ms).
_TAU_INITIAL_GUESS_MS: float = 5.0

#: Guard padding (ms) trimmed *before* a detected spike's threshold crossing
#: when carving out the spike-free portion of a trace.  Spike onset is sharp,
#: so a small pad suffices.
_SPIKE_GUARD_PRE_MS: float = 2.0

#: Guard padding (ms) trimmed *after* a detected spike's peak.  Must be long
#: enough to clear the after-hyperpolarization so the recovered segment carries
#: only the linear subthreshold response.
_SPIKE_GUARD_POST_MS: float = 20.0


@dataclasses.dataclass
class PassiveProperties:
    """Passive membrane properties extracted from a subthreshold current clamp step.

    The per-area density values (``input_resistance``, ``membrane_capacitance``)
    are always populated.  When ``area_cm2`` is supplied to
    :func:`analyze_passive_properties`, the absolute counterparts
    (``input_resistance_mohm``, ``membrane_capacitance_pf``) are filled in too.
    ``time_constant`` is the same numeric value in both modes — τₘ = R · C is
    invariant to the per-area / absolute conversion.

    Attributes:
        input_resistance: Input resistance in kΩ·cm² (R_in = ΔV / ΔI where
            ΔV is in mV and ΔI is in µA/cm²).
        time_constant: Membrane time constant in ms from exponential fit.
            Numerically identical regardless of whether area is supplied.
        membrane_capacitance: Derived C_m = τ_m / R_in in µF/cm², or ``None``
            when R_in is zero.
        fit_converged: ``True`` when the exponential fit converged successfully;
            ``False`` when a fallback 63.2%-crossing estimate was used instead.
        input_resistance_mohm: Absolute input resistance in MΩ when
            ``area_cm2`` is supplied; ``None`` otherwise.
        membrane_capacitance_pf: Absolute membrane capacitance in pF when
            ``area_cm2`` is supplied and the per-area capacitance is finite;
            ``None`` otherwise.
        area_cm2: Membrane surface area in cm² used to compute the absolute
            values.  ``None`` when only density values were requested.
    """

    input_resistance: float
    time_constant: float
    membrane_capacitance: float | None
    fit_converged: bool
    input_resistance_mohm: float | None = None
    membrane_capacitance_pf: float | None = None
    area_cm2: float | None = None


def density_to_absolute_r_in(
    input_resistance_kohm_cm2: float,
    area_cm2: float | None,
) -> float | None:
    """Convert per-area input resistance (kΩ·cm²) to absolute MΩ.

    Derivation: ``R_in [kΩ·cm²] / area [cm²] = R_n [kΩ] = R_n [MΩ] × 1000``,
    so ``R_n [MΩ] = R_in / area / 1000``.

    Args:
        input_resistance_kohm_cm2: Per-area input resistance in kΩ·cm².
        area_cm2: Total membrane surface area in cm².  ``None`` or
            non-positive returns ``None``.

    Returns:
        Absolute input resistance in MΩ, or ``None`` when ``area_cm2`` is
        ``None`` or not strictly positive.
    """
    if area_cm2 is None or area_cm2 <= 0:
        return None
    return input_resistance_kohm_cm2 / area_cm2 / 1000.0


def density_to_absolute_c_m(
    c_m_uf_cm2: float | None,
    area_cm2: float | None,
) -> float | None:
    """Convert per-area membrane capacitance (µF/cm²) to absolute pF.

    Derivation: ``C [pF] = C_m [µF/cm²] × area [cm²] × 1e6``.

    Args:
        c_m_uf_cm2: Per-area membrane capacitance in µF/cm².  ``None`` returns
            ``None``.
        area_cm2: Total membrane surface area in cm².  ``None`` or
            non-positive returns ``None``.

    Returns:
        Absolute capacitance in pF, or ``None`` when either input is ``None``
        or ``area_cm2`` is not strictly positive.
    """
    if c_m_uf_cm2 is None or area_cm2 is None or area_cm2 <= 0:
        return None
    return c_m_uf_cm2 * area_cm2 * 1e6


def is_subthreshold(
    time: np.ndarray,
    voltage: np.ndarray,
    dvdt_threshold: float = 20.0,
    min_spike_height: float = -20.0,
) -> bool:
    """Return ``True`` when a voltage trace contains no detected action potentials.

    Args:
        time: Time array in ms, shape ``(N,)``.
        voltage: Membrane voltage in mV, shape ``(N,)``, same length as
            ``time``.
        dvdt_threshold: dV/dt value (mV/ms) used to identify spike onset.
        min_spike_height: Minimum peak voltage (mV) for a candidate event to
            be classified as a spike.

    Returns:
        ``True`` if no spikes are detected, ``False`` otherwise.
    """
    result = analyze_aps(
        time,
        voltage,
        dvdt_threshold=dvdt_threshold,
        min_spike_height=min_spike_height,
    )
    return result.spike_count == 0


def longest_subthreshold_run(
    time: np.ndarray,
    voltage: np.ndarray,
    *,
    dvdt_threshold: float = 20.0,
    min_spike_height: float = -20.0,
) -> tuple[int, int] | None:
    """Find the longest contiguous spike-free span of a voltage trace.

    Runs :func:`analyze_aps` to locate spikes, excises a guard-padded window
    around each one — from :data:`_SPIKE_GUARD_PRE_MS` before its threshold
    crossing to :data:`_SPIKE_GUARD_POST_MS` after its peak, so the
    after-hyperpolarization transient is removed too — and returns the widest
    remaining gap.  When the trace contains no spikes the whole trace is
    returned.

    Args:
        time: Time array in ms, shape ``(N,)``, ascending.
        voltage: Membrane voltage in mV, shape ``(N,)``, same length as
            ``time``.
        dvdt_threshold: dV/dt value (mV/ms) used to identify spike onset.
        min_spike_height: Minimum peak voltage (mV) for a candidate event to be
            classified as a spike.

    Returns:
        Half-open index bounds ``(start_idx, stop_idx)`` into ``time`` /
        ``voltage`` of the longest spike-free span (``(0, N)`` when there are no
        spikes), or ``None`` when no spike-free span exists — every sample falls
        inside a guard window.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)
    n = time.size
    if n == 0:
        return None

    result = analyze_aps(
        time,
        voltage,
        dvdt_threshold=dvdt_threshold,
        min_spike_height=min_spike_height,
    )
    if result.spike_count == 0:
        return (0, n)

    # Merge the guard-padded "bad" time intervals around each spike.
    intervals = sorted(
        (s.threshold_time - _SPIKE_GUARD_PRE_MS, s.peak_time + _SPIKE_GUARD_POST_MS)
        for s in result.spikes
    )
    merged: list[list[float]] = [list(intervals[0])]
    for lo, hi in intervals[1:]:
        if lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    # Spike-free time gaps: before the first interval, between consecutive
    # intervals, and after the last interval.
    gaps: list[tuple[float, float]] = [(float(time[0]), merged[0][0])]
    for idx in range(len(merged) - 1):
        gaps.append((merged[idx][1], merged[idx + 1][0]))
    gaps.append((merged[-1][1], float(time[-1])))

    best: tuple[int, int] | None = None
    best_span = 0.0
    for gap_start, gap_end in gaps:
        if gap_end <= gap_start:
            continue
        start_idx = int(np.searchsorted(time, gap_start, side="left"))
        stop_idx = int(np.searchsorted(time, gap_end, side="right"))
        if stop_idx - start_idx < 2:
            continue
        span = float(time[stop_idx - 1] - time[start_idx])
        if span > best_span:
            best_span = span
            best = (start_idx, stop_idx)
    return best


def _exponential_model(t: np.ndarray, v_ss: float, a: float, tau: float) -> np.ndarray:
    """Single-exponential model for the voltage step response.

    Models the voltage as ``V(t) = v_ss + a * exp(-t / tau)`` where ``t`` is
    time measured from the stimulus onset.

    Args:
        t: Time array relative to stimulus onset (ms).
        v_ss: Steady-state voltage (mV).
        a: Amplitude of the exponential component (mV); equals
            ``V_baseline - V_ss`` at ``t = 0``.
        tau: Membrane time constant (ms); must be positive.

    Returns:
        Predicted voltage array in mV.
    """
    return v_ss + a * np.exp(-t / tau)


def _fallback_tau(
    t_fit: np.ndarray,
    v_fit: np.ndarray,
    v_baseline: float,
    v_ss: float,
) -> float:
    """Estimate τₘ from the 63.2%-deflection crossing when curve_fit fails.

    Finds the first time at which the voltage has covered 63.2% of the total
    deflection from baseline toward steady state, which equals one time
    constant for a pure exponential.

    Args:
        t_fit: Time array relative to stimulus onset (ms).
        v_fit: Voltage array for the fit window (mV).
        v_baseline: Baseline voltage before the step (mV).
        v_ss: Steady-state voltage during the step (mV).

    Returns:
        Estimated time constant in ms.  Falls back to half the fit window
        duration when no suitable crossing is found.
    """
    deflection = v_ss - v_baseline
    if deflection == 0.0:
        return float(t_fit[-1]) / 2.0

    target = v_baseline + 0.632 * deflection
    # Search for first crossing of the 63.2% level.
    if deflection > 0:
        # Depolarizing: voltage rising toward v_ss
        crossings = np.where(v_fit >= target)[0]
    else:
        # Hyperpolarizing: voltage falling toward v_ss
        crossings = np.where(v_fit <= target)[0]

    if len(crossings) == 0:
        return float(t_fit[-1]) / 2.0

    return float(t_fit[crossings[0]])


def analyze_passive_properties(
    time: np.ndarray,
    voltage: np.ndarray,
    current_amplitude: float,
    stim_start_ms: float,
    stim_end_ms: float,
    max_fit_window_ms: float = _MAX_FIT_WINDOW_MS,
    area_cm2: float | None = None,
) -> PassiveProperties | None:
    """Extract input resistance and membrane time constant from a CC step response.

    Returns ``None`` when the sweep is suprathreshold (contains spikes) or when
    ``current_amplitude`` is zero (R_in cannot be computed).  Also returns
    ``None`` when the stimulus duration is shorter than
    :data:`_MIN_STIM_DURATION_MS`.

    R_in is computed as ΔV / ΔI where ΔV is the steady-state voltage
    deflection from baseline and ΔI is ``current_amplitude``.  The result is
    in kΩ·cm² (mV / µA·cm⁻²).

    τₘ is obtained by fitting ``V(t) = V_ss + A·exp(-t/τ)`` to the rising
    (or falling) phase of the response using
    :func:`scipy.optimize.curve_fit`.  If the fit fails, a fallback estimate
    based on the 63.2%-deflection crossing time is used and
    ``fit_converged`` is set to ``False``.

    Cₘ is derived as τₘ / R_in.  When R_in is zero, ``membrane_capacitance``
    is ``None``.

    Args:
        time: Time array in ms, shape ``(N,)``.
        voltage: Membrane voltage in mV, shape ``(N,)``, same length as
            ``time``.
        current_amplitude: Injected current step amplitude in µA/cm².  Must be
            non-zero for a meaningful R_in estimate.
        stim_start_ms: Time at which the current step begins (ms).
        stim_end_ms: Time at which the current step ends (ms).
        max_fit_window_ms: Maximum duration of the exponential fit window in ms.
            Defaults to :data:`_MAX_FIT_WINDOW_MS` (25 ms).  Pass a larger
            value (e.g. 150 ms) when the trace comes from a passive-only
            simulation where slow gating-variable relaxations are absent.
        area_cm2: Optional membrane surface area in cm².  When supplied, the
            returned :class:`PassiveProperties` carries absolute MΩ / pF
            counterparts in addition to the density values.  When ``None``
            (default), only density values are populated.

    Returns:
        A :class:`PassiveProperties` instance, or ``None`` when analysis is
        not applicable.
    """
    time = np.asarray(time, dtype=float)
    voltage = np.asarray(voltage, dtype=float)

    stim_duration = stim_end_ms - stim_start_ms
    if current_amplitude == 0.0 or stim_duration < _MIN_STIM_DURATION_MS:
        return None

    if not is_subthreshold(time, voltage):
        return None

    # --- Baseline voltage: mean of samples before the step ---
    pre_mask = time < stim_start_ms
    if not np.any(pre_mask):
        return None
    v_baseline = float(np.mean(voltage[pre_mask]))

    # --- Steady-state voltage: mean of last _SS_FRACTION of the step ---
    ss_start_ms = stim_end_ms - _SS_FRACTION * stim_duration
    ss_mask = (time >= ss_start_ms) & (time < stim_end_ms)
    if not np.any(ss_mask):
        return None
    v_ss = float(np.mean(voltage[ss_mask]))

    # --- Input resistance ---
    delta_v = v_ss - v_baseline  # mV
    r_in = delta_v / current_amplitude  # kΩ·cm²

    # --- Exponential fit for τₘ ---
    # Fit window: from stim_start to stim_start + _FIT_FRACTION * stim_duration,
    # capped at max_fit_window_ms so that slow gating-variable relaxations do
    # not distort the single-exponential fit of the fast capacitative transient.
    fit_end_ms = stim_start_ms + min(_FIT_FRACTION * stim_duration, max_fit_window_ms)
    fit_mask = (time >= stim_start_ms) & (time < fit_end_ms)
    if not np.any(fit_mask):
        return None

    t_fit = time[fit_mask] - stim_start_ms  # relative to step onset
    v_fit = voltage[fit_mask]

    a0 = v_baseline - v_ss  # initial amplitude estimate
    p0 = [v_ss, a0, _TAU_INITIAL_GUESS_MS]

    fit_converged = True
    try:
        popt, _ = curve_fit(
            _exponential_model,
            t_fit,
            v_fit,
            p0=p0,
            bounds=(
                [-np.inf, -np.inf, 1e-3],  # tau must be positive
                [np.inf, np.inf, np.inf],
            ),
            maxfev=5000,
        )
        tau_m = float(popt[2])
        if tau_m <= 0:
            raise RuntimeError("Non-positive tau from curve_fit")
    except (RuntimeError, ValueError) as exc:
        logger.debug("Passive property exponential fit failed: %s", exc)
        fit_converged = False
        tau_m = _fallback_tau(t_fit, v_fit, v_baseline, v_ss)

    # --- Membrane capacitance ---
    c_m: float | None = tau_m / r_in if r_in != 0.0 else None

    r_in_mohm = density_to_absolute_r_in(r_in, area_cm2)
    c_m_pf = density_to_absolute_c_m(c_m, area_cm2)

    return PassiveProperties(
        input_resistance=r_in,
        time_constant=tau_m,
        membrane_capacitance=c_m,
        fit_converged=fit_converged,
        input_resistance_mohm=r_in_mohm,
        membrane_capacitance_pf=c_m_pf,
        area_cm2=area_cm2,
    )

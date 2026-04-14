"""Passive membrane property extraction from subthreshold current clamp steps.

Provides :func:`analyze_passive_properties` for computing input resistance and
membrane time constant from a subthreshold voltage response, and
:func:`analyze_passive_from_result` as a convenience wrapper for
:class:`~patch_sim.clamp_simulations.SimulationResult` structured arrays.

Data classes:
    PassiveProperties: Passive membrane property measurements.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

from patch_sim.analysis.ap_metrics import analyze_aps

if TYPE_CHECKING:
    from patch_sim.clamp_simulations import SimulationResult

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


@dataclasses.dataclass
class PassiveProperties:
    """Passive membrane properties extracted from a subthreshold current clamp step.

    Attributes:
        input_resistance: Input resistance in kΩ·cm² (R_in = ΔV / ΔI where
            ΔV is in mV and ΔI is in µA/cm²).
        time_constant: Membrane time constant in ms from exponential fit.
        membrane_capacitance: Derived C_m = τ_m / R_in in µF/cm², or ``None``
            when R_in is zero.
        fit_converged: ``True`` when the exponential fit converged successfully;
            ``False`` when a fallback 63.2%-crossing estimate was used instead.
    """

    input_resistance: float
    time_constant: float
    membrane_capacitance: float | None
    fit_converged: bool


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
    deflection from baseline towards steady state, which equals one time
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
        # Depolarising: voltage rising toward v_ss
        crossings = np.where(v_fit >= target)[0]
    else:
        # Hyperpolarising: voltage falling toward v_ss
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
    # capped at _MAX_FIT_WINDOW_MS so that slow gating-variable relaxations do
    # not distort the single-exponential fit of the fast capacitative transient.
    fit_end_ms = stim_start_ms + min(_FIT_FRACTION * stim_duration, _MAX_FIT_WINDOW_MS)
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

    return PassiveProperties(
        input_resistance=r_in,
        time_constant=tau_m,
        membrane_capacitance=c_m,
        fit_converged=fit_converged,
    )


def analyze_passive_from_result(
    result: SimulationResult,
    current_amplitude: float,
    stim_start_ms: float,
    stim_end_ms: float,
) -> PassiveProperties | None:
    """Extract passive properties from a SimulationResult structured array.

    Convenience wrapper around :func:`analyze_passive_properties` that pulls
    the ``"time"`` and ``"voltage"`` fields from the structured array.  See
    :class:`~patch_sim.clamp_simulations.SimulationResult` for the array
    schema.

    Args:
        result: A structured NumPy array returned by
            :func:`~patch_sim.simulate_current_clamp`.  Must contain
            ``"time"`` and ``"voltage"`` fields.
        current_amplitude: Injected current step amplitude in µA/cm².
        stim_start_ms: Time at which the current step begins (ms).
        stim_end_ms: Time at which the current step ends (ms).

    Returns:
        A :class:`PassiveProperties` instance, or ``None`` when analysis is
        not applicable.
    """
    return analyze_passive_properties(
        result["time"],
        result["voltage"],
        current_amplitude=current_amplitude,
        stim_start_ms=stim_start_ms,
        stim_end_ms=stim_end_ms,
    )

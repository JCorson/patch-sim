"""Steady-state inactivation (h∞) curve analysis for two-pulse voltage clamp.

Computes normalized channel availability h∞(V) = I_peak(V) / I_peak_max from an
existing :class:`~patch_sim.analysis.iv_curve.IVAnalysisResult` (whose stimulus
window is the fixed test pulse of a two-pulse protocol, so each
:class:`~patch_sim.analysis.iv_curve.IVPoint` carries the test-pulse peak inward
current at one conditioning prepulse voltage) and fits a *decreasing*
two-parameter Boltzmann sigmoid to characterise voltage-dependent inactivation.
The Boltzmann math (:func:`~patch_sim.analysis.gv_curve.boltzmann`) and the
fitted-parameter container (:class:`~patch_sim.analysis.gv_curve.BoltzmannFit`)
are shared with the activation g-V curve analysis.

Data classes:
    InactivationPoint: Per-prepulse availability record.
    InactivationAnalysisResult: Aggregated h∞ analysis output.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy.optimize import curve_fit

from .gv_curve import BoltzmannFit, boltzmann
from .iv_curve import IVAnalysisResult

logger = logging.getLogger(__name__)

# Default initial guesses and bounds for the decreasing-Boltzmann curve_fit.
# The half-inactivation voltage of fast Na+ channels is more hyperpolarized than
# their half-activation voltage, hence the more negative guess and wider range.
_VHALF_GUESS = -60.0  # mV — typical Na+ half-inactivation
_K_GUESS = 7.0  # mV — typical slope factor
_VHALF_BOUNDS = (-120.0, 20.0)
_K_BOUNDS = (0.5, 40.0)


@dataclasses.dataclass
class InactivationPoint:
    """Availability measurement for a single conditioning prepulse voltage.

    Attributes:
        prepulse_voltage: Conditioning prepulse voltage held before the fixed
            test pulse (mV).
        peak_inward_current: Most negative (inward) current measured during the
            test pulse for this prepulse (µA/cm²).
        h_normalized: Availability normalized to h∞ = I_peak / I_peak_max, where
            I_peak_max is the most negative peak across all prepulses
            (dimensionless, clamped to 0–1).
    """

    prepulse_voltage: float
    peak_inward_current: float
    h_normalized: float


@dataclasses.dataclass
class InactivationAnalysisResult:
    """Complete steady-state inactivation analysis from a two-pulse simulation.

    Points are sorted in ascending order of prepulse voltage.  The convenience
    properties ``prepulse_voltages``, ``peak_inward_currents``, and
    ``h_normalized_values`` extract the corresponding field from each point on
    demand.

    The fitted Boltzmann parameterizes a *decreasing* sigmoid
    ``h∞(V) = 1 / (1 + exp((V - v_half) / k))`` with ``k > 0`` — equivalently
    ``boltzmann(V, v_half, -k)`` using the shared
    :func:`~patch_sim.analysis.gv_curve.boltzmann` function (which is the
    *increasing* activation form).  Callers reconstructing the fit curve must
    therefore negate ``k``.

    Attributes:
        points: Per-prepulse availability records, sorted by ascending prepulse
            voltage.
        boltzmann: Fitted Boltzmann parameters (may have ``converged=False``).
            ``v_half`` is the half-inactivation voltage (mV) and ``k`` the
            slope factor (mV, positive); the curve they describe is decreasing.
    """

    points: list[InactivationPoint]
    boltzmann: BoltzmannFit

    @property
    def prepulse_voltages(self) -> list[float]:
        """Conditioning prepulse voltages in mV, sorted ascending."""
        return [p.prepulse_voltage for p in self.points]

    @property
    def peak_inward_currents(self) -> list[float]:
        """Test-pulse peak inward currents in µA/cm² at each prepulse."""
        return [p.peak_inward_current for p in self.points]

    @property
    def h_normalized_values(self) -> list[float]:
        """h∞ availability values (dimensionless, 0–1) at each prepulse."""
        return [p.h_normalized for p in self.points]


def _decreasing_boltzmann(
    V: float | np.ndarray,
    v_half: float,
    k: float,
) -> float | np.ndarray:
    """Evaluate the decreasing (inactivation) two-parameter Boltzmann sigmoid.

    Computes ``1 / (1 + exp((V - v_half) / k))`` by negating *k* and delegating
    to the shared (increasing) :func:`~patch_sim.analysis.gv_curve.boltzmann`
    function, so the activation and inactivation analyses share one
    implementation.  Used as the model passed to ``scipy.optimize.curve_fit``.

    Args:
        V: Membrane voltage in mV.  May be a scalar or a 1-D NumPy array.
        v_half: Half-inactivation voltage (mV).
        k: Slope factor (mV); positive for a decreasing sigmoid.

    Returns:
        Sigmoid value in [0, 1].  Returns a NumPy array when *V* is an array,
        or a float when *V* is a scalar.
    """
    return boltzmann(V, v_half, -k)


def compute_inactivation(iv_result: IVAnalysisResult) -> InactivationAnalysisResult:
    """Compute normalized availability vs. prepulse voltage from I-V results.

    Each :class:`~patch_sim.analysis.iv_curve.IVPoint` in *iv_result* is assumed
    to come from one sweep of a two-pulse inactivation protocol: its
    ``voltage_step`` is the conditioning prepulse voltage and its
    ``peak_inward_current`` is the most negative current measured during the
    fixed test pulse.  Availability is::

        h∞(V) = I_peak(V) / I_peak_max

    where ``I_peak_max`` is the most negative peak across all prepulses; the
    ratio is clamped to [0, 1] (a non-negative test-pulse peak — full
    inactivation, possibly net outward — maps to 0.0).

    A decreasing Boltzmann sigmoid ``h∞(V) = 1 / (1 + exp((V - v_half) / k))``
    is then fitted via ``scipy.optimize.curve_fit``.  When the fit does not
    converge, :attr:`BoltzmannFit.converged` is ``False`` and the parameter
    values are ``0.0`` and ``1.0`` respectively.  The returned ``k`` is positive
    and the fit curve is reconstructed as ``boltzmann(V, v_half, -k)``.

    Args:
        iv_result: Pre-computed I-V analysis result whose stimulus window is the
            two-pulse protocol's fixed test pulse, with one point per
            conditioning prepulse voltage.

    Returns:
        An :class:`InactivationAnalysisResult` with sorted per-prepulse
        availability records and Boltzmann fit parameters.  ``points`` is empty
        only when *iv_result* itself is empty.
    """
    # Sentinel returned on failure; parameter values are never exposed to the
    # user because all callers check BoltzmannFit.converged before using them.
    _null_fit = BoltzmannFit(v_half=0.0, k=1.0, converged=False)

    if not iv_result.points:
        return InactivationAnalysisResult(points=[], boltzmann=_null_fit)

    # iv_result.points is already sorted by voltage_step (analyze_iv sorts), but
    # sort defensively so the output ordering does not depend on the caller.
    ordered = sorted(iv_result.points, key=lambda p: p.voltage_step)
    voltages = [p.voltage_step for p in ordered]
    peaks = [p.peak_inward_current for p in ordered]

    i_max = min(peaks)  # most negative ⇒ largest available inward current
    if i_max >= 0.0:
        # No inward current at any prepulse — pathological (e.g. no Na+ channel
        # or test pulse below activation threshold).  Report zero availability.
        points = [
            InactivationPoint(
                prepulse_voltage=v, peak_inward_current=p, h_normalized=0.0
            )
            for v, p in zip(voltages, peaks)
        ]
        return InactivationAnalysisResult(points=points, boltzmann=_null_fit)

    h_norm = [min(1.0, max(0.0, p / i_max)) for p in peaks]
    points = [
        InactivationPoint(prepulse_voltage=v, peak_inward_current=p, h_normalized=hn)
        for v, p, hn in zip(voltages, peaks, h_norm)
    ]

    if len(points) < 2:
        return InactivationAnalysisResult(points=points, boltzmann=_null_fit)

    v_arr = np.array(voltages, dtype=float)
    h_arr = np.array(h_norm, dtype=float)

    fit = _null_fit
    try:
        popt, _ = curve_fit(
            _decreasing_boltzmann,
            v_arr,
            h_arr,
            p0=[_VHALF_GUESS, _K_GUESS],
            bounds=(
                [_VHALF_BOUNDS[0], _K_BOUNDS[0]],
                [_VHALF_BOUNDS[1], _K_BOUNDS[1]],
            ),
            maxfev=2000,
        )
        fit = BoltzmannFit(v_half=float(popt[0]), k=float(popt[1]), converged=True)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Inactivation Boltzmann fit did not converge: %s", exc)

    return InactivationAnalysisResult(points=points, boltzmann=fit)

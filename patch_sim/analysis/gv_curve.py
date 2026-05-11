"""g-V curve analysis for voltage clamp multi-sweep simulations.

Computes normalized peak conductance (G/G_max) at each voltage step from an
existing :class:`~patch_sim.analysis.iv_curve.IVAnalysisResult` and fits a
two-parameter Boltzmann sigmoid to characterise voltage-dependent activation.

Data classes:
    GVPoint: Per-step conductance record.
    BoltzmannFit: Fitted Boltzmann parameters.
    GVAnalysisResult: Aggregated g-V analysis output.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
from scipy.optimize import curve_fit

from .iv_curve import IVAnalysisResult

logger = logging.getLogger(__name__)

# Default initial guesses and bounds for Boltzmann curve_fit.
_VHALF_GUESS = -20.0  # mV — typical Na+ half-activation
_K_GUESS = 6.0  # mV — typical slope factor
_VHALF_BOUNDS = (-100.0, 60.0)
_K_BOUNDS = (0.5, 40.0)


@dataclasses.dataclass
class GVPoint:
    """Conductance measurement for a single voltage step in a g-V protocol.

    Attributes:
        voltage_step: Command voltage applied during the step (mV).
        conductance: Chord conductance at this step (mS/cm²).
        g_normalized: Conductance normalized to G/G_max (dimensionless, 0–1).
    """

    voltage_step: float
    conductance: float
    g_normalized: float


@dataclasses.dataclass
class BoltzmannFit:
    """Fitted parameters for a two-parameter Boltzmann sigmoid.

    The fit uses the function ``1 / (1 + exp(-(V - v_half) / k))``.

    Attributes:
        v_half: Half-activation voltage (mV).
        k: Slope factor (mV). Larger values mean shallower activation.
        converged: ``True`` when ``scipy.optimize.curve_fit`` succeeded.
    """

    v_half: float
    k: float
    converged: bool


@dataclasses.dataclass
class GVAnalysisResult:
    """Complete g-V analysis from a multi-sweep voltage clamp simulation.

    Points are sorted in ascending order of voltage step.  The convenience
    properties ``voltage_steps``, ``conductances``, and ``g_normalized_values``
    extract the corresponding field from each point on demand.

    Attributes:
        points: Per-step conductance records, sorted by ascending voltage.
        boltzmann: Fitted Boltzmann parameters (may have ``converged=False``).
        reversal_potential: Reversal potential used for conductance computation
            (mV).
    """

    points: list[GVPoint]
    boltzmann: BoltzmannFit
    reversal_potential: float

    @property
    def voltage_steps(self) -> list[float]:
        """Command voltages in mV, sorted ascending."""
        return [p.voltage_step for p in self.points]

    @property
    def conductances(self) -> list[float]:
        """Chord conductances in mS/cm² at each voltage step."""
        return [p.conductance for p in self.points]

    @property
    def g_normalized_values(self) -> list[float]:
        """G/G_max values (dimensionless, 0–1) at each voltage step."""
        return [p.g_normalized for p in self.points]


def boltzmann(
    V: float | np.ndarray,
    v_half: float,
    k: float,
) -> float | np.ndarray:
    """Evaluate the two-parameter Boltzmann sigmoid.

    Computes the standard voltage-dependent activation function::

        f(V) = 1 / (1 + exp(-(V - v_half) / k))

    Accepts either a scalar float or a NumPy array for *V*, which allows the
    function to be used directly with ``scipy.optimize.curve_fit``.

    Args:
        V: Membrane voltage in mV.  May be a scalar or a 1-D NumPy array.
        v_half: Half-activation voltage (mV).
        k: Slope factor (mV).

    Returns:
        Sigmoid value in [0, 1].  Returns a NumPy array when *V* is an array,
        or a float when *V* is a scalar.
    """
    return 1.0 / (1.0 + np.exp(-(V - v_half) / k))


def compute_gv(
    iv_result: IVAnalysisResult,
    reversal_potential: float,
    driving_force_threshold: float = 5.0,
) -> GVAnalysisResult:
    """Compute normalized conductance vs. voltage from I-V analysis results.

    For each voltage step the chord conductance is::

        g(V) = I_peak_inward / (V - E_rev)

    where ``E_rev`` is the supplied reversal potential.  Points where the
    absolute driving force ``|V - E_rev|`` is below *driving_force_threshold*
    are excluded to avoid division-by-near-zero artefacts.  Points that
    produce non-positive conductances (physically implausible for the inward
    activation curve) are also excluded.

    The retained conductances are normalized to G/G_max, then a Boltzmann
    sigmoid is fitted via ``scipy.optimize.curve_fit``.  If the fit does not
    converge, :attr:`BoltzmannFit.converged` is ``False`` and the parameter
    values are ``0.0`` and ``1.0`` respectively.

    Args:
        iv_result: Pre-computed I-V analysis result containing per-step peak
            inward currents.
        reversal_potential: Reversal potential used to compute driving force
            (mV).  Typically the Nernst potential for the dominant inward
            current carrier (e.g. E_Na).
        driving_force_threshold: Minimum absolute driving force (mV) required
            for a point to be included.  Defaults to ``5.0``.

    Returns:
        A :class:`GVAnalysisResult` with sorted per-step conductance records
        and Boltzmann fit parameters.
    """
    # Sentinel returned on failure; parameter values are never exposed to the
    # user because all callers check BoltzmannFit.converged before using them.
    _null_fit = BoltzmannFit(v_half=0.0, k=1.0, converged=False)

    if not iv_result.points:
        return GVAnalysisResult(
            points=[], boltzmann=_null_fit, reversal_potential=reversal_potential
        )

    # --- compute chord conductances and filter ---
    valid: list[tuple[float, float]] = []  # (voltage_step, conductance)
    for pt in iv_result.points:
        df = pt.voltage_step - reversal_potential
        if abs(df) < driving_force_threshold:
            continue
        g = pt.peak_inward_current / df
        if g <= 0.0:
            continue
        valid.append((pt.voltage_step, g))

    if len(valid) < 2:
        return GVAnalysisResult(
            points=[], boltzmann=_null_fit, reversal_potential=reversal_potential
        )

    valid.sort(key=lambda x: x[0])
    voltages = [v for v, _ in valid]
    conductances = [g for _, g in valid]

    g_max = max(conductances)
    g_norm = [g / g_max for g in conductances]

    points = [
        GVPoint(voltage_step=v, conductance=g, g_normalized=gn)
        for v, g, gn in zip(voltages, conductances, g_norm)
    ]

    # --- fit Boltzmann ---
    v_arr = np.array(voltages, dtype=float)
    gn_arr = np.array(g_norm, dtype=float)

    fit = _null_fit
    try:
        popt, _ = curve_fit(
            boltzmann,
            v_arr,
            gn_arr,
            p0=[_VHALF_GUESS, _K_GUESS],
            bounds=(
                [_VHALF_BOUNDS[0], _K_BOUNDS[0]],
                [_VHALF_BOUNDS[1], _K_BOUNDS[1]],
            ),
            maxfev=2000,
        )
        fit = BoltzmannFit(v_half=float(popt[0]), k=float(popt[1]), converged=True)
    except (RuntimeError, ValueError) as exc:
        logger.debug("Boltzmann fit did not converge: %s", exc)

    return GVAnalysisResult(
        points=points,
        boltzmann=fit,
        reversal_potential=reversal_potential,
    )

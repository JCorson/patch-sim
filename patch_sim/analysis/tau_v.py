"""τ-V analysis for voltage clamp multi-sweep simulations.

Fits single- and (optionally) double-exponential models to the activation and
inactivation phases of the current trace at each voltage step, extracting time
constants that characterise channel kinetics across voltage.  Complements the
steady-state I-V (:mod:`~patch_sim.analysis.iv_curve`) and G-V
(:mod:`~patch_sim.analysis.gv_curve`) analyses.

Data classes:
    ExponentialFit: Single-exponential fit parameters with convergence flag.
    DoubleExponentialFit: Two-exponential fit parameters with convergence flag.
    TauVPoint: Per-sweep activation and inactivation time constants.
    TauVAnalysisResult: Aggregated τ-V analysis output.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable

import numpy as np
from scipy.optimize import curve_fit

logger = logging.getLogger(__name__)

# Minimum number of samples required in a fit window before curve_fit is
# attempted; below this the τ is reported as None instead of risking spurious
# convergence on too few data points.
_MIN_FIT_POINTS: int = 5

# Fraction of the absolute peak current that the post-peak trace must fall
# below before an inactivation phase is considered present.  When the trace
# stays above this threshold (e.g. sustained K⁺ currents) the inactivation
# fit is skipped and τ_inact is reported as None.  Set to 0.85 so that a
# trace which decays by less than 15% of its peak is treated as "no
# meaningful inactivation" rather than producing a τ from near-flat data.
_INACTIVATION_PEAK_FRACTION: float = 0.85

# Fits whose τ converges within this fraction of either bound are treated
# as "pinned to the bound" and rejected — typically symptoms of an
# under-constrained or noise-only window where curve_fit drifts to the
# corner of the search space rather than locating a real time constant.
_TAU_BOUND_REJECT_FRACTION: float = 0.05

# Lower bound on τ (ms) for fits.  Protects curve_fit from collapsing onto
# values that produce numerical underflow in ``exp(-t/tau)``.
_TAU_FLOOR_MS: float = 1e-3

# τ_max for fits is ``stimulus_duration * _TAU_CEILING_FRACTION``.  Setting
# the ceiling above the stimulus duration allows the fit to express slow
# kinetics that have not fully relaxed within the step window.
_TAU_CEILING_FRACTION: float = 2.0

# Minimum ratio between slow and fast τ for a double-exponential fit to be
# accepted as physiologically meaningful (otherwise it is treated as a
# redundant single-exponential and discarded).
_DOUBLE_TAU_RATIO: float = 3.0

# Minimum fractional reduction in residual sum of squares that the
# double-exponential fit must achieve relative to the single-exponential fit
# to be preferred.
_DOUBLE_FIT_IMPROVEMENT: float = 0.10

# Maximum iterations passed to scipy.optimize.curve_fit; matches the value
# used in :mod:`~patch_sim.analysis.gv_curve`.
_MAXFEV: int = 2000

# Minimum fraction of the global ``|I|`` maximum that a local peak must
# reach before it is accepted as the activation peak.  Filters out tiny
# early wiggles (capacitive / gating transients) without rejecting genuine
# but secondary peaks (e.g. an inward Na⁺ peak that is ~70% of a later
# outward K⁺ peak).
_PEAK_PROMINENCE_FRACTION: float = 0.30


@dataclasses.dataclass
class ExponentialFit:
    """Fitted parameters for a single exponential model.

    Attributes:
        amplitude: Pre-exponential coefficient ``A``.  Sign convention follows
            the chosen model (positive for ``single_exp_rise``, signed for
            ``single_exp_decay``).
        tau_ms: Time constant τ in ms.
        offset: Baseline / asymptote ``C`` (µA/cm²).
        converged: ``True`` when ``scipy.optimize.curve_fit`` succeeded.
        rss: Residual sum of squares of the fit; used to compare against
            the double-exponential fit.  ``inf`` when the fit failed.
    """

    amplitude: float
    tau_ms: float
    offset: float
    converged: bool
    rss: float


@dataclasses.dataclass
class DoubleExponentialFit:
    """Fitted parameters for the two-exponential decay model.

    The model is ``A_fast * exp(-t/τ_fast) + A_slow * exp(-t/τ_slow) + C``,
    with the convention ``τ_fast <= τ_slow`` enforced by
    :func:`compute_tau_v_point` after fitting.

    Attributes:
        amplitude_fast: Fast component pre-exponential.
        tau_fast_ms: Faster of the two time constants (ms).
        amplitude_slow: Slow component pre-exponential.
        tau_slow_ms: Slower of the two time constants (ms).
        offset: Baseline / asymptote ``C`` (µA/cm²).
        converged: ``True`` when ``scipy.optimize.curve_fit`` succeeded.
        rss: Residual sum of squares of the fit.
    """

    amplitude_fast: float
    tau_fast_ms: float
    amplitude_slow: float
    tau_slow_ms: float
    offset: float
    converged: bool
    rss: float


@dataclasses.dataclass
class TauVPoint:
    """Time-constant measurements for a single voltage step.

    All τ fields are ``None`` when the corresponding fit was skipped (window
    too short, no inactivation phase observed, etc.) or did not converge.
    The ``*_converged`` flags distinguish between "skipped" (τ is ``None``
    and converged is ``False``) and "fit succeeded" (τ is a float and
    converged is ``True``).

    Attributes:
        voltage_step: Command voltage applied during the step (mV).
        tau_activation_ms: Time constant of the rising phase (ms), or
            ``None`` when no fit was produced.
        tau_inactivation_ms: Fast time constant of the decaying phase
            (ms), or ``None``.  When ``inactivation_is_double`` is ``True``,
            this is the fast component of the double-exponential fit.
        tau_inactivation_slow_ms: Slow time constant of the decaying phase
            (ms).  Populated only when ``inactivation_is_double`` is ``True``.
        activation_converged: ``True`` when the activation fit succeeded.
        inactivation_converged: ``True`` when the inactivation fit succeeded.
        inactivation_is_double: ``True`` when the double-exponential fit
            was preferred over the single-exponential fit.
    """

    voltage_step: float
    tau_activation_ms: float | None
    tau_inactivation_ms: float | None
    tau_inactivation_slow_ms: float | None
    activation_converged: bool
    inactivation_converged: bool
    inactivation_is_double: bool


@dataclasses.dataclass
class TauVAnalysisResult:
    """Aggregate τ-V analysis from a multi-sweep voltage clamp simulation.

    Points are sorted in ascending order of voltage step.

    Attributes:
        points: Per-step time constants, sorted by ascending voltage.
    """

    points: list[TauVPoint]

    @property
    def voltage_steps(self) -> list[float]:
        """Command voltages in mV, sorted ascending."""
        return [p.voltage_step for p in self.points]

    @property
    def tau_activation_values(self) -> list[float | None]:
        """Activation τ (ms) at each voltage step (``None`` for skipped/failed fits)."""
        return [p.tau_activation_ms for p in self.points]

    @property
    def tau_inactivation_values(self) -> list[float | None]:
        """Inactivation τ (ms) at each voltage step (``None`` for skipped/failed fits).

        When a double-exponential fit was accepted at a step, this is the
        fast component; the slow component is in
        :attr:`tau_inactivation_slow_values`.
        """
        return [p.tau_inactivation_ms for p in self.points]

    @property
    def tau_inactivation_slow_values(self) -> list[float | None]:
        """Slow inactivation τ (ms) at each step where a double-exp fit was accepted."""
        return [p.tau_inactivation_slow_ms for p in self.points]


def single_exp_rise(
    t: float | np.ndarray,
    A: float,
    tau: float,
    C: float,
) -> float | np.ndarray:
    """Evaluate a single-exponential rise to a plateau.

    Computes ``A * (1 - exp(-t / tau)) + C``.

    Args:
        t: Time in ms (scalar or 1-D array, with ``t = 0`` at the start of
            the rising phase).
        A: Plateau amplitude.
        tau: Activation time constant (ms).
        C: Baseline / starting offset.

    Returns:
        The model evaluated at ``t``.
    """
    return A * (1.0 - np.exp(-t / tau)) + C


def single_exp_decay(
    t: float | np.ndarray,
    A: float,
    tau: float,
    C: float,
) -> float | np.ndarray:
    """Evaluate a single-exponential decay toward an asymptote.

    Computes ``A * exp(-t / tau) + C``.

    Args:
        t: Time in ms (scalar or 1-D array, with ``t = 0`` at the start of
            the decaying phase).
        A: Initial amplitude above the asymptote.
        tau: Inactivation time constant (ms).
        C: Asymptote that the trace approaches.

    Returns:
        The model evaluated at ``t``.
    """
    return A * np.exp(-t / tau) + C


def double_exp_decay(
    t: float | np.ndarray,
    A1: float,
    tau1: float,
    A2: float,
    tau2: float,
    C: float,
) -> float | np.ndarray:
    """Evaluate a two-exponential decay toward an asymptote.

    Computes ``A1 * exp(-t / tau1) + A2 * exp(-t / tau2) + C``.  Used to
    capture inactivation processes with two distinct kinetic components
    (e.g. a fast and a slow inactivation gate).

    Args:
        t: Time in ms (scalar or 1-D array).
        A1: First component amplitude.
        tau1: First component time constant (ms).
        A2: Second component amplitude.
        tau2: Second component time constant (ms).
        C: Asymptote.

    Returns:
        The model evaluated at ``t``.
    """
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + C


def _slice_step_window(
    time: np.ndarray,
    current: np.ndarray,
    stim_start_ms: float,
    stim_end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the current trace to the stimulus window with time reset to zero.

    Args:
        time: Time axis array in ms.
        current: Current array in µA/cm², same length as ``time``.
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.

    Returns:
        A 2-tuple ``(t_window, i_window)`` of NumPy arrays.  ``t_window``
        starts at ``0`` so it is suitable for direct use as the independent
        variable in :func:`scipy.optimize.curve_fit`.
    """
    i_start = int(np.searchsorted(time, stim_start_ms))
    i_end = int(np.searchsorted(time, stim_end_ms))
    t_window = np.asarray(time[i_start:i_end], dtype=float) - stim_start_ms
    i_window = np.asarray(current[i_start:i_end], dtype=float)
    return t_window, i_window


def _peak_index_in_window(current_window: np.ndarray) -> int:
    """Return the index that splits activation from inactivation.

    Finds the first local maximum of ``|I(t)|`` whose magnitude is at least
    :data:`_PEAK_PROMINENCE_FRACTION` of the global absolute maximum.  This
    corresponds to the activation peak.  In total-current voltage-clamp
    traces with both inward (Na⁺) and outward (K⁺) components, the global
    ``argmax(|I|)`` can pick a late outward peak with no decay phase after
    it; conversely, the very first local maximum can be a sub-millisecond
    wiggle from the capacitive / gating transient.  Requiring the peak to
    be a meaningful fraction of the global max avoids both pitfalls.  When
    no qualifying local maximum exists (current rises monotonically through
    the window), falls back to the global ``argmax(|I|)``.

    Args:
        current_window: Current values within the stimulus window
            (µA/cm²).

    Returns:
        The integer index of the activation peak, or ``0`` when the
        window is empty.
    """
    if current_window.size == 0:
        return 0
    abs_curr = np.abs(current_window)
    global_max = float(np.max(abs_curr))
    if global_max == 0.0:
        return 0
    threshold = _PEAK_PROMINENCE_FRACTION * global_max
    diffs = np.diff(abs_curr)
    # Local maximum: derivative goes from > 0 to <= 0.  np.where indexing
    # is off by one relative to the original array.
    if diffs.size >= 2:
        rising = diffs[:-1] > 0.0
        falling = diffs[1:] <= 0.0
        sign_changes = np.where(rising & falling)[0]
        for idx in sign_changes:
            peak_idx = int(idx) + 1
            if abs_curr[peak_idx] >= threshold:
                return peak_idx
    return int(np.argmax(abs_curr))


def _fit_single_exp(
    model: Callable[..., float | np.ndarray],
    t: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]],
) -> ExponentialFit:
    """Fit a single-exponential model with a fixed scipy call signature.

    Args:
        model: A function of signature ``(t, A, tau, C) -> y`` such as
            :func:`single_exp_rise` or :func:`single_exp_decay`.
        t: Independent variable (ms), starting at zero.
        y: Dependent variable (current samples).
        p0: Initial guesses for ``[A, tau, C]``.
        bounds: ``(lower, upper)`` bounds for ``[A, tau, C]``.

    Returns:
        An :class:`ExponentialFit`.  ``converged=False`` and ``rss=inf``
        when ``curve_fit`` raises.
    """
    try:
        popt, _ = curve_fit(model, t, y, p0=p0, bounds=bounds, maxfev=_MAXFEV)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("single-exp fit failed: %s", exc)
        return ExponentialFit(
            amplitude=0.0,
            tau_ms=0.0,
            offset=0.0,
            converged=False,
            rss=float("inf"),
        )
    residuals = y - model(t, *popt)
    rss = float(np.sum(residuals * residuals))
    return ExponentialFit(
        amplitude=float(popt[0]),
        tau_ms=float(popt[1]),
        offset=float(popt[2]),
        converged=True,
        rss=rss,
    )


def _fit_double_exp_decay(
    t: np.ndarray,
    y: np.ndarray,
    p0: list[float],
    bounds: tuple[list[float], list[float]],
) -> DoubleExponentialFit:
    """Fit the double-exponential decay model and return ordered components.

    On success, the returned :class:`DoubleExponentialFit` has
    ``tau_fast_ms <= tau_slow_ms`` (and the matching amplitudes swapped
    accordingly).  On failure ``converged`` is ``False``.

    Args:
        t: Independent variable (ms) starting at zero.
        y: Dependent variable (current samples).
        p0: Initial guesses for ``[A1, tau1, A2, tau2, C]``.
        bounds: Bounds for ``[A1, tau1, A2, tau2, C]``.

    Returns:
        A :class:`DoubleExponentialFit`.
    """
    try:
        popt, _ = curve_fit(
            double_exp_decay, t, y, p0=p0, bounds=bounds, maxfev=_MAXFEV
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("double-exp fit failed: %s", exc)
        return DoubleExponentialFit(
            amplitude_fast=0.0,
            tau_fast_ms=0.0,
            amplitude_slow=0.0,
            tau_slow_ms=0.0,
            offset=0.0,
            converged=False,
            rss=float("inf"),
        )
    a1, tau1, a2, tau2, c = (float(p) for p in popt)
    if tau1 <= tau2:
        amp_fast, tau_fast = a1, tau1
        amp_slow, tau_slow = a2, tau2
    else:
        amp_fast, tau_fast = a2, tau2
        amp_slow, tau_slow = a1, tau1
    residuals = y - double_exp_decay(t, *popt)
    rss = float(np.sum(residuals * residuals))
    return DoubleExponentialFit(
        amplitude_fast=amp_fast,
        tau_fast_ms=tau_fast,
        amplitude_slow=amp_slow,
        tau_slow_ms=tau_slow,
        offset=c,
        converged=True,
        rss=rss,
    )


def compute_tau_v_point(
    time: np.ndarray,
    current: np.ndarray,
    voltage_step: float,
    stim_start_ms: float,
    stim_end_ms: float,
    *,
    fit_inactivation: bool = True,
) -> TauVPoint:
    """Fit activation and inactivation exponentials for one voltage-clamp sweep.

    The stimulus window is split at the absolute-value peak of the current.
    The rising segment (start → peak) is fit with :func:`single_exp_rise` to
    extract τ_activation; the decaying segment (peak → end) is fit with
    :func:`single_exp_decay` and, when ``fit_inactivation`` is ``True``, also
    with :func:`double_exp_decay`.  The double-exponential fit is preferred
    only when

    1. its residual sum of squares improves on the single-exponential fit by
       at least :data:`_DOUBLE_FIT_IMPROVEMENT` (10%), and
    2. its two τ values differ by at least :data:`_DOUBLE_TAU_RATIO`
       (a factor of 3),

    so that a marginal second component does not displace the simpler model.

    Failed or skipped fits surface as ``None`` τ values; this function never
    raises on bad data.

    Args:
        time: Shared time axis in ms (full sweep, not just the step window).
        current: Total ionic current array in µA/cm², same length as ``time``.
        voltage_step: Command voltage applied during the step (mV).
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.
        fit_inactivation: When ``False``, skip the inactivation fit entirely
            and report ``tau_inactivation_ms=None``.

    Returns:
        A :class:`TauVPoint` with the activation and (optional) inactivation
        time constants for this step.
    """
    t_win, i_win = _slice_step_window(time, current, stim_start_ms, stim_end_ms)
    stim_duration = stim_end_ms - stim_start_ms

    if i_win.size < _MIN_FIT_POINTS or stim_duration <= 0.0:
        return TauVPoint(
            voltage_step=voltage_step,
            tau_activation_ms=None,
            tau_inactivation_ms=None,
            tau_inactivation_slow_ms=None,
            activation_converged=False,
            inactivation_converged=False,
            inactivation_is_double=False,
        )

    peak_idx = _peak_index_in_window(i_win)
    t_act = t_win[: peak_idx + 1]
    i_act = i_win[: peak_idx + 1]
    t_inact = t_win[peak_idx:] - t_win[peak_idx]
    i_inact = i_win[peak_idx:]

    tau_max = stim_duration * _TAU_CEILING_FRACTION
    # τ values that converge near a bound are typically curve_fit hitting
    # the corner of the search space rather than locating a real time
    # constant; treat them as failed fits.
    tau_lo_reject = _TAU_FLOOR_MS * (1.0 + _TAU_BOUND_REJECT_FRACTION)
    tau_hi_reject = tau_max * (1.0 - _TAU_BOUND_REJECT_FRACTION)

    def _tau_is_bound_pinned(tau: float) -> bool:
        """Return True when ``tau`` lies near either fitting bound."""
        return tau <= tau_lo_reject or tau >= tau_hi_reject

    # ---- activation fit ------------------------------------------------ #
    tau_act: float | None = None
    activation_converged = False
    if t_act.size >= _MIN_FIT_POINTS:
        peak_value = float(i_act[-1])
        baseline = float(i_act[0])
        amp = peak_value - baseline
        # Allow either sign of amplitude; bounds are symmetric around zero
        # to handle both inward and outward currents.
        amp_abs = max(abs(amp), 1e-6)
        a_lo, a_hi = -10.0 * amp_abs, 10.0 * amp_abs
        c_lo, c_hi = baseline - 10.0 * amp_abs, baseline + 10.0 * amp_abs
        # ensure p0 is strictly inside bounds
        amp_seed = float(np.clip(amp, a_lo + 1e-9, a_hi - 1e-9))
        tau_seed = max(_TAU_FLOOR_MS * 10.0, min(stim_duration / 5.0, tau_max / 2.0))
        offset_seed = float(np.clip(baseline, c_lo + 1e-9, c_hi - 1e-9))
        p0 = [amp_seed, tau_seed, offset_seed]
        bounds = ([a_lo, _TAU_FLOOR_MS, c_lo], [a_hi, tau_max, c_hi])
        fit = _fit_single_exp(single_exp_rise, t_act, i_act, p0, bounds)
        if fit.converged and not _tau_is_bound_pinned(fit.tau_ms):
            tau_act = fit.tau_ms
            activation_converged = True

    # ---- inactivation fit ---------------------------------------------- #
    tau_inact: float | None = None
    tau_inact_slow: float | None = None
    inactivation_converged = False
    is_double = False

    if fit_inactivation and t_inact.size >= _MIN_FIT_POINTS:
        peak_value = float(i_inact[0])
        end_value = float(i_inact[-1])
        peak_abs = abs(peak_value)
        end_abs = abs(end_value)
        if peak_abs > 0.0 and end_abs <= _INACTIVATION_PEAK_FRACTION * peak_abs:
            amp = peak_value - end_value
            amp_abs = max(abs(amp), 1e-6)
            a_lo, a_hi = -10.0 * amp_abs, 10.0 * amp_abs
            c_lo, c_hi = end_value - 10.0 * amp_abs, end_value + 10.0 * amp_abs
            amp_seed = float(np.clip(amp, a_lo + 1e-9, a_hi - 1e-9))
            tau_seed = max(
                _TAU_FLOOR_MS * 10.0, min(stim_duration / 5.0, tau_max / 2.0)
            )
            offset_seed = float(np.clip(end_value, c_lo + 1e-9, c_hi - 1e-9))
            p0 = [amp_seed, tau_seed, offset_seed]
            bounds = ([a_lo, _TAU_FLOOR_MS, c_lo], [a_hi, tau_max, c_hi])
            single_fit = _fit_single_exp(single_exp_decay, t_inact, i_inact, p0, bounds)

            if single_fit.converged and not _tau_is_bound_pinned(single_fit.tau_ms):
                tau_inact = single_fit.tau_ms
                inactivation_converged = True

                # Try double-exp; accept only if it clearly improves the fit
                # and the two τ values are distinct.
                tau_fast_seed = max(_TAU_FLOOR_MS * 10.0, single_fit.tau_ms / 3.0)
                tau_slow_seed = min(tau_max / 2.0, single_fit.tau_ms * 3.0)
                if tau_slow_seed <= tau_fast_seed:
                    tau_slow_seed = min(tau_max / 2.0, tau_fast_seed * 3.0)
                p0_d = [
                    amp_seed * 0.5,
                    tau_fast_seed,
                    amp_seed * 0.5,
                    tau_slow_seed,
                    offset_seed,
                ]
                bounds_d = (
                    [a_lo, _TAU_FLOOR_MS, a_lo, _TAU_FLOOR_MS, c_lo],
                    [a_hi, tau_max, a_hi, tau_max, c_hi],
                )
                double_fit = _fit_double_exp_decay(t_inact, i_inact, p0_d, bounds_d)
                if double_fit.converged and single_fit.rss > 0.0:
                    rss_drop = (single_fit.rss - double_fit.rss) / single_fit.rss
                    tau_fast = double_fit.tau_fast_ms
                    tau_slow = double_fit.tau_slow_ms
                    if (
                        rss_drop >= _DOUBLE_FIT_IMPROVEMENT
                        and not _tau_is_bound_pinned(tau_fast)
                        and not _tau_is_bound_pinned(tau_slow)
                        and tau_fast > 0.0
                        and tau_slow / tau_fast >= _DOUBLE_TAU_RATIO
                    ):
                        tau_inact = tau_fast
                        tau_inact_slow = tau_slow
                        is_double = True

    return TauVPoint(
        voltage_step=voltage_step,
        tau_activation_ms=tau_act,
        tau_inactivation_ms=tau_inact,
        tau_inactivation_slow_ms=tau_inact_slow,
        activation_converged=activation_converged,
        inactivation_converged=inactivation_converged,
        inactivation_is_double=is_double,
    )


def analyze_tau_v(
    time: np.ndarray,
    currents: list[np.ndarray],
    voltage_steps: list[float],
    stim_start_ms: float,
    stim_end_ms: float,
    *,
    fit_inactivation: bool = True,
) -> TauVAnalysisResult:
    """Compute τ-V from a multi-sweep voltage clamp simulation.

    Calls :func:`compute_tau_v_point` for each sweep and assembles the points
    sorted in ascending order of voltage step.  Failed individual fits do
    not raise; they surface as ``None`` τ values on their respective
    :class:`TauVPoint`.

    Caveat — total-current vs. isolated-channel kinetics:
        When ``currents`` is the total membrane current (rather than an
        isolated channel current obtained via, e.g. TTX subtraction), the
        recovered τ values reflect the dominant channel at each voltage.
        At strong depolarizations of an HH-style neuron the total current is
        dominated by the much larger sustained K⁺ outward current, which
        masks the brief inward Na⁺ inactivation; the τ_inactivation values
        at those voltages should not be interpreted as the isolated Na⁺
        inactivation time constant.  For unambiguous channel-specific
        kinetics, run this analysis on a pharmacologically isolated current.

    Args:
        time: Shared time axis in ms (same for all sweeps).
        currents: List of total ionic current arrays (µA/cm²), one per sweep.
        voltage_steps: Command voltage (mV) for each sweep, parallel to
            ``currents``.
        stim_start_ms: Start of the stimulus window in ms.
        stim_end_ms: End of the stimulus window in ms.
        fit_inactivation: When ``False``, skip every inactivation fit.

    Returns:
        A :class:`TauVAnalysisResult` with per-step τ values sorted by
        voltage.
    """
    points = [
        compute_tau_v_point(
            time,
            current,
            v,
            stim_start_ms,
            stim_end_ms,
            fit_inactivation=fit_inactivation,
        )
        for current, v in zip(currents, voltage_steps)
    ]
    points.sort(key=lambda p: p.voltage_step)
    return TauVAnalysisResult(points=points)

"""Unit tests for patch_sim.analysis.passive_properties.

Covers subthreshold detection on flat traces, guard conditions with zero/short
stimuli, and exponential fitting on synthetic step responses. Tests that drive
a real HH simulation live in tests/integration/test_passive_properties_simulation.py.
"""

import numpy as np
import pytest

from patch_sim.analysis.passive_properties import (
    _SPIKE_GUARD_POST_MS,
    analyze_passive_properties,
    density_to_absolute_c_m,
    density_to_absolute_r_in,
    is_subthreshold,
    longest_subthreshold_run,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DT = 0.025  # ms — matches SIM_SAMPLING_FREQ = 40 kHz
_PRE_MS = 50.0
_STIM_MS = 200.0
_POST_MS = 50.0


def _make_time(duration_ms: float) -> np.ndarray:
    """Create a time array from 0 to duration_ms with dt = 0.025 ms.

    Args:
        duration_ms: Total duration in ms.

    Returns:
        1-D float array of time points in ms.
    """
    return np.arange(0.0, duration_ms, _DT)


def _flat_trace(
    duration_ms: float = 100.0, v: float = -65.0
) -> tuple[np.ndarray, np.ndarray]:
    """Create a constant-voltage trace.

    Args:
        duration_ms: Total duration in ms.
        v: Voltage level in mV.

    Returns:
        Tuple of (time, voltage) arrays.
    """
    t = _make_time(duration_ms)
    return t, np.full_like(t, v)


def _synthetic_step_trace(
    v_baseline: float,
    v_ss: float,
    tau_ms: float,
    pre_ms: float = _PRE_MS,
    stim_ms: float = _STIM_MS,
    post_ms: float = _POST_MS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic voltage trace with an ideal exponential step response.

    The voltage is held at ``v_baseline`` before the step, follows
    ``V(t) = v_ss + (v_baseline - v_ss) * exp(-t / tau_ms)`` during the step,
    and is held at ``v_ss`` after the step.

    Args:
        v_baseline: Baseline voltage before the step (mV).
        v_ss: Steady-state voltage during the step (mV).
        tau_ms: True membrane time constant (ms).
        pre_ms: Duration of the pre-stimulus period (ms).
        stim_ms: Duration of the current step (ms).
        post_ms: Duration of the post-stimulus period (ms).

    Returns:
        Tuple of (time, voltage) arrays.
    """
    total_ms = pre_ms + stim_ms + post_ms
    time = _make_time(total_ms)
    voltage = np.empty_like(time)

    pre_mask = time < pre_ms
    stim_mask = (time >= pre_ms) & (time < pre_ms + stim_ms)
    post_mask = time >= pre_ms + stim_ms

    voltage[pre_mask] = v_baseline
    t_rel = time[stim_mask] - pre_ms
    voltage[stim_mask] = v_ss + (v_baseline - v_ss) * np.exp(-t_rel / tau_ms)
    voltage[post_mask] = v_ss

    return time, voltage


# ---------------------------------------------------------------------------
# is_subthreshold
# ---------------------------------------------------------------------------


def test_is_subthreshold_flat_trace() -> None:
    """Flat constant-voltage trace contains no spikes and is subthreshold."""
    time, voltage = _flat_trace(duration_ms=100.0, v=-65.0)
    assert is_subthreshold(time, voltage) is True


# ---------------------------------------------------------------------------
# longest_subthreshold_run
# ---------------------------------------------------------------------------


def _inject_spike(voltage: np.ndarray, time: np.ndarray, center_ms: float) -> None:
    """Inject a brief synthetic spike around ``center_ms`` into a flat voltage trace.

    The spike is a 1 ms-wide +30 mV pulse — sharp enough to read as a
    threshold-crossing event for ``analyze_aps`` at the default
    ``dvdt_threshold=20`` mV/ms.

    Args:
        voltage: Voltage trace to modify in place.
        time: Time axis aligned with ``voltage``.
        center_ms: Center of the spike in ms.
    """
    mask = (time >= center_ms) & (time < center_ms + 1.0)
    voltage[mask] = 30.0


def test_longest_subthreshold_run_no_spikes_returns_whole_trace() -> None:
    """A spike-free trace returns the full ``(0, N)`` index range."""
    time, voltage = _flat_trace(duration_ms=200.0)
    run = longest_subthreshold_run(time, voltage)
    assert run == (0, time.size)


def test_longest_subthreshold_run_one_spike_returns_trailing_gap() -> None:
    """One early spike leaves a long trailing spike-free segment past the AHP guard."""
    time, voltage = _flat_trace(duration_ms=500.0)
    _inject_spike(voltage, time, center_ms=50.0)

    run = longest_subthreshold_run(time, voltage)

    assert run is not None
    start_idx, stop_idx = run
    # The recovered segment must start well after the spike's peak time +
    # the post-guard window.
    assert float(time[start_idx]) >= 50.0 + _SPIKE_GUARD_POST_MS
    assert stop_idx == time.size
    # And it should cover most of the remaining trace duration.
    assert float(time[stop_idx - 1] - time[start_idx]) > 400.0


def test_longest_subthreshold_run_spikes_throughout_returns_none() -> None:
    """Tightly-packed spikes leave no spike-free span; the function returns ``None``."""
    time, voltage = _flat_trace(duration_ms=300.0)
    # Refractory at 40 kHz is 1 ms, so 5 ms spacing is comfortably resolved.
    # Spaced 5 ms apart, guard windows (~22 ms wide) fully overlap into one
    # merged excision interval — and we start at 1 ms and extend past the end
    # of the trace so the leading and trailing gaps both collapse to empty.
    for center in np.arange(1.0, 305.0, 5.0):
        _inject_spike(voltage, time, center_ms=float(center))

    assert longest_subthreshold_run(time, voltage) is None


# ---------------------------------------------------------------------------
# analyze_passive_properties — guard conditions
# ---------------------------------------------------------------------------


def test_returns_none_for_zero_current() -> None:
    """analyze_passive_properties returns None when current amplitude is zero."""
    time, voltage = _flat_trace(duration_ms=200.0)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=0.0,
        stim_start_ms=50.0,
        stim_end_ms=150.0,
    )
    assert props is None


def test_returns_none_for_short_stimulus() -> None:
    """analyze_passive_properties returns None when the stimulus is too short."""
    time, voltage = _flat_trace(duration_ms=200.0)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-1.0,
        stim_start_ms=50.0,
        stim_end_ms=51.0,  # only 1 ms — below _MIN_STIM_DURATION_MS
    )
    assert props is None


# ---------------------------------------------------------------------------
# analyze_passive_properties — synthetic exponential trace
# ---------------------------------------------------------------------------


def test_synthetic_exponential_recovers_tau() -> None:
    """Exponential fit recovers the true time constant from a synthetic trace."""
    v_baseline = -65.0
    v_ss = -75.0  # hyperpolarizing step
    true_tau = 10.0  # ms
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    delta_v = v_ss - v_baseline  # -10 mV
    current_amplitude = -2.0  # µA/cm²; R_in = 5 kΩ·cm²
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.fit_converged is True
    assert props.time_constant == pytest.approx(true_tau, rel=0.05)
    expected_rin = delta_v / current_amplitude  # 5.0 kΩ·cm²
    assert props.input_resistance == pytest.approx(expected_rin, rel=0.05)


def test_synthetic_exponential_membrane_capacitance() -> None:
    """Derived Cₘ = τ / R_in matches the analytically expected value."""
    v_baseline = -65.0
    v_ss = -75.0
    true_tau = 8.0
    current_amplitude = -2.0
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.membrane_capacitance is not None
    # R_in = (v_ss - v_baseline) / current_amplitude = (-10) / (-2) = 5 kΩ·cm²
    # C_m = tau / R_in = 8 / 5 = 1.6 µF/cm²
    expected_rin = (v_ss - v_baseline) / current_amplitude  # 5.0
    expected_cm = true_tau / expected_rin  # 1.6
    assert props.membrane_capacitance == pytest.approx(expected_cm, rel=0.05)


def test_depolarizing_step_gives_positive_rin() -> None:
    """Depolarizing subthreshold step yields a positive R_in."""
    v_baseline = -65.0
    v_ss = -60.0  # depolarizing (+5 mV)
    true_tau = 5.0
    current_amplitude = 1.0  # +1 µA/cm²
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.input_resistance > 0.0


def test_hyperpolarizing_step_gives_positive_rin() -> None:
    """Hyperpolarizing step (negative current) also yields a positive R_in."""
    v_baseline = -65.0
    v_ss = -75.0  # hyperpolarizing (-10 mV)
    true_tau = 5.0
    current_amplitude = -2.0  # -2 µA/cm²; R_in = 5.0 (positive)
    time, voltage = _synthetic_step_trace(v_baseline, v_ss, true_tau)

    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=current_amplitude,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.input_resistance > 0.0


# ---------------------------------------------------------------------------
# density_to_absolute conversion helpers
# ---------------------------------------------------------------------------


def test_density_to_absolute_r_in_basic() -> None:
    """R_n [MΩ] = R_in [kΩ·cm²] / area [cm²] / 1000."""
    # R_in 20 kΩ·cm² ÷ 20×10⁻⁶ cm² ÷ 1000 = 1000 MΩ
    assert density_to_absolute_r_in(20.0, 20e-6) == pytest.approx(1000.0)


def test_density_to_absolute_r_in_none_area_returns_none() -> None:
    """``area_cm2=None`` returns ``None`` from density_to_absolute_r_in."""
    assert density_to_absolute_r_in(20.0, None) is None


@pytest.mark.parametrize("area", [0.0, -1e-6])
def test_density_to_absolute_r_in_non_positive_area_returns_none(
    area: float,
) -> None:
    """Zero or negative area returns ``None`` instead of inf / a negative value."""
    assert density_to_absolute_r_in(20.0, area) is None


def test_density_to_absolute_c_m_basic() -> None:
    """C [pF] = C_m [µF/cm²] × area [cm²] × 1e6."""
    # 1.0 µF/cm² × 20×10⁻⁶ cm² × 1e6 = 20 pF
    assert density_to_absolute_c_m(1.0, 20e-6) == pytest.approx(20.0)


def test_density_to_absolute_c_m_none_returns_none() -> None:
    """``c_m_uf_cm2=None`` returns ``None``."""
    assert density_to_absolute_c_m(None, 20e-6) is None


def test_density_to_absolute_c_m_none_area_returns_none() -> None:
    """``area_cm2=None`` returns ``None``."""
    assert density_to_absolute_c_m(1.0, None) is None


# ---------------------------------------------------------------------------
# analyze_passive_properties with area_cm2
# ---------------------------------------------------------------------------


def test_analyze_passive_properties_without_area_leaves_absolute_none() -> None:
    """Without area_cm2, absolute fields stay None and density fields populate."""
    time, voltage = _synthetic_step_trace(-65.0, -75.0, tau_ms=10.0)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-2.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    assert props is not None
    assert props.input_resistance_mohm is None
    assert props.membrane_capacitance_pf is None
    assert props.area_cm2 is None
    assert props.input_resistance == pytest.approx(5.0, rel=0.05)


def test_analyze_passive_properties_with_area_populates_absolute() -> None:
    """Supplying area_cm2 produces matching absolute MΩ / pF outputs."""
    area = 20e-6
    true_tau = 8.0
    time, voltage = _synthetic_step_trace(-65.0, -75.0, tau_ms=true_tau)
    props = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-2.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
        area_cm2=area,
    )
    assert props is not None
    assert props.area_cm2 == pytest.approx(area)
    assert props.input_resistance_mohm == pytest.approx(
        props.input_resistance / area / 1000.0
    )
    assert props.membrane_capacitance is not None
    assert props.membrane_capacitance_pf == pytest.approx(
        props.membrane_capacitance * area * 1e6
    )
    # τ_m = R_n × C identity: in absolute units τ_m [ms] = R_n [MΩ] × C [pF] / 1000
    assert props.input_resistance_mohm is not None
    assert props.membrane_capacitance_pf is not None
    tau_from_absolute = (
        props.input_resistance_mohm * props.membrane_capacitance_pf / 1000.0
    )
    assert tau_from_absolute == pytest.approx(props.time_constant, rel=1e-9)


def test_analyze_passive_properties_tau_invariant_to_area() -> None:
    """τ_m is identical whether or not area_cm2 is supplied."""
    time, voltage = _synthetic_step_trace(-65.0, -75.0, tau_ms=8.0)
    props_density = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-2.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
    )
    props_absolute = analyze_passive_properties(
        time,
        voltage,
        current_amplitude=-2.0,
        stim_start_ms=_PRE_MS,
        stim_end_ms=_PRE_MS + _STIM_MS,
        area_cm2=15e-6,
    )
    assert props_density is not None
    assert props_absolute is not None
    assert props_density.time_constant == props_absolute.time_constant

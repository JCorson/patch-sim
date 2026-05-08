"""Tests for the current clamp simulation."""

import numpy as np
import pytest

from patch_sim.clamp_simulations import SIM_SAMPLING_FREQ, simulate_current_clamp
from patch_sim.constants import DOPAMINERGIC, PURKINJE, STN, TRN
from patch_sim.neuron import Neuron  # noqa: F401  (used in type hints below)
from patch_sim.presets import NEURON_PRESET_NAMES, NEURON_PRESETS, make_squid_giant_axon

# Purkinje is a pacemaker with an UNSTABLE zero-current equilibrium at v_rest.
# Starting from v_rest = −65 mV it drifts below threshold — the stability
# criterion does not apply.  See test_fires_spontaneously in
# tests/integration/test_purkinje.py for the pacemaking behaviour check.
#
# TRN is excluded for the same reason after issue #295 added Ih: the cell is
# spontaneously active at ~3 Hz (consistent with the tonic firing reported in
# HP92 and B&M93 slice recordings) and has no stable zero-current
# equilibrium.  The HP92 rebound-burst phenotype (the key TRN behaviour) is
# exercised by ``test_trn_step_release_produces_hp92_rebound_burst`` in
# tests/integration/test_burst_metrics_simulation.py.
#
# DOPAMINERGIC is excluded after issue #304 — the post-fix preset is a tonic
# pacemaker (~4 Hz at zero current driven by the Cav1.3 + INaP_SNc subthreshold
# ramp and SK-shaped AHP) with no stable zero-current equilibrium.  Pacemaking
# and AP shape are pinned by ``test_da_spontaneous_pacemaking`` in
# tests/integration/test_dopaminergic.py.
#
# STN is excluded post-#305: the autonomous-pacemaker preset uses INaP and
# Ih to destabilise rest near −60 mV and fires spontaneously at ~11 Hz from
# v_rest = −60 mV (Bevan & Wilson 1999: 5–50 Hz autonomous firing in slice).
# See test_stn_spontaneous_pacemaking in tests/integration/test_stn.py.
_STABLE_PRESET_NAMES = [
    p for p in NEURON_PRESET_NAMES if p not in (PURKINJE, TRN, DOPAMINERGIC, STN)
]


def test_simulate_current_clamp_returns_structured_array(hh_model):
    """Test that simulate_current_clamp returns a structured array."""
    # Create a current array for a 10ms simulation
    duration = 10  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    current = np.full(num_steps, 20.0)  # constant current

    result = simulate_current_clamp(
        hh_model,
        current_external=current,
    )

    # Check result type
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None

    # Check that result has a time field
    assert "time" in result.dtype.names

    # Check that result has the expected fields
    expected_columns = [
        "voltage",
        "n",
        "m",
        "h",
    ]
    for col in expected_columns:
        assert col in result.dtype.names

    # Check that gating variables are within physiological bounds (0 to 1)
    gating_vars = ["n", "m", "h"]
    for gating_var in gating_vars:
        assert (result[gating_var] >= 0).all()
        assert (result[gating_var] <= 1).all()

    # Check that voltage is within reasonable physiological bounds
    assert result["voltage"].min() >= -100  # mV
    assert result["voltage"].max() <= 70  # mV


def test_simulation_dynamics():
    """Test that the simulation shows expected dynamics."""
    # K_out=7.8 mM matches HH52 squid seawater; DEFAULT_K_OUT is 4.0 mM (mammalian).
    custom_model = make_squid_giant_axon()

    # Create current array for a 50ms simulation
    duration = 50  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    current = np.full(num_steps, 20.0)  # constant current

    result = simulate_current_clamp(
        custom_model,
        current_external=current,
    )

    # Voltage should change from initial value
    initial_voltage = result["voltage"][0]
    max_voltage = result["voltage"].max()
    assert initial_voltage != max_voltage

    # Should observe an action potential (voltage exceeding threshold)
    threshold = 0  # mV, typical AP threshold
    assert any(result["voltage"] > threshold)

    # Sodium activation should increase during depolarization
    assert result["m"].max() > result["m"][0]


def test_simulate_current_clamp_with_zero_current(hh_model):
    """Test simulate_current_clamp with zero external current.

    Small fluctuations in voltage may occur due to intrinsic dynamics.
    """
    # Create zero current array for a 10ms simulation
    duration = 10  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    zero_current = np.zeros(num_steps)  # zero current array

    result = simulate_current_clamp(
        hh_model,
        current_external=zero_current,
    )

    # Check result type and structure
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    expected_columns = [
        "voltage",
        "n",
        "m",
        "h",
    ]
    for col in expected_columns:
        assert col in result.dtype.names

    # With zero external current, voltage should remain at the resting potential.
    max_allowable_deviation = 1.0  # mV
    voltage_range = result["voltage"].max() - result["voltage"].min()
    assert voltage_range < max_allowable_deviation, (
        f"Voltage range {voltage_range:.3f} mV exceeds {max_allowable_deviation} mV"
    )


@pytest.mark.parametrize("current_amplitude", [10.0, 20.0, 50.0])
def test_simulate_current_clamp_with_non_zero_currents(current_amplitude: float):
    """Test simulate_current_clamp with various non-zero external currents."""
    duration = 10  # ms

    # K_out=7.8 mM matches HH52 squid seawater; DEFAULT_K_OUT is 4.0 mM (mammalian).
    custom_model = make_squid_giant_axon()
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1

    # Create constant current array for the given amplitude
    current_array = np.full(num_steps, current_amplitude)

    result = simulate_current_clamp(
        custom_model,
        current_external=current_array,
    )

    # Check result type and structure
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    expected_columns = [
        "voltage",
        "n",
        "m",
        "h",
    ]
    for col in expected_columns:
        assert col in result.dtype.names

    # For non-zero currents, voltage should change significantly
    initial_voltage = result["voltage"][0]
    max_change = abs(result["voltage"].max() - initial_voltage)

    # Adjusted threshold based on observed behavior
    assert max_change > 3.0, (
        f"Voltage did not change significantly for current {current_amplitude}"
    )

    # Check that gating variables are within physiological bounds
    gating_vars = [
        "n",
        "m",
        "h",
    ]
    for gating_var in gating_vars:
        assert (result[gating_var] >= 0).all()
        assert (result[gating_var] <= 1).all()


def test_physiological_limits_and_action_potentials():
    """Voltage stays in physiological range; higher current generates more APs."""
    # Physiological limits for membrane voltage
    min_physiological_voltage = -100  # mV
    max_physiological_voltage = 70  # mV

    # Parameters for testing action potentials
    duration = 100  # ms, longer simulation to observe multiple APs
    currents = [20.0, 40.0, 60.0]  # increasing external currents in μA/cm²
    ap_threshold = 0  # mV, voltage threshold for counting action potentials

    # Bare Neuron() uses default HH52 squid giant axon kinetics.  Q10=1.0
    # disables temperature scaling so the kinetics match their original
    # parameterisation — this test exercises the HH52 model as published.
    import dataclasses

    custom_model = dataclasses.replace(make_squid_giant_axon(), Q10=1.0)
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1

    ap_counts = []

    for current_value in currents:
        # Create constant current array for each value
        current_array = np.full(num_steps, current_value)

        result = simulate_current_clamp(
            custom_model,
            current_external=current_array,
        )

        # Test that voltage stays within physiological limits
        assert result["voltage"].min() >= min_physiological_voltage, (
            f"Voltage below physiological minimum with current {current_value}"
        )
        assert result["voltage"].max() <= max_physiological_voltage, (
            f"Voltage exceeds physiological maximum with current {current_value}"
        )

        # Count action potentials (threshold crossings from below)
        voltage = result["voltage"]
        threshold_crossings = 0
        above_threshold = False

        for v in voltage:
            if v > ap_threshold and not above_threshold:
                threshold_crossings += 1
                above_threshold = True
            elif v < ap_threshold:
                above_threshold = False

        ap_counts.append(threshold_crossings)

    # Higher current should generally produce more action potentials
    # Allow for some variability in the relationship
    assert ap_counts[-1] >= ap_counts[0], (
        f"Expected more APs with higher current, got {ap_counts}"
    )


def test_simulate_current_clamp_with_different_currents():
    """Test the simulate_current_clamp method with a time-varying current waveform."""
    duration = 50  # ms
    dt = 1000.0 / SIM_SAMPLING_FREQ  # ms per step

    # Create a custom model for testing
    custom_model = make_squid_giant_axon()

    # Calculate the number of time steps
    num_time_steps = int(duration / dt) + 1

    # Create a simple current waveform
    # First half is 10 uA/cm², second half is 50 uA/cm²
    current_waveform = np.concatenate(
        [
            np.full(num_time_steps // 2, 10.0),
            np.full(num_time_steps - num_time_steps // 2, 50.0),
        ]
    )

    # Run the simulation with the time-varying current
    result = simulate_current_clamp(
        custom_model,
        current_external=current_waveform,
    )

    # Check basic properties of the result
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert len(result) == num_time_steps

    # Find the point where current changes
    midpoint_time = (num_time_steps // 2) * dt

    # Get voltages before and after current change
    # (give some time for the effect to manifest)
    before_time = midpoint_time - 5  # 5 ms before the change
    after_time = midpoint_time + 10  # 10 ms after the change

    voltage_before = result["voltage"][result["time"] <= before_time].mean()
    voltage_after = result["voltage"][result["time"] >= after_time].mean()

    # Voltage should be higher after the current increase
    assert voltage_after > voltage_before, (
        f"Expected higher voltage after current increase: "
        f"{voltage_before} -> {voltage_after}"
    )


def test_simulation_time_from_current_waveform():
    """Test that simulation time is derived from the current waveform length."""
    dt = 1000.0 / SIM_SAMPLING_FREQ  # ms per step
    custom_model = make_squid_giant_axon()

    # Create a current waveform of specific length
    duration = 75.0  # ms
    num_steps = int(duration / dt) + 1
    current_waveform = np.ones(num_steps) * 20.0  # constant current

    # Run simulation with only the current waveform, no simulation_time
    result = simulate_current_clamp(
        custom_model,
        current_external=current_waveform,
    )

    # Check that the simulation time matches what we expect from the current array
    # length
    expected_simulation_time = (len(current_waveform) - 1) * dt
    actual_simulation_time = result["time"][-1]

    assert actual_simulation_time == pytest.approx(expected_simulation_time)
    assert len(result) == num_steps


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_empty_current_array_raises(hh_model):
    """An empty current array must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        simulate_current_clamp(hh_model, current_external=np.array([]))


def test_nan_in_current_array_raises(hh_model):
    """A current array containing NaN must raise ValueError."""
    current = np.array([0.0, float("nan"), 0.0])
    with pytest.raises(ValueError, match="NaN"):
        simulate_current_clamp(hh_model, current_external=current)


def test_inf_in_current_array_raises(hh_model):
    """A current array containing Inf must raise ValueError."""
    current = np.array([0.0, float("inf"), 0.0])
    with pytest.raises(ValueError, match="Inf"):
        simulate_current_clamp(hh_model, current_external=current)


# ---------------------------------------------------------------------------
# Coverage-expansion tests (issue #18)
# ---------------------------------------------------------------------------


def test_single_element_current_protocol(hh_model):
    """A single-element current array must return a one-element structured array."""
    result = simulate_current_clamp(hh_model, current_external=np.array([0.0]))

    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert len(result) == 1
    assert result["voltage"][0] == pytest.approx(hh_model.v_rest)


# ---------------------------------------------------------------------------
# Numerical-stability / overflow-protection tests (issue #135)
# ---------------------------------------------------------------------------


def test_large_hyperpolarizing_current_does_not_crash(hh_model):
    """A large hyperpolarizing current must not raise OverflowError or crash.

    Regression test for GitHub issue #135: extreme negative stimulus values
    previously caused math.cosh to overflow inside the RK4 integrator.
    """
    duration = 20  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    current = np.full(num_steps, -200.0)  # large hyperpolarizing current

    result = simulate_current_clamp(hh_model, current_external=current)

    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert len(result) == num_steps
    assert np.isfinite(result["voltage"]).all()


def test_large_hyperpolarizing_current_voltage_stays_bounded(hh_model):
    """Voltage must stay within the numerical safety bounds under extreme input.

    Regression test for GitHub issue #135: extreme negative stimulus values
    previously caused a large positive voltage artifact in the trace.
    """
    duration = 20  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    current = np.full(num_steps, -200.0)  # large hyperpolarizing current

    result = simulate_current_clamp(hh_model, current_external=current)

    assert result["voltage"].min() >= -150.0
    assert result["voltage"].max() <= 150.0


# ---------------------------------------------------------------------------
# Preset stability tests (issue #197)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", _STABLE_PRESET_NAMES)
def test_all_presets_stable_at_rest(preset_name: str) -> None:
    """Every non-pacemaker preset stays within 1 mV of v_rest with zero stimulus.

    Regression test for issue #197: v_rest must equal the zero-current
    equilibrium so that neurons do not drift when unstimulated.  Also
    serves as a standard acceptance criterion for new neuron types.

    Purkinje is excluded because its v_rest is an unstable zero-current
    equilibrium (pacemaker threshold); see test_purkinje.py for pacemaking tests.
    """
    neuron = NEURON_PRESETS[preset_name]()

    duration = 50  # ms
    num_steps = int(duration / (1000.0 / SIM_SAMPLING_FREQ)) + 1
    zero_current = np.zeros(num_steps)

    result = simulate_current_clamp(neuron, current_external=zero_current)

    max_deviation = float(np.max(np.abs(result["voltage"] - neuron.v_rest)))
    assert max_deviation < 1.0, (
        f"{preset_name}: voltage deviated {max_deviation:.2f} mV from "
        f"v_rest={neuron.v_rest:.1f} mV — v_rest may not be the zero-current "
        "equilibrium (run find_zero_current_voltage to compute the correct value)"
    )

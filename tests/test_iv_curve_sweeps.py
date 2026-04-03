"""Tests for I-V curve multi-sweep simulation logic.

Verifies that running each voltage step as an independent simulation
(fresh initial conditions per step) produces the correct number of sweeps
and that each sweep has the expected single-step protocol shape.
"""

import numpy as np
import pytest

import patch_sim
import patch_sim.clamp_simulations
from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS = patch_sim.clamp_simulations.SIM_SAMPLING_FREQ


def _make_iv_protocols(
    voltage_min: float = -100.0,
    voltage_max: float = 60.0,
    voltage_step: float = 10.0,
    step_duration: float = 20.0,
    pre_pulse_duration: float = 5.0,
    post_pulse_duration: float = 5.0,
    holding_voltage: float = -70.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build I-V curve protocol arrays without running simulations.

    Args:
        voltage_min: Minimum test voltage in mV.
        voltage_max: Maximum test voltage in mV.
        voltage_step: Voltage increment between steps in mV.
        step_duration: Duration of the test pulse in milliseconds.
        pre_pulse_duration: Duration of the pre-pulse holding period in ms.
        post_pulse_duration: Duration of the post-pulse holding period in ms.
        holding_voltage: Holding voltage in mV.

    Returns:
        Tuple of (voltages array, list of protocol arrays), one protocol per voltage.
    """
    sweep_duration = pre_pulse_duration + step_duration + post_pulse_duration
    n_steps = round((voltage_max - voltage_min) / voltage_step) + 1
    voltages = np.linspace(voltage_min, voltage_max, n_steps)
    protocols = [
        patch_sim.step_voltage(
            duration=sweep_duration,
            voltage_amplitude=float(v),
            step_start=pre_pulse_duration,
            step_duration=step_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=_FS,
        )
        for v in voltages
    ]
    return voltages, protocols


def _make_iv_sweeps(
    voltage_min: float = -100.0,
    voltage_max: float = 60.0,
    voltage_step: float = 10.0,
    step_duration: float = 20.0,
    pre_pulse_duration: float = 5.0,
    post_pulse_duration: float = 5.0,
    holding_voltage: float = -70.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Run an I-V curve as independent sweeps and return (protocol, result) pairs.

    Args:
        voltage_min: Minimum test voltage in mV.
        voltage_max: Maximum test voltage in mV.
        voltage_step: Voltage increment between steps in mV.
        step_duration: Duration of the test pulse in milliseconds.
        pre_pulse_duration: Duration of the pre-pulse holding period in ms.
        post_pulse_duration: Duration of the post-pulse holding period in ms.
        holding_voltage: Holding voltage in mV.

    Returns:
        List of (protocol_array, result_array) tuples, one per voltage.
    """
    neuron = patch_sim.Neuron()
    sweep_duration = pre_pulse_duration + step_duration + post_pulse_duration
    n_steps = round((voltage_max - voltage_min) / voltage_step) + 1
    voltages = np.linspace(voltage_min, voltage_max, n_steps)
    results: list[tuple[np.ndarray, np.ndarray]] = []
    for voltage in voltages:
        protocol = patch_sim.step_voltage(
            duration=sweep_duration,
            voltage_amplitude=float(voltage),
            step_start=pre_pulse_duration,
            step_duration=step_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=_FS,
        )
        df = patch_sim.simulate_voltage_clamp(neuron, protocol)
        results.append((protocol, df))
    return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_iv_curve_produces_correct_sweep_count() -> None:
    """I-V curve sweeps match the number of voltage steps."""
    voltage_min = -100.0
    voltage_max = 60.0
    voltage_step = 10.0
    expected_count = round((voltage_max - voltage_min) / voltage_step) + 1
    sweeps = _make_iv_sweeps(
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        voltage_step=voltage_step,
    )
    assert len(sweeps) == expected_count


def test_iv_curve_each_sweep_has_single_step_protocol() -> None:
    """Each I-V sweep's protocol contains exactly one voltage step."""
    holding_voltage = -70.0
    step_duration = 20.0
    pre_pulse_duration = 5.0
    sweeps = _make_iv_sweeps(
        voltage_step=20.0,  # fewer sweeps for speed
        step_duration=step_duration,
        pre_pulse_duration=pre_pulse_duration,
        holding_voltage=holding_voltage,
    )
    for protocol, _df in sweeps:
        unique_voltages = np.unique(np.round(protocol, decimals=6))
        # Each protocol should contain at most two distinct voltage levels:
        # the holding voltage and the step voltage (or just one if they match).
        assert unique_voltages.size <= 2, (
            f"Expected at most 2 distinct voltages in a single-step protocol, "
            f"got {unique_voltages.size}: {unique_voltages}"
        )


def test_iv_curve_sweeps_are_independent() -> None:
    """Each I-V sweep starts from the same initial conditions.

    Confirmed by checking that the first time-point total current is the
    same across all sweeps (all start at holding voltage, not carried over
    from a previous step).
    """
    sweeps = _make_iv_sweeps(voltage_step=20.0)  # fewer sweeps for speed
    first_currents = [df["Itotal"][0] for _proto, df in sweeps]
    # All sweeps start at holding voltage → first current values should be
    # nearly identical (same initial state).
    assert max(first_currents) - min(first_currents) == pytest.approx(0.0, abs=1e-6)


def test_iv_curve_protocol_peak_voltage_matches_step_voltage() -> None:
    """The peak voltage in each sweep's protocol equals the commanded step voltage."""
    voltage_min = -80.0
    voltage_max = 40.0
    voltage_step = 20.0
    holding_voltage = -70.0
    n_steps = round((voltage_max - voltage_min) / voltage_step) + 1
    expected_voltages = np.linspace(voltage_min, voltage_max, n_steps)
    sweeps = _make_iv_sweeps(
        voltage_min=voltage_min,
        voltage_max=voltage_max,
        voltage_step=voltage_step,
        holding_voltage=holding_voltage,
    )
    for (protocol, _df), expected_v in zip(sweeps, expected_voltages):
        if expected_v >= holding_voltage:
            peak = float(np.max(protocol))
        else:
            peak = float(np.min(protocol))
        assert peak == pytest.approx(float(expected_v), abs=1e-6)


# ---------------------------------------------------------------------------
# simulate_batch tests
# ---------------------------------------------------------------------------


def test_batch_matches_sequential() -> None:
    """simulate_batch results are identical to sequential simulate_voltage_clamp."""
    neuron = patch_sim.Neuron()
    _voltages, protocols = _make_iv_protocols(
        voltage_min=-60.0,
        voltage_max=0.0,
        voltage_step=20.0,
    )
    sequential = [simulate_voltage_clamp(neuron, p) for p in protocols]
    batch = list(patch_sim.simulate_batch(neuron, protocols, max_workers=2))
    assert len(batch) == len(sequential)
    for seq_df, batch_df in zip(sequential, batch):
        assert seq_df.dtype == batch_df.dtype
        assert seq_df.dtype.names is not None
        for field in seq_df.dtype.names:
            np.testing.assert_array_equal(seq_df[field], batch_df[field])


def test_batch_single_sweep() -> None:
    """A single-protocol batch yields exactly one correct DataFrame."""
    neuron = patch_sim.Neuron()
    _voltages, protocols = _make_iv_protocols(
        voltage_min=0.0,
        voltage_max=0.0,
        voltage_step=10.0,
    )
    results = list(patch_sim.simulate_batch(neuron, protocols, max_workers=2))
    assert len(results) == 1
    expected = simulate_voltage_clamp(neuron, protocols[0])
    assert results[0].dtype == expected.dtype
    assert expected.dtype.names is not None
    for field in expected.dtype.names:
        np.testing.assert_array_equal(results[0][field], expected[field])


def test_batch_empty_protocols() -> None:
    """An empty protocol list yields no results."""
    neuron = patch_sim.Neuron()
    results = list(patch_sim.simulate_batch(neuron, [], max_workers=2))
    assert results == []


def test_batch_preserves_order() -> None:
    """Results come back in voltage order (same order as input protocols)."""
    neuron = patch_sim.Neuron()
    voltages, protocols = _make_iv_protocols(
        voltage_min=-80.0,
        voltage_max=40.0,
        voltage_step=20.0,
    )
    results = list(patch_sim.simulate_batch(neuron, protocols, max_workers=2))
    assert len(results) == len(voltages)
    for df, voltage in zip(results, voltages):
        # Each protocol steps to its commanded voltage — check the peak matches.
        holding = -70.0
        if voltage >= holding:
            peak = float(np.max(df["voltage"]))
        else:
            peak = float(np.min(df["voltage"]))
        assert peak == pytest.approx(float(voltage), abs=1e-6)


def test_batch_with_calcium_channels() -> None:
    """simulate_batch works with channels using boltzmann_cosh_rates (issue #105).

    ICaL uses rate functions produced by boltzmann_cosh_rates. Before the fix,
    these were unpicklable closures that caused simulate_batch (which relies on
    ProcessPoolExecutor) to raise AttributeError. This test runs an IV curve
    simulation with ICaL to verify the fix.
    """
    ical = patch_sim.make_ical_channel()
    cd = patch_sim.CalciumDynamics()
    neuron = patch_sim.Neuron(
        additional_channels=(ical,),
        calcium_dynamics=cd,
    )
    voltages = [-40.0, 0.0, 40.0]
    protocols = [
        patch_sim.step_voltage(
            duration=30.0,
            voltage_amplitude=float(v),
            step_start=5.0,
            step_duration=20.0,
            holding_voltage=-70.0,
            sampling_frequency=_FS,
        )
        for v in voltages
    ]

    results = list(patch_sim.simulate_batch(neuron, protocols, max_workers=2))

    assert len(results) == len(voltages)
    for df in results:
        assert df.dtype.names is not None
        assert "ICaL" in df.dtype.names
        assert "ca_i" in df.dtype.names
        assert "d" in df.dtype.names
        assert "f" in df.dtype.names
    # Calcium should accumulate above resting level during the depolarising
    # step (ICaL activates with depolarisation).
    ca_rest = cd.ca_rest
    assert any(float(df["ca_i"].max()) > ca_rest for df in results)


def test_batch_with_current_clamp() -> None:
    """simulate_batch with simulate_fn=simulate_current_clamp works correctly."""
    neuron = patch_sim.Neuron()
    num_steps = int(_FS * 0.05)  # 50 ms
    stimulus = np.zeros(num_steps)
    stimulus[int(_FS * 0.01) : int(_FS * 0.04)] = 10.0

    results = list(
        patch_sim.simulate_batch(
            neuron,
            [stimulus],
            simulate_fn=simulate_current_clamp,
            max_workers=2,
        )
    )
    assert len(results) == 1
    expected = simulate_current_clamp(neuron, stimulus)
    assert results[0].dtype == expected.dtype
    assert expected.dtype.names is not None
    for field in expected.dtype.names:
        np.testing.assert_array_equal(results[0][field], expected[field])

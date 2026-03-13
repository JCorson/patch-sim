"""Tests for I-V curve multi-sweep simulation logic.

Verifies that running each voltage step as an independent simulation
(fresh initial conditions per step) produces the correct number of sweeps
and that each sweep has the expected single-step protocol shape.
"""

import numpy as np
import pandas as pd
import pytest

import ap_sim
import ap_sim.clamp_simulations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FS = ap_sim.clamp_simulations.SIM_SAMPLING_FREQ


def _make_iv_sweeps(
    voltage_min: float = -100.0,
    voltage_max: float = 60.0,
    voltage_step: float = 10.0,
    step_duration: float = 20.0,
    pre_pulse_duration: float = 5.0,
    post_pulse_duration: float = 5.0,
    holding_voltage: float = -70.0,
) -> list[tuple[np.ndarray, pd.DataFrame]]:
    """Run an I-V curve as independent sweeps and return (protocol, df) pairs.

    Args:
        voltage_min: Minimum test voltage in mV.
        voltage_max: Maximum test voltage in mV.
        voltage_step: Voltage increment between steps in mV.
        step_duration: Duration of the test pulse in milliseconds.
        pre_pulse_duration: Duration of the pre-pulse holding period in ms.
        post_pulse_duration: Duration of the post-pulse holding period in ms.
        holding_voltage: Holding voltage in mV.

    Returns:
        List of (protocol_array, result_dataframe) tuples, one per voltage.
    """
    neuron = ap_sim.HodgkinHuxley()
    sweep_duration = pre_pulse_duration + step_duration + post_pulse_duration
    voltages = np.arange(voltage_min, voltage_max + voltage_step, voltage_step)
    results: list[tuple[np.ndarray, pd.DataFrame]] = []
    for voltage in voltages:
        protocol = ap_sim.step_voltage(
            duration=sweep_duration,
            voltage_amplitude=float(voltage),
            step_start=pre_pulse_duration,
            step_duration=step_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=_FS,
        )
        df = ap_sim.simulate_voltage_clamp(neuron, protocol)
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
    expected_count = len(
        np.arange(voltage_min, voltage_max + voltage_step, voltage_step)
    )
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
    first_currents = [df["total_current"].iloc[0] for _proto, df in sweeps]
    # All sweeps start at holding voltage → first current values should be
    # nearly identical (same initial state).
    assert max(first_currents) - min(first_currents) == pytest.approx(0.0, abs=1e-6)


def test_iv_curve_protocol_peak_voltage_matches_step_voltage() -> None:
    """The peak voltage in each sweep's protocol equals the commanded step voltage."""
    voltage_min = -80.0
    voltage_max = 40.0
    voltage_step = 20.0
    holding_voltage = -70.0
    expected_voltages = np.arange(voltage_min, voltage_max + voltage_step, voltage_step)
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

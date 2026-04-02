"""Integration tests for the protocol → simulation pipeline.

Each test generates a stimulus protocol via patch_sim.protocols and feeds it
directly into simulate_current_clamp() or simulate_voltage_clamp(), exercising
the full end-to-end workflow rather than either component in isolation.
"""

import numpy as np

from patch_sim.clamp_simulations import simulate_current_clamp, simulate_voltage_clamp
from patch_sim.hodgkin_huxley import HodgkinHuxley
from patch_sim.protocols import (
    chirp_current,
    noise_current,
    pulse_train,
    pulse_train_voltage,
    ramp_current,
    ramp_voltage,
    sinusoidal_current,
    step_current,
    step_voltage,
)

# ---------------------------------------------------------------------------
# Shared constants and helpers
# ---------------------------------------------------------------------------

CURRENT_CLAMP_COLUMNS = [
    "time",
    "voltage",
    "INa",
    "IK",
    "Ileak",
    "Itotal",
    "n",
    "m",
    "h",
]

VOLTAGE_CLAMP_COLUMNS = [
    "time",
    "voltage",
    "Itotal",
    "INa",
    "IK",
    "Ileak",
    "n",
    "m",
    "h",
]


def _assert_current_clamp_result(result: np.ndarray) -> None:
    """Shared structural assertions for simulate_current_clamp results."""
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert "time" in result.dtype.names
    for col in CURRENT_CLAMP_COLUMNS:
        assert col in result.dtype.names
    assert all(np.isfinite(result[f]).all() for f in result.dtype.names)
    assert result["voltage"].min() >= -100.0
    assert result["voltage"].max() <= 60.0
    for gate in ("n", "m", "h"):
        assert (result[gate] >= 0.0).all()
        assert (result[gate] <= 1.0).all()


def _assert_voltage_clamp_result(result: np.ndarray) -> None:
    """Shared structural assertions for simulate_voltage_clamp results."""
    assert isinstance(result, np.ndarray)
    assert result.dtype.names is not None
    assert "time" in result.dtype.names
    for col in VOLTAGE_CLAMP_COLUMNS:
        assert col in result.dtype.names
    assert all(np.isfinite(result[f]).all() for f in result.dtype.names)
    for gate in ("n", "m", "h"):
        assert (result[gate] >= 0.0).all()
        assert (result[gate] <= 1.0).all()


def _count_action_potentials(voltage: np.ndarray, threshold: float = 0.0) -> int:
    """Count upward threshold crossings in a voltage trace."""
    count = 0
    above = False
    for v in voltage:
        if v > threshold and not above:
            count += 1
            above = True
        elif v < threshold:
            above = False
    return count


# ---------------------------------------------------------------------------
# Section 1 — Current protocol → simulate_current_clamp
# ---------------------------------------------------------------------------


def test_step_current_to_simulation(hh_model: HodgkinHuxley) -> None:
    """step_current → simulate_current_clamp: structural checks and AP generation."""
    protocol = step_current(duration=50.0, current_amplitude=15.0)
    result = simulate_current_clamp(hh_model, current_external=protocol)

    _assert_current_clamp_result(result)
    assert len(result) == len(protocol)

    # A 15 µA/cm² step is suprathreshold — at least one AP expected
    assert result["voltage"].max() > 0.0


def test_ramp_current_to_simulation(hh_model: HodgkinHuxley) -> None:
    """ramp_current → simulate_current_clamp: structural checks."""
    protocol = ramp_current(duration=50.0, start_current=0.0, end_current=30.0)
    result = simulate_current_clamp(hh_model, current_external=protocol)

    _assert_current_clamp_result(result)
    assert len(result) == len(protocol)

    # Voltage should depart from resting during the ramp
    assert result["voltage"].max() - result["voltage"].min() > 5.0


def test_pulse_train_current_to_simulation(hh_model: HodgkinHuxley) -> None:
    """pulse_train → simulate_current_clamp: multiple APs expected."""
    protocol = pulse_train(
        duration=100.0,
        pulse_amplitude=20.0,
        pulse_width=2.0,
        pulse_interval=10.0,
    )
    result = simulate_current_clamp(hh_model, current_external=protocol)

    _assert_current_clamp_result(result)
    assert len(result) == len(protocol)

    # Multiple suprathreshold pulses should evoke multiple APs
    ap_count = _count_action_potentials(result["voltage"])
    assert ap_count >= 2


def test_sinusoidal_current_to_simulation(hh_model: HodgkinHuxley) -> None:
    """sinusoidal_current → simulate_current_clamp: structural checks."""
    protocol = sinusoidal_current(
        duration=50.0,
        dc_offset=15.0,
        amplitude=10.0,
        frequency=50.0,
    )
    result = simulate_current_clamp(hh_model, current_external=protocol)

    _assert_current_clamp_result(result)
    assert len(result) == len(protocol)

    # DC offset ensures net depolarizing drive; voltage should shift from resting
    initial_voltage = result["voltage"][0]
    assert result["voltage"].max() - initial_voltage > 3.0


def test_noise_current_to_simulation() -> None:
    """noise_current → simulate_current_clamp: reproducibility with fixed seed."""
    protocol_a = noise_current(
        duration=50.0, mean_current=10.0, std_current=2.0, seed=42
    )
    protocol_b = noise_current(
        duration=50.0, mean_current=10.0, std_current=2.0, seed=42
    )
    protocol_c = noise_current(
        duration=50.0, mean_current=10.0, std_current=2.0, seed=99
    )

    # Use separate model instances so tests are not order-dependent
    result_a = simulate_current_clamp(HodgkinHuxley(), current_external=protocol_a)
    result_b = simulate_current_clamp(HodgkinHuxley(), current_external=protocol_b)
    result_c = simulate_current_clamp(HodgkinHuxley(), current_external=protocol_c)

    _assert_current_clamp_result(result_a)

    # Same seed → identical results
    assert result_a.dtype.names is not None
    for field in result_a.dtype.names:
        np.testing.assert_array_equal(result_a[field], result_b[field])

    # Different seed → different voltage trace
    assert not np.array_equal(result_a["voltage"], result_c["voltage"])


def test_chirp_current_to_simulation(hh_model: HodgkinHuxley) -> None:
    """chirp_current → simulate_current_clamp: structural checks with DC offset."""
    protocol = chirp_current(
        duration=50.0,
        dc_offset=15.0,
        amplitude=10.0,
        start_frequency=1.0,
        end_frequency=100.0,
    )
    result = simulate_current_clamp(hh_model, current_external=protocol)

    _assert_current_clamp_result(result)
    assert len(result) == len(protocol)

    # DC offset provides a sustained depolarising drive; voltage should depart
    # noticeably from resting potential
    assert result["voltage"].max() - result["voltage"].min() > 5.0


# ---------------------------------------------------------------------------
# Section 2 — Voltage protocol → simulate_voltage_clamp
# ---------------------------------------------------------------------------


def test_step_voltage_to_simulation(hh_model: HodgkinHuxley) -> None:
    """step_voltage → simulate_voltage_clamp: Na⁺ inward current during step."""
    protocol = step_voltage(
        duration=20.0,
        voltage_amplitude=0.0,
        step_start=5.0,
        step_duration=10.0,
    )
    result = simulate_voltage_clamp(hh_model, voltage_protocol=protocol)

    _assert_voltage_clamp_result(result)
    assert len(result) == len(protocol)

    # During the depolarising step there should be inward Na⁺ current
    step_mask = np.isclose(result["voltage"], 0.0)
    assert step_mask.any(), "No rows matched the step voltage — mask is empty"
    assert result["INa"][step_mask].min() < -10.0


def test_ramp_voltage_to_simulation(hh_model: HodgkinHuxley) -> None:
    """ramp_voltage → simulate_voltage_clamp: K⁺ current higher in second half."""
    protocol = ramp_voltage(
        duration=20.0,
        start_voltage=-80.0,
        end_voltage=40.0,
    )
    result = simulate_voltage_clamp(hh_model, voltage_protocol=protocol)

    _assert_voltage_clamp_result(result)
    assert len(result) == len(protocol)

    # Voltage in result should rise monotonically (mirrors the ramp)
    assert np.isclose(result["voltage"][0], -80.0)
    assert np.isclose(result["voltage"][-1], 40.0)

    # K⁺ current (outward) should be higher in second half of ramp
    n = len(result)
    k_first_half = result["IK"][: n // 2].mean()
    k_second_half = result["IK"][n // 2 :].mean()
    assert k_second_half > k_first_half


def test_pulse_train_voltage_to_simulation(hh_model: HodgkinHuxley) -> None:
    """pulse_train_voltage → simulate_voltage_clamp: repeated Na⁺ transients."""
    protocol = pulse_train_voltage(
        duration=50.0,
        pulse_amplitude=0.0,
        pulse_width=2.0,
        pulse_interval=10.0,
        holding_voltage=-80.0,
    )
    result = simulate_voltage_clamp(hh_model, voltage_protocol=protocol)

    _assert_voltage_clamp_result(result)
    assert len(result) == len(protocol)

    # Each depolarising pulse should produce an inward Na⁺ transient
    # Verify at least one strong inward Na⁺ event occurred
    assert result["INa"].min() < -10.0


# ---------------------------------------------------------------------------
# Section 3 — Physiological plausibility
# ---------------------------------------------------------------------------


def test_action_potential_with_step_current(hh_model: HodgkinHuxley) -> None:
    """Suprathreshold step current produces a full action potential."""
    protocol = step_current(duration=50.0, current_amplitude=15.0)
    result = simulate_current_clamp(hh_model, current_external=protocol)

    voltage: np.ndarray = result["voltage"]

    # AP peak must exceed +20 mV
    assert float(voltage.max()) > 20.0

    # Repolarisation: voltage must return below -50 mV after the peak
    peak_idx = int(np.argmax(voltage))
    post_peak_voltage = voltage[peak_idx:]
    assert float(post_peak_voltage.min()) < -50.0


def test_no_ap_below_threshold_current(hh_model: HodgkinHuxley) -> None:
    """Sub-threshold step current should not trigger an action potential."""
    protocol = step_current(duration=50.0, current_amplitude=5.0)
    result = simulate_current_clamp(hh_model, current_external=protocol)

    # Voltage must never cross 0 mV (AP threshold)
    assert float(result["voltage"].max()) < 0.0

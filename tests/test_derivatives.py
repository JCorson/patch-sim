"""Tests for patch_sim.analysis.derivatives."""

import numpy as np

from patch_sim.analysis.derivatives import compute_dvdt


def test_compute_dvdt_constant_voltage_returns_zero_dvdt() -> None:
    """Constant voltage produces a zero dV/dt everywhere."""
    time = np.array([0.0, 1.0, 2.0, 3.0])
    voltage = np.array([-65.0, -65.0, -65.0, -65.0])
    _, dvdt = compute_dvdt(time, voltage)
    assert len(dvdt) == len(time)
    np.testing.assert_allclose(dvdt, 0.0, atol=1e-12)


def test_compute_dvdt_linear_ramp_returns_constant_dvdt() -> None:
    """Linearly increasing voltage at 1 mV/ms gives dV/dt = 1 everywhere."""
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    voltage = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    _, dvdt = compute_dvdt(time, voltage)
    np.testing.assert_allclose(dvdt, 1.0, atol=1e-12)


def test_compute_dvdt_output_length_matches_input() -> None:
    """Output arrays have the same length as the input arrays."""
    time = np.linspace(0.0, 10.0, 100)
    voltage = np.sin(time)
    v_out, dvdt = compute_dvdt(time, voltage)
    assert len(v_out) == len(time)
    assert len(dvdt) == len(time)


def test_compute_dvdt_returns_voltage_unchanged() -> None:
    """The first returned array is the voltage passed in, unchanged."""
    time = np.array([0.0, 1.0, 2.0])
    voltage = np.array([-70.0, -65.0, -60.0])
    v_out, _ = compute_dvdt(time, voltage)
    np.testing.assert_array_equal(v_out, voltage)


def test_compute_dvdt_single_point_returns_voltage_and_empty_dvdt() -> None:
    """A single-point input returns the voltage and an empty dvdt array."""
    time = np.array([0.0])
    voltage = np.array([-65.0])
    v_out, dvdt = compute_dvdt(time, voltage)
    np.testing.assert_array_equal(v_out, voltage)
    assert len(dvdt) == 0


def test_compute_dvdt_two_points_returns_same_length() -> None:
    """A two-point input is the minimum for a derivative; output lengths match."""
    time = np.array([0.0, 1.0])
    voltage = np.array([0.0, 2.0])
    v_out, dvdt = compute_dvdt(time, voltage)
    assert len(v_out) == 2  # noqa: PLR2004
    assert len(dvdt) == 2  # noqa: PLR2004
    # With only two points np.gradient uses forward/backward difference.
    np.testing.assert_allclose(dvdt, 2.0, atol=1e-12)


def test_compute_dvdt_empty_input_returns_empty() -> None:
    """Empty input arrays return an empty voltage and an empty dvdt array."""
    v_out, dvdt = compute_dvdt(np.array([]), np.array([]))
    assert len(v_out) == 0
    assert len(dvdt) == 0


def test_compute_dvdt_mismatched_lengths_raise() -> None:
    """Mismatched time and voltage lengths raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="same length"):
        compute_dvdt(np.array([0.0, 1.0, 2.0]), np.array([-65.0, -64.0]))


def test_compute_dvdt_nonuniform_time_steps() -> None:
    """Non-uniform time steps are handled correctly by np.gradient."""
    # voltage = 2 * time => dV/dt = 2 everywhere
    time = np.array([0.0, 0.5, 1.5, 3.0])
    voltage = 2.0 * time
    _, dvdt = compute_dvdt(time, voltage)
    np.testing.assert_allclose(dvdt, 2.0, atol=1e-12)


def test_compute_dvdt_accepts_lists() -> None:
    """Plain Python lists are accepted as input without error."""
    _, dvdt = compute_dvdt([0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert len(dvdt) == 3  # noqa: PLR2004
    np.testing.assert_allclose(dvdt, 0.0, atol=1e-12)


def test_compute_dvdt_large_array_performance() -> None:
    """compute_dvdt runs without error on a large array (10 000 points)."""
    time = np.linspace(0.0, 100.0, 10_000)
    voltage = np.sin(time)
    v_out, dvdt = compute_dvdt(time, voltage)
    assert len(dvdt) == 10_000
    # Verify the approximation is reasonable: d/dt sin(t) ≈ cos(t)
    np.testing.assert_allclose(dvdt[1:-1], np.cos(time)[1:-1], atol=1e-3)

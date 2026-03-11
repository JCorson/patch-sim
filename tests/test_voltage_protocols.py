"""Tests for voltage clamp protocols.

This module contains unit tests for all voltage clamp protocol functions.
"""

import numpy as np
import pytest

from ap_sim.protocols import (
    activation_protocol,
    iv_curve_protocol,
    pulse_train_voltage,
    ramp_voltage,
    step_voltage,
)


class TestStepVoltage:
    """Test cases for step_voltage function."""

    def test_basic_step_voltage(self):
        """Test basic step voltage generation."""
        duration = 10.0  # ms
        voltage_amplitude = -30.0  # mV
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = step_voltage(
            duration=duration,
            voltage_amplitude=voltage_amplitude,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Check that all values are equal to the amplitude
        assert np.allclose(voltage, voltage_amplitude)

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(voltage) == expected_length

    def test_step_with_delay_and_holding(self):
        """Test step voltage with delayed start and holding voltage."""
        duration = 10.0  # ms
        voltage_amplitude = 0.0  # mV
        step_start = 2.0  # ms
        step_duration = 5.0  # ms
        holding_voltage = -80.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = step_voltage(
            duration=duration,
            voltage_amplitude=voltage_amplitude,
            step_start=step_start,
            step_duration=step_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Create time array for verification
        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        # Check holding voltage before step
        pre_step_mask = time_array < step_start
        assert np.allclose(voltage[pre_step_mask], holding_voltage)

        # Check step voltage during step
        step_mask = (time_array >= step_start) & (
            time_array <= step_start + step_duration
        )
        assert np.allclose(voltage[step_mask], voltage_amplitude)

        # Check holding voltage after step
        post_step_mask = time_array > step_start + step_duration
        assert np.allclose(voltage[post_step_mask], holding_voltage)


class TestRampVoltage:
    """Test cases for ramp_voltage function."""

    def test_basic_ramp_voltage(self):
        """Test basic ramp voltage generation."""
        duration = 10.0  # ms
        start_voltage = -80.0  # mV
        end_voltage = 40.0  # mV
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = ramp_voltage(
            duration=duration,
            start_voltage=start_voltage,
            end_voltage=end_voltage,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Check that voltage starts at start_voltage and ends at end_voltage
        assert np.isclose(voltage[0], start_voltage, atol=1e-10)
        assert np.isclose(voltage[-1], end_voltage, atol=1e-10)

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(voltage) == expected_length

    def test_ramp_with_delay(self):
        """Test ramp voltage with delayed start."""
        duration = 20.0  # ms
        start_voltage = -100.0  # mV
        end_voltage = 60.0  # mV
        ramp_start = 5.0  # ms
        ramp_duration = 10.0  # ms
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = ramp_voltage(
            duration=duration,
            start_voltage=start_voltage,
            end_voltage=end_voltage,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Create time array for verification
        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        # Check holding voltage before ramp
        pre_ramp_mask = time_array < ramp_start
        assert np.allclose(voltage[pre_ramp_mask], holding_voltage)

        # Check ramp progression
        ramp_mask = (time_array >= ramp_start) & (
            time_array <= ramp_start + ramp_duration
        )
        ramp_voltages = voltage[ramp_mask]
        # end_voltage > start_voltage, so ramp should be monotonically increasing
        assert np.all(np.diff(ramp_voltages) >= -1e-10)

        # Check holding voltage after ramp
        post_ramp_mask = time_array > ramp_start + ramp_duration
        assert np.allclose(voltage[post_ramp_mask], holding_voltage)

    def test_ramp_decreasing(self):
        """Test ramp voltage with decreasing values."""
        duration = 20.0  # ms
        start_voltage = 60.0  # mV
        end_voltage = -100.0  # mV
        ramp_start = 5.0  # ms
        ramp_duration = 10.0  # ms
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = ramp_voltage(
            duration=duration,
            start_voltage=start_voltage,
            end_voltage=end_voltage,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        ramp_mask = (time_array >= ramp_start) & (
            time_array <= ramp_start + ramp_duration
        )
        ramp_voltages = voltage[ramp_mask]
        # end_voltage < start_voltage, so ramp should be monotonically decreasing
        assert np.all(np.diff(ramp_voltages) <= 1e-10)


class TestPulseTrainVoltage:
    """Test cases for pulse_train_voltage function."""

    def test_basic_pulse_train_voltage(self):
        """Test basic pulse train voltage generation."""
        duration = 20.0  # ms
        pulse_amplitude = 0.0  # mV
        pulse_width = 2.0  # ms
        pulse_interval = 5.0  # ms
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = pulse_train_voltage(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(voltage) == expected_length

        # Check that values are either holding voltage or pulse amplitude
        unique_values = np.unique(voltage)
        expected_values = {holding_voltage, pulse_amplitude}
        for val in unique_values:
            assert any(
                np.isclose(val, exp_val, atol=1e-10) for exp_val in expected_values
            )

    def test_pulse_timing(self):
        """Test pulse train timing."""
        duration = 15.0  # ms
        pulse_amplitude = 20.0  # mV
        pulse_width = 1.0  # ms
        pulse_interval = 4.0  # ms
        train_start = 2.0  # ms
        holding_voltage = -80.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = pulse_train_voltage(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=train_start,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Create time array for verification
        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        # Check that pulses occur at expected times
        expected_pulse_starts = [train_start + i * pulse_interval for i in range(4)]

        for pulse_start in expected_pulse_starts:
            if pulse_start < duration:
                pulse_mask = (time_array >= pulse_start) & (
                    time_array <= pulse_start + pulse_width
                )
                if np.any(pulse_mask):
                    assert np.allclose(voltage[pulse_mask], pulse_amplitude)


class TestIVCurveProtocol:
    """Test cases for iv_curve_protocol function."""

    def test_basic_iv_curve(self):
        """Test basic I-V curve protocol generation."""
        step_duration = 5.0  # ms
        voltage_min = -80.0  # mV
        voltage_max = 40.0  # mV
        voltage_step = 20.0  # mV
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = iv_curve_protocol(
            step_duration=step_duration,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            voltage_step=voltage_step,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Check that protocol contains expected voltage levels
        expected_voltages = np.arange(
            voltage_min, voltage_max + voltage_step, voltage_step
        )
        unique_voltages = np.unique(voltage)

        # Should contain holding voltage and all test voltages
        expected_unique = set([holding_voltage] + list(expected_voltages))
        actual_unique = set()
        for val in unique_voltages:
            for exp_val in expected_unique:
                if np.isclose(val, exp_val, atol=1e-10):
                    actual_unique.add(exp_val)
                    break

        assert len(actual_unique) >= len(expected_voltages)

    def test_iv_curve_timing(self):
        """Test I-V curve protocol timing."""
        step_duration = 3.0  # ms
        voltage_min = -60.0  # mV
        voltage_max = 0.0  # mV
        voltage_step = 30.0  # mV  # Only 3 steps: -60, -30, 0
        pre_pulse_duration = 2.0  # ms
        post_pulse_duration = 1.0  # ms
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = iv_curve_protocol(
            step_duration=step_duration,
            voltage_min=voltage_min,
            voltage_max=voltage_max,
            voltage_step=voltage_step,
            pre_pulse_duration=pre_pulse_duration,
            post_pulse_duration=post_pulse_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Calculate expected duration
        expected_voltages = np.arange(
            voltage_min, voltage_max + voltage_step, voltage_step
        )
        sweep_duration = pre_pulse_duration + step_duration + post_pulse_duration
        expected_total_duration = sweep_duration * len(expected_voltages)

        # Check total duration
        time_step = 1000.0 / sampling_frequency
        actual_duration = (len(voltage) - 1) * time_step
        assert np.isclose(actual_duration, expected_total_duration, atol=time_step)


class TestActivationProtocol:
    """Test cases for activation_protocol function."""

    def test_basic_activation_protocol(self):
        """Test basic activation protocol generation."""
        test_duration = 10.0  # ms
        prepulse_voltage = -120.0  # mV
        prepulse_duration = 100.0  # ms
        test_voltage_min = -80.0  # mV
        test_voltage_max = 40.0  # mV
        voltage_step = 40.0  # mV  # Only 4 steps: -80, -40, 0, 40
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = activation_protocol(
            test_duration=test_duration,
            prepulse_voltage=prepulse_voltage,
            prepulse_duration=prepulse_duration,
            test_voltage_min=test_voltage_min,
            test_voltage_max=test_voltage_max,
            voltage_step=voltage_step,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Check that protocol contains expected voltage levels
        expected_test_voltages = np.arange(
            test_voltage_min, test_voltage_max + voltage_step, voltage_step
        )
        unique_voltages = np.unique(voltage)

        # Should contain holding voltage, prepulse voltage, and all test
        # voltages
        expected_unique = set(
            [holding_voltage, prepulse_voltage] + list(expected_test_voltages)
        )
        actual_unique = set()
        for val in unique_voltages:
            for exp_val in expected_unique:
                if np.isclose(val, exp_val, atol=1e-10):
                    actual_unique.add(exp_val)
                    break

        # +1 for prepulse
        assert len(actual_unique) >= len(expected_test_voltages) + 1

    def test_activation_protocol_timing(self):
        """Test activation protocol timing structure."""
        test_duration = 5.0  # ms
        prepulse_voltage = -100.0  # mV
        prepulse_duration = 50.0  # ms
        test_voltage_min = -60.0  # mV
        test_voltage_max = 0.0  # mV
        voltage_step = 30.0  # mV  # Only 3 steps: -60, -30, 0
        interpulse_duration = 5.0  # ms
        holding_voltage = -70.0  # mV
        sampling_frequency = 10000.0  # Hz

        voltage = activation_protocol(
            test_duration=test_duration,
            prepulse_voltage=prepulse_voltage,
            prepulse_duration=prepulse_duration,
            test_voltage_min=test_voltage_min,
            test_voltage_max=test_voltage_max,
            voltage_step=voltage_step,
            interpulse_duration=interpulse_duration,
            holding_voltage=holding_voltage,
            sampling_frequency=sampling_frequency,
        )

        # Calculate expected duration
        expected_test_voltages = np.arange(
            test_voltage_min, test_voltage_max + voltage_step, voltage_step
        )
        sweep_duration = (
            prepulse_duration
            + interpulse_duration
            + test_duration
            + interpulse_duration
        )
        expected_total_duration = sweep_duration * len(expected_test_voltages)

        # Check total duration
        time_step = 1000.0 / sampling_frequency
        actual_duration = (len(voltage) - 1) * time_step
        assert np.isclose(actual_duration, expected_total_duration, atol=time_step)


class TestVoltageProtocolIntegration:
    """Integration tests for voltage clamp protocols."""

    def test_all_voltage_protocols_generate_arrays(self):
        """Test that all voltage protocols generate valid arrays."""
        duration = 5.0  # ms
        sampling_frequency = 10000.0  # Hz

        protocols = [
            lambda: step_voltage(
                duration, -30.0, sampling_frequency=sampling_frequency
            ),
            lambda: ramp_voltage(
                duration, -80.0, 40.0, sampling_frequency=sampling_frequency
            ),
            lambda: pulse_train_voltage(
                duration, 0.0, 1.0, 3.0, sampling_frequency=sampling_frequency
            ),
        ]

        for protocol_func in protocols:
            voltage = protocol_func()

            # Check that protocol generates a valid array
            assert isinstance(voltage, np.ndarray)
            assert len(voltage) > 0
            assert np.all(np.isfinite(voltage))

            # Check array length
            expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
            assert len(voltage) == expected_length

    def test_voltage_protocols_with_different_frequencies(self):
        """Test voltage protocols with different sampling frequencies."""
        duration = 3.0  # ms
        frequencies = [1000.0, 10000.0, 100000.0]  # 1 kHz, 10 kHz, 100 kHz

        for freq in frequencies:
            voltage = step_voltage(duration, -20.0, sampling_frequency=freq)
            expected_length = int(duration / (1000.0 / freq)) + 1
            assert len(voltage) == expected_length

    def test_voltage_protocol_edge_cases(self):
        """Test edge cases for voltage protocol functions."""
        # Very short duration
        short_duration = 0.1  # ms
        voltage = step_voltage(short_duration, -30.0, sampling_frequency=10000.0)
        assert len(voltage) >= 1

        # Very small voltage difference
        small_diff = 1e-6  # mV
        voltage = ramp_voltage(
            1.0, -70.0, -70.0 + small_diff, sampling_frequency=10000.0
        )
        assert np.all(np.isfinite(voltage))

        # Test that voltage range is respected
        voltage_range = np.max(voltage) - np.min(voltage)
        assert voltage_range >= 0  # Should not be negative


# =============================================================================
# Error-path tests (voltage clamp)
# =============================================================================


class TestVoltageProtocolValidation:
    """Tests that voltage clamp protocol functions raise ValueError on invalid inputs."""  # noqa: E501

    # --- duration and sampling_frequency (shared path via _calculate_time_parameters)

    def test_non_positive_duration_voltage_protocol(self):
        """Duration <= 0 must raise ValueError for voltage protocols."""
        with pytest.raises(ValueError, match="duration"):
            step_voltage(0.0, -30.0)

    def test_non_positive_sampling_frequency_voltage_protocol(self):
        """sampling_frequency <= 0 must raise ValueError for voltage protocols."""
        with pytest.raises(ValueError, match="sampling_frequency"):
            step_voltage(10.0, -30.0, sampling_frequency=0.0)

    # --- ramp_duration == 0

    def test_zero_ramp_duration_raises_voltage(self):
        """ramp_duration=0 causes division by zero — must raise ValueError."""
        with pytest.raises(ValueError, match="ramp_duration"):
            ramp_voltage(10.0, -80.0, 40.0, ramp_duration=0.0)

    # --- pulse_width >= pulse_interval

    @pytest.mark.parametrize(
        "pulse_width, pulse_interval",
        [
            (2.0, 2.0),
            (3.0, 2.0),
        ],
    )
    def test_overlapping_pulse_width_raises_voltage(
        self, pulse_width: float, pulse_interval: float
    ):
        """pulse_width >= pulse_interval must raise ValueError."""
        with pytest.raises(ValueError, match="pulse_width"):
            pulse_train_voltage(20.0, 0.0, pulse_width, pulse_interval)

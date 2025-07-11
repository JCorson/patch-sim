"""
Tests for the protocols module.

This module contains unit tests for all protocol functions.
"""

import numpy as np
from ap_sim.protocols import (
    step_current,
    ramp_current,
    pulse_train,
    sinusoidal_current,
    chirp_current,
    noise_current,
    # Voltage clamp protocols
    step_voltage,
    ramp_voltage,
    pulse_train_voltage,
    iv_curve_protocol,
    activation_protocol,
)


class TestStepCurrent:
    """Test cases for step_current function."""

    def test_basic_step_current(self):
        """Test basic step current generation."""
        duration = 10.0  # ms
        current_amplitude = 5.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz (10 kHz for faster testing)

        current = step_current(
            duration=duration,
            current_amplitude=current_amplitude,
            sampling_frequency=sampling_frequency,
        )

        # Check that all values are equal to the amplitude
        assert np.allclose(current, current_amplitude)

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

    def test_step_with_delay(self):
        """Test step current with delayed start."""
        duration = 10.0  # ms
        current_amplitude = 3.0  # uA/cm^2
        step_start = 2.0  # ms
        step_duration = 5.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = step_current(
            duration=duration,
            current_amplitude=current_amplitude,
            step_start=step_start,
            step_duration=step_duration,
            sampling_frequency=sampling_frequency,
        )

        # Create time array for verification
        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        # Check zero current before step
        pre_step_mask = time_array < step_start
        assert np.allclose(current[pre_step_mask], 0.0)

        # Check step current during step
        step_mask = (time_array >= step_start) & (
            time_array <= step_start + step_duration
        )
        assert np.allclose(current[step_mask], current_amplitude)

        # Check zero current after step
        post_step_mask = time_array > step_start + step_duration
        assert np.allclose(current[post_step_mask], 0.0)

    def test_step_current_parameters(self):
        """Test step current with various parameters."""
        duration = 20.0  # ms
        current_amplitude = 10.0  # uA/cm^2

        current = step_current(duration=duration, current_amplitude=current_amplitude)

        # Check that current is within expected range
        assert np.min(current) >= min(0.0, current_amplitude) - 1e-10
        assert np.max(current) <= max(0.0, current_amplitude) + 1e-10


class TestRampCurrent:
    """Test cases for ramp_current function."""

    def test_basic_ramp_current(self):
        """Test basic ramp current generation."""
        duration = 10.0  # ms
        start_current = 0.0  # uA/cm^2
        end_current = 5.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            sampling_frequency=sampling_frequency,
        )

        # Check that current starts at start_current and ends at end_current
        assert np.isclose(current[0], start_current, atol=1e-10)
        assert np.isclose(current[-1], end_current, atol=1e-10)

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

    def test_ramp_with_delay(self):
        """Test ramp current with delayed start."""
        duration = 20.0  # ms
        start_current = 1.0  # uA/cm^2
        end_current = 6.0  # uA/cm^2
        ramp_start = 5.0  # ms
        ramp_duration = 10.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
            sampling_frequency=sampling_frequency,
        )

        # Create time array for verification
        time_step = 1000.0 / sampling_frequency
        time_array = np.arange(0, duration + time_step, time_step)

        # Check start current before ramp
        pre_ramp_mask = time_array < ramp_start
        assert np.allclose(current[pre_ramp_mask], start_current)

        # Check ramp progression
        ramp_mask = (time_array >= ramp_start) & (
            time_array <= ramp_start + ramp_duration
        )
        ramp_currents = current[ramp_mask]
        # Should be monotonically increasing (or decreasing)
        if end_current > start_current:
            # Allow for numerical precision
            assert np.all(np.diff(ramp_currents) >= -1e-10)
        else:
            assert np.all(np.diff(ramp_currents) <= 1e-10)

        # Check end current after ramp
        post_ramp_mask = time_array > ramp_start + ramp_duration
        assert np.allclose(current[post_ramp_mask], start_current)

    def test_linearity(self):
        """Test that ramp current is linear."""
        duration = 10.0  # ms
        start_current = -2.0  # uA/cm^2
        end_current = 8.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            sampling_frequency=sampling_frequency,
        )

        # Check linearity using linear fit
        time_array = np.linspace(0, duration, len(current))
        coefficients = np.polyfit(time_array, current, 1)
        expected_slope = (end_current - start_current) / duration

        assert np.isclose(coefficients[0], expected_slope, rtol=1e-6)
        assert np.isclose(coefficients[1], start_current, atol=1e-6)


class TestPulseTrain:
    """Test cases for pulse_train function."""

    def test_basic_pulse_train(self):
        """Test basic pulse train generation."""
        duration = 20.0  # ms
        pulse_amplitude = 4.0  # uA/cm^2
        pulse_width = 2.0  # ms
        pulse_interval = 5.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            sampling_frequency=sampling_frequency,
        )

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

        # Check that values are either zero or pulse amplitude
        unique_values = np.unique(current)
        expected_values = {0.0, pulse_amplitude}
        for val in unique_values:
            assert any(
                np.isclose(val, exp_val, atol=1e-10) for exp_val in expected_values
            )

    def test_pulse_timing(self):
        """Test pulse train timing."""
        duration = 15.0  # ms
        pulse_amplitude = 3.0  # uA/cm^2
        pulse_width = 1.0  # ms
        pulse_interval = 4.0  # ms
        train_start = 2.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=train_start,
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
                    assert np.allclose(current[pulse_mask], pulse_amplitude)

    def test_limited_pulses(self):
        """Test pulse train with limited number of pulses."""
        duration = 20.0  # ms
        pulse_amplitude = 2.0  # uA/cm^2
        pulse_width = 1.0  # ms
        pulse_interval = 3.0  # ms
        num_pulses = 3

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            num_pulses=num_pulses,
        )

        # Count number of transitions from zero to pulse amplitude
        transitions = np.sum(np.diff(current) != 0)
        # Should have transitions for each pulse (up and down)
        assert transitions <= 2 * num_pulses


class TestSinusoidalCurrent:
    """Test cases for sinusoidal_current function."""

    def test_basic_sinusoidal(self):
        """Test basic sinusoidal current generation."""
        duration = 10.0  # ms
        dc_offset = 2.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        frequency = 100.0  # Hz
        sampling_frequency = 10000.0  # Hz

        current = sinusoidal_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            sampling_frequency=sampling_frequency,
        )

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

        # Check DC component (mean should be close to dc_offset)
        assert np.isclose(np.mean(current), dc_offset, atol=0.1)

        # Check amplitude (range should be approximately 2 * amplitude)
        current_range = np.max(current) - np.min(current)
        assert np.isclose(current_range, 2 * amplitude, rtol=0.1)

    def test_sinusoidal_frequency(self):
        """Test sinusoidal frequency content."""
        duration = 100.0  # ms (longer for better frequency resolution)
        dc_offset = 0.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        frequency = 50.0  # Hz
        sampling_frequency = 5000.0  # Hz

        current = sinusoidal_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            sampling_frequency=sampling_frequency,
        )

        # Check that the signal oscillates around the DC offset
        assert np.abs(np.mean(current) - dc_offset) < 0.01

        # Check zero crossings (approximate frequency check)
        zero_crossings = np.sum(np.diff(np.sign(current - dc_offset)) != 0)
        expected_crossings = 2 * frequency * (duration / 1000.0)
        assert np.abs(zero_crossings - expected_crossings) < 4

    def test_phase_offset(self):
        """Test sinusoidal with phase offset."""
        duration = 10.0  # ms
        dc_offset = 0.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        frequency = 100.0  # Hz
        phase = np.pi / 2  # 90 degrees
        sampling_frequency = 10000.0  # Hz

        current = sinusoidal_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            phase=phase,
            sampling_frequency=sampling_frequency,
        )

        # With 90-degree phase shift, signal should start at maximum
        assert np.isclose(current[0], amplitude, atol=0.1)


class TestChirpCurrent:
    """Test cases for chirp_current function."""

    def test_basic_chirp(self):
        """Test basic chirp current generation."""
        duration = 20.0  # ms
        dc_offset = 1.0  # uA/cm^2
        amplitude = 0.5  # uA/cm^2
        start_frequency = 10.0  # Hz
        end_frequency = 100.0  # Hz
        sampling_frequency = 10000.0  # Hz

        current = chirp_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            start_frequency=start_frequency,
            end_frequency=end_frequency,
            sampling_frequency=sampling_frequency,
        )

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

        # Check DC component
        assert np.isclose(np.mean(current), dc_offset, atol=0.1)

        # Check amplitude bounds
        assert np.max(current) <= dc_offset + amplitude + 1e-10
        assert np.min(current) >= dc_offset - amplitude - 1e-10

    def test_chirp_frequency_progression(self):
        """Test that chirp frequency increases over time."""
        duration = 50.0  # ms
        dc_offset = 0.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        start_frequency = 5.0  # Hz
        end_frequency = 50.0  # Hz
        sampling_frequency = 2000.0  # Hz

        current = chirp_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            start_frequency=start_frequency,
            end_frequency=end_frequency,
            sampling_frequency=sampling_frequency,
        )

        # Analyze frequency content in different time windows
        first_half = current[: len(current) // 2]
        second_half = current[len(current) // 2 :]

        # Second half should have higher frequency content
        # This is a simplified check - in practice, you'd use FFT
        first_crossings = np.sum(np.diff(np.sign(first_half)) != 0)
        second_crossings = np.sum(np.diff(np.sign(second_half)) != 0)

        assert second_crossings > first_crossings


class TestNoiseCurrent:
    """Test cases for noise_current function."""

    def test_basic_noise(self):
        """Test basic noise current generation."""
        duration = 10.0  # ms
        mean_current = 2.0  # uA/cm^2
        std_current = 0.5  # uA/cm^2
        sampling_frequency = 10000.0  # Hz
        seed = 42

        current = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
            seed=seed,
        )

        # Check array length
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
        assert len(current) == expected_length

        # Check statistical properties
        assert np.isclose(np.mean(current), mean_current, atol=0.1)
        assert np.isclose(np.std(current), std_current, atol=0.1)

    def test_noise_reproducibility(self):
        """Test that noise is reproducible with same seed."""
        duration = 5.0  # ms
        mean_current = 1.0  # uA/cm^2
        std_current = 0.2  # uA/cm^2
        seed = 123

        current1 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=seed,
        )

        current2 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=seed,
        )

        # Should be identical with same seed
        assert np.allclose(current1, current2)

    def test_noise_different_seeds(self):
        """Test that different seeds produce different noise."""
        duration = 5.0  # ms
        mean_current = 1.0  # uA/cm^2
        std_current = 0.2  # uA/cm^2

        current1 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=1,
        )

        current2 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=2,
        )

        # Should be different with different seeds
        assert not np.allclose(current1, current2)

    def test_noise_distribution(self):
        """Test that noise follows Gaussian distribution."""
        duration = 100.0  # ms (longer for better statistics)
        mean_current = 0.0  # uA/cm^2
        std_current = 1.0  # uA/cm^2
        sampling_frequency = 1000.0  # Hz
        seed = 456

        current = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
            seed=seed,
        )

        # Check that values roughly follow normal distribution
        # About 68% should be within 1 std, 95% within 2 std
        within_1_std = np.sum(np.abs(current - mean_current) <= std_current)
        within_2_std = np.sum(np.abs(current - mean_current) <= 2 * std_current)

        assert within_1_std / len(current) > 0.6  # Should be ~68%
        assert within_2_std / len(current) > 0.9  # Should be ~95%


class TestProtocolIntegration:
    """Integration tests for current clamp protocols."""

    def test_all_protocols_generate_arrays(self):
        """Test that all protocols generate valid arrays."""
        duration = 5.0  # ms
        sampling_frequency = 10000.0  # Hz

        protocols = [
            lambda: step_current(duration, 1.0, sampling_frequency=sampling_frequency),
            lambda: ramp_current(
                duration, 0.0, 2.0, sampling_frequency=sampling_frequency
            ),
            lambda: pulse_train(
                duration, 1.0, 1.0, 2.5, sampling_frequency=sampling_frequency
            ),
            lambda: sinusoidal_current(
                duration, 1.0, 0.5, 50.0, sampling_frequency=sampling_frequency
            ),
            lambda: chirp_current(
                duration, 1.0, 0.5, 10.0, 100.0, sampling_frequency=sampling_frequency
            ),
            lambda: noise_current(
                duration, 1.0, 0.1, sampling_frequency=sampling_frequency, seed=42
            ),
        ]

        for protocol_func in protocols:
            current = protocol_func()

            # Check that protocol generates a valid array
            assert isinstance(current, np.ndarray)
            assert len(current) > 0
            assert np.all(np.isfinite(current))

            # Check array length
            expected_length = int(duration / (1000.0 / sampling_frequency)) + 1
            assert len(current) == expected_length

    def test_different_sampling_frequencies(self):
        """Test protocols with different sampling frequencies."""
        duration = 5.0  # ms
        frequencies = [1000.0, 10000.0, 100000.0]  # 1 kHz, 10 kHz, 100 kHz

        for freq in frequencies:
            current = step_current(duration, 1.0, sampling_frequency=freq)
            expected_length = int(duration / (1000.0 / freq)) + 1
            assert len(current) == expected_length

    def test_edge_cases(self):
        """Test edge cases for protocol functions."""
        # Very short duration
        short_duration = 0.1  # ms
        current = step_current(short_duration, 1.0, sampling_frequency=10000.0)
        assert len(current) >= 1

        # Very small amplitude
        small_amp = 1e-6  # uA/cm^2
        current = step_current(1.0, small_amp, sampling_frequency=10000.0)
        assert np.allclose(current, small_amp)


# =============================================================================
# Voltage Clamp Protocol Tests
# =============================================================================


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
        # Should be monotonically increasing (or decreasing)
        if end_voltage > start_voltage:
            # Allow for numerical precision
            assert np.all(np.diff(ramp_voltages) >= -1e-10)
        else:
            assert np.all(np.diff(ramp_voltages) <= 1e-10)

        # Check holding voltage after ramp
        post_ramp_mask = time_array > ramp_start + ramp_duration
        assert np.allclose(voltage[post_ramp_mask], holding_voltage)


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

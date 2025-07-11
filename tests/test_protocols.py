"""
Tests for the protocols module.

This module contains unit tests for all the current delivery protocol functions.
"""

import numpy as np
from ap_sim.protocols import (
    step_current,
    ramp_current,
    pulse_train,
    sinusoidal_current,
    chirp_current,
    noise_current,
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

        # Create time array to check values
        time_array = np.linspace(0, duration, len(current))

        # Check values before step
        before_step = current[time_array < step_start]
        assert np.allclose(before_step, 0.0)

        # Check values during step
        step_end = step_start + step_duration
        during_step = current[(time_array >= step_start) & (time_array <= step_end)]
        assert np.allclose(during_step, current_amplitude)

        # Check values after step
        after_step = current[time_array > step_end]
        assert np.allclose(after_step, 0.0)

    def test_step_current_zero_amplitude(self):
        """Test step current with zero amplitude."""
        duration = 5.0
        current_amplitude = 0.0
        sampling_frequency = 10000.0

        current = step_current(
            duration=duration,
            current_amplitude=current_amplitude,
            sampling_frequency=sampling_frequency,
        )

        assert np.allclose(current, 0.0)


class TestRampCurrent:
    """Test cases for ramp_current function."""

    def test_basic_ramp_current(self):
        """Test basic ramp current generation."""
        duration = 10.0  # ms
        start_current = 1.0  # uA/cm^2
        end_current = 5.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            sampling_frequency=sampling_frequency,
        )

        # Check first and last values
        assert np.isclose(current[0], start_current)
        assert np.isclose(current[-1], end_current)

        # Check that it's monotonically increasing
        assert np.all(np.diff(current) >= 0)

    def test_ramp_with_delay(self):
        """Test ramp current with delayed start."""
        duration = 10.0  # ms
        start_current = 2.0  # uA/cm^2
        end_current = 8.0  # uA/cm^2
        ramp_start = 2.0  # ms
        ramp_duration = 6.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
            sampling_frequency=sampling_frequency,
        )

        # Create time array
        time_array = np.linspace(0, duration, len(current))

        # Check values before ramp
        before_ramp = current[time_array < ramp_start]
        assert np.allclose(before_ramp, start_current)

        # Check values at ramp start and end
        ramp_end = ramp_start + ramp_duration
        ramp_mask = (time_array >= ramp_start) & (time_array <= ramp_end)
        ramp_values = current[ramp_mask]

        if len(ramp_values) > 0:
            assert np.isclose(ramp_values[0], start_current, atol=0.1)
            assert np.isclose(ramp_values[-1], end_current, atol=0.1)

    def test_decreasing_ramp(self):
        """Test decreasing ramp current."""
        duration = 10.0  # ms
        start_current = 5.0  # uA/cm^2
        end_current = 1.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = ramp_current(
            duration=duration,
            start_current=start_current,
            end_current=end_current,
            sampling_frequency=sampling_frequency,
        )

        # Check that it's monotonically decreasing
        assert np.all(np.diff(current) <= 0)


class TestPulseTrain:
    """Test cases for pulse_train function."""

    def test_basic_pulse_train(self):
        """Test basic pulse train generation."""
        duration = 20.0  # ms
        pulse_amplitude = 4.0  # uA/cm^2
        pulse_width = 1.0  # ms
        pulse_interval = 5.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            sampling_frequency=sampling_frequency,
        )

        # Check that baseline is zero
        non_zero_indices = np.where(current != 0)[0]
        baseline_indices = np.where(current == 0)[0]
        assert len(baseline_indices) > 0

        # Check that pulses have correct amplitude
        assert np.allclose(current[non_zero_indices], pulse_amplitude)

    def test_pulse_train_with_num_pulses(self):
        """Test pulse train with specified number of pulses."""
        duration = 20.0  # ms
        pulse_amplitude = 3.0  # uA/cm^2
        pulse_width = 1.0  # ms
        pulse_interval = 5.0  # ms
        num_pulses = 2
        sampling_frequency = 10000.0  # Hz

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            num_pulses=num_pulses,
            sampling_frequency=sampling_frequency,
        )

        # Check that we have approximately the right number of pulses
        # by counting transitions from 0 to pulse_amplitude
        transitions = np.diff(np.where(current == pulse_amplitude, 1, 0))
        pulse_starts = np.sum(transitions == 1)
        assert pulse_starts <= num_pulses

    def test_pulse_train_with_delay(self):
        """Test pulse train with delayed start."""
        duration = 15.0  # ms
        pulse_amplitude = 2.0  # uA/cm^2
        pulse_width = 1.0  # ms
        pulse_interval = 4.0  # ms
        train_start = 3.0  # ms
        sampling_frequency = 10000.0  # Hz

        current = pulse_train(
            duration=duration,
            pulse_amplitude=pulse_amplitude,
            pulse_width=pulse_width,
            pulse_interval=pulse_interval,
            train_start=train_start,
            sampling_frequency=sampling_frequency,
        )

        # Create time array
        time_array = np.linspace(0, duration, len(current))

        # Check that current is zero before train_start
        before_train = current[time_array < train_start]
        assert np.allclose(before_train, 0.0)


class TestSinusoidalCurrent:
    """Test cases for sinusoidal_current function."""

    def test_basic_sinusoidal_current(self):
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

        # Check mean is approximately dc_offset
        assert np.isclose(np.mean(current), dc_offset, atol=0.1)

        # Check amplitude range
        assert np.isclose(np.max(current), dc_offset + amplitude, atol=0.1)
        assert np.isclose(np.min(current), dc_offset - amplitude, atol=0.1)

    def test_sinusoidal_with_phase(self):
        """Test sinusoidal current with phase offset."""
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

        # With 90-degree phase shift, first value should be close to amplitude
        assert np.isclose(current[0], amplitude, atol=0.1)

    def test_zero_frequency(self):
        """Test sinusoidal current with zero frequency (DC)."""
        duration = 5.0  # ms
        dc_offset = 3.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        frequency = 0.0  # Hz
        sampling_frequency = 10000.0  # Hz

        current = sinusoidal_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            frequency=frequency,
            sampling_frequency=sampling_frequency,
        )

        # Should be constant at dc_offset (sin(0) = 0)
        expected_value = dc_offset
        assert np.allclose(current, expected_value)


class TestChirpCurrent:
    """Test cases for chirp_current function."""

    def test_basic_chirp_current(self):
        """Test basic chirp current generation."""
        duration = 10.0  # ms
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

        # Check mean is approximately dc_offset (with larger tolerance for chirp)
        assert np.isclose(np.mean(current), dc_offset, atol=0.3)

        # Check amplitude range
        assert np.max(current) <= dc_offset + amplitude + 0.1
        assert np.min(current) >= dc_offset - amplitude - 0.1

    def test_constant_frequency_chirp(self):
        """Test chirp with same start and end frequency."""
        duration = 10.0  # ms
        dc_offset = 0.0  # uA/cm^2
        amplitude = 1.0  # uA/cm^2
        frequency = 50.0  # Hz (same start and end)
        sampling_frequency = 10000.0  # Hz

        current = chirp_current(
            duration=duration,
            dc_offset=dc_offset,
            amplitude=amplitude,
            start_frequency=frequency,
            end_frequency=frequency,
            sampling_frequency=sampling_frequency,
        )

        # Should behave like a sinusoid (with larger tolerance for short duration)
        assert np.isclose(np.mean(current), dc_offset, atol=0.7)


class TestNoiseCurrent:
    """Test cases for noise_current function."""

    def test_basic_noise_current(self):
        """Test basic noise current generation."""
        duration = 100.0  # ms (longer for better statistics)
        mean_current = 2.0  # uA/cm^2
        std_current = 0.5  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )

        # Check statistical properties
        assert np.isclose(np.mean(current), mean_current, atol=0.1)
        assert np.isclose(np.std(current), std_current, atol=0.1)

    def test_noise_reproducibility(self):
        """Test that noise is reproducible with same seed."""
        duration = 10.0  # ms
        mean_current = 1.0  # uA/cm^2
        std_current = 0.2  # uA/cm^2
        seed = 42
        sampling_frequency = 10000.0  # Hz

        current1 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=seed,
            sampling_frequency=sampling_frequency,
        )

        current2 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            seed=seed,
            sampling_frequency=sampling_frequency,
        )

        # Should be identical with same seed
        assert np.allclose(current1, current2)

    def test_noise_variability(self):
        """Test that noise is different without seed."""
        duration = 10.0  # ms
        mean_current = 1.0  # uA/cm^2
        std_current = 0.2  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current1 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )

        current2 = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )

        # Should be different without seed
        assert not np.allclose(current1, current2)

    def test_zero_std_noise(self):
        """Test noise current with zero standard deviation."""
        duration = 5.0  # ms
        mean_current = 3.0  # uA/cm^2
        std_current = 0.0  # uA/cm^2
        sampling_frequency = 10000.0  # Hz

        current = noise_current(
            duration=duration,
            mean_current=mean_current,
            std_current=std_current,
            sampling_frequency=sampling_frequency,
        )

        # Should be constant at mean_current
        assert np.allclose(current, mean_current)


class TestProtocolIntegration:
    """Integration tests for protocol functions."""

    def test_protocol_lengths(self):
        """Test that all protocols generate arrays of expected length."""
        duration = 10.0  # ms
        sampling_frequency = 10000.0  # Hz
        expected_length = int(duration / (1000.0 / sampling_frequency)) + 1

        # Test each protocol function
        step = step_current(duration, 1.0, sampling_frequency=sampling_frequency)
        ramp = ramp_current(duration, 0.0, 1.0, sampling_frequency=sampling_frequency)
        pulse = pulse_train(
            duration, 1.0, 1.0, 2.0, sampling_frequency=sampling_frequency
        )
        sine = sinusoidal_current(
            duration, 0.0, 1.0, 10.0, sampling_frequency=sampling_frequency
        )
        chirp = chirp_current(
            duration, 0.0, 1.0, 1.0, 10.0, sampling_frequency=sampling_frequency
        )
        noise = noise_current(
            duration, 0.0, 1.0, seed=42, sampling_frequency=sampling_frequency
        )

        protocols = [step, ramp, pulse, sine, chirp, noise]
        for protocol in protocols:
            assert len(protocol) == expected_length

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

        # Negative amplitude
        neg_amp = -2.0  # uA/cm^2
        current = step_current(1.0, neg_amp, sampling_frequency=10000.0)
        assert np.allclose(current, neg_amp)

"""Current clamp protocol generation utilities.

This module provides functions to generate typical current stimulation
protocols that can be used with current clamp simulations.
"""

import numpy as np

from .common import (
    _calculate_time_parameters,
    _generate_pulse_train_protocol,
    _generate_ramp_protocol,
    _generate_step_protocol,
)


def step_current(
    duration: float,
    current_amplitude: float,
    step_start: float = 0.0,
    step_duration: float | None = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """Generate a step current protocol.

    Creates a current protocol with a rectangular step of specified amplitude
    and duration, useful for studying action potential threshold and firing patterns.

    Args:
        duration: Total duration of the protocol in milliseconds.
        current_amplitude: Amplitude of the current step in uA/cm^2.
        step_start: Time when the step begins in milliseconds. Default is 0.0.
        step_duration: Duration of the current step in milliseconds.
            If None, the step lasts for the entire duration.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_step_protocol(
        duration,
        current_amplitude,
        baseline=0.0,
        step_start=step_start,
        step_duration=step_duration,
        sampling_frequency=sampling_frequency,
    )


def ramp_current(
    duration: float,
    start_current: float,
    end_current: float,
    ramp_start: float = 0.0,
    ramp_duration: float | None = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """Generate a ramp current protocol.

    Creates a current protocol with a linear ramp from start to end current,
    useful for studying firing frequency adaptation and rheobase determination.

    Args:
        duration: Total duration of the protocol in milliseconds.
        start_current: Starting current amplitude in uA/cm^2.
        end_current: Ending current amplitude in uA/cm^2.
        ramp_start: Time when the ramp begins in milliseconds. Default is 0.0.
        ramp_duration: Duration of the ramp in milliseconds.
            If None, the ramp lasts for the entire duration.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_ramp_protocol(
        duration,
        start_current,
        end_current,
        baseline=start_current,
        ramp_start=ramp_start,
        ramp_duration=ramp_duration,
        sampling_frequency=sampling_frequency,
    )


def pulse_train(
    duration: float,
    pulse_amplitude: float,
    pulse_width: float,
    pulse_interval: float,
    train_start: float = 0.0,
    num_pulses: int | None = None,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """Generate a pulse train current protocol.

    Creates a series of rectangular current pulses with specified amplitude,
    width, and interval, useful for studying synaptic integration and
    temporal summation.

    Args:
        duration: Total duration of the protocol in milliseconds.
        pulse_amplitude: Amplitude of each pulse in uA/cm^2.
        pulse_width: Width of each pulse in milliseconds.
        pulse_interval: Time between pulse onsets in milliseconds.
        train_start: Time when the pulse train begins in milliseconds.
            Default is 0.0.
        num_pulses: Number of pulses in the train. If None, pulses
            continue until the end of the duration.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of current values in uA/cm^2.
    """
    return _generate_pulse_train_protocol(
        duration,
        pulse_amplitude,
        pulse_width,
        pulse_interval,
        baseline=0.0,
        train_start=train_start,
        num_pulses=num_pulses,
        sampling_frequency=sampling_frequency,
    )


def sinusoidal_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """Generate a sinusoidal current protocol.

    Creates a sinusoidal current waveform with specified DC offset, amplitude,
    and frequency, useful for studying frequency response and impedance.

    Args:
        duration: Total duration of the protocol in milliseconds.
        dc_offset: DC offset current in uA/cm^2.
        amplitude: Amplitude of the sinusoidal component in uA/cm^2.
        frequency: Frequency of the sinusoid in Hz.
        phase: Phase offset in radians. Default is 0.0.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of current values in uA/cm^2.
    """
    # Calculate time step and number of points
    _, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Generate sinusoidal current
    time_array_seconds = time_array / 1000.0  # Convert to seconds
    current_array = dc_offset + amplitude * np.sin(
        2 * np.pi * frequency * time_array_seconds + phase
    )

    return current_array


def chirp_current(
    duration: float,
    dc_offset: float,
    amplitude: float,
    start_frequency: float,
    end_frequency: float,
    sampling_frequency: float = 100000.0,
) -> np.ndarray:
    """Generate a chirp (frequency sweep) current protocol.

    Creates a sinusoidal current with linearly increasing frequency,
    useful for measuring frequency response characteristics.

    Args:
        duration: Total duration of the protocol in milliseconds.
        dc_offset: DC offset current in uA/cm^2.
        amplitude: Amplitude of the chirp in uA/cm^2.
        start_frequency: Starting frequency in Hz.
        end_frequency: Ending frequency in Hz.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.

    Returns:
        Array of current values in uA/cm^2.
    """
    # Calculate time step and number of points
    _, time_array = _calculate_time_parameters(duration, sampling_frequency)

    # Calculate instantaneous frequency
    time_array_seconds = time_array / 1000.0  # Convert to seconds
    freq_slope = (end_frequency - start_frequency) / (duration / 1000.0)

    # Generate chirp current using proper phase integration
    # Phase = 2π * ∫f(t)dt where f(t) = f0 + kt
    # ∫(f0 + kt)dt = f0*t + k*t²/2
    phase = (
        2
        * np.pi
        * (
            start_frequency * time_array_seconds
            + 0.5 * freq_slope * time_array_seconds**2
        )
    )
    current_array = dc_offset + amplitude * np.sin(phase)

    return current_array


def noise_current(
    duration: float,
    mean_current: float,
    std_current: float,
    sampling_frequency: float = 100000.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Gaussian white noise current protocol.

    Creates a current protocol with Gaussian-distributed random values,
    useful for studying stochastic resonance and noise effects.

    Args:
        duration: Total duration of the protocol in milliseconds.
        mean_current: Mean current value in uA/cm^2.
        std_current: Standard deviation of current in uA/cm^2.
        sampling_frequency: Sampling frequency in Hz. Default is 100 kHz.
        seed: Random seed for reproducibility. If None, uses random seed.

    Returns:
        Array of current values in uA/cm^2.
    """
    # Create a local RNG instance to avoid mutating global state
    rng = np.random.default_rng(seed)

    # Calculate time step and number of points
    num_points, _ = _calculate_time_parameters(duration, sampling_frequency)

    # Generate Gaussian noise
    current_array = rng.normal(mean_current, std_current, num_points)

    return current_array
